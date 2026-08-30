"""
Read-only completeness/consistency audit of completions_en_full572_corrected.json
for all 3 models -- run BEFORE trusting that old No-defence harmful responses
can be reused for the Exp3 defence protocol (per explicit instruction: do not
infer completeness from 31_merge_corrected_completions.py's write-time guard
alone, actually check the file). CPU-only, no `pipeline` import.

Checks per model (each independently pass/fail, none inferred from another):
  1. total record count == 4576 (572 instructions x 8 conditions)
  2. unique instruction id count == 572
  3. every id has exactly 8 condition rows
  4. the condition set is exactly {plain, placebo} union active_mechanisms (V2,
     read dynamically via _taxonomy_v2_loader -- not hardcoded)
  5. no duplicate (id, condition) pairs
  6. direction_ids/validation_ids/test_ids coverage counts (each id present at
     all, regardless of condition completeness, which check 3 already covers
     per-id) equal 300/72/200 respectively
  7. no missing or empty 'response' field on any record
  8. every record has a 'generation_tokens' field
  9. for active-mechanism rows, the record's 'mechanism' field (if present)
     matches the V2 taxonomy's category for that condition
  10. neither 'instruction_hierarchy' nor 'fictional_framing' appears as a
      condition anywhere
Cross-model check: all 3 models used the exact same id set and the exact
same condition set (not just the same counts).

Usage:
  python scripts/audits/audit_corrected_completions.py --output_dir output --lang en --suffix _full572_corrected
"""
import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
from _taxonomy_v2_loader import load_taxonomy_v2

SPLITS_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'data', 'splits.json')
MODELS = ['Qwen2.5-7B-Instruct', 'Meta-Llama-3.1-8B-Instruct', 'gemma-2-9b-it']
NON_MECHANISM_CONDITIONS = {'plain', 'placebo'}
STALE_MECHANISMS = {'instruction_hierarchy', 'fictional_framing'}


def audit_one(path, active_mechanisms, mechanism_of, direction_ids, validation_ids, test_ids):
    result = {'path': path, 'exists': os.path.exists(path)}
    if not result['exists']:
        result['checks'] = {'FATAL': 'file does not exist'}
        result['all_passed'] = False
        return result

    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    completions = data['completions'] if isinstance(data, dict) and 'completions' in data else data

    checks = {}
    expected_conditions = NON_MECHANISM_CONDITIONS | set(active_mechanisms)

    # 1. total record count
    checks['total_record_count'] = {'value': len(completions), 'expected': 4576, 'pass': len(completions) == 4576}

    # 2/3. unique ids and per-id condition completeness
    ids_seen = {}
    for c in completions:
        ids_seen.setdefault(c['id'], []).append(c.get('condition'))
    checks['unique_id_count'] = {'value': len(ids_seen), 'expected': 572, 'pass': len(ids_seen) == 572}
    incomplete_ids = {i: len(conds) for i, conds in ids_seen.items() if len(conds) != 8}
    checks['every_id_has_8_conditions'] = {'pass': len(incomplete_ids) == 0,
                                            'n_incomplete': len(incomplete_ids),
                                            'sample_incomplete': dict(list(incomplete_ids.items())[:10])}

    # 4. condition set exactly matches expected
    conditions_present = set()
    for conds in ids_seen.values():
        conditions_present.update(conds)
    checks['condition_set_matches_v2'] = {
        'present': sorted(conditions_present), 'expected': sorted(expected_conditions),
        'missing': sorted(expected_conditions - conditions_present),
        'unexpected': sorted(conditions_present - expected_conditions),
        'pass': conditions_present == expected_conditions,
    }

    # 5. duplicate (id, condition) pairs
    pair_counts = {}
    for c in completions:
        key = (c['id'], c.get('condition'))
        pair_counts[key] = pair_counts.get(key, 0) + 1
    dupes = {f"{k[0]}|{k[1]}": v for k, v in pair_counts.items() if v > 1}
    checks['no_duplicate_id_condition_pairs'] = {'pass': len(dupes) == 0, 'n_duplicates': len(dupes),
                                                  'sample_duplicates': dict(list(dupes.items())[:10])}

    # 6. split coverage
    id_set = set(ids_seen.keys())
    cov = {
        'direction_ids': len(id_set & direction_ids), 'expected_direction_ids': len(direction_ids),
        'validation_ids': len(id_set & validation_ids), 'expected_validation_ids': len(validation_ids),
        'test_ids': len(id_set & test_ids), 'expected_test_ids': len(test_ids),
    }
    cov['pass'] = (cov['direction_ids'] == len(direction_ids) and
                   cov['validation_ids'] == len(validation_ids) and
                   cov['test_ids'] == len(test_ids))
    checks['split_coverage'] = cov

    # 7. missing/empty response
    bad_response = [c['id'] + '|' + str(c.get('condition')) for c in completions
                    if not c.get('response') or not str(c.get('response')).strip()]
    checks['no_missing_or_empty_response'] = {'pass': len(bad_response) == 0, 'n_bad': len(bad_response),
                                               'sample': bad_response[:10]}

    # 8. generation_tokens field present
    missing_gt = [c['id'] + '|' + str(c.get('condition')) for c in completions if 'generation_tokens' not in c]
    checks['generation_tokens_field_present'] = {'pass': len(missing_gt) == 0, 'n_missing': len(missing_gt),
                                                  'sample': missing_gt[:10]}

    # 9. mechanism field consistency with V2 taxonomy (only for active-mechanism rows that carry a 'mechanism' field)
    mismatches = []
    for c in completions:
        cond = c.get('condition')
        if cond in active_mechanisms and 'mechanism' in c:
            expected_mech = mechanism_of[cond]
            if c['mechanism'] != expected_mech:
                mismatches.append(f"{c['id']}|{cond}: field={c['mechanism']!r} expected={expected_mech!r}")
    checks['mechanism_field_consistent_with_v2'] = {'pass': len(mismatches) == 0, 'n_mismatches': len(mismatches),
                                                     'sample': mismatches[:10]}

    # 10. no stale mechanisms
    stale_present = conditions_present & STALE_MECHANISMS
    checks['no_stale_mechanisms'] = {'pass': len(stale_present) == 0, 'found': sorted(stale_present)}

    result['checks'] = checks
    result['all_passed'] = all(v.get('pass', False) for v in checks.values())
    result['id_set_for_cross_model_check'] = sorted(id_set)
    result['condition_set_for_cross_model_check'] = sorted(conditions_present)
    return result


def main(args):
    taxonomy = load_taxonomy_v2()
    active_mechanisms = taxonomy['active_mechanisms']
    mechanism_of = taxonomy['mechanism_of']
    print(f"Taxonomy v2: active_mechanisms={active_mechanisms}\n")

    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    direction_ids, validation_ids, test_ids = set(splits['direction_ids']), set(splits['validation_ids']), set(splits['test_ids'])

    results = {'lang': args.lang, 'suffix': args.suffix, 'per_model': {}}
    for model_alias in MODELS:
        path = os.path.join(args.output_dir, model_alias, f'completions_{args.lang}{args.suffix}.json')
        print(f"=== {model_alias} ===")
        r = audit_one(path, active_mechanisms, mechanism_of, direction_ids, validation_ids, test_ids)
        results['per_model'][model_alias] = r
        if not r['exists']:
            print("  MISSING FILE\n")
            continue
        for name, c in r['checks'].items():
            status = 'PASS' if c.get('pass') else 'FAIL'
            print(f"  [{status}] {name}: {c}")
        print(f"  ALL_PASSED: {r['all_passed']}\n")

    # cross-model check
    existing = {m: r for m, r in results['per_model'].items() if r['exists']}
    cross = {}
    if len(existing) == len(MODELS):
        id_sets = [set(r['id_set_for_cross_model_check']) for r in existing.values()]
        cond_sets = [set(r['condition_set_for_cross_model_check']) for r in existing.values()]
        cross['same_id_set_across_models'] = all(s == id_sets[0] for s in id_sets)
        cross['same_condition_set_across_models'] = all(s == cond_sets[0] for s in cond_sets)
    else:
        cross['skipped_reason'] = f"only {len(existing)}/{len(MODELS)} model files exist"
    results['cross_model_check'] = cross
    print("=== cross-model check ===")
    print(json.dumps(cross, indent=2))

    overall_pass = all(r.get('all_passed', False) for r in results['per_model'].values()) and \
        cross.get('same_id_set_across_models', False) and cross.get('same_condition_set_across_models', False)
    results['overall_pass'] = overall_pass
    print(f"\nOVERALL_PASS: {overall_pass}")

    out_dir = os.path.join(args.output_dir, 'canonical_v2')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'experiment3_corrected_completions_audit.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=os.path.join(SCRIPT_DIR, '..', '..', 'output'))
    parser.add_argument('--lang',       type=str, default='en')
    parser.add_argument('--suffix',     type=str, default='_full572_corrected')
    args = parser.parse_args()
    main(args)
