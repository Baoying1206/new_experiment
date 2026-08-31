"""
Joins one (model, method) validation run's generation JSONL (3,648 full
per-record rows: 4 alphas x (72 harmful + 80 benign) x 6 templates) with its
deduplicated WildGuard judgement JSONL (one row per UNIQUE (instruction_en,
response) content, per the content-cache design already authorized for
run_judge), producing a fully-expanded 3,648-row joined table and a
per-alpha/per-template summary. Pure CPU/Python -- no GPU, does not call
run_judge or generate anything new.

The canonical judge_cache_key function is imported from _defence_metrics.py
(NEVER reimplemented) -- every generation record's key is recomputed fresh
via that same function; the file never trusts row order or any key stored
on the generation side (there isn't one). This script deliberately imports
ONLY _defence_metrics (torch-free) rather than 40_defence_generation_driver.py
directly -- importing script 40 pulls in torch (its own, plus
35_common_direction_coverage_audit.py's and 37_defence_directions_and_hooks.py's,
both imported eagerly at module level by script 40), which is unavailable
outside the GPU venv and unnecessary for this purely CPU/JSON join step.
direction_config_hash/generation_config_hash are read off the generation
records themselves (already embedded per-record at generation time) rather
than recomputed, both for this reason and because it doubles as an extra
consistency check (all 3,648 records must agree on one value each).

Alpha is NOT selected here. This script produces exactly one (model,
method)'s validation summary; alpha freezing requires the No-defence
benign baseline + all 4 methods' summaries for the same model (a later,
separate step).

Usage:
  python scripts/41_join_and_summarize_defence_validation.py \
      --model_idx 1 --method global --output_path output
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)
import _defence_metrics as dm  # torch-free: canonical judge_cache_key, load_jsonl, metric fns
from _taxonomy_v2_loader import load_taxonomy_v2, DEFAULT_TEMPLATES_PATH

MODELS = dm.MODEL_PATHS
ALPHAS = dm.VALIDATION_ALPHAS
N_HARMFUL_IDS, N_BENIGN_IDS, N_TEMPLATES = 72, 80, 6
EXPECTED_HARMFUL = len(ALPHAS) * N_HARMFUL_IDS * N_TEMPLATES   # 1728
EXPECTED_BENIGN = len(ALPHAS) * N_BENIGN_IDS * N_TEMPLATES     # 1920
EXPECTED_TOTAL = EXPECTED_HARMFUL + EXPECTED_BENIGN            # 3648


def sha256_of_file(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


class JudgeKeyCollisionError(RuntimeError):
    pass


def check_no_key_collisions(gen_rows):
    """For every judge_cache_key shared by multiple generation records, their
    (instruction_en, response) must be byte-identical -- otherwise the hash
    has a real collision or the key computation is inconsistent between
    generate-time and this script. Raises immediately (does not degrade to
    a soft check) since a collision would silently corrupt every downstream
    metric."""
    key_to_content = {}
    collisions = []
    for r in gen_rows:
        key = dm.judge_cache_key(r['instruction_en'], r['response'])
        content = (r['instruction_en'], r['response'])
        if key in key_to_content and key_to_content[key] != content:
            collisions.append({'judge_cache_key': key, 'content_a': key_to_content[key], 'content_b': content})
        else:
            key_to_content[key] = content
    if collisions:
        raise JudgeKeyCollisionError(
            f"{len(collisions)} judge_cache_key collision(s): same key maps to different "
            f"(instruction_en, response) content -- hash collision or key-computation bug. "
            f"Stopping immediately, no output written. Sample: {collisions[:3]}"
        )
    return key_to_content  # key -> canonical (instruction_en, response)


def preflight_generation_checks(gen_rows):
    checks = {}
    checks['total_count'] = {'value': len(gen_rows), 'expected': EXPECTED_TOTAL, 'pass': len(gen_rows) == EXPECTED_TOTAL}

    harmful = [r for r in gen_rows if r['benign_or_harmful'] == 'harmful']
    benign = [r for r in gen_rows if r['benign_or_harmful'] == 'benign']
    checks['harmful_count'] = {'value': len(harmful), 'expected': EXPECTED_HARMFUL, 'pass': len(harmful) == EXPECTED_HARMFUL}
    checks['benign_count'] = {'value': len(benign), 'expected': EXPECTED_BENIGN, 'pass': len(benign) == EXPECTED_BENIGN}

    keys = [r['record_key'] for r in gen_rows]
    dupes = [k for k, c in Counter(keys).items() if c > 1]
    checks['no_duplicate_record_key'] = {'pass': len(dupes) == 0, 'n_duplicates': len(dupes), 'sample': dupes[:5]}

    harmful_counts = Counter((r['alpha'], r['template']) for r in harmful)
    bad_harmful = {k: v for k, v in harmful_counts.items() if v != N_HARMFUL_IDS}
    checks['harmful_alpha_template_counts'] = {
        'pass': len(harmful_counts) == len(ALPHAS) * N_TEMPLATES and len(bad_harmful) == 0,
        'n_combos_found': len(harmful_counts), 'n_combos_expected': len(ALPHAS) * N_TEMPLATES,
        'bad_combos': {str(k): v for k, v in bad_harmful.items()},
    }

    benign_counts = Counter((r['alpha'], r['template']) for r in benign)
    bad_benign = {k: v for k, v in benign_counts.items() if v != N_BENIGN_IDS}
    checks['benign_alpha_template_counts'] = {
        'pass': len(benign_counts) == len(ALPHAS) * N_TEMPLATES and len(bad_benign) == 0,
        'n_combos_found': len(benign_counts), 'n_combos_expected': len(ALPHAS) * N_TEMPLATES,
        'bad_combos': {str(k): v for k, v in bad_benign.items()},
    }
    return checks


def preflight_judgement_checks(judge_rows):
    checks = {}
    keys = [j['judge_cache_key'] for j in judge_rows]
    dupes = [k for k, c in Counter(keys).items() if c > 1]
    checks['no_duplicate_judge_key'] = {'pass': len(dupes) == 0, 'n_duplicates': len(dupes), 'sample': dupes[:5]}

    n_parse_success = sum(1 for j in judge_rows if j['parse_success'])
    n_parse_failure = len(judge_rows) - n_parse_success
    checks['parse_success_count'] = {'value': n_parse_success, 'total': len(judge_rows)}
    checks['parse_failure_count'] = {'value': n_parse_failure, 'pass': n_parse_failure == 0}
    return checks, n_parse_failure


def join_generation_and_judgement(gen_rows, judge_lookup):
    """judge_lookup: judge_cache_key -> judgement dict. Every generation
    record's key is recomputed via dm.judge_cache_key -- never assumed."""
    joined, missing_keys = [], []
    for r in gen_rows:
        key = dm.judge_cache_key(r['instruction_en'], r['response'])
        if key not in judge_lookup:
            missing_keys.append(key)
            continue
        j = judge_lookup[key]
        joined_row = dict(r)
        joined_row.update({
            'judge_cache_key': key,
            'request_harmful': j.get('request_harmful'), 'refusal': j.get('refusal'),
            'response_harmful': j.get('response_harmful'), 'parse_success': j.get('parse_success'),
            'parse_error_reason': j.get('parse_error_reason'),
            'judge_model_version': dm.JUDGE_MODEL_VERSION, 'judge_prompt_version': dm.JUDGE_PROMPT_VERSION,
        })
        joined.append(joined_row)
    return joined, missing_keys


def build_join_audit(gen_path, judge_path, gen_rows, judge_rows, joined_rows, missing_keys,
                      gen_checks, judge_checks, n_parse_failure, model_alias, method,
                      taxonomy, direction_config_hash, generation_config_hash, git_commit):
    unique_gen_keys = {dm.judge_cache_key(r['instruction_en'], r['response']) for r in gen_rows}
    judge_key_set = {j['judge_cache_key'] for j in judge_rows}
    orphan_keys = sorted(judge_key_set - unique_gen_keys)

    joined_keys = [r['record_key'] for r in joined_rows]
    joined_dupes = [k for k, c in Counter(joined_keys).items() if c > 1]
    n_missing_labels = sum(1 for r in joined_rows if r.get('parse_success') is None
                            or r.get('refusal') is None or r.get('request_harmful') is None
                            or r.get('response_harmful') is None)

    n_cache_hits = len(gen_rows) - len(unique_gen_keys)
    cache_hit_rate = n_cache_hits / len(gen_rows) if gen_rows else None

    join_checks = {
        'joined_rows_equal_generation_rows': {'pass': len(joined_rows) == len(gen_rows),
                                               'joined': len(joined_rows), 'generation': len(gen_rows)},
        'no_missing_keys': {'pass': len(missing_keys) == 0, 'n_missing': len(missing_keys), 'sample': missing_keys[:5]},
        'no_orphan_keys': {'pass': len(orphan_keys) == 0, 'n_orphan': len(orphan_keys), 'sample': orphan_keys[:5]},
        'no_duplicate_joined_record_key': {'pass': len(joined_dupes) == 0, 'n_duplicates': len(joined_dupes)},
        'no_missing_labels': {'pass': n_missing_labels == 0, 'n_missing_labels': n_missing_labels},
        'unique_judge_keys_match': {'pass': len(unique_gen_keys) == len(judge_key_set) == len(judge_rows),
                                     'unique_gen_keys': len(unique_gen_keys), 'judge_rows': len(judge_rows)},
    }

    all_checks = {**gen_checks, **judge_checks, **join_checks}
    overall_pass = all(v.get('pass', False) for v in all_checks.values())

    warnings = []
    if n_parse_failure > 0:
        warnings.append(f"{n_parse_failure} parse failures present in judgement file.")
    if missing_keys:
        warnings.append(f"{len(missing_keys)} generation records have no matching judgement -- judging incomplete.")
    if orphan_keys:
        warnings.append(f"{len(orphan_keys)} judgement rows are not referenced by any generation record.")

    with open(os.path.join(SCRIPT_DIR, '..', 'data', 'splits.json')) as f:
        splits_hash = hashlib.sha256(f.read().encode('utf-8')).hexdigest()

    return {
        'model': model_alias, 'method': method,
        'generation_input_path': gen_path, 'generation_input_sha256': sha256_of_file(gen_path),
        'judgement_input_path': judge_path, 'judgement_input_sha256': sha256_of_file(judge_path),
        'judge_key_function': 'judge_cache_key (40_defence_generation_driver.py, imported not reimplemented)',
        'judge_model_version': dm.JUDGE_MODEL_VERSION, 'judge_prompt_version': dm.JUDGE_PROMPT_VERSION,
        'generation_rows': len(gen_rows), 'unique_judgement_rows': len(judge_rows),
        'expanded_joined_rows': len(joined_rows),
        'cache_hits': n_cache_hits, 'cache_hit_rate': cache_hit_rate,
        'missing_keys_count': len(missing_keys), 'orphan_keys_count': len(orphan_keys),
        'duplicate_generation_record_keys': gen_checks['no_duplicate_record_key']['n_duplicates'],
        'parse_failures': n_parse_failure,
        'split_ids_sha256': splits_hash,
        'taxonomy_config_sha256': sha256_of_file(DEFAULT_TEMPLATES_PATH),
        'taxonomy_version': taxonomy['taxonomy_version'],
        'direction_config_hash': direction_config_hash, 'generation_config_hash': generation_config_hash,
        'git_commit': git_commit, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'checks': all_checks, 'warnings': warnings, 'overall_pass': overall_pass,
    }


# ── grouped summary (alpha x template) ───────────────────────────────────

def _length_stats(rows):
    lens = sorted(r['generation_length'] for r in rows)
    n = len(lens)
    if n == 0:
        return {'mean': None, 'median': None, 'p90': None, 'eos_rate': None}
    p90 = lens[min(n - 1, int(round(0.9 * (n - 1))))]
    eos_rate = sum(1 for r in rows if r['stop_reason'] == 'eos') / n
    return {'mean': sum(lens) / n, 'median': lens[n // 2], 'p90': p90, 'eos_rate': eos_rate}


def summarize_alpha_template(joined_rows):
    """Returns {alpha: {template: {...}}} -- one combined row per (alpha,
    template) with ASR/compliance/response_harmful_rate from the harmful
    subset, benign_FRR from the benign subset, and length/cache stats over
    the full combined (harmful+benign) subset for that (alpha,template)."""
    by_alpha_template = defaultdict(list)
    for r in joined_rows:
        by_alpha_template[(r['alpha'], r['template'])].append(r)

    out = defaultdict(dict)
    for (alpha, template), rows in by_alpha_template.items():
        harmful_rows = [r for r in rows if r['benign_or_harmful'] == 'harmful']
        benign_rows = [r for r in rows if r['benign_or_harmful'] == 'benign']

        asr_stats = dm.compute_template_asr(harmful_rows)
        frr_stats = dm.compute_template_frr(benign_rows)
        n_request_harmful_among_benign = sum(
            1 for r in benign_rows if r['parse_success'] and r['request_harmful'] == 1)

        n_valid = sum(1 for r in rows if r['parse_success'])
        n_parse_failure = len(rows) - n_valid
        n_request_harmful = sum(1 for r in rows if r['parse_success'] and r['request_harmful'] == 1)

        unique_responses = {r['judge_cache_key'] for r in rows}
        n_unique = len(unique_responses)
        n_cache_hits = len(rows) - n_unique
        length = _length_stats(rows)

        out[alpha][template] = {
            'n_total': len(rows), 'n_valid': n_valid, 'n_parse_failure': n_parse_failure,
            'n_request_harmful': n_request_harmful,
            'asr': asr_stats['asr'], 'compliance_rate': asr_stats['compliance_rate'],
            'response_harmful_rate': asr_stats['response_harmful_rate'],
            'asr_valid_denominator': asr_stats['valid_denominator'],
            'benign_frr': frr_stats['frr'], 'benign_frr_valid_denominator': frr_stats['valid_denominator'],
            'n_request_harmful_among_benign': n_request_harmful_among_benign,
            'mean_generation_tokens': length['mean'], 'median_generation_tokens': length['median'],
            'p90_generation_tokens': length['p90'], 'eos_rate': length['eos_rate'],
            'unique_response_count': n_unique, 'judge_cache_hit_count': n_cache_hits,
            'judge_cache_hit_rate': n_cache_hits / len(rows) if rows else None,
        }
    return dict(out)


def summarize_per_alpha_macro(per_alpha_template):
    """macro_* = simple mean across the 6 per-template rates (NEVER pooled --
    pooled is reported separately alongside, under a different key)."""
    out = {}
    for alpha, by_template in per_alpha_template.items():
        asr_vals = [v['asr'] for v in by_template.values() if v['asr'] is not None]
        comp_vals = [v['compliance_rate'] for v in by_template.values() if v['compliance_rate'] is not None]
        rhr_vals = [v['response_harmful_rate'] for v in by_template.values() if v['response_harmful_rate'] is not None]
        frr_vals = [v['benign_frr'] for v in by_template.values() if v['benign_frr'] is not None]

        def _avg(vals):
            return sum(vals) / len(vals) if vals else None

        out[alpha] = {
            'macro_asr': _avg(asr_vals), 'macro_compliance_rate': _avg(comp_vals),
            'macro_response_harmful_rate': _avg(rhr_vals), 'macro_benign_frr': _avg(frr_vals),
            'n_templates_contributing_asr': len(asr_vals), 'n_templates_contributing_frr': len(frr_vals),
            'per_template_valid_denominators': {
                t: {'asr_denom': v['asr_valid_denominator'], 'frr_denom': v['benign_frr_valid_denominator']}
                for t, v in by_template.items()
            },
        }
    return out


def summarize_pooled_per_alpha(joined_rows):
    """Explicitly separate from macro -- pooling all templates' records together
    before computing a rate (NOT the same as averaging 6 template-level rates)."""
    by_alpha = defaultdict(list)
    for r in joined_rows:
        by_alpha[r['alpha']].append(r)
    out = {}
    for alpha, rows in by_alpha.items():
        harmful_rows = [r for r in rows if r['benign_or_harmful'] == 'harmful']
        benign_rows = [r for r in rows if r['benign_or_harmful'] == 'benign']
        asr_stats = dm.compute_template_asr(harmful_rows)
        frr_stats = dm.compute_template_frr(benign_rows)
        out[alpha] = {
            'pooled_asr': asr_stats['asr'], 'pooled_compliance_rate': asr_stats['compliance_rate'],
            'pooled_response_harmful_rate': asr_stats['response_harmful_rate'],
            'pooled_asr_valid_denominator': asr_stats['valid_denominator'],
            'pooled_benign_frr': frr_stats['frr'], 'pooled_frr_valid_denominator': frr_stats['valid_denominator'],
        }
    return out


def extract_and_verify_config_hashes(gen_rows):
    """direction_config_hash/generation_config_hash are read off the generation
    records (already embedded per-record at generation time) rather than
    recomputed -- this avoids needing exp3_coverage/hooks_mod (torch) here,
    and doubles as a consistency check: every one of the 3,648 records for a
    given (model, method) job must agree on exactly one value each."""
    direction_hashes = {r.get('direction_config_hash') for r in gen_rows}
    generation_hashes = {r.get('generation_config_hash') for r in gen_rows}
    consistent = len(direction_hashes) == 1 and len(generation_hashes) == 1
    return (direction_hashes.pop() if len(direction_hashes) == 1 else None,
            generation_hashes.pop() if len(generation_hashes) == 1 else None,
            consistent, sorted(str(h) for h in direction_hashes if len(direction_hashes) > 1),
            sorted(str(h) for h in generation_hashes if len(generation_hashes) > 1))


def main(args):
    model_alias, model_path = MODELS[args.model_idx]
    taxonomy = load_taxonomy_v2()
    git_commit = dm.git_commit_hash()

    out_dir = os.path.join(args.output_path, 'canonical_v2')
    gen_path = os.path.join(out_dir, f'experiment3_validation_generations_{model_alias}_{args.method}.jsonl')
    judge_path = os.path.join(out_dir, f'experiment3_validation_judgements_{model_alias}_{args.method}.jsonl')

    print(f"=== Join & summarize: {model_alias} x {args.method} ===")
    gen_rows = dm.load_jsonl(gen_path)
    judge_rows = dm.load_jsonl(judge_path)
    print(f"Loaded {len(gen_rows)} generation rows from {gen_path}")
    print(f"Loaded {len(judge_rows)} judgement rows from {judge_path}")

    # Hard gate -- must pass before anything else runs.
    check_no_key_collisions(gen_rows)
    print("No judge_cache_key collisions detected.")

    direction_config_hash, generation_config_hash, hashes_consistent, dup_dir, dup_gen = \
        extract_and_verify_config_hashes(gen_rows)
    print(f"direction_config_hash={direction_config_hash!r} generation_config_hash={generation_config_hash!r} "
          f"consistent_across_all_records={hashes_consistent}")

    gen_checks = preflight_generation_checks(gen_rows)
    gen_checks['config_hashes_consistent'] = {
        'pass': hashes_consistent, 'direction_config_hash': direction_config_hash,
        'generation_config_hash': generation_config_hash,
        'distinct_direction_hashes_found': dup_dir, 'distinct_generation_hashes_found': dup_gen,
    }
    judge_checks, n_parse_failure = preflight_judgement_checks(judge_rows)
    for name, c in {**gen_checks, **judge_checks}.items():
        print(f"  [{'PASS' if c.get('pass', True) else 'FAIL'}] {name}: {c}")

    judge_lookup = {j['judge_cache_key']: j for j in judge_rows}
    joined_rows, missing_keys = join_generation_and_judgement(gen_rows, judge_lookup)

    audit = build_join_audit(gen_path, judge_path, gen_rows, judge_rows, joined_rows, missing_keys,
                              gen_checks, judge_checks, n_parse_failure, model_alias, args.method,
                              taxonomy, direction_config_hash, generation_config_hash, git_commit)
    print(f"\nJoin audit: generation_rows={audit['generation_rows']} "
          f"unique_judgement_rows={audit['unique_judgement_rows']} "
          f"expanded_joined_rows={audit['expanded_joined_rows']} "
          f"cache_hit_rate={audit['cache_hit_rate']}")
    print(f"OVERALL_PASS: {audit['overall_pass']}")
    if audit['warnings']:
        print(f"WARNINGS: {audit['warnings']}")

    per_alpha_template = summarize_alpha_template(joined_rows)
    per_alpha_macro = summarize_per_alpha_macro(per_alpha_template)
    per_alpha_pooled = summarize_pooled_per_alpha(joined_rows)

    print("\n=== Per-alpha macro (NOT pooled) ===")
    for alpha in ALPHAS:
        if alpha in per_alpha_macro:
            print(f"  alpha={alpha}: {per_alpha_macro[alpha]}")

    os.makedirs(out_dir, exist_ok=True)
    joined_path = os.path.join(out_dir, f'experiment3_validation_joined_{model_alias}_{args.method}.jsonl')
    with open(joined_path, 'w', encoding='utf-8') as f:
        for r in joined_rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    summary = {
        'model': model_alias, 'method': args.method,
        'per_alpha_template': {str(a): t for a, t in per_alpha_template.items()},
        'per_alpha_macro': {str(a): v for a, v in per_alpha_macro.items()},
        'per_alpha_pooled': {str(a): v for a, v in per_alpha_pooled.items()},
        'note': 'macro_* = simple mean of 6 per-template rates; pooled_* = all records merged '
                'before computing the rate -- these are intentionally different and must not be conflated.',
        'alpha_not_selected_here': True,
    }
    summary_path = os.path.join(out_dir, f'experiment3_validation_summary_{model_alias}_{args.method}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    csv_path = os.path.join(out_dir, f'experiment3_validation_summary_{model_alias}_{args.method}.csv')
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['alpha', 'template', 'n_total', 'n_valid', 'n_parse_failure', 'n_request_harmful',
                      'asr', 'compliance_rate', 'response_harmful_rate', 'asr_valid_denominator',
                      'benign_frr', 'benign_frr_valid_denominator', 'n_request_harmful_among_benign',
                      'mean_generation_tokens', 'median_generation_tokens', 'p90_generation_tokens',
                      'eos_rate', 'unique_response_count', 'judge_cache_hit_count', 'judge_cache_hit_rate']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for alpha in sorted(per_alpha_template.keys()):
            for template in sorted(per_alpha_template[alpha].keys()):
                row = {'alpha': alpha, 'template': template, **per_alpha_template[alpha][template]}
                writer.writerow({k: row.get(k) for k in fieldnames})

    audit_path = os.path.join(out_dir, f'experiment3_validation_join_audit_{model_alias}_{args.method}.json')
    with open(audit_path, 'w') as f:
        json.dump(audit, f, indent=2)

    print(f"\nSaved: {joined_path}\n       {summary_path}\n       {csv_path}\n       {audit_path}")
    if not audit['overall_pass']:
        print("\n*** OVERALL_PASS is False -- this summary must NOT be used for alpha selection "
              "or reporting until the failing checks are resolved. ***")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_idx', type=int, required=True)
    parser.add_argument('--method', choices=['placebo', 'global', 'fixed_wei', 'adaptive'], required=True)
    parser.add_argument('--output_path', type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    args = parser.parse_args()
    main(args)
