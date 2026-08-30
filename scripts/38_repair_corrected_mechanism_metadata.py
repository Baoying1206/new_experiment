"""
Deterministic, assert-gated repair of the ONE known metadata defect found by
scripts/audits/audit_corrected_completions.py: completions_en_full572_corrected.json's
per-record 'mechanism' field for condition=='persona_roleplay' is still
'mismatched_generalization' (the pre-correction taxonomy value) instead of
'competing_objectives' (the current V2 taxonomy value), for all 3 models.

This is a metadata-only repair. It does NOT regenerate response,
generation_tokens, or any paired activation -- and none of scripts
18/33/34/35's results depend on this field (they read CO/MG membership
dynamically via _taxonomy_v2_loader.py, never from a completion record's
'mechanism' field), so those results are unaffected either way.

Repair rule (fixed, not derived from any geometric result):
  if record['condition'] == 'persona_roleplay':
      record['mechanism'] = taxonomy.mechanism_of['persona_roleplay']  # == 'competing_objectives'

Refuses to run (raises, writes nothing) unless ALL of the following hold
for a given model's file, checked BEFORE any write:
  - total records == 4576
  - persona_roleplay record count == 572
  - exactly 0 or exactly 572 persona_roleplay mismatches (0 = already
    repaired, idempotent re-run; 572 = the known unrepaired defect; any
    other count is an unrecognized state, refuse)
  - all stale values (if any) are exactly 'mismatched_generalization'
  - taxonomy_v2_loader's expected value for persona_roleplay is exactly
    'competing_objectives' (catches an unexpectedly-changed taxonomy config)
  - zero mismatches on any OTHER active-mechanism condition (out of scope
    for this repair -- a differently-shaped defect must not be silently
    patched by this script)
  - no duplicate (id, condition) pairs; 572 unique ids each with exactly 8
    condition rows

After repair, verifies (before writing) that removing the 'mechanism' key
from every record leaves the before/after records identical -- i.e. proves
only that one field, on exactly the intended rows, changed.

Default is DRY RUN (no write). Only writes with --apply, and only via a
temp-file-then-os.replace atomic swap (never edits the file in place while
reading it).

Usage:
  python scripts/38_repair_corrected_mechanism_metadata.py --output_dir output           # dry run, all 3 models
  python scripts/38_repair_corrected_mechanism_metadata.py --output_dir output --apply    # actually writes
"""
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)
from _taxonomy_v2_loader import load_taxonomy_v2, DEFAULT_TEMPLATES_PATH

MODELS = ['Qwen2.5-7B-Instruct', 'Meta-Llama-3.1-8B-Instruct', 'gemma-2-9b-it']
TARGET_CONDITION = 'persona_roleplay'
EXPECTED_STALE_VALUE = 'mismatched_generalization'


def sha256_of_file(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def sha256_of_text_file(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def get_completions_list(data):
    return data['completions'] if isinstance(data, dict) and 'completions' in data else data


def preflight_checks(completions, active_mechanisms, mechanism_of):
    total = len(completions)
    assert total == 4576, f"expected 4576 total records, got {total}"

    persona_records = [c for c in completions if c.get('condition') == TARGET_CONDITION]
    assert len(persona_records) == 572, (
        f"expected {572} {TARGET_CONDITION} records, got {len(persona_records)}"
    )

    pair_counts = {}
    for c in completions:
        key = (c['id'], c.get('condition'))
        pair_counts[key] = pair_counts.get(key, 0) + 1
    dupes = {k: v for k, v in pair_counts.items() if v > 1}
    assert len(dupes) == 0, f"duplicate (id,condition) pairs found: {list(dupes.items())[:5]}"

    ids_seen = {}
    for c in completions:
        ids_seen.setdefault(c['id'], set()).add(c.get('condition'))
    assert len(ids_seen) == 572, f"expected 572 unique ids, got {len(ids_seen)}"
    incomplete = {i: len(conds) for i, conds in ids_seen.items() if len(conds) != 8}
    assert len(incomplete) == 0, f"some ids do not have exactly 8 conditions: {list(incomplete.items())[:5]}"

    stale_persona, other_mismatches = [], []
    for c in completions:
        cond = c.get('condition')
        if cond not in active_mechanisms or 'mechanism' not in c:
            continue
        expected = mechanism_of[cond]
        if c['mechanism'] != expected:
            (stale_persona if cond == TARGET_CONDITION else other_mismatches).append(c)

    assert len(other_mismatches) == 0, (
        f"non-{TARGET_CONDITION} mechanism mismatches found (out of scope for this repair, refusing to run): "
        f"{[(c['id'], c['condition'], c['mechanism']) for c in other_mismatches[:5]]}"
    )
    assert len(stale_persona) in (0, 572), (
        f"expected exactly 0 (already repaired) or 572 (unrepaired) stale {TARGET_CONDITION} "
        f"mismatches, found {len(stale_persona)} -- unrecognized state, refusing to run"
    )
    if stale_persona:
        stale_values = set(c['mechanism'] for c in stale_persona)
        assert stale_values == {EXPECTED_STALE_VALUE}, (
            f"expected all stale values to equal {EXPECTED_STALE_VALUE!r}, found: {stale_values}"
        )
    expected_value = mechanism_of[TARGET_CONDITION]
    assert expected_value == 'competing_objectives', (
        f"taxonomy loader says {TARGET_CONDITION}'s expected mechanism is {expected_value!r}, "
        f"not 'competing_objectives' -- taxonomy config changed unexpectedly, refusing to run"
    )
    return {
        'total_records': total, 'persona_roleplay_records': len(persona_records),
        'stale_mismatch_count': len(stale_persona),
        'stale_values': sorted(set(c['mechanism'] for c in stale_persona)),
        'expected_value': expected_value, 'other_mismatches_count': len(other_mismatches),
    }


def repair(completions, mechanism_of):
    expected = mechanism_of[TARGET_CONDITION]
    changed = 0
    for c in completions:
        if c.get('condition') == TARGET_CONDITION and c.get('mechanism') != expected:
            c['mechanism'] = expected
            changed += 1
    return changed


def assert_semantic_equivalence(before_list, after_list):
    assert len(before_list) == len(after_list), "record count changed across repair -- must never happen"
    for b, a in zip(before_list, after_list):
        b2 = {k: v for k, v in b.items() if k != 'mechanism'}
        a2 = {k: v for k, v in a.items() if k != 'mechanism'}
        assert b2 == a2, (
            f"non-'mechanism' field changed for id={b.get('id')} condition={b.get('condition')} -- "
            f"repair must only ever touch 'mechanism'"
        )


def atomic_write_json(path, data):
    tmp_path = path + '.tmp_repair'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def git_commit_hash():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=SCRIPT_DIR,
                               capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return 'unknown'


def process_model(model_alias, path, active_mechanisms, mechanism_of, apply):
    sha_before = sha256_of_file(path)
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    completions = get_completions_list(data)
    before_snapshot = copy.deepcopy(completions)

    preflight = preflight_checks(completions, active_mechanisms, mechanism_of)
    changed = repair(completions, mechanism_of)
    assert changed == preflight['stale_mismatch_count'], (
        f"repair changed {changed} rows but preflight predicted {preflight['stale_mismatch_count']}"
    )
    assert_semantic_equivalence(before_snapshot, completions)

    result = {
        'input_path': path, 'sha256_before': sha_before,
        'total_rows': preflight['total_records'], 'changed_rows': changed,
        'changed_field': 'mechanism', 'old_value': EXPECTED_STALE_VALUE if changed else None,
        'new_value': mechanism_of[TARGET_CONDITION],
        'unchanged_field_semantic_validation': 'passed (all non-mechanism fields identical, verified per-record)',
        'preflight': preflight,
        'applied': False, 'sha256_after': None,
    }
    if apply and changed > 0:
        atomic_write_json(path, data)
        result['sha256_after'] = sha256_of_file(path)
        result['applied'] = True
    elif apply and changed == 0:
        result['sha256_after'] = sha_before
        result['applied'] = False
        result['note'] = 'nothing to change (already repaired) -- file left untouched, no write performed'
    else:
        result['note'] = 'dry run -- no file written'
    return result


def main(args):
    taxonomy = load_taxonomy_v2()
    active_mechanisms, mechanism_of = taxonomy['active_mechanisms'], taxonomy['mechanism_of']
    print(f"Taxonomy v2: active_mechanisms={active_mechanisms}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN (no files will be written)'}\n")

    taxonomy_config_hash = sha256_of_text_file(DEFAULT_TEMPLATES_PATH)
    report = {
        'taxonomy_version': taxonomy['taxonomy_version'],
        'taxonomy_config_path': taxonomy['config_path'],
        'taxonomy_config_sha256': taxonomy_config_hash,
        'git_commit': git_commit_hash(),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'mode': 'apply' if args.apply else 'dry_run',
        'target_condition': TARGET_CONDITION,
        'warnings': [],
        'per_model': {},
    }

    for model_alias in MODELS:
        path = os.path.join(args.output_dir, model_alias, f'completions_{args.lang}{args.suffix}.json')
        print(f"=== {model_alias} ===")
        if not os.path.exists(path):
            msg = f"{model_alias}: file not found at {path}"
            print(f"  MISSING: {path}")
            report['warnings'].append(msg)
            report['per_model'][model_alias] = {'exists': False, 'path': path}
            continue
        r = process_model(model_alias, path, active_mechanisms, mechanism_of, args.apply)
        r['exists'] = True
        report['per_model'][model_alias] = r
        print(f"  total_rows={r['total_rows']}  changed_rows={r['changed_rows']}  applied={r['applied']}")
        print(f"  sha256_before={r['sha256_before'][:16]}...  sha256_after={(r['sha256_after'] or '')[:16]}...")

    out_dir = os.path.join(args.output_dir, 'canonical_v2')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'corrected_completions_metadata_repair.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out_path}")
    if not args.apply:
        print("This was a DRY RUN. Re-run with --apply to actually write the repaired files.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    parser.add_argument('--lang',       type=str, default='en')
    parser.add_argument('--suffix',     type=str, default='_full572_corrected')
    parser.add_argument('--apply',      action='store_true', help='Actually write the repaired files (default: dry run only)')
    args = parser.parse_args()
    main(args)
