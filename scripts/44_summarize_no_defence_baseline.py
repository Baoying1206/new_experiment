"""
Computes the No-defence baseline needed for alpha freezing (select_alpha's
no_defence_macro_frr argument), for one model:
  - benign macro FRR, from experiment3_validation_generations_{model}_
    no_defence_benign.jsonl (480 rows, no alpha dimension) joined against its
    judgements.
  - harmful macro ASR, from completions_en_full572_corrected.json filtered to
    validation_ids x active_mechanisms (432 rows, the same reused-completions
    set run_no_defence_harmful_rejudge judges) joined against
    experiment3_validation_judgements_{model}_no_defence_harmful.jsonl.

Does NOT select alpha -- alpha freezing (a separate step) needs this
baseline plus all of Fixed Wei/Adaptive's per-alpha summaries for the same
model. Pure CPU/Python: reuses 41_join_and_summarize_defence_validation.py's
already-tested check_no_key_collisions/join_generation_and_judgement/
preflight_judgement_checks (imported, never reimplemented) plus
_defence_metrics.py's compute_template_frr/compute_template_asr/
compute_macro_frr/compute_macro_asr. No torch, no --phase validation's
alpha-sweep assumptions (EXPECTED_HARMFUL etc, which do not apply here).

Usage:
  python scripts/44_summarize_no_defence_baseline.py --model_idx 1 --output_path output
"""
import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from importlib import import_module

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)
import _defence_metrics as dm  # torch-free
from _taxonomy_v2_loader import load_taxonomy_v2
join_mod = import_module('41_join_and_summarize_defence_validation')  # torch-free, reused not reimplemented

MODELS = dm.MODEL_PATHS
N_HARMFUL_IDS, N_BENIGN_IDS, N_TEMPLATES = 72, 80, 6
SPLITS_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'splits.json')


def sha256_of_file(path):
    return join_mod.sha256_of_file(path)


def load_benign_records(output_path, model_alias):
    out_dir = os.path.join(output_path, 'canonical_v2')
    gen_path = os.path.join(out_dir, f'experiment3_validation_generations_{model_alias}_no_defence_benign.jsonl')
    judge_path = os.path.join(out_dir, f'experiment3_validation_judgements_{model_alias}_no_defence_benign.jsonl')
    gen_rows = dm.load_jsonl(gen_path)
    judge_rows = dm.load_jsonl(judge_path)
    return gen_path, judge_path, gen_rows, judge_rows


def load_harmful_reused_records(output_path, model_alias, active_mechanisms):
    """Mirrors 40_defence_generation_driver.py's run_no_defence_harmful_rejudge
    (JSON-reading portion only -- no judging here, that already happened on
    the cluster): filters completions_en_full572_corrected.json to
    validation_ids x active_mechanisms, tags instruction_id/benign_or_harmful/
    template exactly as that function does, so judge_cache_key recomputes
    identically."""
    comp_path = os.path.join(output_path, model_alias, 'completions_en_full572_corrected.json')
    with open(comp_path, encoding='utf-8') as f:
        data = json.load(f)
    completions = data['completions'] if isinstance(data, dict) and 'completions' in data else data

    with open(SPLITS_PATH) as f:
        validation_ids = set(json.load(f)['validation_ids'])

    records = [dict(c) for c in completions if c['id'] in validation_ids and c['condition'] in active_mechanisms]
    assert len(records) == len(validation_ids) * len(active_mechanisms) == N_HARMFUL_IDS * N_TEMPLATES, (
        f"expected {N_HARMFUL_IDS * N_TEMPLATES} reused no-defence harmful records, got {len(records)}"
    )
    for r in records:
        r['instruction_id'], r['benign_or_harmful'], r['template'] = r['id'], 'harmful', r['condition']
    return comp_path, records


def summarize_benign(gen_rows, judge_rows):
    join_mod.check_no_key_collisions(gen_rows)
    judge_checks, n_parse_failure = join_mod.preflight_judgement_checks(judge_rows)
    judge_lookup = {j['judge_cache_key']: j for j in judge_rows}
    joined_rows, missing_keys = join_mod.join_generation_and_judgement(gen_rows, judge_lookup)

    by_template = defaultdict(list)
    for r in joined_rows:
        by_template[r['template']].append(r)
    macro = dm.compute_macro_frr(by_template)

    checks = {
        'total_count': {'value': len(gen_rows), 'expected': N_BENIGN_IDS * N_TEMPLATES,
                         'pass': len(gen_rows) == N_BENIGN_IDS * N_TEMPLATES},
        'joined_rows_equal_generation_rows': {'pass': len(joined_rows) == len(gen_rows),
                                               'joined': len(joined_rows), 'generation': len(gen_rows)},
        'no_missing_keys': {'pass': len(missing_keys) == 0, 'n_missing': len(missing_keys)},
        **judge_checks,
    }
    per_template_counts = {t: len(rows) for t, rows in by_template.items()}
    checks['per_template_counts_correct'] = {
        'pass': all(c == N_BENIGN_IDS for c in per_template_counts.values()) and len(per_template_counts) == N_TEMPLATES,
        'counts': per_template_counts, 'expected_per_template': N_BENIGN_IDS,
    }
    return macro, checks, joined_rows


def summarize_harmful(records, judge_rows):
    join_mod.check_no_key_collisions(records)
    judge_checks, n_parse_failure = join_mod.preflight_judgement_checks(judge_rows)
    judge_lookup = {j['judge_cache_key']: j for j in judge_rows}
    joined_rows, missing_keys = join_mod.join_generation_and_judgement(records, judge_lookup)

    by_template = defaultdict(list)
    for r in joined_rows:
        by_template[r['template']].append(r)
    macro = dm.compute_macro_asr(by_template)

    checks = {
        'total_count': {'value': len(records), 'expected': N_HARMFUL_IDS * N_TEMPLATES,
                         'pass': len(records) == N_HARMFUL_IDS * N_TEMPLATES},
        'joined_rows_equal_generation_rows': {'pass': len(joined_rows) == len(records),
                                               'joined': len(joined_rows), 'generation': len(records)},
        'no_missing_keys': {'pass': len(missing_keys) == 0, 'n_missing': len(missing_keys)},
        **judge_checks,
    }
    per_template_counts = {t: len(rows) for t, rows in by_template.items()}
    checks['per_template_counts_correct'] = {
        'pass': all(c == N_HARMFUL_IDS for c in per_template_counts.values()) and len(per_template_counts) == N_TEMPLATES,
        'counts': per_template_counts, 'expected_per_template': N_HARMFUL_IDS,
    }
    return macro, checks, joined_rows


def main(args):
    model_alias, _ = MODELS[args.model_idx]
    taxonomy = load_taxonomy_v2()
    active_mechanisms = taxonomy['active_mechanisms']
    git_commit = dm.git_commit_hash()

    print(f"=== No-defence baseline: {model_alias} ===")

    gen_path, judge_path, gen_rows, judge_rows = load_benign_records(args.output_path, model_alias)
    print(f"Benign: loaded {len(gen_rows)} generation rows, {len(judge_rows)} judgement rows")
    benign_macro, benign_checks, benign_joined = summarize_benign(gen_rows, judge_rows)

    comp_path, harmful_records = load_harmful_reused_records(args.output_path, model_alias, active_mechanisms)
    harmful_judge_path = os.path.join(args.output_path, 'canonical_v2',
                                       f'experiment3_validation_judgements_{model_alias}_no_defence_harmful.jsonl')
    harmful_judge_rows = dm.load_jsonl(harmful_judge_path)
    print(f"Harmful: loaded {len(harmful_records)} reused records, {len(harmful_judge_rows)} judgement rows")
    harmful_macro, harmful_checks, harmful_joined = summarize_harmful(harmful_records, harmful_judge_rows)

    pass_bearing_benign = {k: v for k, v in benign_checks.items() if 'pass' in v}
    pass_bearing_harmful = {k: v for k, v in harmful_checks.items() if 'pass' in v}
    overall_pass = all(v['pass'] for v in pass_bearing_benign.values()) and \
        all(v['pass'] for v in pass_bearing_harmful.values())

    print("\n-- Benign checks --")
    for name, c in benign_checks.items():
        label = 'PASS' if c.get('pass') else ('INFO' if 'pass' not in c else 'FAIL')
        print(f"  [{label}] {name}: {c}")
    print("-- Harmful checks --")
    for name, c in harmful_checks.items():
        label = 'PASS' if c.get('pass') else ('INFO' if 'pass' not in c else 'FAIL')
        print(f"  [{label}] {name}: {c}")

    print(f"\nOVERALL_PASS: {overall_pass}")
    print(f"benign_macro_frr = {benign_macro['macro_frr']}")
    print(f"harmful_macro_asr = {harmful_macro['macro_asr']}  "
          f"(macro_compliance_rate={harmful_macro['macro_compliance_rate']}, "
          f"macro_response_harmful_rate={harmful_macro['macro_response_harmful_rate']})")

    out_dir = os.path.join(args.output_path, 'canonical_v2')
    os.makedirs(out_dir, exist_ok=True)

    for name, rows in (('benign', benign_joined), ('harmful', harmful_joined)):
        joined_path = os.path.join(out_dir, f'experiment3_no_defence_baseline_joined_{model_alias}_{name}.jsonl')
        with open(joined_path, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

    result = {
        'model': model_alias,
        'benign_macro_frr': benign_macro['macro_frr'],
        'benign_per_template': benign_macro['per_template'],
        'harmful_macro_asr': harmful_macro['macro_asr'],
        'harmful_macro_compliance_rate': harmful_macro['macro_compliance_rate'],
        'harmful_macro_response_harmful_rate': harmful_macro['macro_response_harmful_rate'],
        'harmful_per_template': harmful_macro['per_template'],
        'overall_pass': overall_pass,
        'benign_checks': benign_checks, 'harmful_checks': harmful_checks,
        'benign_generation_input_path': gen_path, 'benign_generation_input_sha256': sha256_of_file(gen_path),
        'benign_judgement_input_path': judge_path, 'benign_judgement_input_sha256': sha256_of_file(judge_path),
        'harmful_completions_input_path': comp_path, 'harmful_completions_input_sha256': sha256_of_file(comp_path),
        'harmful_judgement_input_path': harmful_judge_path,
        'harmful_judgement_input_sha256': sha256_of_file(harmful_judge_path),
        'git_commit': git_commit, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    }
    result_path = os.path.join(out_dir, f'experiment3_no_defence_baseline_{model_alias}.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {result_path}")
    if not overall_pass:
        print("\n*** OVERALL_PASS is False -- this baseline must NOT be used for alpha selection "
              "until the failing checks are resolved. ***")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_idx', type=int, required=True)
    parser.add_argument('--output_path', type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    args = parser.parse_args()
    main(args)
