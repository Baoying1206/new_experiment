"""
Joins one (model, method) TEST run's generation + judgement JSONLs into a
fully-expanded table and a per-template summary. method in {no_defence,
fixed_wei, adaptive} -- global/placebo are not test-phase methods.

Unlike 41_join_and_summarize_defence_validation.py, there is NO alpha
dimension here (each method uses exactly one frozen alpha, or none for
no_defence) -- expected row count is 1800 per (model, method): 1200
harmful (200 test_ids x 6 templates) + 600 benign (100 benign_test ids x
6 templates), for ALL THREE methods (no_defence's harmful/benign happen
to sum to the same 1800 as fixed_wei/adaptive's).

Reuses (imports, never reimplements) 41_join_and_summarize_defence_validation.py's
check_no_key_collisions/join_generation_and_judgement/preflight_judgement_checks.
For no_defence, harmful records are reconstructed from
completions_en_full572_corrected.json filtered to test_ids, mirroring
49_defence_test_driver.py's run_test_no_defence_harmful_rejudge (read-only
here, no judging). Pure CPU/Python, no GPU.

Usage:
  python scripts/50_join_and_summarize_test.py --model_idx 1 --method fixed_wei --output_path output
  python scripts/50_join_and_summarize_test.py --model_idx 1 --method no_defence --output_path output
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib import import_module

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)
import _defence_metrics as dm  # torch-free
from _taxonomy_v2_loader import load_taxonomy_v2
join_mod = import_module('41_join_and_summarize_defence_validation')  # torch-free, reused not reimplemented

MODELS = dm.MODEL_PATHS
N_HARMFUL_IDS, N_BENIGN_IDS, N_TEMPLATES = 200, 100, 6
EXPECTED_HARMFUL = N_HARMFUL_IDS * N_TEMPLATES   # 1200
EXPECTED_BENIGN = N_BENIGN_IDS * N_TEMPLATES     # 600
EXPECTED_TOTAL = EXPECTED_HARMFUL + EXPECTED_BENIGN  # 1800
TEST_METHODS = ('no_defence', 'fixed_wei', 'adaptive')
SPLITS_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'splits.json')


def sha256_of_file(path):
    return join_mod.sha256_of_file(path)


def load_no_defence_harmful_reused(output_path, model_alias, active_mechanisms):
    """Mirrors 49_defence_test_driver.py's run_test_no_defence_harmful_rejudge
    (JSON-reading portion only)."""
    comp_path = os.path.join(output_path, model_alias, 'completions_en_full572_corrected.json')
    with open(comp_path, encoding='utf-8') as f:
        data = json.load(f)
    completions = data['completions'] if isinstance(data, dict) and 'completions' in data else data
    with open(SPLITS_PATH) as f:
        test_ids = set(json.load(f)['test_ids'])
    records = [dict(c) for c in completions if c['id'] in test_ids and c['condition'] in active_mechanisms]
    assert len(records) == EXPECTED_HARMFUL, f"expected {EXPECTED_HARMFUL} reused records, got {len(records)}"
    for r in records:
        r['instruction_id'], r['benign_or_harmful'], r['template'] = r['id'], 'harmful', r['condition']
        r.setdefault('method', 'no_defence')
        r.setdefault('alpha', None)
    return comp_path, records


def load_and_join(output_path, model_alias, method, active_mechanisms):
    out_dir = os.path.join(output_path, 'canonical_v2')
    if method == 'no_defence':
        benign_gen_path = os.path.join(out_dir, f'experiment3_test_generations_{model_alias}_no_defence_benign.jsonl')
        benign_judge_path = os.path.join(out_dir, f'experiment3_test_judgements_{model_alias}_no_defence_benign.jsonl')
        benign_gen_rows = dm.load_jsonl(benign_gen_path)
        benign_judge_rows = dm.load_jsonl(benign_judge_path)

        comp_path, harmful_gen_rows = load_no_defence_harmful_reused(output_path, model_alias, active_mechanisms)
        harmful_judge_path = os.path.join(out_dir, f'experiment3_test_judgements_{model_alias}_no_defence_harmful.jsonl')
        harmful_judge_rows = dm.load_jsonl(harmful_judge_path)

        gen_rows = harmful_gen_rows + benign_gen_rows
        judge_rows = harmful_judge_rows + benign_judge_rows
        gen_path_display = f"{comp_path} (harmful, filtered to test_ids) + {benign_gen_path}"
        judge_path_display = f"{harmful_judge_path} + {benign_judge_path}"
    else:
        gen_path = os.path.join(out_dir, f'experiment3_test_generations_{model_alias}_{method}.jsonl')
        judge_path = os.path.join(out_dir, f'experiment3_test_judgements_{model_alias}_{method}.jsonl')
        gen_rows = dm.load_jsonl(gen_path)
        judge_rows = dm.load_jsonl(judge_path)
        gen_path_display, judge_path_display = gen_path, judge_path

    join_mod.check_no_key_collisions(gen_rows)
    judge_checks, n_parse_failure = join_mod.preflight_judgement_checks(judge_rows)
    judge_lookup = {j['judge_cache_key']: j for j in judge_rows}
    joined_rows, missing_keys = join_mod.join_generation_and_judgement(gen_rows, judge_lookup)

    return {
        'gen_rows': gen_rows, 'judge_rows': judge_rows, 'joined_rows': joined_rows,
        'missing_keys': missing_keys, 'judge_checks': judge_checks, 'n_parse_failure': n_parse_failure,
        'gen_path_display': gen_path_display, 'judge_path_display': judge_path_display,
    }


def preflight_generation_checks(gen_rows):
    checks = {}
    checks['total_count'] = {'value': len(gen_rows), 'expected': EXPECTED_TOTAL, 'pass': len(gen_rows) == EXPECTED_TOTAL}
    harmful = [r for r in gen_rows if r['benign_or_harmful'] == 'harmful']
    benign = [r for r in gen_rows if r['benign_or_harmful'] == 'benign']
    checks['harmful_count'] = {'value': len(harmful), 'expected': EXPECTED_HARMFUL, 'pass': len(harmful) == EXPECTED_HARMFUL}
    checks['benign_count'] = {'value': len(benign), 'expected': EXPECTED_BENIGN, 'pass': len(benign) == EXPECTED_BENIGN}

    harmful_template_counts = Counter(r['template'] for r in harmful)
    bad_harmful = {t: c for t, c in harmful_template_counts.items() if c != N_HARMFUL_IDS}
    checks['harmful_template_counts'] = {
        'pass': len(harmful_template_counts) == N_TEMPLATES and len(bad_harmful) == 0,
        'counts': dict(harmful_template_counts), 'expected_per_template': N_HARMFUL_IDS,
    }
    benign_template_counts = Counter(r['template'] for r in benign)
    bad_benign = {t: c for t, c in benign_template_counts.items() if c != N_BENIGN_IDS}
    checks['benign_template_counts'] = {
        'pass': len(benign_template_counts) == N_TEMPLATES and len(bad_benign) == 0,
        'counts': dict(benign_template_counts), 'expected_per_template': N_BENIGN_IDS,
    }
    return checks


def _length_stats(rows):
    """no_defence's harmful subset comes from completions_en_full572_corrected.json
    (script 03's own output schema), which may not carry generation_length/
    stop_reason the same way _build_generation_record's freshly-generated rows
    do -- rows missing either field are excluded from the stat, not treated as
    a crash; the count excluded is reported so it's never a silent gap."""
    usable = [r for r in rows if 'generation_length' in r and 'stop_reason' in r]
    n_excluded = len(rows) - len(usable)
    lens = sorted(r['generation_length'] for r in usable)
    n = len(lens)
    if n == 0:
        return {'mean': None, 'median': None, 'p90': None, 'eos_rate': None, 'n_excluded_missing_fields': n_excluded}
    p90 = lens[min(n - 1, int(round(0.9 * (n - 1))))]
    eos_rate = sum(1 for r in usable if r['stop_reason'] == 'eos') / n
    return {'mean': sum(lens) / n, 'median': lens[n // 2], 'p90': p90, 'eos_rate': eos_rate,
            'n_excluded_missing_fields': n_excluded}


def summarize_per_template(joined_rows):
    by_template = defaultdict(list)
    for r in joined_rows:
        by_template[r['template']].append(r)

    out = {}
    for template, rows in by_template.items():
        harmful_rows = [r for r in rows if r['benign_or_harmful'] == 'harmful']
        benign_rows = [r for r in rows if r['benign_or_harmful'] == 'benign']
        asr_stats = dm.compute_template_asr(harmful_rows)
        frr_stats = dm.compute_template_frr(benign_rows)
        n_request_harmful_among_benign = sum(1 for r in benign_rows if r['parse_success'] and r['request_harmful'] == 1)
        length = _length_stats(rows)
        n_valid = sum(1 for r in rows if r['parse_success'])
        out[template] = {
            'n_total': len(rows), 'n_valid': n_valid, 'n_parse_failure': len(rows) - n_valid,
            'asr': asr_stats['asr'], 'compliance_rate': asr_stats['compliance_rate'],
            'response_harmful_rate': asr_stats['response_harmful_rate'],
            'asr_valid_denominator': asr_stats['valid_denominator'],
            'asr_n_excluded_not_request_harmful_or_parse_fail': asr_stats['n_excluded'],
            'benign_frr': frr_stats['frr'], 'benign_frr_valid_denominator': frr_stats['valid_denominator'],
            'n_request_harmful_among_benign': n_request_harmful_among_benign,
            'mean_generation_tokens': length['mean'], 'median_generation_tokens': length['median'],
            'p90_generation_tokens': length['p90'], 'eos_rate': length['eos_rate'],
            'length_stats_n_excluded_missing_fields': length['n_excluded_missing_fields'],
        }
    return out


def summarize_macro(per_template):
    def _avg(field):
        vals = [v[field] for v in per_template.values() if v[field] is not None]
        return sum(vals) / len(vals) if vals else None
    return {
        'macro_asr': _avg('asr'), 'macro_compliance_rate': _avg('compliance_rate'),
        'macro_response_harmful_rate': _avg('response_harmful_rate'), 'macro_benign_frr': _avg('benign_frr'),
    }


def build_audit(model_alias, method, join_result, gen_checks, alpha):
    gen_rows, judge_rows, joined_rows = join_result['gen_rows'], join_result['judge_rows'], join_result['joined_rows']
    missing_keys, judge_checks = join_result['missing_keys'], join_result['judge_checks']

    unique_gen_keys = {dm.judge_cache_key(r['instruction_en'], r['response']) for r in gen_rows}
    judge_key_set = {j['judge_cache_key'] for j in judge_rows}
    orphan_keys = sorted(judge_key_set - unique_gen_keys)
    joined_dupes = [k for k, c in Counter(r['record_key'] for r in joined_rows).items() if c > 1] \
        if joined_rows and 'record_key' in joined_rows[0] else []
    n_missing_labels = sum(1 for r in joined_rows if r.get('parse_success') is None
                            or r.get('refusal') is None or r.get('request_harmful') is None
                            or r.get('response_harmful') is None)
    n_cache_hits = len(gen_rows) - len(unique_gen_keys)

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
    pass_bearing = {k: v for k, v in all_checks.items() if 'pass' in v}
    overall_pass = all(v['pass'] for v in pass_bearing.values())

    return {
        'model': model_alias, 'method': method, 'alpha': alpha, 'split': 'test',
        'generation_input': join_result['gen_path_display'], 'judgement_input': join_result['judge_path_display'],
        'generation_rows': len(gen_rows), 'unique_judgement_rows': len(judge_rows),
        'expanded_joined_rows': len(joined_rows), 'cache_hits': n_cache_hits,
        'cache_hit_rate': n_cache_hits / len(gen_rows) if gen_rows else None,
        'parse_failures': join_result['n_parse_failure'],
        'checks': all_checks, 'overall_pass': overall_pass,
        'git_commit': dm.git_commit_hash(), 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    }


def main(args):
    model_alias, _ = MODELS[args.model_idx]
    taxonomy = load_taxonomy_v2()
    active_mechanisms = taxonomy['active_mechanisms']

    alpha = None
    if args.method != 'no_defence':
        frozen_path = os.path.join(args.output_path, 'canonical_v2', 'experiment3_defence_frozen_config.json')
        with open(frozen_path) as f:
            frozen = json.load(f)
        alpha = frozen['per_model'][model_alias]['alpha'][args.method]

    print(f"=== Test join & summarize: {model_alias} x {args.method} (alpha={alpha}) ===")
    join_result = load_and_join(args.output_path, model_alias, args.method, active_mechanisms)
    print(f"Loaded {len(join_result['gen_rows'])} generation rows, {len(join_result['judge_rows'])} judgement rows.")

    gen_checks = preflight_generation_checks(join_result['gen_rows'])
    audit = build_audit(model_alias, args.method, join_result, gen_checks, alpha)

    for name, c in audit['checks'].items():
        label = 'PASS' if c.get('pass') else ('INFO' if 'pass' not in c else 'FAIL')
        print(f"  [{label}] {name}: {c}")
    print(f"\nOVERALL_PASS: {audit['overall_pass']}")

    per_template = summarize_per_template(join_result['joined_rows'])
    macro = summarize_macro(per_template)
    print(f"\nmacro: {macro}")

    out_dir = os.path.join(args.output_path, 'canonical_v2')
    os.makedirs(out_dir, exist_ok=True)
    joined_path = os.path.join(out_dir, f'experiment3_test_joined_{model_alias}_{args.method}.jsonl')
    with open(joined_path, 'w', encoding='utf-8') as f:
        for r in join_result['joined_rows']:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    summary = {'model': model_alias, 'method': args.method, 'alpha': alpha,
               'per_template': per_template, 'macro': macro}
    summary_path = os.path.join(out_dir, f'experiment3_test_summary_{model_alias}_{args.method}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    audit_path = os.path.join(out_dir, f'experiment3_test_join_audit_{model_alias}_{args.method}.json')
    with open(audit_path, 'w') as f:
        json.dump(audit, f, indent=2)

    print(f"\nSaved: {joined_path}\n       {summary_path}\n       {audit_path}")
    if not audit['overall_pass']:
        print("\n*** OVERALL_PASS is False -- do not use for the bootstrap comparison. ***")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_idx', type=int, required=True)
    parser.add_argument('--method', choices=list(TEST_METHODS), required=True)
    parser.add_argument('--output_path', type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    args = parser.parse_args()
    main(args)
