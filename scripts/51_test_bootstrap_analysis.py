"""
Paired bootstrap analysis for the Exp3 TEST phase. For one model, reads
the 3 methods' (no_defence, fixed_wei, adaptive) joined test data
(experiment3_test_joined_{model}_{method}.jsonl, written by
50_join_and_summarize_test.py) and computes:

  - per method: macro ASR/FRR point estimate + 95% bootstrap CI, plus
    per-template detail, parse failures, and generation length/EOS stats
    (pulled from 50's own summary.json, never recomputed independently --
    only cross-checked for consistency, see the assertion below).
  - 4 paired deltas, each from bootstrap replicates that use the SAME
    resampled instruction-ID draw across every method compared in that
    replicate (never independent per-method resampling, which would
    understate how correlated the methods' estimates are since they're
    evaluated on the identical underlying instructions):
      1. Adaptive - Fixed Wei   delta ASR  (harmful test_ids resampled)
      2. Adaptive - Fixed Wei   delta FRR  (benign_test ids resampled)
      3. Fixed Wei - No-defence delta ASR  (harmful test_ids resampled)
      4. Adaptive - No-defence  delta ASR  (harmful test_ids resampled)

Resampling unit is the INSTRUCTION ID, never a (instruction, template)
row: the 6 template rows for one instruction are NEVER treated as
independent samples. Each bootstrap replicate draws instruction IDs with
replacement (sample size = original N); every one of a drawn instruction's
6 template rows is included (with repetition if drawn more than once) in
that replicate's per-template rate, and macro = mean of the 6 per-template
rates for that replicate.

Pure CPU/Python (stdlib only -- random, math -- no numpy/torch). A fixed,
documented seed (BOOTSTRAP_SEED) makes every run byte-for-byte
reproducible. Refuses to run (RuntimeError) if any method's join_audit has
overall_pass != True.

Usage:
  python scripts/51_test_bootstrap_analysis.py --model_idx 1 --output_path output --n_boot 10000
"""
import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)
import _defence_metrics as dm  # torch-free

MODELS = dm.MODEL_PATHS
METHODS = ('no_defence', 'fixed_wei', 'adaptive')
BOOTSTRAP_SEED = 20260901  # fixed, documented -- every run is reproducible


def load_joined(output_path, model_alias, method):
    path = os.path.join(output_path, 'canonical_v2', f'experiment3_test_joined_{model_alias}_{method}.jsonl')
    return dm.load_jsonl(path)


def build_outcome_index(joined_rows, boh):
    """{template: {instruction_id: outcome_dict}} for rows matching boh.
    Exactly one row per (template, instruction_id) by construction (each
    instruction is rendered once per template)."""
    idx = defaultdict(dict)
    for r in joined_rows:
        if r['benign_or_harmful'] != boh:
            continue
        idx[r['template']][r['instruction_id']] = {
            'parse_success': r['parse_success'], 'refusal': r['refusal'],
            'response_harmful': r['response_harmful'], 'request_harmful': r['request_harmful'],
        }
    return dict(idx)


def macro_metric_for_ids(outcome_index, resampled_ids, metric):
    """metric in {'asr', 'frr'}. Macro = mean over templates of the
    per-template rate computed over resampled_ids (WITH repetition -- an id
    drawn twice contributes its outcome twice). A template with zero valid
    (parse_success, and for ASR request_harmful==1) rows in this particular
    resample is excluded from that replicate's macro mean (not treated as 0)."""
    rates = []
    for template, id_to_outcome in outcome_index.items():
        num = denom = 0
        for iid in resampled_ids:
            o = id_to_outcome.get(iid)
            if o is None or not o['parse_success']:
                continue
            if metric == 'asr':
                if o['request_harmful'] != 1:
                    continue
                denom += 1
                if o['refusal'] == 0 and o['response_harmful'] == 1:
                    num += 1
            else:  # frr
                denom += 1
                if o['refusal'] == 1:
                    num += 1
        if denom > 0:
            rates.append(num / denom)
    return sum(rates) / len(rates) if rates else None


def bootstrap_estimates(outcome_indices_by_method, ids, metric, n_boot, rng):
    """Returns {method: [n_boot estimates]}. ALL methods in
    outcome_indices_by_method share the SAME resampled-id draw within each
    replicate -- this is what makes the resulting per-replicate deltas
    'paired'."""
    n = len(ids)
    estimates = {m: [] for m in outcome_indices_by_method}
    for _ in range(n_boot):
        resampled = [ids[rng.randrange(n)] for _ in range(n)]
        for method, idx in outcome_indices_by_method.items():
            estimates[method].append(macro_metric_for_ids(idx, resampled, metric))
    return estimates


def _percentile(vals_sorted, p):
    n = len(vals_sorted)
    if n == 0:
        return None
    k = (p / 100) * (n - 1)
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return vals_sorted[int(k)]
    return vals_sorted[f] + (vals_sorted[c] - vals_sorted[f]) * (k - f)


def ci_from_values(vals, lo=2.5, hi=97.5):
    clean = sorted(v for v in vals if v is not None)
    return {'ci_lo': _percentile(clean, lo), 'ci_hi': _percentile(clean, hi), 'n_valid_replicates': len(clean)}


def paired_delta(estimates, point_dict, method_a, method_b):
    deltas = [a - b for a, b in zip(estimates[method_a], estimates[method_b]) if a is not None and b is not None]
    return {
        'point_delta': point_dict[method_a] - point_dict[method_b],
        **ci_from_values(deltas),
        'ci_method': 'percentile bootstrap of the paired (same-replicate-resample) per-replicate delta',
    }


def main(args):
    model_alias, _ = MODELS[args.model_idx]
    print(f"=== Test bootstrap analysis: {model_alias} (n_boot={args.n_boot}, seed={BOOTSTRAP_SEED}) ===")

    with open(os.path.join(SCRIPT_DIR, '..', 'data', 'splits.json')) as f:
        harmful_ids = sorted(json.load(f)['test_ids'])
    with open(os.path.join(SCRIPT_DIR, '..', 'data', 'benign_test_100.json')) as f:
        benign_ids = sorted(b['benign_id'] for b in json.load(f))
    assert len(harmful_ids) == 200 and len(benign_ids) == 100

    joined, audits, summaries = {}, {}, {}
    for method in METHODS:
        rows = load_joined(args.output_path, model_alias, method)
        assert rows, f"no joined rows for {model_alias} x {method} -- run 50_join_and_summarize_test.py first"
        joined[method] = rows
        audit_path = os.path.join(args.output_path, 'canonical_v2',
                                   f'experiment3_test_join_audit_{model_alias}_{method}.json')
        with open(audit_path) as f:
            audit = json.load(f)
        if not audit.get('overall_pass'):
            raise RuntimeError(f"{model_alias} x {method} join_audit has overall_pass={audit.get('overall_pass')!r} "
                                f"-- refusing to bootstrap from unverified data.")
        audits[method] = audit
        summary_path = os.path.join(args.output_path, 'canonical_v2',
                                     f'experiment3_test_summary_{model_alias}_{method}.json')
        with open(summary_path) as f:
            summaries[method] = json.load(f)

    harmful_idx = {m: build_outcome_index(joined[m], 'harmful') for m in METHODS}
    benign_idx = {m: build_outcome_index(joined[m], 'benign') for m in METHODS}

    point_asr = {m: macro_metric_for_ids(harmful_idx[m], harmful_ids, 'asr') for m in METHODS}
    point_frr = {m: macro_metric_for_ids(benign_idx[m], benign_ids, 'frr') for m in METHODS}
    for m in METHODS:
        assert abs(point_asr[m] - summaries[m]['macro']['macro_asr']) < 1e-9, \
            f"{m}: bootstrap point ASR {point_asr[m]} != 50's summary {summaries[m]['macro']['macro_asr']}"
        assert abs(point_frr[m] - summaries[m]['macro']['macro_benign_frr']) < 1e-9, \
            f"{m}: bootstrap point FRR {point_frr[m]} != 50's summary {summaries[m]['macro']['macro_benign_frr']}"
    print("Point estimates cross-checked against 50_join_and_summarize_test.py's own summary -- consistent.")

    rng = random.Random(BOOTSTRAP_SEED)
    asr_estimates = bootstrap_estimates(harmful_idx, harmful_ids, 'asr', args.n_boot, rng)
    frr_estimates = bootstrap_estimates(benign_idx, benign_ids, 'frr', args.n_boot, rng)

    per_method = {}
    for m in METHODS:
        per_method[m] = {
            'macro_asr': point_asr[m], 'macro_asr_ci': ci_from_values(asr_estimates[m]),
            'macro_benign_frr': point_frr[m], 'macro_benign_frr_ci': ci_from_values(frr_estimates[m]),
        }

    comparisons = {
        'adaptive_minus_fixed_wei_ASR': paired_delta(asr_estimates, point_asr, 'adaptive', 'fixed_wei'),
        'adaptive_minus_fixed_wei_FRR': paired_delta(frr_estimates, point_frr, 'adaptive', 'fixed_wei'),
        'fixed_wei_minus_no_defence_ASR': paired_delta(asr_estimates, point_asr, 'fixed_wei', 'no_defence'),
        'adaptive_minus_no_defence_ASR': paired_delta(asr_estimates, point_asr, 'adaptive', 'no_defence'),
    }

    print("\n-- Per-method macro ASR/FRR + 95% CI --")
    for m in METHODS:
        d = per_method[m]
        print(f"  {m}: ASR={d['macro_asr']:.4f} CI=[{d['macro_asr_ci']['ci_lo']:.4f}, {d['macro_asr_ci']['ci_hi']:.4f}]  "
              f"FRR={d['macro_benign_frr']:.4f} CI=[{d['macro_benign_frr_ci']['ci_lo']:.4f}, {d['macro_benign_frr_ci']['ci_hi']:.4f}]")

    print("\n-- Paired comparisons (same-replicate resampled instruction IDs) --")
    for name, c in comparisons.items():
        print(f"  {name}: delta={c['point_delta']:.4f}  95% CI=[{c['ci_lo']:.4f}, {c['ci_hi']:.4f}]")

    result = {
        'model': model_alias, 'n_boot': args.n_boot, 'bootstrap_seed': BOOTSTRAP_SEED,
        'harmful_test_ids_count': len(harmful_ids), 'benign_test_ids_count': len(benign_ids),
        'per_method': per_method,
        'per_template': {m: summaries[m]['per_template'] for m in METHODS},
        'parse_failures': {m: audits[m]['parse_failures'] for m in METHODS},
        'comparisons': comparisons,
        'join_audit_overall_pass': {m: audits[m]['overall_pass'] for m in METHODS},
        'git_commit': dm.git_commit_hash(), 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    }

    out_dir = os.path.join(args.output_path, 'canonical_v2')
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, f'experiment3_test_bootstrap_{model_alias}.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {result_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_idx', type=int, required=True)
    parser.add_argument('--output_path', type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    parser.add_argument('--n_boot', type=int, default=10000)
    args = parser.parse_args()
    main(args)
