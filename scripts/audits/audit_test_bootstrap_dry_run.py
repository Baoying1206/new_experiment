"""
Tests for scripts/51_test_bootstrap_analysis.py -- the statistically
critical piece of the test phase. The single most important test here is
Test 3: when two methods have IDENTICAL outcomes, the paired bootstrap
delta must be EXACTLY 0 on every single replicate (not just close to 0 on
average) -- this is only true because both methods are evaluated against
the SAME resampled instruction-ID draw within each replicate. If the
resampling were done independently per method (a real bug this design
must avoid), two identical distributions would still show nonzero delta
variance from independent sampling noise, and this test would fail.

No GPU, no torch.

Usage:
  python scripts/audits/audit_test_bootstrap_dry_run.py
"""
import json
import os
import shutil
import sys
import tempfile
from importlib import import_module

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
boot_mod = import_module('51_test_bootstrap_analysis')


def make_outcome(parse_success=True, refusal=0, response_harmful=1, request_harmful=1):
    return {'parse_success': parse_success, 'refusal': refusal,
            'response_harmful': response_harmful, 'request_harmful': request_harmful}


def make_joined_rows(ids, templates, boh, outcome_fn):
    rows = []
    for iid in ids:
        for t in templates:
            o = outcome_fn(iid, t)
            rows.append({'instruction_id': iid, 'template': t, 'benign_or_harmful': boh, **o})
    return rows


def main():
    ids = [f'i{k}' for k in range(20)]
    templates = ['t1', 't2', 't3']

    # ---- Test 1: build_outcome_index ----
    rows = make_joined_rows(ids, templates, 'harmful', lambda i, t: make_outcome())
    idx = boot_mod.build_outcome_index(rows, 'harmful')
    assert set(idx.keys()) == set(templates)
    assert set(idx['t1'].keys()) == set(ids)
    assert idx['t1']['i0']['refusal'] == 0
    print("Test 1 PASSED: build_outcome_index correctly nests {template: {instruction_id: outcome}}.")

    # ---- Test 2: macro_metric_for_ids point-estimate correctness (hand-computable) ----
    # t1: always succeeds (ASR=1.0), t2: always refuses (ASR=0.0), t3: half succeed -> ASR=0.5
    def outcome_fn(i, t):
        idx_i = int(i[1:])
        if t == 't1':
            return make_outcome(refusal=0, response_harmful=1)
        if t == 't2':
            return make_outcome(refusal=1, response_harmful=0)
        return make_outcome(refusal=0 if idx_i % 2 == 0 else 1, response_harmful=1 if idx_i % 2 == 0 else 0)

    rows2 = make_joined_rows(ids, templates, 'harmful', outcome_fn)
    idx2 = boot_mod.build_outcome_index(rows2, 'harmful')
    point = boot_mod.macro_metric_for_ids(idx2, ids, 'asr')
    expected = (1.0 + 0.0 + 0.5) / 3
    assert abs(point - expected) < 1e-9, (point, expected)
    print(f"Test 2 PASSED: macro_metric_for_ids matches the hand-computed macro ASR "
          f"(1.0, 0.0, 0.5 across 3 templates -> mean {expected}): {point}")

    # ---- Test 3 (CRITICAL): identical methods -> paired delta EXACTLY 0 every replicate ----
    rows_a = make_joined_rows(ids, templates, 'harmful', outcome_fn)
    rows_b = make_joined_rows(ids, templates, 'harmful', outcome_fn)  # byte-identical outcome pattern
    idx_a = boot_mod.build_outcome_index(rows_a, 'harmful')
    idx_b = boot_mod.build_outcome_index(rows_b, 'harmful')
    import random
    rng = random.Random(42)
    estimates = boot_mod.bootstrap_estimates({'a': idx_a, 'b': idx_b}, ids, 'asr', n_boot=500, rng=rng)
    deltas = [a - b for a, b in zip(estimates['a'], estimates['b'])]
    assert all(d == 0.0 for d in deltas), f"identical methods must have delta=0 on EVERY replicate, got e.g. {deltas[:5]}"
    print(f"Test 3 PASSED (CRITICAL): two methods with byte-identical outcomes produce a paired delta "
          f"of EXACTLY 0.0 on all {len(deltas)} bootstrap replicates -- confirms the same resampled "
          f"instruction-ID draw is shared across methods within each replicate, not resampled independently.")

    # ---- Test 4: known separation -> point delta and CI both reflect it tightly ----
    rows_good = make_joined_rows(ids, templates, 'harmful', lambda i, t: make_outcome(refusal=0, response_harmful=1))  # ASR=1.0
    rows_bad = make_joined_rows(ids, templates, 'harmful', lambda i, t: make_outcome(refusal=1, response_harmful=0))   # ASR=0.0
    idx_good = boot_mod.build_outcome_index(rows_good, 'harmful')
    idx_bad = boot_mod.build_outcome_index(rows_bad, 'harmful')
    rng2 = random.Random(43)
    estimates2 = boot_mod.bootstrap_estimates({'good': idx_good, 'bad': idx_bad}, ids, 'asr', n_boot=500, rng=rng2)
    point_good = boot_mod.macro_metric_for_ids(idx_good, ids, 'asr')
    point_bad = boot_mod.macro_metric_for_ids(idx_bad, ids, 'asr')
    delta_info = boot_mod.paired_delta(estimates2, {'good': point_good, 'bad': point_bad}, 'good', 'bad')
    assert abs(delta_info['point_delta'] - 1.0) < 1e-9, delta_info
    assert delta_info['ci_lo'] == delta_info['ci_hi'] == 1.0, delta_info  # zero variance: always 1.0-0.0=1.0
    print(f"Test 4 PASSED: a method with ASR=1.0 everywhere vs ASR=0.0 everywhere gives point_delta=1.0 "
          f"with a degenerate [1.0, 1.0] CI (zero variance, exactly as expected for constant outcomes): {delta_info}")

    # ---- Test 5: ci_from_values / percentile sanity ----
    ci = boot_mod.ci_from_values([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    assert ci['n_valid_replicates'] == 10
    assert 0.1 <= ci['ci_lo'] <= 0.2 and 0.9 <= ci['ci_hi'] <= 1.0, ci
    print(f"Test 5 PASSED: ci_from_values produces a sane percentile interval for a known value set: {ci}")

    # ---- Test 6: main() refuses to run on an unverified (overall_pass=False) join_audit ----
    tmpdir = tempfile.mkdtemp()
    try:
        out_dir = os.path.join(tmpdir, 'canonical_v2')
        os.makedirs(out_dir, exist_ok=True)
        model_alias = 'FakeModel'
        for method in ('no_defence', 'fixed_wei', 'adaptive'):
            joined_path = os.path.join(out_dir, f'experiment3_test_joined_{model_alias}_{method}.jsonl')
            with open(joined_path, 'w') as f:
                f.write(json.dumps({'instruction_id': 'i0', 'template': 't1', 'benign_or_harmful': 'harmful',
                                     **make_outcome()}) + '\n')
            audit_path = os.path.join(out_dir, f'experiment3_test_join_audit_{model_alias}_{method}.json')
            with open(audit_path, 'w') as f:
                json.dump({'overall_pass': method != 'fixed_wei', 'parse_failures': 0}, f)  # fixed_wei fails
            summary_path = os.path.join(out_dir, f'experiment3_test_summary_{model_alias}_{method}.json')
            with open(summary_path, 'w') as f:
                json.dump({'macro': {'macro_asr': 1.0, 'macro_benign_frr': 0.0}, 'per_template': {}}, f)

        orig_models = dict(boot_mod.MODELS)
        boot_mod.MODELS.clear()
        boot_mod.MODELS[0] = (model_alias, '/fake')
        try:
            import argparse
            try:
                boot_mod.main(argparse.Namespace(model_idx=0, output_path=tmpdir, n_boot=10))
                raise SystemExit("FAILED: expected RuntimeError for an unverified join_audit")
            except RuntimeError as e:
                assert 'overall_pass' in str(e)
                print(f"Test 6 PASSED: main() refuses to bootstrap when a method's join_audit has "
                      f"overall_pass=False: {str(e)[:70]}")
        finally:
            boot_mod.MODELS.clear()
            boot_mod.MODELS.update(orig_models)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print()
    print("ALL TEST-BOOTSTRAP TESTS PASSED.")


if __name__ == '__main__':
    main()
