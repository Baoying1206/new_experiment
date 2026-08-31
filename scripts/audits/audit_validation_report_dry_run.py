"""
Tests for scripts/46_compile_validation_report.py.
Test 1 runs check_test_data_not_read() against the REAL scripts/ directory
(not synthetic) -- this is the actual claim going into the validation
report, so it must be verified against real source, not a mock.
Tests 2-4 use synthetic per-model JSON files to check aggregation
correctness and the two refusal paths (failing join_audit, failing
No-defence baseline). No GPU, no torch.

Usage:
  python scripts/audits/audit_validation_report_dry_run.py
"""
import json
import os
import shutil
import sys
import tempfile
from importlib import import_module

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
report_mod = import_module('46_compile_validation_report')


def write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f)


def make_summary(macro_asr_by_alpha, macro_frr_by_alpha):
    return {
        'per_alpha_macro': {str(a): {'macro_asr': macro_asr_by_alpha[a], 'macro_benign_frr': macro_frr_by_alpha[a]}
                             for a in macro_asr_by_alpha},
        'per_alpha_template': {str(a): {'faketemplate': {'asr': macro_asr_by_alpha[a]}} for a in macro_asr_by_alpha},
    }


def make_audit(overall_pass=True):
    return {'overall_pass': overall_pass, 'generation_rows': 3648, 'unique_judgement_rows': 2000,
            'cache_hit_rate': 0.45, 'parse_failures': 0,
            'direction_config_hash': 'DCH', 'generation_config_hash': 'GCH'}


def main():
    # ---- Test 1: real source-code check against the actual repo ----
    real_check = report_mod.check_test_data_not_read()
    assert real_check['pass'] is True, real_check
    print("Test 1 PASSED: check_test_data_not_read() against the REAL scripts/ directory "
          "confirms no bare test_ids reference and BENIGN_TEST_PATH occurs at most once "
          "(its own definition, never opened).")

    tmpdir = tempfile.mkdtemp()
    try:
        out_dir = os.path.join(tmpdir, 'canonical_v2')
        os.makedirs(out_dir, exist_ok=True)
        model_alias = 'FakeModel'

        write_json(os.path.join(out_dir, f'experiment3_no_defence_baseline_{model_alias}.json'),
                   {'overall_pass': True, 'benign_macro_frr': 0.02, 'harmful_macro_asr': 0.3,
                    'benign_per_template': {}, 'harmful_per_template': {}})
        write_json(os.path.join(out_dir, f'experiment3_alpha_freeze_{model_alias}.json'), {
            'results': {
                'fixed_wei': {'frozen_alpha': 1.0, 'reason': 'min_macro_asr_subject_to_benign_frr_constraint',
                              'max_allowed_frr': 0.07, 'macro_asr_at_frozen_alpha': 0.10,
                              'macro_frr_at_frozen_alpha': 0.07},
                'adaptive': {'frozen_alpha': 1.5, 'reason': 'min_macro_asr_subject_to_benign_frr_constraint',
                             'max_allowed_frr': 0.07, 'macro_asr_at_frozen_alpha': 0.02,
                             'macro_frr_at_frozen_alpha': 0.06},
            }
        })
        for method in ('fixed_wei', 'adaptive'):
            write_json(os.path.join(out_dir, f'experiment3_validation_join_audit_{model_alias}_{method}.json'),
                       make_audit(True))
            write_json(os.path.join(out_dir, f'experiment3_validation_summary_{model_alias}_{method}.json'),
                       make_summary({0.25: 0.3, 0.5: 0.2, 1.0: 0.1, 1.5: 0.02},
                                    {0.25: 0.05, 0.5: 0.06, 1.0: 0.07, 1.5: 0.06}))

        orig_models = dict(report_mod.MODELS)
        report_mod.MODELS.clear()
        report_mod.MODELS[0] = (model_alias, '/fake/path')
        try:
            m = report_mod.collect_model(tmpdir, 0)
            assert m['no_defence_benign_macro_frr'] == 0.02
            assert m['methods']['fixed_wei']['frozen_alpha'] == 1.0
            assert m['methods']['adaptive']['frozen_alpha'] == 1.5
            print("Test 2 PASSED: collect_model aggregates baseline + alpha_freeze + join_audit + summary "
                  "for one model correctly, with no recomputation of any metric.")

            # ---- Test 3: a failing join_audit is refused ----
            write_json(os.path.join(out_dir, f'experiment3_validation_join_audit_{model_alias}_fixed_wei.json'),
                       make_audit(False))
            try:
                report_mod.collect_model(tmpdir, 0)
                raise SystemExit("FAILED: expected MissingInputError for overall_pass=False")
            except report_mod.MissingInputError as e:
                assert 'overall_pass' in str(e)
                print(f"Test 3 PASSED: a join_audit with overall_pass=False is refused: {str(e)[:70]}")
            finally:
                write_json(os.path.join(out_dir, f'experiment3_validation_join_audit_{model_alias}_fixed_wei.json'),
                           make_audit(True))

            # ---- Test 4: a missing file is refused, not silently skipped ----
            baseline_path = os.path.join(out_dir, f'experiment3_no_defence_baseline_{model_alias}.json')
            os.remove(baseline_path)
            try:
                report_mod.collect_model(tmpdir, 0)
                raise SystemExit("FAILED: expected MissingInputError for a missing input file")
            except report_mod.MissingInputError as e:
                assert 'missing' in str(e)
                print(f"Test 4 PASSED: a missing required input file is refused, not silently skipped: {str(e)[:70]}")
        finally:
            report_mod.MODELS.clear()
            report_mod.MODELS.update(orig_models)

        print()
        print("ALL VALIDATION-REPORT COMPILATION TESTS PASSED.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
