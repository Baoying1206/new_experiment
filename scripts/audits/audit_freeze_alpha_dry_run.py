"""
Synthetic-data tests for scripts/45_freeze_alpha.py. Builds fake
join_audit/summary/no_defence_baseline files and checks: correct alpha
frozen via the real select_alpha function, refusal when a join_audit's
overall_pass is False, refusal when the No-defence baseline's overall_pass
is False. No GPU, no torch.

Usage:
  python scripts/audits/audit_freeze_alpha_dry_run.py
"""
import json
import os
import shutil
import sys
import tempfile
from importlib import import_module

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
freeze_mod = import_module('45_freeze_alpha')


def write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f)


def make_summary(macro_asr_by_alpha, macro_frr_by_alpha):
    return {'per_alpha_macro': {
        str(a): {'macro_asr': macro_asr_by_alpha[a], 'macro_benign_frr': macro_frr_by_alpha[a]}
        for a in macro_asr_by_alpha
    }}


def main():
    tmpdir = tempfile.mkdtemp()
    try:
        model_alias = 'FakeModel'
        out_dir = os.path.join(tmpdir, 'canonical_v2')
        os.makedirs(out_dir, exist_ok=True)

        # fixed_wei: alpha=1.0 is the best ASR that stays within the FRR budget
        fw_asr = {0.25: 0.30, 0.5: 0.20, 1.0: 0.10, 1.5: 0.05}
        fw_frr = {0.25: 0.05, 0.5: 0.06, 1.0: 0.07, 1.5: 0.20}  # 1.5 blows the budget
        write_json(os.path.join(out_dir, f'experiment3_validation_join_audit_{model_alias}_fixed_wei.json'),
                   {'overall_pass': True})
        write_json(os.path.join(out_dir, f'experiment3_validation_summary_{model_alias}_fixed_wei.json'),
                   make_summary(fw_asr, fw_frr))

        # adaptive: better ASR everywhere, same FRR pattern
        ad_asr = {0.25: 0.25, 0.5: 0.15, 1.0: 0.05, 1.5: 0.02}
        ad_frr = {0.25: 0.05, 0.5: 0.06, 1.0: 0.07, 1.5: 0.20}
        write_json(os.path.join(out_dir, f'experiment3_validation_join_audit_{model_alias}_adaptive.json'),
                   {'overall_pass': True})
        write_json(os.path.join(out_dir, f'experiment3_validation_summary_{model_alias}_adaptive.json'),
                   make_summary(ad_asr, ad_frr))

        write_json(os.path.join(out_dir, f'experiment3_no_defence_baseline_{model_alias}.json'),
                   {'overall_pass': True, 'benign_macro_frr': 0.02, 'harmful_macro_asr': 0.30})
        # no_defence_macro_frr=0.02 -> max_allowed = 0.02+0.05 = 0.07
        # fixed_wei eligible: {0.25(frr .05), 0.5(.06), 1.0(.07)} -> min asr among these -> 1.0 (asr .10)
        # adaptive eligible: same alphas eligible -> min asr -> 1.0 (asr .05)

        import argparse
        args = argparse.Namespace(model_idx=999, output_path=tmpdir)
        orig_models = dict(freeze_mod.MODELS)
        freeze_mod.MODELS[999] = (model_alias, '/fake/path')
        try:
            freeze_mod.main(args)
        finally:
            freeze_mod.MODELS.clear()
            freeze_mod.MODELS.update(orig_models)

        result_path = os.path.join(out_dir, f'experiment3_alpha_freeze_{model_alias}.json')
        with open(result_path) as f:
            result = json.load(f)

        assert result['results']['fixed_wei']['frozen_alpha'] == 1.0, result['results']['fixed_wei']
        assert result['results']['fixed_wei']['reason'] == 'min_macro_asr_subject_to_benign_frr_constraint'
        assert result['results']['adaptive']['frozen_alpha'] == 1.0, result['results']['adaptive']
        print("Test 1 PASSED: alpha=1.0 frozen for both fixed_wei and adaptive "
              "(lowest ASR among FRR-eligible alphas, matching the hand-derived expectation).")

        assert result['no_defence_benign_macro_frr'] == 0.02
        assert result['protocol_version'] == 'exp3_reduced_v1'
        assert result['alpha_eligible_methods'] == ['fixed_wei', 'adaptive']
        print("Test 2 PASSED: output records protocol_version, the No-defence baseline used, "
              "and confirms only fixed_wei/adaptive were eligible.")

        # ---- Test 3: a failing join_audit must refuse, not silently proceed ----
        write_json(os.path.join(out_dir, f'experiment3_validation_join_audit_{model_alias}_fixed_wei.json'),
                   {'overall_pass': False})
        try:
            freeze_mod.load_method_summary(tmpdir, model_alias, 'fixed_wei')
            raise SystemExit("FAILED: expected UnverifiedInputError for overall_pass=False")
        except freeze_mod.UnverifiedInputError as e:
            assert 'overall_pass' in str(e)
            print(f"Test 3 PASSED: a join_audit with overall_pass=False is refused: {str(e)[:70]}")
        finally:
            write_json(os.path.join(out_dir, f'experiment3_validation_join_audit_{model_alias}_fixed_wei.json'),
                       {'overall_pass': True})

        # ---- Test 4: a failing No-defence baseline must refuse, not silently proceed ----
        write_json(os.path.join(out_dir, f'experiment3_no_defence_baseline_{model_alias}.json'),
                   {'overall_pass': False, 'benign_macro_frr': 0.02, 'harmful_macro_asr': 0.30})
        try:
            freeze_mod.load_no_defence_baseline(tmpdir, model_alias)
            raise SystemExit("FAILED: expected UnverifiedInputError for a failing No-defence baseline")
        except freeze_mod.UnverifiedInputError as e:
            assert 'overall_pass' in str(e)
            print(f"Test 4 PASSED: a No-defence baseline with overall_pass=False is refused: {str(e)[:70]}")

        print()
        print("ALL ALPHA-FREEZE TESTS PASSED.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
