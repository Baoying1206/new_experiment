"""
Tests for scripts/47_freeze_test_config.py.
Test 1 verifies the REAL scripts/37_defence_directions_and_hooks.py against
the hardcoded EXPECTED_ADAPTIVE_GROUPING (not a mock -- this is the actual
claim going into the frozen test config). Tests 2-4 use synthetic
alpha_freeze/join_audit files to check the three refusal paths (wrong
alpha, hash mismatch between fixed_wei/adaptive) and the happy path.
No GPU, no torch.

Usage:
  python scripts/audits/audit_freeze_test_config_dry_run.py
"""
import json
import os
import shutil
import sys
import tempfile
from importlib import import_module

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
freeze_mod = import_module('47_freeze_test_config')


def write_json(path, obj):
    with open(path, 'w') as f:
        json.dump(obj, f)


def main():
    # ---- Test 1: real source-code + git-log verification ----
    commit = freeze_mod.verify_adaptive_grouping_unmodified()
    assert commit, "expected a non-empty commit hash"
    print(f"Test 1 PASSED: verify_adaptive_grouping_unmodified() against the REAL "
          f"scripts/37_defence_directions_and_hooks.py confirms it matches the hardcoded "
          f"EXPECTED_ADAPTIVE_GROUPING and has exactly 1 commit in its git history: {commit}")

    tmpdir = tempfile.mkdtemp()
    try:
        out_dir = os.path.join(tmpdir, 'canonical_v2')
        os.makedirs(out_dir, exist_ok=True)

        def write_model_artifacts(model_alias, fw_alpha, ad_alpha, fw_hash, ad_hash):
            write_json(os.path.join(out_dir, f'experiment3_alpha_freeze_{model_alias}.json'), {
                'results': {
                    'fixed_wei': {'frozen_alpha': fw_alpha, 'reason': 'r'},
                    'adaptive': {'frozen_alpha': ad_alpha, 'reason': 'r'},
                }
            })
            write_json(os.path.join(out_dir, f'experiment3_validation_join_audit_{model_alias}_fixed_wei.json'),
                       {'direction_config_hash': fw_hash[0], 'generation_config_hash': fw_hash[1]})
            write_json(os.path.join(out_dir, f'experiment3_validation_join_audit_{model_alias}_adaptive.json'),
                       {'direction_config_hash': ad_hash[0], 'generation_config_hash': ad_hash[1]})

        orig_expected = freeze_mod.EXPECTED_ALPHA
        orig_models = dict(freeze_mod.MODELS)
        freeze_mod.MODELS.clear()
        freeze_mod.MODELS[0] = ('Qwen2.5-7B-Instruct', '/fake')
        freeze_mod.MODELS[1] = ('Meta-Llama-3.1-8B-Instruct', '/fake')
        freeze_mod.MODELS[2] = ('gemma-2-9b-it', '/fake')
        freeze_mod.EXPECTED_ALPHA = {
            'Qwen2.5-7B-Instruct': {'fixed_wei': 1.5, 'adaptive': 1.5},
            'Meta-Llama-3.1-8B-Instruct': {'fixed_wei': 0.5, 'adaptive': 1.5},
            'gemma-2-9b-it': {'fixed_wei': 0.5, 'adaptive': 0.25},
        }
        try:
            for alias, fw_a, ad_a in (('Qwen2.5-7B-Instruct', 1.5, 1.5),
                                       ('Meta-Llama-3.1-8B-Instruct', 0.5, 1.5),
                                       ('gemma-2-9b-it', 0.5, 0.25)):
                write_model_artifacts(alias, fw_a, ad_a, ('DCH_' + alias, 'GCH_' + alias),
                                       ('DCH_' + alias, 'GCH_' + alias))

            import argparse
            freeze_mod.main(argparse.Namespace(output_path=tmpdir))
            config_path = os.path.join(out_dir, 'experiment3_defence_frozen_config.json')
            with open(config_path) as f:
                config = json.load(f)
            assert config['per_model']['Qwen2.5-7B-Instruct']['alpha'] == {'fixed_wei': 1.5, 'adaptive': 1.5}
            assert config['per_model']['gemma-2-9b-it']['alpha'] == {'fixed_wei': 0.5, 'adaptive': 0.25}
            print("Test 2 PASSED: happy path writes a frozen config with the correct alpha per model.")

            # ---- Test 3: a wrong alpha is refused ----
            write_model_artifacts('gemma-2-9b-it', 0.5, 1.5,  # WRONG adaptive alpha (should be 0.25)
                                   ('DCH_gemma-2-9b-it', 'GCH_gemma-2-9b-it'),
                                   ('DCH_gemma-2-9b-it', 'GCH_gemma-2-9b-it'))
            try:
                freeze_mod.main(argparse.Namespace(output_path=tmpdir))
                raise SystemExit("FAILED: expected FrozenConfigMismatchError for a wrong alpha")
            except freeze_mod.FrozenConfigMismatchError as e:
                assert 'does not match the authorized expected value' in str(e)
                print(f"Test 3 PASSED: a frozen alpha not matching the authorized value is refused: {str(e)[:70]}")
            finally:
                write_model_artifacts('gemma-2-9b-it', 0.5, 0.25,
                                       ('DCH_gemma-2-9b-it', 'GCH_gemma-2-9b-it'),
                                       ('DCH_gemma-2-9b-it', 'GCH_gemma-2-9b-it'))

            # ---- Test 4: a hash mismatch between fixed_wei/adaptive is refused ----
            write_model_artifacts('gemma-2-9b-it', 0.5, 0.25,
                                   ('DCH_gemma-2-9b-it', 'GCH_gemma-2-9b-it'),
                                   ('DIFFERENT_HASH', 'GCH_gemma-2-9b-it'))
            try:
                freeze_mod.main(argparse.Namespace(output_path=tmpdir))
                raise SystemExit("FAILED: expected FrozenConfigMismatchError for a hash mismatch")
            except freeze_mod.FrozenConfigMismatchError as e:
                assert 'differ between fixed_wei and adaptive' in str(e)
                print(f"Test 4 PASSED: a direction/generation_config_hash mismatch between "
                      f"fixed_wei and adaptive is refused: {str(e)[:70]}")
        finally:
            freeze_mod.MODELS.clear()
            freeze_mod.MODELS.update(orig_models)
            freeze_mod.EXPECTED_ALPHA = orig_expected

        print()
        print("ALL FREEZE-TEST-CONFIG TESTS PASSED.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
