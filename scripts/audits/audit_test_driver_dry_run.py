"""
GPU-free tests for scripts/49_defence_test_driver.py: test-split prompt
builders (against the REAL data/splits.json test_ids and
data/benign_test_100.json, not synthetic), and verify_against_frozen_config's
four independent stop conditions (direction hash, generation hash, layer,
Adaptive grouping). Does NOT exercise run_test_intervention_method/
run_test_no_defence_benign/run_test_no_defence_harmful_rejudge themselves --
those need a real model+GPU, exactly like 40_defence_generation_driver.py's
own validation-phase functions; their hook/chunking logic is the SAME
already-tested code from 40 (see audit_defence_driver_chunking_dry_run.py),
reused here via direct import, not reimplemented.

Usage:
  python scripts/audits/audit_test_driver_dry_run.py
"""
import os
import sys
from importlib import import_module

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
test_drv = import_module('49_defence_test_driver')
from _taxonomy_v2_loader import load_taxonomy_v2

taxonomy = load_taxonomy_v2()
active_mechs = taxonomy['active_mechanisms']

# 1. prompt builders -- exact counts against REAL data
harmful_prompts, test_ids = test_drv.build_test_harmful_prompts(active_mechs)
assert len(harmful_prompts) == 1200
assert len(test_ids) == 200
assert all(p['benign_or_harmful'] == 'harmful' for p in harmful_prompts)
print("Test 1 PASSED: 1200 harmful test prompts (200 real test_ids x 6 templates).")

benign_prompts, benign_ids = test_drv.build_test_benign_prompts(active_mechs)
assert len(benign_prompts) == 600
assert len(benign_ids) == 100
assert all(p['benign_or_harmful'] == 'benign' for p in benign_prompts)
print("Test 2 PASSED: 600 benign test prompts (100 real benign_test_100 ids x 6 templates).")

# 2. verify_against_frozen_config -- happy path + 4 independent stop conditions
FAKE_FROZEN = {
    'per_model': {
        'FakeModel': {
            'direction_config_hash': 'DCH', 'generation_config_hash': 'GCH',
            'fixed_layer_0based': 19, 'adaptive_grouping': {'template_specific': ['a'], 'subgroups': {}},
        }
    }
}


class FakeExp3Coverage:
    FIXED_LAYERS = {'FakeModel': 19}


class FakeHooksMod:
    FROZEN_ADAPTIVE_GROUPING = {'FakeModel': {'template_specific': ['a'], 'subgroups': {}}}


orig_exp3, orig_hooks = test_drv.drv.exp3_coverage, test_drv.drv.hooks_mod
test_drv.drv.exp3_coverage, test_drv.drv.hooks_mod = FakeExp3Coverage(), FakeHooksMod()
try:
    test_drv.verify_against_frozen_config(FAKE_FROZEN, 'FakeModel', 'DCH', 'GCH')
    print("Test 3 PASSED: happy path (all 4 checks match) does not raise.")

    try:
        test_drv.verify_against_frozen_config(FAKE_FROZEN, 'FakeModel', 'WRONG_DCH', 'GCH')
        raise SystemExit("FAILED: expected FrozenConfigViolationError for a direction hash mismatch")
    except test_drv.FrozenConfigViolationError as e:
        assert 'direction_config_hash' in str(e)
        print(f"Test 4 PASSED: a direction_config_hash mismatch stops immediately: {str(e)[:70]}")

    try:
        test_drv.verify_against_frozen_config(FAKE_FROZEN, 'FakeModel', 'DCH', 'WRONG_GCH')
        raise SystemExit("FAILED: expected FrozenConfigViolationError for a generation hash mismatch")
    except test_drv.FrozenConfigViolationError as e:
        assert 'generation_config_hash' in str(e)
        print(f"Test 5 PASSED: a generation_config_hash mismatch stops immediately: {str(e)[:70]}")

    FakeExp3Coverage.FIXED_LAYERS = {'FakeModel': 999}  # wrong layer
    try:
        test_drv.verify_against_frozen_config(FAKE_FROZEN, 'FakeModel', 'DCH', 'GCH')
        raise SystemExit("FAILED: expected FrozenConfigViolationError for a layer mismatch")
    except test_drv.FrozenConfigViolationError as e:
        assert 'fixed_layer' in str(e)
        print(f"Test 6 PASSED: a fixed-layer mismatch stops immediately: {str(e)[:70]}")
    finally:
        FakeExp3Coverage.FIXED_LAYERS = {'FakeModel': 19}

    FakeHooksMod.FROZEN_ADAPTIVE_GROUPING = {'FakeModel': {'template_specific': ['DIFFERENT'], 'subgroups': {}}}
    try:
        test_drv.verify_against_frozen_config(FAKE_FROZEN, 'FakeModel', 'DCH', 'GCH')
        raise SystemExit("FAILED: expected FrozenConfigViolationError for an Adaptive grouping mismatch")
    except test_drv.FrozenConfigViolationError as e:
        assert 'Adaptive grouping' in str(e)
        print(f"Test 7 PASSED: an Adaptive grouping mismatch stops immediately: {str(e)[:70]}")
finally:
    test_drv.drv.exp3_coverage, test_drv.drv.hooks_mod = orig_exp3, orig_hooks

# 3. frozen_alpha_for
FAKE_FROZEN_ALPHA = {'per_model': {'FakeModel': {'alpha': {'fixed_wei': 0.5, 'adaptive': 1.5}}}}
assert test_drv.frozen_alpha_for(FAKE_FROZEN_ALPHA, 'FakeModel', 'fixed_wei') == 0.5
assert test_drv.frozen_alpha_for(FAKE_FROZEN_ALPHA, 'FakeModel', 'adaptive') == 1.5
print("Test 8 PASSED: frozen_alpha_for reads the correct per-method alpha, never a sweep.")

print()
print("ALL TEST-DRIVER LOGIC TESTS PASSED.")
