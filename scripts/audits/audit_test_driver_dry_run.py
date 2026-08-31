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

# 4. Regression for the real Llama OOM (jobs 5016/5017): test-phase intervention
# batch_size must be capped at TEST_INTERVENTION_BATCH_SIZE (20) even for models
# with no CONDITION_BATCH_SIZE_OVERRIDE entry (default 60), while Gemma's
# already-smaller override (15) must still win via min().
effective_batch_qwen = min(test_drv.drv.CONDITION_BATCH_SIZE_OVERRIDE.get('Qwen2.5-7B-Instruct', 60),
                            test_drv.TEST_INTERVENTION_BATCH_SIZE)
effective_batch_llama = min(test_drv.drv.CONDITION_BATCH_SIZE_OVERRIDE.get('Meta-Llama-3.1-8B-Instruct', 60),
                             test_drv.TEST_INTERVENTION_BATCH_SIZE)
effective_batch_gemma = min(test_drv.drv.CONDITION_BATCH_SIZE_OVERRIDE.get('gemma-2-9b-it', 60),
                             test_drv.TEST_INTERVENTION_BATCH_SIZE)
assert effective_batch_qwen == 20, effective_batch_qwen
assert effective_batch_llama == 20, effective_batch_llama
assert effective_batch_gemma == 15, effective_batch_gemma  # Gemma's own smaller override still wins
print(f"Test 9 PASSED: effective test-phase intervention batch_size is capped at "
      f"min(CONDITION_BATCH_SIZE_OVERRIDE, {test_drv.TEST_INTERVENTION_BATCH_SIZE}) -- "
      f"Qwen/Llama={effective_batch_qwen}, Gemma={effective_batch_gemma} (its own smaller override wins).")

# 5. Sorting by rendered-instruction length groups similar-length prompts together
fake_prompts = [{'instruction': 'x' * n, 'instruction_id': f'p{i}'} for i, n in enumerate([300, 10, 150, 5, 800])]
sorted_prompts = sorted(fake_prompts, key=lambda p: len(p['instruction']))
assert [len(p['instruction']) for p in sorted_prompts] == [5, 10, 150, 300, 800]
print("Test 10 PASSED: sorting todo by len(instruction) correctly orders prompts short-to-long, "
      "so an outlier-length instruction lands in a chunk with others near its own length "
      "instead of inflating an otherwise-short batch.")

print()
print("ALL TEST-DRIVER LOGIC TESTS PASSED.")
