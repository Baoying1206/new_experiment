"""
GPU-free synthetic dry run for scripts/37_defence_directions_and_hooks.py.
No real model, no `pipeline` import -- uses a mock nn.Module to simulate a
prefill call (seq_len>1) followed by several decode calls (seq_len==1) and
checks the hook's single-fire, last-token-only, non-fatal-anomaly behavior
exactly as it will be relied on by the (not yet written) generation driver.

Usage:
  python scripts/audits/audit_defence_hooks_dry_run.py
"""
import os
import sys
from importlib import import_module

import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
mod = import_module('37_defence_directions_and_hooks')


class MockLayer(nn.Module):
    def forward(self, x):
        return (x,)  # mimic an HF decoder layer returning a tuple


def main():
    torch.manual_seed(0)
    d_model = 8
    dtilde = {m: torch.randn(d_model) * (i + 1) for i, m in enumerate(mod.ACTIVE_MECHANISMS)}
    dtilde['placebo'] = torch.randn(d_model) * 0.3

    # 1. direction construction sanity across all 3 models' frozen groupings
    for model_alias in mod.FROZEN_ADAPTIVE_GROUPING:
        conds = mod.build_all_condition_directions(dtilde, model_alias)
        print(f"=== {model_alias} ===")
        for cond, mapping in conds.items():
            if '*' in mapping:
                print(f"  {cond}: shared vector, norm={mapping['*'].norm().item():.4f}")
            else:
                norms = {m: round(v.norm().item(), 4) for m, v in mapping.items()}
                print(f"  {cond}: {norms}")
        g = conds['global']['*']
        fw_co = conds['fixed_wei']['prefix_injection']
        assert not torch.allclose(g, fw_co), "global and fixed_wei CO direction should differ"
    print()

    # 2. c_G formula manual check
    group = ['prefix_injection', 'refusal_suppression', 'persona_roleplay']
    us = [mod.normalize(dtilde[m]) for m in group]
    g_manual = mod.normalize(torch.stack(us, 0).mean(0))
    s_manual = torch.tensor([dtilde[m].norm().item() for m in group]).median()
    c_manual = s_manual * g_manual
    c_fn = mod.build_c_G(dtilde, group)
    assert torch.allclose(c_manual, c_fn, atol=1e-6), "build_c_G mismatch vs manual formula"
    print("build_c_G matches manual formula exactly. OK.")
    print()

    # 3. hook firing behavior: 1 prefill (seq_len=5) + 4 decode steps (seq_len=1)
    batch_size = 3
    layer = MockLayer()
    per_row_c = torch.stack(
        [dtilde['prefix_injection'], dtilde['refusal_suppression'], dtilde['persona_roleplay']], 0)
    audit_log = []
    hook_fn, state = mod.make_prefill_last_token_hook(per_row_c, audit_log=audit_log)
    handle = layer.register_forward_hook(hook_fn)

    prefill_input = torch.randn(batch_size, 5, d_model)
    prefill_out = layer(prefill_input)
    last_pos_before = prefill_input[:, -1, :].clone()
    last_pos_after = prefill_out[0][:, -1, :]
    expected = last_pos_before - per_row_c
    assert torch.allclose(last_pos_after, expected, atol=1e-5), "prefill last-token subtraction incorrect"
    assert torch.allclose(prefill_out[0][:, :-1, :], prefill_input[:, :-1, :]), \
        "non-last prefill positions were modified!"
    print("Prefill intervention: last token modified correctly, other positions untouched. OK.")

    for step in range(4):
        decode_input = torch.randn(batch_size, 1, d_model)
        decode_out = layer(decode_input)
        assert torch.allclose(decode_out[0], decode_input), f"decode step {step} was modified! leakage detected."
    print("4 decode steps: none modified (has_intervened guard held). OK.")

    mod.assert_single_intervention(state)
    print(f"intervention_count == 1 confirmed. state={state}")
    assert len(audit_log) == 0
    print(f"audit_log empty as expected: {audit_log}")
    handle.remove()

    # 4. seq_len==1-before-prefill anomaly: warning, not a crash
    audit_log2 = []
    hook_fn2, state2 = mod.make_prefill_last_token_hook(per_row_c, audit_log=audit_log2)
    handle2 = layer.register_forward_hook(hook_fn2)
    bad_input = torch.randn(batch_size, 1, d_model)
    bad_out = layer(bad_input)
    assert torch.allclose(bad_out[0], bad_input), "should have skipped intervention on seq_len==1 first call"
    assert state2['intervention_count'] == 0
    assert len(audit_log2) == 1
    print(f"seq_len==1-first-call anomaly correctly skipped with warning: {audit_log2[0]}")
    handle2.remove()

    # 5. left-padding assertion helper
    am_good = torch.tensor([[0, 1, 1], [1, 1, 1]])
    mod.assert_left_padded(am_good)
    am_bad = torch.tensor([[1, 1, 0], [1, 1, 1]])
    try:
        mod.assert_left_padded(am_bad)
        raise SystemExit("FAILED: should have raised on bad (right-padded) attention_mask")
    except AssertionError:
        print("assert_left_padded correctly rejects a right-padded row. OK.")

    print()
    print("ALL DRY-RUN CHECKS PASSED.")


if __name__ == '__main__':
    main()
