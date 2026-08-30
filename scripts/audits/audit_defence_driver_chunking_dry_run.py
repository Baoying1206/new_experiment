"""
Mock-model_base tests for 40_defence_generation_driver.py's run_condition()
sub-batch chunking -- added after Gemma OOM'd at batch_size=60 (job 4979,
"Tried to allocate 16.82 GiB" from Gemma2's full-sequence float32 logits
during prefill) and CONDITION_BATCH_SIZE_OVERRIDE was introduced. Confirms
that splitting into multiple internal batches still creates a FRESH hook
(fresh has_intervened state) per batch -- reusing one hook/state object
across generate_completions() calls would silently skip the intervention
on every batch after the first.

No GPU needed -- uses a mock nn.Module standing in for the real decoder
layer and a mock model_base standing in for pipeline's ModelBase.

Usage:
  python scripts/audits/audit_defence_driver_chunking_dry_run.py
"""
import os
import sys
from importlib import import_module

import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
drv = import_module('40_defence_generation_driver')

# This test runs on machines without CUDA; run_condition's peak-memory calls
# are only meaningful on the real GPU cluster (already proven working there
# via the real Qwen/Llama job logs) -- stub them out to isolate chunking logic.
torch.cuda.reset_peak_memory_stats = lambda *a, **k: None
torch.cuda.max_memory_allocated = lambda *a, **k: -1

D_MODEL = 8
EOS_ID = 999


class MockLayer(nn.Module):
    def forward(self, x):
        return (x,)


class MockTokenizeResult:
    def __init__(self, n):
        self.attention_mask = torch.ones(n, 5, dtype=torch.long)


class MockModelBase:
    def __init__(self):
        self.tokenizer = type('T', (), {'eos_token_id': EOS_ID})()

    def tokenize_instructions_fn(self, instructions, system=None):
        return MockTokenizeResult(len(instructions))

    def generate_completions(self, dataset, fwd_pre_hooks, fwd_hooks, batch_size, max_new_tokens):
        for module, hook_fn in fwd_hooks:
            handle = module.register_forward_hook(hook_fn)
            try:
                fake_hidden = torch.randn(len(dataset), 7, D_MODEL)  # prefill, seq_len=7>1
                module(fake_hidden)
                for _ in range(2):  # simulated decode steps
                    module(torch.randn(len(dataset), 1, D_MODEL))
            finally:
                handle.remove()
        return [{'response': f'resp{i}', 'generation_tokens': f'1 2 {EOS_ID}', 'instruction_en': f'instr{i}'}
                for i in range(len(dataset))]


class BrokenModelBase(MockModelBase):
    """Never actually calls the hook -- simulates a wrong layer_module reference."""
    def generate_completions(self, dataset, fwd_pre_hooks, fwd_hooks, batch_size, max_new_tokens):
        return [{'response': 'r', 'generation_tokens': f'1 2 {EOS_ID}', 'instruction_en': 'i'} for _ in dataset]


def make_prompts(n):
    mechs = ['prefix_injection', 'refusal_suppression', 'persona_roleplay',
             'encoding_obfuscation', 'payload_splitting', 'distractors_negated']
    return [{'instruction_id': f'p{i}', 'benign_or_harmful': 'harmful', 'template': mechs[i % 6],
             'instruction': f'fake prompt {i}'} for i in range(n)]


def main():
    layer_module = MockLayer()
    model_base = MockModelBase()
    c_G = torch.randn(D_MODEL)
    prompts = make_prompts(10)

    per_record, metrics, states, audit_log = drv.run_condition(
        model_base, layer_module, prompts, 'hook_alpha_zero', c_G, batch_size=None)
    assert len(per_record) == 10
    assert metrics['n_batches'] == 1
    assert metrics['intervention_count_distribution'] == [1]
    assert metrics['intervention_count_all_batches_equal_one'] is True
    print("Test A PASSED: single-batch chunking -- 1 batch, intervention_count_distribution=[1].")

    per_record_b, metrics_b, states_b, audit_log_b = drv.run_condition(
        model_base, layer_module, prompts, 'global_alpha_one', c_G, batch_size=3)
    assert len(per_record_b) == 10
    assert metrics_b['n_batches'] == 4, f"expected 4 batches (3+3+3+1), got {metrics_b['n_batches']}"
    assert metrics_b['intervention_count_distribution'] == [1, 1, 1, 1]
    assert metrics_b['intervention_count_all_batches_equal_one'] is True
    assert len(metrics_b['per_batch_wall_seconds']) == 4
    print("Test B PASSED: multi-batch chunking (3+3+3+1) -- fresh hook fired exactly once per batch.")

    for i, r in enumerate(per_record_b):
        assert r['instruction_id'] == f'p{i}', f"record {i} out of order after chunking: {r['instruction_id']}"
    print("Test B2 PASSED: record order preserved exactly across chunk boundaries.")

    per_record_c, metrics_c, states_c, _ = drv.run_condition(
        model_base, layer_module, prompts, 'no_hook', c_G, batch_size=3)
    assert states_c == []
    assert metrics_c['intervention_count_distribution'] == []
    assert metrics_c['intervention_count_all_batches_equal_one'] is None
    print("Test C PASSED: no_hook creates no hook state regardless of batch_size.")

    broken = BrokenModelBase()
    try:
        drv.run_condition(broken, layer_module, prompts, 'hook_alpha_zero', c_G, batch_size=None)
        raise SystemExit("FAILED: expected AssertionError when the hook never fires")
    except AssertionError as e:
        print(f"Test D PASSED: run_condition raises immediately when a batch's hook never fires: {str(e)[:80]}")

    print()
    print("ALL run_condition CHUNKING TESTS PASSED (mock model_base, no GPU).")


if __name__ == '__main__':
    main()
