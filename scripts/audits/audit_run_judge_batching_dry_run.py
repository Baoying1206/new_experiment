"""
Confirms run_judge batches its WildGuard calls (never one giant forward pass)
and that on_new_batch fires with fully-merged, incrementally-persistable
judgement dicts per batch. Added after job 4983 (Llama x global judge stage)
OOM'd trying to judge all 3648 new records in a single WildGuard call
("Tried to allocate 36.68 GiB").

No GPU needed -- fakes WildGuard's tokenizer/model and records the batch
sizes it was actually called with.

Usage:
  python scripts/audits/audit_run_judge_batching_dry_run.py
"""
import os
import sys
import types
from importlib import import_module

import torch

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
drv = import_module('40_defence_generation_driver')


class FakeEnc(dict):
    def __init__(self, n):
        super().__init__(input_ids=torch.zeros(n, 3, dtype=torch.long))
        self.input_ids = self['input_ids']

    def to(self, device):
        return self


class FakeGuardTok:
    padding_side = 'left'
    eos_token_id = 0

    def __call__(self, texts, **kw):
        return FakeEnc(len(texts))

    def decode(self, ids, **kw):
        return 'Harmful request: yes\nResponse refusal: no\nHarmful response: yes'


class FakeGuardModel:
    device = 'cpu'

    def __init__(self):
        self.call_sizes = []

    def generate(self, **kw):
        n = kw['input_ids'].shape[0]
        self.call_sizes.append(n)
        return torch.zeros(n, 5, dtype=torch.long)


def main():
    script03 = types.SimpleNamespace(WILDGUARD_PROMPT='PROMPT {prompt} RESPONSE {response}')
    guard_tok, guard_model = FakeGuardTok(), FakeGuardModel()

    # 100 records, all-new (no cache), default judge_batch_size=16 -> 7 calls (16*6+4)
    records = [{'instruction_id': f'p{i}', 'template': 'persona_roleplay', 'method': 'global',
                'alpha': 1.0, 'benign_or_harmful': 'harmful', 'model': 'Llama',
                'instruction_en': f'x{i}', 'response': f'y{i}'} for i in range(100)]

    persisted_batches = []
    judgements, jmetrics, cache = drv.run_judge(
        records, guard_model, guard_tok, script03, on_new_batch=persisted_batches.append)

    assert guard_model.call_sizes == [16, 16, 16, 16, 16, 16, 4], guard_model.call_sizes
    print(f"Test 1 PASSED: 100 records batched into {len(guard_model.call_sizes)} WildGuard calls "
          f"of sizes {guard_model.call_sizes} (never one 100-row call).")

    assert sum(len(b) for b in persisted_batches) == 100
    assert len(persisted_batches) == 7
    for b in persisted_batches:
        for j in b:
            assert 'method' in j and 'instruction_id' in j and 'parse_success' in j
    print(f"Test 2 PASSED: on_new_batch fired {len(persisted_batches)} times with fully-merged, "
          f"persistable judgement dicts (100 total).")

    assert len(judgements) == 100
    assert jmetrics['n_judged_new'] == 100 and jmetrics['n_cache_hits'] == 0
    print("Test 3 PASSED: final return value still has all 100 judgements + correct metrics.")

    # Second call with the now-populated cache -- everything should be a cache hit,
    # zero new WildGuard calls, on_new_batch never fires.
    guard_model2 = FakeGuardModel()
    persisted_batches2 = []
    judgements2, jmetrics2, _ = drv.run_judge(
        records, guard_model2, guard_tok, script03, cache=cache, on_new_batch=persisted_batches2.append)
    assert guard_model2.call_sizes == []
    assert persisted_batches2 == []
    assert jmetrics2['n_cache_hits'] == 100 and jmetrics2['n_judged_new'] == 0
    print("Test 4 PASSED: fully-cached re-run makes zero WildGuard calls and never fires on_new_batch.")

    # Regression test for the real Llama x global rejudge (job 4985): 54 pairs of
    # records shared a judge_cache_key (identical instruction_en+response across
    # different templates/alphas) and landed in the SAME judge_batch_size chunk,
    # so both got judged and both got written -- 84 duplicate rows in the output
    # file. Build a batch with an intra-chunk duplicate key and confirm it's
    # judged exactly once and appears exactly once per on_new_batch call.
    dup_records = [
        {'instruction_id': 'a', 'template': 'persona_roleplay', 'method': 'global', 'alpha': 1.0,
         'benign_or_harmful': 'harmful', 'model': 'Llama', 'instruction_en': 'same text', 'response': 'same resp'},
        {'instruction_id': 'b', 'template': 'prefix_injection', 'method': 'global', 'alpha': 0.5,
         'benign_or_harmful': 'harmful', 'model': 'Llama', 'instruction_en': 'same text', 'response': 'same resp'},
        {'instruction_id': 'c', 'template': 'payload_splitting', 'method': 'global', 'alpha': 1.5,
         'benign_or_harmful': 'harmful', 'model': 'Llama', 'instruction_en': 'other text', 'response': 'other resp'},
    ]
    guard_model3 = FakeGuardModel()
    persisted_batches3 = []
    judgements3, jmetrics3, cache3 = drv.run_judge(
        dup_records, guard_model3, guard_tok, script03, on_new_batch=persisted_batches3.append)

    assert guard_model3.call_sizes == [2], (
        f"expected exactly 1 WildGuard call of size 2 (2 unique keys among 3 records), got {guard_model3.call_sizes}")
    assert jmetrics3['n_judged_new'] == 2, jmetrics3
    all_persisted = [j for b in persisted_batches3 for j in b]
    persisted_keys = [j['judge_cache_key'] for j in all_persisted]
    assert len(persisted_keys) == len(set(persisted_keys)) == 2, (
        f"on_new_batch must never emit the same judge_cache_key twice: {persisted_keys}")
    assert len(judgements3) == 3, "the returned per-record judgements list must still cover all 3 input records"
    assert judgements3[0]['judge_cache_key'] == judgements3[1]['judge_cache_key']
    assert judgements3[0]['request_harmful'] == judgements3[1]['request_harmful']
    print("Test 5 PASSED: two records sharing a judge_cache_key within the same call/batch are judged "
          "exactly once and written exactly once, while both still receive the merged judgement in the "
          "returned per-record list.")

    print()
    print("ALL run_judge BATCHING TESTS PASSED.")


if __name__ == '__main__':
    main()
