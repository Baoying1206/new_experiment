"""
Confirms run_judge's field-passthrough works for BOTH record schemas it's
actually called with: timing-pilot records (use 'condition') and validation
records (use 'method'/'benign_or_harmful', no 'condition'). Added after
discovering run_judge originally hardcoded r['condition'], which would have
KeyError'd the first time run_validation_judge/run_no_defence_harmful_rejudge
called it on validation-schema records missing that key.

No GPU needed -- fakes WildGuard's tokenizer/model minimally.

Usage:
  python scripts/audits/audit_run_judge_schema_dry_run.py
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

    def generate(self, **kw):
        n = kw['input_ids'].shape[0]
        return torch.zeros(n, 5, dtype=torch.long)


def main():
    script03 = types.SimpleNamespace(WILDGUARD_PROMPT='PROMPT {prompt} RESPONSE {response}')
    guard_tok, guard_model = FakeGuardTok(), FakeGuardModel()

    val_records = [{'instruction_id': 'p000', 'template': 'persona_roleplay', 'method': 'global',
                     'alpha': 1.0, 'benign_or_harmful': 'harmful', 'model': 'Llama',
                     'instruction_en': 'x', 'response': 'y'}]
    judgements, _, _ = drv.run_judge(val_records, guard_model, guard_tok, script03)
    j = judgements[0]
    assert j['method'] == 'global' and j['benign_or_harmful'] == 'harmful' and j['template'] == 'persona_roleplay'
    assert 'condition' not in j
    print(f"Test 1 PASSED: run_judge passes through method/benign_or_harmful for validation-schema records: {j}")

    tp_records = [{'instruction_id': 'p001', 'template': 'prefix_injection', 'condition': 'hook_alpha_zero',
                   'alpha': 0.0, 'instruction_en': 'x2', 'response': 'y2'}]
    judgements2, _, _ = drv.run_judge(tp_records, guard_model, guard_tok, script03)
    j2 = judgements2[0]
    assert j2['condition'] == 'hook_alpha_zero' and 'method' not in j2
    print(f"Test 2 PASSED: run_judge still works for timing-pilot-schema records (condition field): {j2}")

    print()
    print("ALL run_judge SCHEMA TESTS PASSED.")


if __name__ == '__main__':
    main()
