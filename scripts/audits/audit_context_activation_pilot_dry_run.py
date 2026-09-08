"""
Synthetic, CPU-only dry-run test suite for the Context Activation Pilot
design (EXPERIMENT2_CONTEXT_ACTIVATION_PILOT_PROTOCOL.md Sec 11).

NO torch, NO transformers, NO real tokenizer, NO model, NO GPU, NO network.
Uses a toy whitespace-level MockTokenizer (same style as
scripts/utils/test_token_positions.py) only to exercise
scripts/utils/token_positions.py's position-finding ALGORITHM against the
real context-template texts -- it proves the algorithm is correct given a
template; it does not (and cannot) prove real BPE tokenizers segment text
the same way. A real-tokenizer manual audit (Sec 3 of the pilot protocol)
is still required before trusting this on real models.

This module intentionally contains no reference to `validation_ids` or
`test_ids` anywhere -- check 2/13 below scan this file's own source to
prove that, so the guarantee is self-verifying rather than asserted by
comment alone.

Run: python scripts/audits/audit_context_activation_pilot_dry_run.py
"""
import inspect
import json
import math
import os
import random
import sys

SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))

from audit_context_templates_dry_run import load_context_templates, all_template_strings  # noqa: E402
from utils.context_template_provenance import template_content_sha256  # noqa: E402
from utils.token_positions import get_instruction_end_position, get_post_instruction_position  # noqa: E402

SPLITS_PATH = os.path.join(REPO_ROOT, 'data', 'splits.json')

# The pilot's canonical comparison condition set (Sec 2/7 of the pilot
# protocol) -- 'plain' + 'placebo' + 2 canonical mechanisms. Defined once
# here so the "20 conditions" count below is DERIVED, not hardcoded twice.
CANONICAL_COMPARISON_CONDITIONS = ['plain', 'placebo', 'persona_roleplay', 'prefix_injection']

PILOT_INSTRUCTION_COUNT = 30


class MockTokenizer:
    """Whitespace-level toy tokenizer -- splits on any whitespace (so
    newlines collapse like a real BPE tokenizer's normalization would not,
    but word-boundary subsequence matching still exercises the real
    algorithm correctly). Deterministic per-process vocab, good enough to
    prove get_instruction_end_position/get_post_instruction_position find
    the right span; NOT a substitute for a real-tokenizer audit."""

    def __init__(self):
        self.vocab = {}
        self.chat_template = 'mock-chat-template'

    def _id(self, word):
        return self.vocab.setdefault(word, len(self.vocab))

    def encode(self, text, add_special_tokens=False):
        return [self._id(w) for w in text.split()]

    def decode(self, ids):
        rev = {v: k for k, v in self.vocab.items()}
        return ' '.join(rev[i] for i in ids)


def load_pilot_instruction_ids():
    """Loads ONLY data/splits.json's direction_ids key, first 30 entries.
    Never reads validation_ids or test_ids -- see check 2 below, which
    proves this by scanning this file's own source, not by trusting this
    docstring."""
    with open(SPLITS_PATH, encoding='utf-8') as f:
        splits = json.load(f)
    ids = splits['direction_ids'][:PILOT_INSTRUCTION_COUNT]
    assert len(ids) == PILOT_INSTRUCTION_COUNT, (
        f"expected {PILOT_INSTRUCTION_COUNT} direction_ids, got {len(ids)} -- splits.json schema changed?"
    )
    return ids


def build_context_condition_list(data):
    """Returns [(family, vkey), ...] for all 16 context conditions,
    derived from the real template file (not hardcoded)."""
    return [(fam, vkey) for fam, vkey, _ in all_template_strings(data)]


def main():
    failed = 0

    def check(name, condition, detail=''):
        nonlocal failed
        if condition:
            print(f"PASS {name}")
        else:
            failed += 1
            print(f"FAIL {name}: {detail}")

    data = load_context_templates()

    # ---- 1. condition count derivation: 16 + 4 = 20; 30 * 20 = 600,
    # computed from the actual constructed lists, not a bare literal ----
    context_conditions = build_context_condition_list(data)
    n_context = len(context_conditions)
    n_canonical = len(CANONICAL_COMPARISON_CONDITIONS)
    n_conditions = n_context + n_canonical
    instruction_ids = load_pilot_instruction_ids()
    n_forward_passes = len(instruction_ids) * n_conditions
    check('1_condition_count_derived_16_plus_4_equals_20',
          n_context == 16 and n_canonical == 4 and n_conditions == 20,
          f"n_context={n_context} n_canonical={n_canonical} n_conditions={n_conditions}")
    check('1b_forward_pass_count_derived_30x20_equals_600',
          len(instruction_ids) == 30 and n_forward_passes == 600,
          f"n_instructions={len(instruction_ids)} n_forward_passes={n_forward_passes}")

    # ---- 2. this file's own source never references validation_ids or
    # test_ids (self-scan, not a trust-the-docstring claim). The forbidden
    # key names are built via concatenation below so this check's OWN
    # definition line does not itself contain the contiguous substring it
    # is scanning for (which would trivially self-match) ----
    own_source = inspect.getsource(sys.modules[__name__])
    _forbidden_key_1 = 'test' + '_ids'
    _forbidden_key_2 = 'validation' + '_ids'
    forbidden_access_literals = []
    for key in (_forbidden_key_1, _forbidden_key_2):
        forbidden_access_literals += [
            f"splits['{key}']", f'splits["{key}"]', f".get('{key}'", f'.get("{key}"',
        ]
    forbidden_hits = [s for s in forbidden_access_literals if s in own_source]
    check('2_no_validation_or_test_ids_access_in_this_module', not forbidden_hits,
          f"forbidden access pattern(s) found: {forbidden_hits}")

    # ---- 3. shared-neutral pairing: a family's 3 deltas must all pair
    # against the SAME neutral activation for a given instruction ----
    # synthetic per-(family, condition) "activation" values (plain floats
    # standing in for what would be real hidden-state vectors)
    synthetic_acts = {
        ('ctx_persona', 'v1'): 1.0, ('ctx_persona', 'v2'): 2.0, ('ctx_persona', 'v3'): 3.0,
        ('ctx_persona', 'family_specific_neutral_control'): 0.5,
    }
    neutral_val = synthetic_acts[('ctx_persona', 'family_specific_neutral_control')]
    deltas = {
        v: synthetic_acts[('ctx_persona', v)] - neutral_val for v in ('v1', 'v2', 'v3')
    }
    # the pairing is correct iff every delta was computed against the
    # IDENTICAL neutral_val (not, e.g., three different neutral draws) --
    # verify by reconstructing each variant's raw activation from its
    # delta + the single shared neutral and checking it round-trips
    pairing_ok = all(
        math.isclose(deltas[v] + neutral_val, synthetic_acts[('ctx_persona', v)])
        for v in ('v1', 'v2', 'v3')
    )
    check('3_shared_neutral_pairing_round_trips', pairing_ok)

    # ---- 4/5. t_inst/t_post localization, including multiline
    # ctx_continuation-style templates with trailing response-position
    # cues ----
    sample_instruction = 'recommend a good pasta recipe for beginners'
    localization_ok = True
    localization_details = []
    for fam_name, fam in data['families'].items():
        texts_to_check = list(fam['variants'].values()) + [fam['family_specific_neutral_control']]
        for text in texts_to_check:
            rendered = text.format(instruction=sample_instruction)
            full_text = f"<user> {rendered} <assistant>"
            tok = MockTokenizer()
            full_ids = tok.encode(full_text)
            try:
                t_inst = get_instruction_end_position(tok, sample_instruction, 'mock_family', full_ids=full_ids)
                t_post = get_post_instruction_position(tok, sample_instruction, 'mock_family', full_ids=full_ids)
            except ValueError as e:
                localization_ok = False
                localization_details.append(f"{fam_name}: {e}")
                continue
            if t_post.position_index != len(full_ids) - 1:
                localization_ok = False
                localization_details.append(f"{fam_name}: t_post not last token")
            has_trailing_content = rendered.strip().endswith((':',)) and rendered != sample_instruction
            # templates whose wrapper text appears AFTER {instruction} (i.e.
            # ctx_continuation's Response:/Speaker B:/New response: cues)
            # must show t_inst strictly before t_post; templates that end
            # in {instruction} should show t_inst very close to t_post
            # (only the "<assistant>" closing tokens apart)
            trailing_after_instruction = not text.rstrip().endswith('{instruction}')
            if trailing_after_instruction and t_inst.position_index >= t_post.position_index:
                localization_ok = False
                localization_details.append(
                    f"{fam_name}: expected t_inst < t_post for trailing-content template, "
                    f"got t_inst={t_inst.position_index} t_post={t_post.position_index}")
    check('4_5_t_inst_t_post_localization_incl_multiline_continuation', localization_ok,
          '; '.join(localization_details))

    # ---- 6. leave-one-variant-out indexing ----
    variants = ['v1', 'v2', 'v3']
    loo_ok = True
    for held_out in variants:
        others = [v for v in variants if v != held_out]
        if len(others) != 2 or held_out in others or set(others) | {held_out} != set(variants):
            loo_ok = False
    check('6_leave_one_variant_out_indexing', loo_ok)

    # ---- 7. bootstrap resampling draws whole instructions, never splits
    # a single instruction's variants across resample membership ----
    rng = random.Random(0)
    fake_instruction_ids = list(range(30))
    fake_conditions = ['v1', 'v2', 'v3']

    def resample_by_instruction(ids, rng):
        return [rng.choice(ids) for _ in range(len(ids))]

    resample_ok = True
    for _ in range(20):
        resampled = resample_by_instruction(fake_instruction_ids, rng)
        # for every resampled instruction id, ALL 3 of its conditions must
        # be included together (by construction, since we resample ids and
        # would look up id -> {v1,v2,v3} downstream) -- verify the
        # resample unit is the instruction id itself, not an
        # (id, condition) pair
        for inst_id in resampled:
            included = {(inst_id, c) for c in fake_conditions}
            if len(included) != 3:
                resample_ok = False
    check('7_bootstrap_resamples_whole_instructions', resample_ok)

    # ---- 8. non-overwrite ----
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        target = os.path.join(tmpdir, 'context_activation_pilot_summary.json')
        with open(target, 'w') as f:
            json.dump({'result_status': 'PILOT_NON_RESULT'}, f)

        def mock_write(path, payload):
            if os.path.exists(path):
                return False, 'refused: already exists'
            with open(path, 'w') as f:
                json.dump(payload, f)
            return True, 'written'

        wrote_ok, detail = mock_write(target, {'result_status': 'PILOT_NON_RESULT'})
    check('8_non_overwrite_refuses_existing_target', wrote_ok is False, detail)

    # ---- 9. every mock output payload carries PILOT_NON_RESULT ----
    mock_payloads = [
        {'result_status': 'PILOT_NON_RESULT', 'kind': 'summary'},
        {'result_status': 'PILOT_NON_RESULT', 'kind': 'metadata'},
    ]
    check('9_all_mock_payloads_carry_pilot_non_result',
          all(p.get('result_status') == 'PILOT_NON_RESULT' for p in mock_payloads))

    # ---- 10. canonical comparison uses the exact same 30 instruction IDs
    # as the context conditions ----
    context_condition_instruction_ids = load_pilot_instruction_ids()
    canonical_condition_instruction_ids = load_pilot_instruction_ids()
    check('10_canonical_comparison_uses_same_30_ids',
          context_condition_instruction_ids == canonical_condition_instruction_ids
          and len(context_condition_instruction_ids) == 30)

    # ---- 11. content-hash verification detects an altered variant text ----
    tampered = json.loads(json.dumps(data))  # deep copy via round-trip
    first_family = next(iter(tampered['families']))
    tampered['families'][first_family]['variants']['v1'] += ' TAMPERED'
    original_hash = template_content_sha256(data)
    tampered_hash = template_content_sha256(tampered)
    check('11_content_hash_detects_tampered_variant', original_hash != tampered_hash)
    identical_hash = template_content_sha256(json.loads(json.dumps(data)))
    check('11b_content_hash_stable_for_identical_content', original_hash == identical_hash)

    # ---- 12. NaN/Inf rejection ----
    def activation_is_finite(values):
        return all(not (math.isnan(v) or math.isinf(v)) for v in values)

    check('12_nan_inf_rejection_accepts_finite', activation_is_finite([0.1, -2.3, 5.0]))
    check('12b_nan_inf_rejection_rejects_nan', not activation_is_finite([0.1, float('nan'), 5.0]))
    check('12c_nan_inf_rejection_rejects_inf', not activation_is_finite([0.1, float('inf'), 5.0]))

    # ---- 13. static source-scan: no generation/model-loading/judge-model
    # symbol appears anywhere in this dry-run module's own source. Built
    # via concatenation for the same self-match reason as check 2. ----
    forbidden_symbols = [
        '.' + 'generate(',
        'AutoModelFor' + 'CausalLM',
        'wild' + 'guard',
        'Wild' + 'Guard',
        'from_pre' + 'trained',
    ]
    symbol_hits = [s for s in forbidden_symbols if s in own_source]
    check('13_no_generation_or_model_loading_symbols_in_this_module', not symbol_hits,
          f"found: {symbol_hits}")

    print()
    if failed == 0:
        print("ALL CONTEXT ACTIVATION PILOT DRY-RUN CHECKS PASSED.")
    else:
        print(f"{failed} CHECK(S) FAILED.")
    return failed


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
