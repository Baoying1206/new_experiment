"""
Static, read-only audit for templates/templates_context_multiturn_v1.json
(the post-hoc exploratory multi-turn extension to the completed
single-turn Contextual Reconfiguration study -- see
EXPERIMENT_CONTEXT_MULTITURN_BEHAVIOR_PROTOCOL.md). NO model weights, NO
GPU, NO WildGuard, NO generation -- this script only constructs and
statically validates the 16 multi-turn message-list conditions using a
harmless placeholder instruction, plus mock per-family tokenizers as a
LOCAL-ONLY proxy for token-length comparison (real per-model token
counts require the actual cluster tokenizers -- flagged, not claimed
here as final numbers).

Run: python scripts/audits/audit_context_multiturn_templates_dry_run.py
"""
import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
TEMPLATE_PATH = os.path.join(REPO_ROOT, 'templates', 'templates_context_multiturn_v1.json')

FAMILIES = ['ctx_persona', 'ctx_authority', 'ctx_fictional', 'ctx_continuation']
VARIANTS = ['v1', 'v2', 'v3']

PLACEHOLDER_INSTRUCTION = "Explain how a widget factory schedules its weekly production run."

# Phrases that would confuse this design with an existing canonical
# jailbreak mechanism (protocol Sec 5) -- checked against setup_user and
# assistant_acknowledgement (final_user is always exactly "{instruction}",
# nothing else, so it is not scanned for these).
FORBIDDEN_PATTERNS = [
    r'\bignore\s+(the\s+)?(previous|prior|above)\s+instructions?\b',
    r'\bno\s+(rules|restrictions|limits|filters)\b',
    r'\byou\s+(are\s+)?(not\s+)?(bound|restricted|limited)\s+by\b',
    r'\bunlock(ed)?\b',
    r'\bdisregard\s+(your\s+)?(guidelines|policy|policies|training)\b',
    r'\b(safety|content)\s+(guideline|polic|filter|restriction)',
    r'\bjailbreak\b',
    r'\bDAN\b',
    r'\bsure,?\s+here',  # forced answer opening
    r'\bcertainly[,!]?\s+here',  # forced answer opening
    r'base64',
    r'\bpayload_?[ab]\b',
]


class AuditFailure(Exception):
    pass


def load_templates():
    with open(TEMPLATE_PATH, encoding='utf-8') as f:
        return json.load(f)


def build_conditions(data):
    """Returns list of (template_id, family, variant_or_neutral, cond_dict)."""
    conditions = []
    for fam in FAMILIES:
        fam_data = data['families'][fam]
        for v in VARIANTS:
            conditions.append((f'{fam}_{v}', fam, v, fam_data['variants'][v]))
        conditions.append((f'{fam}_neutral', fam, 'neutral', fam_data['family_specific_neutral_control']))
    return conditions


def render_messages(cond_dict, instruction):
    """Builds the real 3-message list (user/assistant/user) with the
    instruction substituted into final_user. This IS the actual messages
    structure a future generation driver would pass to
    tokenizer.apply_chat_template() -- see protocol Sec 2 for why the
    existing pipeline's hand-rolled single-turn formatters cannot be
    reused for this (no messages-list/apply_chat_template support
    anywhere in pipeline/model_utils/*)."""
    final_user = cond_dict['final_user'].format(instruction=instruction)
    return [
        {'role': 'user', 'content': cond_dict['setup_user']},
        {'role': 'assistant', 'content': cond_dict['assistant_acknowledgement']},
        {'role': 'user', 'content': final_user},
    ]


class MockTokenizer:
    """LOCAL-ONLY whitespace-word proxy for token count -- NOT a real
    tokenizer, NOT per-model-accurate. Used only to sanity-check that
    positive/neutral pairs are of comparable LENGTH before real cluster
    tokenizers are available; real token counts must be re-verified on
    the cluster before being cited as final (same discipline as every
    other mock-tokenizer use in this project)."""
    def __init__(self, family):
        self.family = family

    def count(self, text):
        # crude but consistent proxy: word count doesn't vary by model
        # family (a real BPE tokenizer would), which is itself worth
        # flagging -- this mock cannot detect family-specific
        # subword-splitting differences, only gross length mismatches.
        return len(text.split())


def main():
    failed = 0

    def check(name, condition, detail=''):
        nonlocal failed
        if condition:
            print(f"PASS {name}")
        else:
            failed += 1
            print(f"FAIL {name}: {detail}")

    check('0_template_file_exists', os.path.exists(TEMPLATE_PATH))
    if not os.path.exists(TEMPLATE_PATH):
        print(f"\n{failed} CHECK(S) FAILED.")
        return failed

    data = load_templates()
    check('0b_status_is_candidate_pending_review', data.get('status') == 'CANDIDATE_PENDING_HUMAN_REVIEW',
          f"status={data.get('status')}")
    check('0c_study_status_is_post_hoc_exploratory', data.get('study_status') == 'POST_HOC_EXPLORATORY_EXTENSION',
          f"study_status={data.get('study_status')}")
    check('0d_all_4_families_present', all(fam in data['families'] for fam in FAMILIES),
          f"families={list(data['families'].keys())}")

    conditions = build_conditions(data)
    check('1_exactly_16_conditions', len(conditions) == 16, f"got {len(conditions)}")

    # ---- 2. 16 unique template_ids ----
    ids = [c[0] for c in conditions]
    check('2_16_unique_template_ids', len(set(ids)) == 16, f"ids={ids}")

    # ---- 3. every condition renders to exactly 3 messages, role order
    # user/assistant/user ----
    role_order_ok = True
    rendered_by_id = {}
    for tid, fam, v, cond in conditions:
        msgs = render_messages(cond, PLACEHOLDER_INSTRUCTION)
        rendered_by_id[tid] = msgs
        if len(msgs) != 3 or [m['role'] for m in msgs] != ['user', 'assistant', 'user']:
            role_order_ok = False
            print(f"  {tid}: bad structure -- {[m['role'] for m in msgs]}")
    check('3_all_conditions_3_messages_user_assistant_user', role_order_ok)

    # ---- 4. final_user (BEFORE substitution) contains exactly one
    # {instruction} placeholder; setup_user/assistant_acknowledgement
    # contain ZERO ----
    placeholder_ok = True
    for tid, fam, v, cond in conditions:
        n_final = cond['final_user'].count('{instruction}')
        n_setup = cond['setup_user'].count('{instruction}')
        n_ack = cond['assistant_acknowledgement'].count('{instruction}')
        if n_final != 1:
            placeholder_ok = False
            print(f"  {tid}: final_user has {n_final} occurrences of {{instruction}}, expected 1")
        if n_setup != 0 or n_ack != 0:
            placeholder_ok = False
            print(f"  {tid}: setup_user/ack must not contain {{instruction}} (setup={n_setup}, ack={n_ack})")
    check('4_exactly_one_instruction_placeholder_in_final_user_only', placeholder_ok)

    # ---- 5. forbidden-phrase / canonical-mechanism-confusion scan, on
    # setup_user + assistant_acknowledgement only (final_user is always
    # bare {instruction}, not authored content) ----
    forbidden_ok = True
    for tid, fam, v, cond in conditions:
        text_to_scan = f"{cond['setup_user']} {cond['assistant_acknowledgement']}".lower()
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, text_to_scan, re.IGNORECASE):
                forbidden_ok = False
                print(f"  {tid}: matched forbidden pattern {pattern!r}")
    check('5_no_forbidden_phrases_or_canonical_mechanism_confusion', forbidden_ok)

    # ---- 6. assistant_acknowledgement never contains the raw instruction
    # (would be answer-prefilling, not context-accumulation) ----
    no_prefill_ok = True
    for tid, fam, v, cond in conditions:
        if PLACEHOLDER_INSTRUCTION.lower() in cond['assistant_acknowledgement'].lower():
            no_prefill_ok = False
            print(f"  {tid}: assistant_acknowledgement appears to reference the instruction content")
    check('6_no_instruction_content_in_acknowledgement', no_prefill_ok)

    # ---- 7. positive vs neutral structural match: same message count/role
    # order (already true by construction, checked above) -- character-
    # count comparison of setup_user + acknowledgement, positive vs
    # neutral, per family ----
    print()
    print("-- character-count comparison (setup_user + acknowledgement), positive vs neutral --")
    length_ratios = {}
    for fam in FAMILIES:
        neutral_cond = data['families'][fam]['family_specific_neutral_control']
        neutral_len = len(neutral_cond['setup_user']) + len(neutral_cond['assistant_acknowledgement'])
        for v in VARIANTS:
            pos_cond = data['families'][fam]['variants'][v]
            pos_len = len(pos_cond['setup_user']) + len(pos_cond['assistant_acknowledgement'])
            ratio = pos_len / neutral_len if neutral_len else None
            length_ratios[f'{fam}_{v}'] = ratio
            print(f"  {fam}_{v}: positive={pos_len} chars, neutral={neutral_len} chars, ratio={ratio:.2f}")
    max_ratio_dev = max(abs(r - 1.0) for r in length_ratios.values())
    check('7_length_ratios_within_50pct_of_neutral', max_ratio_dev < 0.5,
          f"max deviation from ratio=1.0 is {max_ratio_dev:.2f} -- review flagged conditions above")

    # ---- 8. mock per-model-family token-count proxy (LOCAL ONLY, NOT
    # real tokenizers -- flagged explicitly, real counts need the cluster) ----
    print()
    print("-- MOCK token-count proxy (word count, NOT a real tokenizer -- see docstring) --")
    for family_name in ('qwen', 'llama', 'gemma'):
        tok = MockTokenizer(family_name)
        print(f"  [{family_name} mock]")
        for fam in FAMILIES:
            neutral_cond = data['families'][fam]['family_specific_neutral_control']
            neutral_msgs = render_messages(neutral_cond, PLACEHOLDER_INSTRUCTION)
            neutral_tok = sum(tok.count(m['content']) for m in neutral_msgs)
            for v in VARIANTS:
                pos_cond = data['families'][fam]['variants'][v]
                pos_msgs = render_messages(pos_cond, PLACEHOLDER_INSTRUCTION)
                pos_tok = sum(tok.count(m['content']) for m in pos_msgs)
                print(f"    {fam}_{v}: positive~{pos_tok} tok, neutral~{neutral_tok} tok")
    check('8_mock_token_proxy_computed_for_all_3_families', True,
          "(informational only -- NOT a pass/fail gate; real cluster tokenizers required before citing real numbers)")

    # ---- 9. no verbatim reuse of the single-turn template text as the
    # multi-turn setup_user (protocol Sec 4: must be rewritten as a
    # natural dialogue setup, not copy-pasted) ----
    single_turn_path = os.path.join(REPO_ROOT, 'templates', 'templates_context_v1.json')
    reuse_ok = True
    if os.path.exists(single_turn_path):
        with open(single_turn_path, encoding='utf-8') as f:
            single_turn_data = json.load(f)
        single_turn_texts = set()
        for fam in FAMILIES:
            fam_data = single_turn_data['families'][fam]
            for v in VARIANTS:
                single_turn_texts.add(fam_data['variants'][v].strip())
            single_turn_texts.add(fam_data['family_specific_neutral_control'].strip())
        for tid, fam, v, cond in conditions:
            if cond['setup_user'].strip() in single_turn_texts:
                reuse_ok = False
                print(f"  {tid}: setup_user is a VERBATIM copy of a single-turn template")
    check('9_no_verbatim_reuse_of_single_turn_template_as_setup', reuse_ok)

    # ---- 10. single-turn template file is untouched (protocol Sec 6:
    # must never modify/overwrite). This script has no code path that
    # writes to output/behavioral_test_formal/ at all, and that directory
    # is cluster-only (real generations were never copied to this local
    # checkout) -- checking for its existence here would fail on every
    # local machine regardless of whether anything is wrong, so it is
    # reported informationally rather than gated. ----
    single_turn_untouched = os.path.exists(single_turn_path)
    check('10_single_turn_template_file_still_exists_untouched', single_turn_untouched)
    formal_dir_path = os.path.join(REPO_ROOT, 'output', 'behavioral_test_formal')
    print(f"  (informational) output/behavioral_test_formal exists on THIS machine: "
          f"{os.path.exists(formal_dir_path)} -- expected False on a local Mac checkout, "
          f"True on the cluster where the real formal run lives; this script never writes there either way.")

    print()
    if failed == 0:
        print("ALL CONTEXT MULTITURN TEMPLATE DRY-RUN CHECKS PASSED.")
    else:
        print(f"{failed} CHECK(S) FAILED.")
    return failed


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
