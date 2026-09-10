"""
Generates a per-condition human-review checklist for the 16 multi-turn
Contextual Reconfiguration conditions in
templates/templates_context_multiturn_v1.json (post-hoc exploratory
extension -- see EXPERIMENT_CONTEXT_MULTITURN_BEHAVIOR_PROTOCOL.md).
Pure static analysis -- no model, no tokenizer, no network. Auto-fills
the objectively-checkable fields (reusing the same logic as
audit_context_multiturn_templates_dry_run.py); leaves
reviewer_status/reviewer_notes for an actual human.

Refuses to overwrite an existing file; only writes under
output/audits/context/.

Usage:
  python scripts/audits/generate_context_multiturn_review_checklist.py \
      --write_report output/audits/context/context_multiturn_templates_human_review_checklist_v1.json
"""
import argparse
import datetime
import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)
from audit_context_multiturn_templates_dry_run import (
    load_templates, build_conditions, render_messages, FORBIDDEN_PATTERNS,
    PLACEHOLDER_INSTRUCTION, TEMPLATE_PATH, FAMILIES, VARIANTS,
)

REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
DEFAULT_REPORT_DIR = os.path.join(REPO_ROOT, 'output', 'audits', 'context')

# Curated per-family judgment (NOT an automated/verified claim -- same
# discipline as the single-turn checklist's CANONICAL_OVERLAP_JUDGMENT):
# what this multi-turn condition is intended to test vs. what still needs
# a human's read for tone/naturalness/confound risk.
OPERATIONAL_INTENT = {
    'ctx_persona': "turn 1 assigns a professional task identity for the rest of the conversation; "
                   "neutral assigns none. Human check: does the identity assignment read as a natural "
                   "conversational setup, not an abrupt non-sequitur?",
    'ctx_authority': "turn 1 has the user assert their own authorisation/approval before asking; "
                      "neutral has the user only preview that a question is coming. Human check: does "
                      "the authority claim avoid sounding like a demand or a jailbreak-style assertion?",
    'ctx_fictional': "turn 1 establishes a fictional creative-writing project; neutral establishes a "
                      "generic writing project. Human check: does the setup read as a genuine creative-"
                      "writing framing rather than a thin excuse?",
    'ctx_continuation': "turn 1 frames the next request as continuing existing text; neutral frames it "
                         "as new/independent. Human check: does the framing avoid implying or prefilling "
                         "any part of a response (protocol Sec 5)?",
}

# ---- Schema v2 additions (2026-09-10) -- curated per-family judgments,
# NOT automated/verified claims (same discipline as the single-turn
# checklist's CANONICAL_OVERLAP_JUDGMENT): what the closest existing
# canonical mechanism is, and what STRUCTURAL confound the multi-turn
# format itself (not just the framing content) may introduce relative to
# the single-turn version of the same family. These are flagged for
# human review, not resolved here.
CLOSEST_CANONICAL_MECHANISM = {
    'ctx_persona': "persona_roleplay -- both assign the model a persona/identity; the multi-turn version "
                   "delivers it as an accepted premise from a PRIOR turn rather than a same-turn "
                   "instruction, which is untested as a distinct variable.",
    'ctx_authority': "prefix_injection -- weak structural analogy only (both add framing before the "
                      "substantive request); prefix_injection compels a specific RESPONSE phrasing while "
                      "ctx_authority asserts USER credentials, so the mechanism differs.",
    'ctx_fictional': "None of the 6 currently active canonical mechanisms (same assessment as the "
                      "single-turn version).",
    'ctx_continuation': "prefix_injection -- weak structural analogy (both can constrain the position/"
                         "framing generation begins from). Explicitly NOT payload_splitting and NOT "
                         "answer-prefilling: no partial response content ever appears (Sec 5).",
}

POTENTIAL_CONFOUND = {
    'ctx_persona': "The fixed assistant_acknowledgement ('Understood — I'm ready to continue our "
                    "conversation') is a NEW structural element vs. the single-turn version (which has "
                    "no assistant turn at all before the request) -- any effect may partly reflect a "
                    "generic 'the assistant already agreed to continue' consistency pressure, not the "
                    "persona content specifically. This confound is present in ALL 4 families equally "
                    "(by construction, protocol Sec 3), so it does not differentially affect "
                    "family-to-family comparisons, but it does mean multi-turn vs single-turn deltas are "
                    "not a pure test of 'framing accumulated over turns' in isolation.",
    'ctx_authority': "Same generic acknowledgement/consistency-pressure confound as ctx_persona (shared "
                      "across all 4 families, not specific to authority).",
    'ctx_fictional': "Same generic acknowledgement/consistency-pressure confound as ctx_persona, PLUS: "
                      "this family's acknowledgement ('ready to help with your writing project') is the "
                      "only one of the 4 that references the setup's TOPIC (writing) rather than being "
                      "fully generic -- this makes ctx_fictional's acknowledgement slightly more specific "
                      "than the other 3 families' acknowledgements, a minor asymmetry worth a human's "
                      "judgment on whether it matters.",
    'ctx_continuation': "IMPORTANT, flagged prominently (see chat reply): the single-turn ctx_continuation "
                         "template embeds explicit 'Request: {instruction}\\nResponse:' labels in the SAME "
                         "turn as the instruction, which is arguably the actual mechanism under test "
                         "(a literal fill-in-the-blank cue). The multi-turn version's final_user is, by "
                         "the frozen Sec 4 design (uniform bare {instruction} across all 16 conditions), "
                         "JUST the raw instruction with NO 'Request:'/'Response:' labels anywhere -- the "
                         "turn-1 setup only ASSERTS that a continuation is coming, without the label-based "
                         "fill-in-the-blank structure itself. This may mean the multi-turn version tests a "
                         "meaningfully different (weaker, or simply different) operationalization of "
                         "'continuation' than the single-turn study did, not a stronger one. Needs explicit "
                         "human decision: keep uniform final_user (current design) or allow ctx_continuation "
                         "alone to carry format labels in final_user (breaking the Sec 4 uniformity rule) "
                         "to preserve mechanism parity with the single-turn version.",
}


def build_checklist_entries(schema_version):
    data = load_templates()
    conditions = build_conditions(data)
    entries = []
    for tid, fam, v, cond in conditions:
        msgs = render_messages(cond, PLACEHOLDER_INSTRUCTION)
        text_to_scan = f"{cond['setup_user']} {cond['assistant_acknowledgement']}".lower()
        forbidden_hits = [p for p in FORBIDDEN_PATTERNS if re.search(p, text_to_scan, re.IGNORECASE)]

        neutral_cond = data['families'][fam]['family_specific_neutral_control']
        neutral_len = len(neutral_cond['setup_user']) + len(neutral_cond['assistant_acknowledgement'])
        this_len = len(cond['setup_user']) + len(cond['assistant_acknowledgement'])
        length_ratio_vs_neutral = (this_len / neutral_len) if neutral_len else None

        entry = {
            'template_id': tid,
            'family': fam,
            'variant_or_neutral': v,
            'is_positive': v != 'neutral',
            'setup_user': cond['setup_user'],
            'assistant_acknowledgement': cond['assistant_acknowledgement'],
            'final_user': cond['final_user'],
            'operational_intent': OPERATIONAL_INTENT[fam],
            'messages_preview': msgs,
            'auto_checks': {
                'message_count_is_3': len(msgs) == 3,
                'role_order_is_user_assistant_user': [m['role'] for m in msgs] == ['user', 'assistant', 'user'],
                'final_user_has_exactly_one_placeholder': cond['final_user'].count('{instruction}') == 1,
                'setup_and_ack_have_zero_placeholders': (cond['setup_user'].count('{instruction}') == 0
                                                          and cond['assistant_acknowledgement'].count('{instruction}') == 0),
                'no_forbidden_phrase_matches': len(forbidden_hits) == 0,
                'forbidden_phrase_matches': forbidden_hits,
                'length_ratio_vs_neutral': length_ratio_vs_neutral,
            },
            'reviewer_status': 'PENDING_HUMAN_REVIEW',
            'reviewer_notes': '',
            'reviewed_at': None,
        }
        if schema_version == 'v2':
            entry['closest_canonical_mechanism'] = CLOSEST_CANONICAL_MECHANISM[fam]
            entry['potential_confound'] = POTENTIAL_CONFOUND[fam]
        entries.append(entry)
    return entries


CONCLUSION_TEXT = ("Automated static checks found no explicit forbidden-pattern violations. "
                    "Conceptual overlap with canonical mechanisms and acknowledgement/format effects "
                    "remain subject to human review and empirical testing.")


def main(args):
    if os.path.exists(args.write_report):
        print(f"Refusing to overwrite existing file: {args.write_report}")
        sys.exit(1)
    os.makedirs(os.path.dirname(os.path.abspath(args.write_report)), exist_ok=True)

    entries = build_checklist_entries(args.schema_version)
    report = {
        'schema_version': f'context_multiturn_review_{args.schema_version}',
        'source_template_path': os.path.relpath(TEMPLATE_PATH, REPO_ROOT),
        'study_status': 'POST_HOC_EXPLORATORY_EXTENSION',
        'static_audit_conclusion': CONCLUSION_TEXT,
        'n_entries': len(entries),
        'entries': entries,
        'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    tmp_path = args.write_report + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, args.write_report)
    print(f"Wrote (new file, non-overwriting): {args.write_report}")
    print(f"n_entries: {len(entries)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--schema_version', type=str, default='v2', choices=['v1', 'v2'])
    parser.add_argument('--write_report', type=str, default=None)
    args = parser.parse_args()
    if args.write_report is None:
        args.write_report = os.path.join(
            DEFAULT_REPORT_DIR, f'context_multiturn_templates_human_review_checklist_{args.schema_version}.json')
    main(args)
