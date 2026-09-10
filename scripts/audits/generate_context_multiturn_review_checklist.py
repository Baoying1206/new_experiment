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
    'ctx_continuation': "RESOLVED via the B2 revision (2026-09-10, human decision): final_user is now "
                         "IDENTICAL ('Request: {instruction}\\nResponse:') across all 4 conditions, so the "
                         "format label is a SHARED, CONTROLLED variable rather than a manipulated one -- "
                         "the family-internal positive-vs-neutral comparison isolates the turn-1 setup "
                         "difference only. Residual limitation (protocol Sec 12, not resolved by B2): the "
                         "EXISTING single-turn ctx_continuation study did NOT use an identical final format "
                         "across its own positive/neutral (positive uses 'Response:'/'Speaker B:', neutral "
                         "uses 'New response:'), so the single-vs-multi INTERACTION test for this family is "
                         "a sensitivity analysis only, not equal in strength to the other 3 families' "
                         "interaction tests. This revision does NOT establish that ctx_continuation is now "
                         "empirically distinguished from prefix_injection.",
}

# ---- Schema v3 additions (2026-09-10) -- records the human's actual
# round-2 review decisions (per family; all 4 conditions in a family
# share the same decision in this round). NOT auto-generated -- these are
# the user's own stated decisions, transcribed here for a durable,
# machine-readable record.
REVIEWER_DECISIONS = {
    'ctx_persona': {
        'reviewer_status': 'APPROVED_FOR_PILOT_WITH_CANONICAL_OVERLAP_LIMITATION',
        'reviewer_notes': "Overlaps conceptually with canonical persona_roleplay (both assign the AI an "
                           "identity). No permission-removal language present, so approved for pilot, but "
                           "must not be pre-claimed as a distinct mechanism from persona_roleplay. Future "
                           "work must compare multi-turn ctx_persona to persona_roleplay both behaviorally "
                           "and at the activation level.",
    },
    'ctx_authority': {
        'reviewer_status': 'APPROVED_FOR_PILOT',
        'reviewer_notes': "Positive manipulates the user's claimed authorisation/approval/duty; neutral "
                           "contains no such claim. No apparent canonical-mechanism confusion found.",
    },
    'ctx_fictional': {
        'reviewer_status': 'APPROVED_FOR_PILOT_WITH_ACKNOWLEDGEMENT_LIMITATION',
        'reviewer_notes': "This family's assistant_acknowledgement explicitly references 'writing project', "
                           "reinforcing the setup's semantics. Positive and neutral share the IDENTICAL "
                           "acknowledgement, so the family-internal paired comparison remains valid; but "
                           "cross-family strength comparisons must be interpreted conservatively given this "
                           "family's acknowledgement is topically specific while the other 3 are generic.",
    },
    'ctx_continuation': {
        'reviewer_status': 'APPROVED_FOR_PILOT_WITH_FORMAT_CUE_LIMITATION',
        'reviewer_notes': ("The B2 revision is approved for the exploratory pilot because all three "
                            "positive variants and the neutral control use the identical final-user "
                            "format, \"Request: {instruction}\\nResponse:\". The within-family "
                            "positive--neutral contrast therefore controls the direct effect of this "
                            "format cue. However, \"Response:\" remains an artificial response-position "
                            "cue and may interact with the preceding dialogue context. Approval does "
                            "not establish a pure continuation mechanism, independence from prefix "
                            "injection, or comparability with the primary single-versus-multi "
                            "interaction tests for the other families."),
    },
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
        }
        if schema_version == 'v1':
            entry['reviewer_status'] = 'PENDING_HUMAN_REVIEW'
            entry['reviewer_notes'] = ''
            entry['reviewed_at'] = None
        elif schema_version == 'v2':
            entry['closest_canonical_mechanism'] = CLOSEST_CANONICAL_MECHANISM[fam]
            entry['potential_confound'] = POTENTIAL_CONFOUND[fam]
            entry['reviewer_status'] = 'PENDING_HUMAN_REVIEW'
            entry['reviewer_notes'] = ''
            entry['reviewed_at'] = None
        elif schema_version == 'v3':
            entry['closest_canonical_mechanism'] = CLOSEST_CANONICAL_MECHANISM[fam]
            entry['potential_confound'] = POTENTIAL_CONFOUND[fam]
            decision = REVIEWER_DECISIONS[fam]
            entry['reviewer_status'] = decision['reviewer_status']
            entry['reviewer_notes'] = decision['reviewer_notes']
            entry['reviewed_at'] = REVIEW_ROUND_2_DATE
        entries.append(entry)
    return entries


CONCLUSION_TEXT = ("Automated static checks found no explicit forbidden-pattern violations. "
                    "Conceptual overlap with canonical mechanisms and acknowledgement/format effects "
                    "remain subject to human review and empirical testing.")

REVIEW_ROUND_2_DATE = '2026-09-10'


def main(args):
    if os.path.exists(args.write_report) and not args.force:
        print(f"Refusing to overwrite existing file (use --force if this is intentional): {args.write_report}")
        sys.exit(1)
    os.makedirs(os.path.dirname(os.path.abspath(args.write_report)), exist_ok=True)

    entries = build_checklist_entries(args.schema_version)

    import hashlib
    with open(TEMPLATE_PATH, 'rb') as f:
        template_content_sha256 = hashlib.sha256(f.read()).hexdigest()

    report = {
        'schema_version': f'context_multiturn_review_{args.schema_version}',
        'source_template_path': os.path.relpath(TEMPLATE_PATH, REPO_ROOT),
        'template_content_sha256': template_content_sha256,
        'study_status': 'POST_HOC_EXPLORATORY_EXTENSION',
        'static_audit_conclusion': CONCLUSION_TEXT,
        'n_entries': len(entries),
        'entries': entries,
        'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    if args.schema_version == 'v3':
        # Explicit provenance fields requested for v3 (2026-09-10, round 3
        # human decision): v3 is the CURRENT, mutable "latest checklist"
        # within this review lineage -- v1/v2 are frozen historical
        # snapshots and are never modified. "supersedes" names only which
        # file's REVIEW STATUS is now stale, it does NOT mean v2 is
        # deleted or edited.
        report['checklist_version'] = 'v3'
        report['source_template_version'] = 'context_multiturn_v2'
        report['supersedes_checklist'] = 'context_multiturn_templates_human_review_checklist_v2.json'

    tmp_path = args.write_report + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, args.write_report)
    print(f"Wrote ({'FORCED overwrite' if args.force else 'new file, non-overwriting'}): {args.write_report}")
    print(f"n_entries: {len(entries)}")
    print(f"template_content_sha256: {template_content_sha256}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--schema_version', type=str, default='v3', choices=['v1', 'v2', 'v3'])
    parser.add_argument('--write_report', type=str, default=None)
    parser.add_argument('--force', action='store_true',
                         help="Allow overwriting an existing file. Reserved for updating v3 (the current, "
                              "mutable checklist) with new review decisions -- NEVER use this for v1/v2, "
                              "which are frozen historical snapshots.")
    args = parser.parse_args()
    if args.write_report is None:
        args.write_report = os.path.join(
            DEFAULT_REPORT_DIR, f'context_multiturn_templates_human_review_checklist_{args.schema_version}.json')
    if args.force and args.schema_version in ('v1', 'v2'):
        print(f"Refusing --force for schema_version={args.schema_version!r} -- v1/v2 are frozen historical "
              f"snapshots and must never be overwritten, even with --force.")
        sys.exit(1)
    main(args)
