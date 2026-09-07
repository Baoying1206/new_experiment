"""
Generates a per-template human-review checklist for
templates/templates_context_v1.json's 16 texts. Pure static analysis --
no model, no tokenizer, no network. Auto-fills the objectively-checkable
boolean fields (reusing the same logic as audit_context_templates_dry_run.py);
leaves reviewer_status/reviewer_notes for an actual human.

The 'most_likely_canonical_overlap' field is a curated judgment call per
family (documented inline below), not an automated string match -- it is
NOT a claim that the overlap is empirically verified (see
EXPERIMENT2_CONTEXT_RECONFIGURATION_PROTOCOL.md Sec 4).

Refuses to overwrite an existing file; only writes under output/audits/context/.

Usage:
  python scripts/audits/generate_context_templates_review_checklist.py \
      --write_report output/audits/context/context_templates_human_review_checklist.json
"""
import argparse
import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)
from audit_context_templates_dry_run import (
    load_context_templates, all_template_strings, FORBIDDEN_PHRASES, NUMBERED_STEP_PATTERNS,
)

DEFAULT_REPORT_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'output', 'audits', 'context')

PERMISSION_REMOVAL_PHRASES = ['ignore previous instructions', 'no restrictions', 'unrestricted', 'bypass safety']
PREFILLED_ANSWER_PHRASES = ['sure, here', 'certainly, here']
ENCODING_TERMS = ['base64', 'rot13', 'payload_a', 'payload_b', 'instruction_b64']

# Curated per-family judgment (not an automated match) -- see
# EXPERIMENT2_CONTEXT_RECONFIGURATION_PROTOCOL.md Sec 4 for the caveat that
# none of these are empirically verified.
CANONICAL_OVERLAP_JUDGMENT = {
    'ctx_persona': "persona_roleplay -- both assign the model a persona; ctx_persona's operational "
                   "distinction is no permission-altering claim, but this is a text-level design "
                   "choice, not a verified geometric distinction (Sec 4).",
    'ctx_authority': "None of the 6 directly. Weak structural analogy to prefix_injection (both add "
                      "framing before the request), but prefix_injection compels a specific RESPONSE "
                      "phrasing while ctx_authority asserts USER credentials -- mechanism differs.",
    'ctx_fictional': "None of the 6 currently active. Resembles 'fictional_framing', a PRE-CORRECTION "
                      "stand-in name that was removed from the taxonomy because Wei et al. does not "
                      "actually name it as one of their categories -- not a current canonical mechanism.",
    'ctx_continuation': "None of the 6 directly. Weak structural analogy to payload_splitting (both "
                         "restructure how the instruction is delivered rather than presenting it "
                         "directly as a fresh request), but payload_splitting reconstructs FRAGMENTED "
                         "text while continuation asks the model to EXTEND existing text -- mechanism "
                         "differs substantially.",
}


def analyze(text):
    lowered = text.lower()
    return {
        'instruction_placeholder_count': len(re.findall(r'\{instruction\}', text)),
        'contains_permission_removal_claim': any(p in lowered for p in PERMISSION_REMOVAL_PHRASES),
        'contains_prefilled_answer': (
            any(p in lowered for p in PREFILLED_ANSWER_PHRASES)
            or any(p.search(text) for p in NUMBERED_STEP_PATTERNS)
        ),
        'contains_encoding_or_payload_split': any(t in lowered for t in ENCODING_TERMS),
        'contains_step_by_step': any(p.search(text) for p in NUMBERED_STEP_PATTERNS),
    }


def main(args):
    data = load_context_templates()
    checklist = []
    for fam_name, vkey, text in all_template_strings(data):
        is_positive = vkey != 'family_specific_neutral_control'
        template_id = f"{fam_name}_{vkey}" if is_positive else f"{fam_name}_neutral"
        entry = {
            'template_id': template_id,
            'family': fam_name,
            'positive_or_neutral': 'positive' if is_positive else 'neutral',
            'full_template_text': text,
        }
        entry.update(analyze(text))
        entry['most_likely_canonical_overlap'] = CANONICAL_OVERLAP_JUDGMENT[fam_name]
        entry['reviewer_status'] = 'PENDING_HUMAN_REVIEW'
        entry['reviewer_notes'] = ''
        checklist.append(entry)

    assert len(checklist) == 16, f"expected 16 entries, got {len(checklist)}"

    print(f"Generated {len(checklist)} checklist entries:")
    for e in checklist:
        print(f"  {e['template_id']:28s} perm_removal={e['contains_permission_removal_claim']}  "
              f"prefilled={e['contains_prefilled_answer']}  encoding={e['contains_encoding_or_payload_split']}  "
              f"steps={e['contains_step_by_step']}  placeholder_count={e['instruction_placeholder_count']}")

    if args.write_report:
        if os.path.exists(args.write_report):
            print(f"\nRefusing to write: {args.write_report} already exists.")
            sys.exit(1)
        report_dir = os.path.dirname(os.path.abspath(args.write_report))
        expected_dir = os.path.abspath(DEFAULT_REPORT_DIR)
        if report_dir != expected_dir:
            print(f"\nRefusing to write outside {DEFAULT_REPORT_DIR}: got {args.write_report}")
            sys.exit(1)
        os.makedirs(report_dir, exist_ok=True)
        with open(args.write_report, 'w') as f:
            json.dump({'result_status': 'HUMAN_REVIEW_CHECKLIST_PENDING', 'entries': checklist}, f, indent=2)
        print(f"\nWrote checklist (new file, non-overwriting): {args.write_report}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write_report', type=str, default=None)
    args = parser.parse_args()
    main(args)
