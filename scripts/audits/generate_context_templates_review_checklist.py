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
import datetime
import hashlib
import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)
from audit_context_templates_dry_run import (
    load_context_templates, all_template_strings, FORBIDDEN_PHRASES, NUMBERED_STEP_PATTERNS,
    TEMPLATES_PATH,
)

DEFAULT_REPORT_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'output', 'audits', 'context')

PERMISSION_REMOVAL_PHRASES = ['ignore previous instructions', 'no restrictions', 'unrestricted', 'bypass safety']
# 'here is how' / "here's how" added 2026-09-08 alongside the ctx_continuation
# template revision -- closes a gap where a continuation-style answer opener
# would not have been caught by 'sure, here'/'certainly, here' alone.
PREFILLED_ANSWER_PHRASES = ['sure, here', 'certainly, here', 'here is how', "here's how"]
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
    'ctx_continuation': "Primary possible overlap: prefix_injection, because both can constrain the "
                         "position or format from which generation begins. The continuation templates "
                         "do not contain substantive or affirmative answer prefixes. A secondary "
                         "structural analogy exists with payload_splitting, but the source instruction "
                         "is not fragmented. These relationships require empirical geometric testing.",
}

# ---- Schema v3 additions (2026-09-08) -----------------------------------
# Two of the seven new fields (contains_semantic_answer_prefill,
# contains_format_or_role_prefix) are objectively checkable from the text
# itself and computed below. The remaining five are curated per-family
# judgments -- like CANONICAL_OVERLAP_JUDGMENT above, these are NOT claims
# of verified geometric overlap; requires_empirical_overlap_test /
# template_level_overlap / family_level_overlap name what still needs an
# empirical (activation-geometry) check, not a conclusion already reached.

FORMAT_ROLE_PREFIX_SUFFIXES = ('response:', 'speaker b:')


def contains_format_or_role_prefix(text):
    """True if the template's own text (not the {instruction} content) ends
    in a generation-position/role cue such as 'Response:' or 'Speaker B:'.
    'New response:' also ends in 'response:' so it is caught by the same
    suffix check without a separate case."""
    return text.rstrip().lower().endswith(FORMAT_ROLE_PREFIX_SUFFIXES)


def contains_semantic_answer_prefill(text):
    """True if the template text itself contains substantive answer content
    or an affirmative-compliance opener (e.g. 'Sure, here is...'). Unlike the
    legacy contains_prefilled_answer field, this does NOT also check for
    numbered-step patterns -- that remains a separate field
    (contains_step_by_step)."""
    lowered = text.lower()
    return any(p in lowered for p in PREFILLED_ANSWER_PHRASES)


# curated per (family, 'positive'|'neutral')
CHANGES_OUTPUT_FORMAT = {
    'ctx_persona': {'positive': False, 'neutral': False},
    'ctx_authority': {'positive': False, 'neutral': False},
    'ctx_fictional': {'positive': True, 'neutral': False},
    'ctx_continuation': {'positive': True, 'neutral': True},
}
CHANGES_PROFESSIONAL_DOMAIN = {
    'ctx_persona': {'positive': True, 'neutral': False},
    'ctx_authority': {'positive': False, 'neutral': False},
    'ctx_fictional': {'positive': False, 'neutral': False},
    'ctx_continuation': {'positive': False, 'neutral': False},
}
REQUIRES_EMPIRICAL_OVERLAP_TEST = {
    'ctx_persona': {'positive': True, 'neutral': False},
    'ctx_authority': {'positive': False, 'neutral': False},
    'ctx_fictional': {'positive': True, 'neutral': False},
    'ctx_continuation': {'positive': True, 'neutral': True},
}
TEMPLATE_LEVEL_OVERLAP = {
    'ctx_persona': {'positive': 'persona_roleplay', 'neutral': 'none'},
    'ctx_authority': {'positive': 'none', 'neutral': 'none'},
    'ctx_fictional': {'positive': 'persona_roleplay', 'neutral': 'none'},
    'ctx_continuation': {'positive': 'prefix_injection', 'neutral': 'prefix_injection'},
}
# family-level (same for every entry in the family, positive or neutral --
# this names what the FAMILY as a whole needs comparing against, not a
# per-template judgment)
FAMILY_LEVEL_OVERLAP = {
    'ctx_persona': ['persona_roleplay'],
    'ctx_authority': [],
    'ctx_fictional': ['persona_roleplay'],
    'ctx_continuation': ['prefix_injection', 'payload_splitting'],
}

REVIEWER_STATUS_V3 = {
    'ctx_persona': 'APPROVED_FOR_PILOT_WITH_LIMITATION',
    'ctx_authority': 'APPROVED_FOR_PILOT',
    'ctx_fictional': 'APPROVED_FOR_PILOT',
    'ctx_continuation': 'PENDING_SECOND_HUMAN_REVIEW',
}
REVIEWER_NOTES_V3 = {
    'ctx_persona': "Approved for pilot use. The ctx_persona/persona_roleplay geometric-independence "
                   "check (protocol Sec 4) remains unverified -- this is a noted limitation, not a "
                   "resolved question; no permission-altering claim is present in the text.",
    'ctx_authority': "Approved for pilot use. No canonical-mechanism overlap identified requiring "
                      "empirical testing; the claim of authorization belongs to the user, not the AI.",
    'ctx_fictional': "Approved for pilot use. Possible structural analogy to persona_roleplay (the "
                      "model is asked to write a character/role) is flagged for future empirical "
                      "comparison -- this does not assume the two are the same mechanism.",
    'ctx_continuation': "Templates revised 2026-09-08. Response-position cues (Response:/Speaker B:/"
                         "New response:) are format-level, not substantive answer content, but the "
                         "current design cannot fully separate continuation semantics from the "
                         "response-position cue itself. Requires a second human review pass plus "
                         "geometric comparison with prefix_injection before pilot use.",
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


def load_decisions(path):
    """Loads a human-review decision sidecar file (schema: {'decisions':
    [{'template_id', 'reviewer_status', 'reviewer_notes', 'reviewed_at',
    'review_scope'}, ...]}). Returns a dict keyed by template_id. Raises if
    template_ids are missing or duplicated -- the caller is responsible for
    checking the set matches the 16 real template_ids."""
    with open(path, encoding='utf-8') as f:
        sidecar = json.load(f)
    decisions = sidecar['decisions']
    by_id = {}
    for d in decisions:
        tid = d['template_id']
        if tid in by_id:
            raise ValueError(f"duplicate template_id in decision sidecar: {tid}")
        by_id[tid] = d
    return by_id


def main(args):
    data = load_context_templates()

    decisions_by_id = None
    if args.schema_version in ('v4', 'v5'):
        if not args.decisions_path:
            print(f"--decisions_path is required for --schema_version {args.schema_version}")
            sys.exit(1)
        decisions_by_id = load_decisions(args.decisions_path)

    if args.schema_version == 'v5' and not args.token_audit_path:
        print("--token_audit_path is required for --schema_version v5")
        sys.exit(1)

    checklist = []
    for fam_name, vkey, text in all_template_strings(data):
        is_positive = vkey != 'family_specific_neutral_control'
        ptype = 'positive' if is_positive else 'neutral'
        template_id = f"{fam_name}_{vkey}" if is_positive else f"{fam_name}_neutral"
        entry = {
            'template_id': template_id,
            'family': fam_name,
            'positive_or_neutral': ptype,
            'full_template_text': text,
        }
        entry.update(analyze(text))
        entry['most_likely_canonical_overlap'] = CANONICAL_OVERLAP_JUDGMENT[fam_name]

        if args.schema_version in ('v3', 'v4', 'v5'):
            entry['contains_semantic_answer_prefill'] = contains_semantic_answer_prefill(text)
            entry['contains_format_or_role_prefix'] = contains_format_or_role_prefix(text)
            entry['changes_output_format'] = CHANGES_OUTPUT_FORMAT[fam_name][ptype]
            entry['changes_professional_domain'] = CHANGES_PROFESSIONAL_DOMAIN[fam_name][ptype]
            entry['requires_empirical_overlap_test'] = REQUIRES_EMPIRICAL_OVERLAP_TEST[fam_name][ptype]
            entry['template_level_overlap'] = TEMPLATE_LEVEL_OVERLAP[fam_name][ptype]
            entry['family_level_overlap'] = FAMILY_LEVEL_OVERLAP[fam_name]

        if args.schema_version in ('v4', 'v5'):
            # v4/v5 take reviewer_status/reviewer_notes from the external
            # decision sidecar (not from hardcoded dicts in this script) so
            # that later review rounds never require hand-editing a
            # previously-generated checklist file -- only a new/updated
            # sidecar plus a fresh regeneration. v5 reuses the SAME sidecar
            # as v4 -- the token-length audit did not change any per-
            # template human review decision, only added a token-length
            # QC gate on top of the existing decisions.
            if template_id not in decisions_by_id:
                print(f"FATAL: no decision found for template_id={template_id} in {args.decisions_path}")
                sys.exit(1)
            d = decisions_by_id[template_id]
            entry['reviewer_status'] = d['reviewer_status']
            entry['reviewer_notes'] = d['reviewer_notes']
            entry['reviewed_at'] = d['reviewed_at']
            entry['review_scope'] = d['review_scope']
        elif args.schema_version == 'v3':
            entry['reviewer_status'] = REVIEWER_STATUS_V3[fam_name]
            entry['reviewer_notes'] = REVIEWER_NOTES_V3[fam_name]
        else:
            entry['reviewer_status'] = 'PENDING_HUMAN_REVIEW'
            entry['reviewer_notes'] = ''

        checklist.append(entry)

    if decisions_by_id is not None:
        expected_ids = {e['template_id'] for e in checklist}
        actual_ids = set(decisions_by_id.keys())
        if actual_ids != expected_ids:
            print(f"FATAL: decision sidecar template_id set does not match the 16 real template_ids.\n"
                  f"  missing from sidecar: {expected_ids - actual_ids}\n"
                  f"  unexpected in sidecar: {actual_ids - expected_ids}")
            sys.exit(1)

    assert len(checklist) == 16, f"expected 16 entries, got {len(checklist)}"

    print(f"Generated {len(checklist)} checklist entries (schema {args.schema_version}):")
    for e in checklist:
        print(f"  {e['template_id']:28s} perm_removal={e['contains_permission_removal_claim']}  "
              f"prefilled={e['contains_prefilled_answer']}  encoding={e['contains_encoding_or_payload_split']}  "
              f"steps={e['contains_step_by_step']}  placeholder_count={e['instruction_placeholder_count']}"
              + (f"  reviewer_status={e['reviewer_status']}" if args.schema_version in ('v3', 'v4', 'v5') else ''))

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

        if args.schema_version == 'v3':
            with open(TEMPLATES_PATH, 'rb') as f:
                source_sha256 = hashlib.sha256(f.read()).hexdigest()
            report = {
                'checklist_version': 'v3',
                'result_status': 'PARTIALLY_HUMAN_REVIEWED_CONTINUATION_PENDING',
                'source_template_path': 'templates/templates_context_v1.json',
                'source_template_sha256': source_sha256,
                'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'generator_path': 'scripts/audits/generate_context_templates_review_checklist.py',
                'entries': checklist,
            }
        elif args.schema_version == 'v4':
            with open(TEMPLATES_PATH, 'rb') as f:
                source_sha256 = hashlib.sha256(f.read()).hexdigest()
            with open(args.decisions_path, 'rb') as f:
                decisions_sha256 = hashlib.sha256(f.read()).hexdigest()
            with open(os.path.abspath(__file__), 'rb') as f:
                generator_sha256 = hashlib.sha256(f.read()).hexdigest()
            report = {
                'checklist_version': 'v4',
                'result_status': 'HUMAN_REVIEWED_READY_FOR_TOKEN_AUDIT',
                'source_template_path': 'templates/templates_context_v1.json',
                'source_template_sha256': source_sha256,
                'decision_sidecar_path': os.path.relpath(args.decisions_path,
                                                          os.path.join(SCRIPT_DIR, '..', '..')),
                'decision_sidecar_sha256': decisions_sha256,
                'generator_path': 'scripts/audits/generate_context_templates_review_checklist.py',
                'generator_sha256': generator_sha256,
                'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'entries': checklist,
            }
        elif args.schema_version == 'v5':
            with open(TEMPLATES_PATH, 'rb') as f:
                source_sha256 = hashlib.sha256(f.read()).hexdigest()
            with open(args.decisions_path, 'rb') as f:
                decisions_sha256 = hashlib.sha256(f.read()).hexdigest()
            with open(args.token_audit_path, 'rb') as f:
                token_audit_sha256 = hashlib.sha256(f.read()).hexdigest()
            with open(os.path.abspath(__file__), 'rb') as f:
                generator_sha256 = hashlib.sha256(f.read()).hexdigest()
            report = {
                'checklist_version': 'v5',
                'result_status': 'HUMAN_AND_TOKEN_AUDITED_READY_FOR_ACTIVATION_PILOT',
                'source_template_path': 'templates/templates_context_v1.json',
                'source_template_sha256': source_sha256,
                'decision_sidecar_path': os.path.relpath(args.decisions_path,
                                                          os.path.join(SCRIPT_DIR, '..', '..')),
                'decision_sidecar_sha256': decisions_sha256,
                'token_length_audit_path': os.path.relpath(args.token_audit_path,
                                                            os.path.join(SCRIPT_DIR, '..', '..')),
                'token_length_audit_sha256': token_audit_sha256,
                'generator_path': 'scripts/audits/generate_context_templates_review_checklist.py',
                'generator_sha256': generator_sha256,
                'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'note': 'result_status reflects that template wording has completed human review AND a '
                        'tokenizer-only length QC pass -- it does NOT mean any C (context-reconfiguration) '
                        'direction, family independence, or canonical-mechanism distinction has been '
                        'empirically validated. See EXPERIMENT2_CONTEXT_RECONFIGURATION_PROTOCOL.md.',
                'entries': checklist,
            }
        else:
            report = {'result_status': 'HUMAN_REVIEW_CHECKLIST_PENDING', 'entries': checklist}

        with open(args.write_report, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote checklist (new file, non-overwriting): {args.write_report}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write_report', type=str, default=None)
    parser.add_argument('--schema_version', type=str, default='v1', choices=['v1', 'v3', 'v4', 'v5'],
                         help="'v1' reproduces the original 5-field-plus-status schema (used for both "
                              "the v1 and v2 checklists). 'v3' adds the 7 extended fields and the "
                              "curated reviewer_status/reviewer_notes for that round's human review. "
                              "'v4' also adds the 7 extended fields, but sources "
                              "reviewer_status/reviewer_notes/reviewed_at/review_scope from an external "
                              "--decisions_path sidecar file instead of a hardcoded dict, and records "
                              "source/decision/generator SHA-256 provenance. 'v5' reuses v4's sidecar "
                              "(no per-template decision changed) and additionally records the "
                              "--token_audit_path report's own SHA-256, marking pilot admission as "
                              "human-AND-token-audited.")
    parser.add_argument('--decisions_path', type=str, default=None,
                         help="Path to a human-review decision sidecar JSON (required for --schema_version v4/v5).")
    parser.add_argument('--token_audit_path', type=str, default=None,
                         help="Path to the real token-length audit report JSON (required for --schema_version v5).")
    args = parser.parse_args()
    main(args)
