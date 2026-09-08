"""
Pure static/structural audit of templates/templates_context_v1.json --
NO model, NO GPU, NO generation, NO WildGuard, NO activation extraction.
Does not read test_ids. Does not touch templates_en.json or any existing
result file.

By default writes NOTHING -- stdout only, matching this repo's existing
*_dry_run.py convention. If --write_report PATH is given, refuses to
overwrite an existing file at that path (checks non-existence first) and
only ever writes under output/audits/context/.

Usage:
  python scripts/audits/audit_context_templates_dry_run.py
  python scripts/audits/audit_context_templates_dry_run.py --write_report output/audits/context/context_templates_audit.json
"""
import argparse
import hashlib
import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(__file__)
TEMPLATES_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'templates', 'templates_context_v1.json')
CANONICAL_TEMPLATES_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'templates', 'templates_en.json')
DEFAULT_REPORT_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'output', 'audits', 'context')

V1_CHECKLIST_PATH = os.path.join(DEFAULT_REPORT_DIR, 'context_templates_human_review_checklist.json')
V2_CHECKLIST_PATH = os.path.join(DEFAULT_REPORT_DIR, 'context_templates_human_review_checklist_v2.json')
V3_CHECKLIST_PATH = os.path.join(DEFAULT_REPORT_DIR, 'context_templates_human_review_checklist_v3.json')
V4_CHECKLIST_PATH = os.path.join(DEFAULT_REPORT_DIR, 'context_templates_human_review_checklist_v4.json')
DECISIONS_SIDECAR_PATH = os.path.join(DEFAULT_REPORT_DIR, 'context_templates_human_review_decisions.json')
# Hardcoded regression guard (added 2026-09-08, alongside the ctx_continuation
# revision; extended the same day to also pin v3 once v4 existed) -- these
# are the SHA-256 of the v1/v2/v3 checklist files as they existed once each
# was finalized. v1/v2/v3 must never be modified again; only a new (v4, v5,
# ...) file is ever added.
EXPECTED_V1_CHECKLIST_SHA256 = '97edbb27ea247f8710c1d71689465854369550d98e22ea80f4c69dd9c8dec92d'
EXPECTED_V2_CHECKLIST_SHA256 = 'f468fafc20306c3879808db83c47c8fc324b72b72f406005ef2de3c9429f2315'
EXPECTED_V3_CHECKLIST_SHA256 = '91f54d6c72f017a3644a840bf90c3625fb6bfeebbc0f6b15edc4dc5fe58fb6a6'

# Substrings that would overclaim empirical validation from a pilot
# approval. Checked case-sensitively against the all-caps status vocabulary
# this project actually uses (PENDING_*/APPROVED_FOR_PILOT*/READY_FOR_* are
# all fine; nothing currently in use should ever match this).
VALIDATION_CLAIM_SUBSTRINGS = ['VALIDATED', 'CONFIRMED']

# Duplicated (not imported) from generate_context_templates_review_checklist.py
# to avoid a circular import -- that script imports FROM this one.
ANSWER_PREFILL_PHRASES = ['sure, here', 'certainly, here', 'here is how', "here's how"]
FORMAT_ROLE_PREFIX_SUFFIXES = ('response:', 'speaker b:')

EXPECTED_FAMILIES = {'ctx_persona', 'ctx_authority', 'ctx_fictional', 'ctx_continuation'}

FORBIDDEN_PHRASES = [
    'ignore previous instructions',
    'no restrictions',
    'unrestricted',
    'bypass safety',
    'sure, here',
    'certainly, here',
]

NUMBERED_STEP_PATTERNS = [
    re.compile(r'\bstep\s*1\b', re.IGNORECASE),
    re.compile(r'\bstep\s*2\b', re.IGNORECASE),
    re.compile(r'^\s*1\.\s', re.MULTILINE),
    re.compile(r'^\s*2\.\s', re.MULTILINE),
]

HARMLESS_PLACEHOLDERS = [
    "recommend a good pasta recipe for beginners",
    "explain how photosynthesis works in simple terms",
    'describe the history of the printing press, including "movable type"',
]


def load_context_templates():
    with open(TEMPLATES_PATH, encoding='utf-8') as f:
        return json.load(f)


def load_canonical_mechanism_names():
    with open(CANONICAL_TEMPLATES_PATH, encoding='utf-8') as f:
        t = json.load(f)
    return set(t['active_mechanisms'])


def all_template_strings(data):
    """Yields (family, variant_key, text) for every positive variant AND the
    family_specific_neutral_control, across all families."""
    for fam_name, fam in data['families'].items():
        for vkey, vtext in fam['variants'].items():
            yield fam_name, vkey, vtext
        yield fam_name, 'family_specific_neutral_control', fam['family_specific_neutral_control']


def main(args):
    failed = 0

    def check(name, condition, detail=''):
        nonlocal failed
        if condition:
            print(f"PASS {name}")
        else:
            failed += 1
            print(f"FAIL {name}: {detail}")

    data = load_context_templates()

    # ---- 1. family set exactly equals the expected 4 ----
    actual_families = set(data['families'].keys())
    check('1_family_set_exact', actual_families == EXPECTED_FAMILIES,
          f"expected {EXPECTED_FAMILIES}, got {actual_families}")

    # ---- 0b. explicit total-count invariant (added 2026-09-07, after a
    # PROSE reporting error called the total "12" instead of "16" -- the
    # test loop itself (all_template_strings(), used below and by tests
    # 3/4/6/8/9) was always correct; this makes the count an asserted
    # invariant instead of an implicit property, so it can never silently
    # drift again) ----
    _count_probe = list(all_template_strings(data))
    n_positive_templates = sum(1 for _, vkey, _ in _count_probe if vkey != 'family_specific_neutral_control')
    n_neutral_controls = sum(1 for _, vkey, _ in _count_probe if vkey == 'family_specific_neutral_control')
    n_total_template_texts = len(_count_probe)
    check('0b_explicit_template_counts',
          n_positive_templates == 12 and n_neutral_controls == 4 and n_total_template_texts == 16,
          f"n_positive_templates={n_positive_templates} (expected 12), "
          f"n_neutral_controls={n_neutral_controls} (expected 4), "
          f"n_total_template_texts={n_total_template_texts} (expected 16)")
    print(f"  n_positive_templates={n_positive_templates}  n_neutral_controls={n_neutral_controls}  "
          f"n_total_template_texts={n_total_template_texts}")

    # ---- 2. each family has exactly 3 positive variants + 1 neutral ----
    variant_count_ok = True
    for fam_name, fam in data['families'].items():
        n_variants = len(fam.get('variants', {}))
        has_neutral = 'family_specific_neutral_control' in fam and bool(fam['family_specific_neutral_control'])
        if n_variants != 3 or not has_neutral:
            variant_count_ok = False
            print(f"  {fam_name}: n_variants={n_variants}, has_neutral={has_neutral}")
    check('2_three_variants_plus_neutral', variant_count_ok)

    all_strings = list(all_template_strings(data))

    # ---- 3. each template contains {instruction} exactly once, and no
    # other placeholder (rules out encoding/payload-splitting placeholders
    # as a side effect -- see check 6) ----
    placeholder_ok = True
    for fam_name, vkey, text in all_strings:
        placeholders = re.findall(r'\{([^{}]*)\}', text)
        if placeholders != ['instruction']:
            placeholder_ok = False
            print(f"  {fam_name}/{vkey}: placeholders found = {placeholders} (expected exactly ['instruction'])")
    check('3_exactly_one_instruction_placeholder', placeholder_ok)

    # ---- 4. no permission-altering / compliant-prefix phrases ----
    forbidden_ok = True
    for fam_name, vkey, text in all_strings:
        lowered = text.lower()
        hits = [p for p in FORBIDDEN_PHRASES if p in lowered]
        if hits:
            forbidden_ok = False
            print(f"  {fam_name}/{vkey}: contains forbidden phrase(s) {hits}")
    check('4_no_forbidden_phrases', forbidden_ok)

    # ---- 5. no canonical mechanism name used as a new family name ----
    canonical_mechs = load_canonical_mechanism_names()
    overlap = actual_families & canonical_mechs
    check('5_no_canonical_name_reuse', not overlap, f"overlap: {overlap}")

    # ---- 6. no encoding/ROT13/Base64/payload-splitting placeholders or
    # operation mentions ----
    encoding_terms = ['base64', 'rot13', 'payload_a', 'payload_b', 'instruction_b64']
    encoding_ok = True
    for fam_name, vkey, text in all_strings:
        lowered = text.lower()
        hits = [t for t in encoding_terms if t in lowered]
        if hits:
            encoding_ok = False
            print(f"  {fam_name}/{vkey}: contains encoding-operation term(s) {hits}")
    check('6_no_encoding_or_payload_split_ops', encoding_ok)

    # ---- 7. ctx_continuation templates contain no compliant prefix
    # (already covered by check 4's 'sure, here'/'certainly, here'), no
    # numbered steps, and are not otherwise pre-filled with a substantive
    # answer ----
    cont_ok = True
    cont_fam = data['families'].get('ctx_continuation', {})
    cont_texts = list(cont_fam.get('variants', {}).items()) + \
        [('family_specific_neutral_control', cont_fam.get('family_specific_neutral_control', ''))]
    for vkey, text in cont_texts:
        step_hits = [p.pattern for p in NUMBERED_STEP_PATTERNS if p.search(text)]
        if step_hits:
            cont_ok = False
            print(f"  ctx_continuation/{vkey}: numbered-step pattern(s) matched {step_hits}")
    check('7_continuation_no_numbered_steps_or_prefilled_answer', cont_ok)

    # ---- 8. schema/id uniqueness: no duplicate template text anywhere,
    # family names unique (guaranteed by JSON object keys, checked anyway) ----
    texts_seen = {}
    dup_ok = True
    for fam_name, vkey, text in all_strings:
        key = (fam_name, vkey)
        if text in texts_seen:
            dup_ok = False
            print(f"  duplicate template text: {fam_name}/{vkey} == {texts_seen[text]}")
        texts_seen[text] = key
    family_names_unique = len(data['families'].keys()) == len(set(data['families'].keys()))
    check('8_unique_ids_and_texts', dup_ok and family_names_unique)

    # ---- 9. render with harmless placeholders, confirm formatting (newlines,
    # quotes) survives correctly ----
    render_ok = True
    for fam_name, vkey, text in all_strings:
        for placeholder in HARMLESS_PLACEHOLDERS:
            try:
                rendered = text.format(instruction=placeholder)
            except Exception as e:
                render_ok = False
                print(f"  {fam_name}/{vkey}: render raised {type(e).__name__}: {e}")
                continue
            if placeholder not in rendered:
                render_ok = False
                print(f"  {fam_name}/{vkey}: rendered output does not contain the placeholder instruction")
            # newline count in the template must be preserved verbatim in the render
            if text.count('\n') != rendered.count('\n'):
                render_ok = False
                print(f"  {fam_name}/{vkey}: newline count changed after render "
                      f"({text.count(chr(10))} -> {rendered.count(chr(10))})")
    check('9_renders_correctly_with_harmless_placeholders', render_ok)

    # ---- 10. this audit script itself never overwrites an existing result
    # file; report-writing (if requested) is refused unless the target is new ----
    write_ok = True
    if args.write_report:
        if os.path.exists(args.write_report):
            write_ok = False
            print(f"  refusing to write: {args.write_report} already exists")
        else:
            report_dir = os.path.dirname(os.path.abspath(args.write_report))
            expected_dir = os.path.abspath(DEFAULT_REPORT_DIR)
            if report_dir != expected_dir:
                write_ok = False
                print(f"  refusing to write outside {DEFAULT_REPORT_DIR}: got {args.write_report}")
    check('10_never_overwrites_report_output_dir_restricted', write_ok)

    # ---- 11. ctx_continuation contains no affirmative/compliant answer
    # opener ('sure, here' / 'certainly, here' / 'here is how' / "here's
    # how") anywhere in its 4 texts (added 2026-09-08 with the
    # ctx_continuation template revision) ----
    cont_prefill_ok = True
    for vkey, text in cont_texts:
        lowered = text.lower()
        hits = [p for p in ANSWER_PREFILL_PHRASES if p in lowered]
        if hits:
            cont_prefill_ok = False
            print(f"  ctx_continuation/{vkey}: contains answer-prefill phrase(s) {hits}")
    check('11_continuation_no_answer_prefill_phrases', cont_prefill_ok)

    # ---- 12. all 4 ctx_continuation texts end in a generation-position /
    # role cue ('Response:' / 'Speaker B:' / 'New response:') ----
    cont_format_prefix_ok = True
    for vkey, text in cont_texts:
        if not text.rstrip().lower().endswith(FORMAT_ROLE_PREFIX_SUFFIXES):
            cont_format_prefix_ok = False
            print(f"  ctx_continuation/{vkey}: does not end in a format/role-position cue")
    check('12_continuation_format_prefix_present', cont_format_prefix_ok)

    # The "latest" checklist is whichever of v4/v3 currently exists, most
    # recent first. Pinning checks 13/15 to a single hardcoded version
    # (originally v3) would make them fail forever after any later round adds
    # a new checklist without regenerating the old one -- that's expected
    # staleness for an archived snapshot, not a bug. Checking "the latest one
    # that exists" keeps the invariant meaningful across rounds.
    if os.path.exists(V4_CHECKLIST_PATH):
        latest_checklist_path = V4_CHECKLIST_PATH
    elif os.path.exists(V3_CHECKLIST_PATH):
        latest_checklist_path = V3_CHECKLIST_PATH
    else:
        latest_checklist_path = None

    # ---- 13 & 15. read the latest checklist (if present) and verify (a) its
    # ctx_continuation entries name prefix_injection as the primary overlap,
    # and (b) its recorded source_template_sha256 matches the CURRENT
    # templates_context_v1.json on disk (i.e. the checklist was generated
    # from -- and hasn't drifted from -- this exact template file) ----
    latest_overlap_ok = False
    latest_hash_ok = False
    latest = None
    if latest_checklist_path is not None:
        with open(latest_checklist_path, encoding='utf-8') as f:
            latest = json.load(f)
        cont_entries = [e for e in latest.get('entries', []) if e.get('family') == 'ctx_continuation']
        latest_overlap_ok = (
            len(cont_entries) == 4
            and all(e.get('template_level_overlap') == 'prefix_injection' for e in cont_entries)
            and all('prefix_injection' in e.get('most_likely_canonical_overlap', '') for e in cont_entries)
        )
        if not latest_overlap_ok:
            print(f"  {latest_checklist_path} ctx_continuation entries do not all name prefix_injection "
                  f"as the primary overlap: "
                  f"{[(e.get('template_id'), e.get('template_level_overlap')) for e in cont_entries]}")
        with open(TEMPLATES_PATH, 'rb') as f:
            current_template_sha256 = hashlib.sha256(f.read()).hexdigest()
        latest_hash_ok = latest.get('source_template_sha256') == current_template_sha256
        if not latest_hash_ok:
            print(f"  {latest_checklist_path} source_template_sha256={latest.get('source_template_sha256')} "
                  f"does not match current templates_context_v1.json sha256={current_template_sha256}")
    else:
        print(f"  neither {V3_CHECKLIST_PATH} nor {V4_CHECKLIST_PATH} exists yet")
    check('13_latest_checklist_continuation_overlap_is_prefix_injection', latest_overlap_ok)
    check('15_latest_checklist_source_hash_matches_current_template', latest_hash_ok)

    # ---- 14. v1, v2, and v3 checklists were never modified by this or any
    # later round -- only new files (v4, v5, ...) are ever added ----
    v1_v2_v3_unmodified_ok = True
    for path, expected in [(V1_CHECKLIST_PATH, EXPECTED_V1_CHECKLIST_SHA256),
                            (V2_CHECKLIST_PATH, EXPECTED_V2_CHECKLIST_SHA256),
                            (V3_CHECKLIST_PATH, EXPECTED_V3_CHECKLIST_SHA256)]:
        if not os.path.exists(path):
            v1_v2_v3_unmodified_ok = False
            print(f"  {path} is missing")
            continue
        with open(path, 'rb') as f:
            actual = hashlib.sha256(f.read()).hexdigest()
        if actual != expected:
            v1_v2_v3_unmodified_ok = False
            print(f"  {path} sha256={actual} does not match expected={expected} -- file was modified")
    check('14_v1_v2_v3_checklists_unmodified', v1_v2_v3_unmodified_ok)

    # ---- 16. the decision sidecar (if present) contains exactly 16 unique
    # template_ids, matching the 16 real template_ids derived from the
    # current template file ----
    sidecar_ids_ok = False
    if os.path.exists(DECISIONS_SIDECAR_PATH):
        with open(DECISIONS_SIDECAR_PATH, encoding='utf-8') as f:
            sidecar = json.load(f)
        sidecar_ids = [d.get('template_id') for d in sidecar.get('decisions', [])]
        expected_ids = {
            f"{fam}_{vkey}" if vkey != 'family_specific_neutral_control' else f"{fam}_neutral"
            for fam, vkey, _ in all_strings
        }
        sidecar_ids_ok = len(sidecar_ids) == 16 and len(set(sidecar_ids)) == 16 and set(sidecar_ids) == expected_ids
        if not sidecar_ids_ok:
            print(f"  decision sidecar template_ids: n={len(sidecar_ids)}, "
                  f"n_unique={len(set(sidecar_ids))}, set_matches_expected={set(sidecar_ids) == expected_ids}")
    else:
        print(f"  {DECISIONS_SIDECAR_PATH} does not exist")
    check('16_decision_sidecar_has_16_unique_template_ids', sidecar_ids_ok)

    # ---- 17. if the latest checklist includes ctx_continuation, all 4 of
    # its entries carry the expected activation-pilot-with-format-control
    # status (only meaningful once that review round has happened; skipped
    # cleanly if no checklist exists yet) ----
    cont_status_ok = False
    if latest is not None:
        cont_entries = [e for e in latest.get('entries', []) if e.get('family') == 'ctx_continuation']
        cont_status_ok = len(cont_entries) == 4 and all(
            e.get('reviewer_status') == 'APPROVED_FOR_ACTIVATION_PILOT_WITH_FORMAT_CONTROL'
            for e in cont_entries
        )
        if not cont_status_ok:
            print(f"  ctx_continuation reviewer_status values in {latest_checklist_path}: "
                  f"{[(e.get('template_id'), e.get('reviewer_status')) for e in cont_entries]}")
    check('17_continuation_reviewer_status_is_activation_pilot_with_format_control', cont_status_ok)

    # ---- 18. the latest checklist's top-level result_status and every
    # entry's reviewer_status avoid language that would overclaim empirical
    # validation from what is still only a pilot-use approval ----
    no_overclaim_ok = False
    if latest is not None:
        strings_to_check = [latest.get('result_status', '')] + [
            e.get('reviewer_status', '') for e in latest.get('entries', [])
        ]
        offenders = [s for s in strings_to_check if any(sub in s for sub in VALIDATION_CLAIM_SUBSTRINGS)]
        no_overclaim_ok = len(offenders) == 0
        if not no_overclaim_ok:
            print(f"  status string(s) overclaiming empirical validation: {offenders}")
    check('18_no_validation_claim_language_in_latest_checklist', no_overclaim_ok)

    print()
    if failed == 0:
        print("ALL CONTEXT-TEMPLATE STATIC AUDIT CHECKS PASSED.")
    else:
        print(f"{failed} CHECK(S) FAILED.")

    if args.write_report and write_ok:
        os.makedirs(os.path.dirname(args.write_report), exist_ok=True)
        report = {
            'result_status': 'STATIC_AUDIT_NON_RESULT',
            'note': 'Pure structural/lexical audit of templates_context_v1.json. No model, no data.',
            'families_checked': sorted(actual_families),
            'checks_failed': failed,
        }
        with open(args.write_report, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote report (new file, non-overwriting): {args.write_report}")

    return failed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write_report', type=str, default=None,
                         help=f"Optional path under {DEFAULT_REPORT_DIR} to write a summary report. "
                              f"Refuses if the file already exists. Default: stdout only, no file written.")
    args = parser.parse_args()
    sys.exit(1 if main(args) else 0)
