"""
Generates a standalone, independently-checkable attestation that the formal
token-length audit (output/audits/context/context_templates_token_length_audit.json)
was run against the exact same template CONTENT (the 16 rendered texts) as
the current templates/templates_context_v1.json -- distinguishing that from
the whole-file hash, which differs because the `status` field was bumped
after the audit ran (see EXPERIMENT2_CONTEXT_RECONFIGURATION_PROTOCOL.md
Sec 9).

Pure Python, no torch/transformers. Uses a single read-only `git show` (via
scripts/utils/context_template_provenance.get_git_show_text) -- never
checks out or modifies the working tree. Refuses to overwrite an existing
attestation file.

Usage:
  python scripts/audits/generate_context_template_provenance_attestation.py \
      --write_report output/audits/context/context_templates_token_audit_provenance_attestation.json
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)
from audit_context_templates_dry_run import (  # noqa: E402
    TEMPLATES_PATH, TEMPLATES_REL_PATH, TOKEN_AUDIT_PATH, TOKEN_AUDIT_COMMIT,
    REPO_ROOT, CONTENT_DIFF_PATH_PATTERN,
)
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
from utils.context_template_provenance import (  # noqa: E402
    template_content_sha256, canonical_template_content_payload, diff_json_paths, get_git_show_text,
    CONTENT_HASH_SCHEMA_VERSION, INCLUDED_FIELDS, EXCLUDED_FIELDS,
)

DEFAULT_REPORT_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'output', 'audits', 'context')


def main(args):
    if os.path.exists(args.write_report):
        print(f"Refusing to write: {args.write_report} already exists.")
        sys.exit(1)
    report_dir = os.path.dirname(os.path.abspath(args.write_report))
    expected_dir = os.path.abspath(DEFAULT_REPORT_DIR)
    if report_dir != expected_dir:
        print(f"Refusing to write outside {DEFAULT_REPORT_DIR}: got {args.write_report}")
        sys.exit(1)

    if not os.path.exists(TOKEN_AUDIT_PATH):
        print(f"FATAL: formal token audit report not found at {TOKEN_AUDIT_PATH} -- run it first.")
        sys.exit(1)
    with open(TOKEN_AUDIT_PATH, 'rb') as f:
        audit_bytes = f.read()
    audit_report_sha256 = hashlib.sha256(audit_bytes).hexdigest()
    token_audit = json.loads(audit_bytes)
    audit_time_source_file_sha256 = token_audit.get('source_template_sha256')
    if not audit_time_source_file_sha256:
        print("FATAL: token audit report has no source_template_sha256 field.")
        sys.exit(1)

    with open(TEMPLATES_PATH, 'rb') as f:
        current_bytes = f.read()
    current_source_file_sha256 = hashlib.sha256(current_bytes).hexdigest()
    current_data = json.loads(current_bytes)
    current_template_content_sha256 = template_content_sha256(current_data)

    historical_text, git_err = get_git_show_text(TOKEN_AUDIT_COMMIT, TEMPLATES_REL_PATH, REPO_ROOT)
    if git_err is not None:
        print(f"FATAL: PROVENANCE_UNVERIFIED -- could not read historical template via git show: {git_err}")
        sys.exit(1)
    try:
        historical_data = json.loads(historical_text)
    except json.JSONDecodeError as e:
        print(f"FATAL: PROVENANCE_UNVERIFIED -- historical file at {TOKEN_AUDIT_COMMIT} is not valid JSON: {e}")
        sys.exit(1)

    hist_full_sha256 = hashlib.sha256(historical_text.encode('utf-8')).hexdigest()
    if hist_full_sha256 != audit_time_source_file_sha256:
        print(f"FATAL: historical full-file sha256={hist_full_sha256} does not match the audit's own "
              f"recorded source_template_sha256={audit_time_source_file_sha256} -- TOKEN_AUDIT_COMMIT is "
              f"wrong, or the audit report does not correspond to that commit. Refusing to attest.")
        sys.exit(1)

    audit_time_template_content_sha256 = template_content_sha256(historical_data)
    template_texts_identical = audit_time_template_content_sha256 == current_template_content_sha256

    hist_payload = canonical_template_content_payload(historical_data)
    cur_payload = canonical_template_content_payload(current_data)
    mismatched_template_ids = []
    for fam_name in sorted(set(hist_payload) | set(cur_payload)):
        hist_fam = hist_payload.get(fam_name, {})
        cur_fam = cur_payload.get(fam_name, {})
        for vkey in sorted(set(hist_fam.get('variants', {})) | set(cur_fam.get('variants', {}))):
            if hist_fam.get('variants', {}).get(vkey) != cur_fam.get('variants', {}).get(vkey):
                mismatched_template_ids.append(f"{fam_name}_{vkey}")
        if hist_fam.get('family_specific_neutral_control') != cur_fam.get('family_specific_neutral_control'):
            mismatched_template_ids.append(f"{fam_name}_neutral")

    changed_json_paths = diff_json_paths(historical_data, current_data)
    content_changed_paths = [p for p in changed_json_paths if CONTENT_DIFF_PATH_PATTERN.match(p)]
    metadata_only_change = len(content_changed_paths) == 0

    attestation = {
        'result_status': 'PROVENANCE_ATTESTATION',
        'audited_git_commit': TOKEN_AUDIT_COMMIT,
        'audit_report_path': os.path.relpath(TOKEN_AUDIT_PATH, REPO_ROOT),
        'audit_report_sha256': audit_report_sha256,
        'audit_time_source_file_sha256': audit_time_source_file_sha256,
        'current_source_file_sha256': current_source_file_sha256,
        'audit_time_template_content_sha256': audit_time_template_content_sha256,
        'current_template_content_sha256': current_template_content_sha256,
        'content_hash_schema_version': CONTENT_HASH_SCHEMA_VERSION,
        'included_fields': INCLUDED_FIELDS,
        'excluded_fields': EXCLUDED_FIELDS,
        'template_texts_identical': template_texts_identical,
        'mismatched_template_ids': mismatched_template_ids,
        'metadata_only_change': metadata_only_change,
        'changed_json_paths': changed_json_paths,
        'content_changed_paths': content_changed_paths,
        'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    print(json.dumps(attestation, indent=2))

    if not template_texts_identical or not metadata_only_change:
        print("\nWARNING: attestation indicates the audited template content and the current template "
              "content are NOT provably identical. Writing the attestation anyway (it documents the "
              "actual finding), but this must be investigated before treating pilot admission as valid.")

    os.makedirs(report_dir, exist_ok=True)
    with open(args.write_report, 'w') as f:
        json.dump(attestation, f, indent=2)
    print(f"\nWrote attestation (new file, non-overwriting): {args.write_report}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write_report', type=str, required=True)
    args = parser.parse_args()
    main(args)
