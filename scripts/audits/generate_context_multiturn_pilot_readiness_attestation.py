"""
Generates output/audits/context/context_multiturn_pilot_readiness_attestation.json
-- a pilot-readiness admission sidecar for the multi-turn Contextual
Reconfiguration extension (EXPERIMENT_CONTEXT_MULTITURN_BEHAVIOR_PROTOCOL.md).

Per explicit user decision (2026-09-10): templates/templates_context_multiturn_v2.json
is NEVER modified, including its top-level `status` field -- that file is
the exact frozen input the real-tokenizer audit measured, and changing its
bytes would invalidate the `source_template_sha256` already recorded in
output/audits/context/context_multiturn_token_length_audit_v2.json. This
sidecar is the durable, SEPARATE "is this ready for a pilot" record
instead of mutating the template.

Pure read/verify/write -- no model weights, no GPU, no generation, no
WildGuard, no test_ids. Reads (does not modify):
  - templates/templates_context_multiturn_v2.json                          (frozen)
  - output/audits/context/context_multiturn_templates_human_review_checklist_v3.json
  - output/audits/context/context_multiturn_token_length_audit_v2.json     (CLUSTER-ONLY --
    produced by audit_context_multiturn_token_lengths_cluster.py; this
    script must be run on the cluster too, from the same checkout, AFTER
    that audit has completed)

Verifies 6 conditions before writing anything. Any failure raises
AttestationGateFailure and NO sidecar (partial or otherwise) is written:
  1. current template file's raw-byte SHA-256 == the token audit's
     recorded source_template_sha256
  2. current template's normalized content SHA-256 (recomputed here, same
     method as the token audit script) == the token audit's recorded
     template_content_sha256
  3. all 16 checklist v3 entries carry an APPROVED_FOR_PILOT* reviewer_status
  4. token audit's result_status == HUMAN_AND_TOKEN_AUDITED_READY_FOR_PILOT
  5. all 3 tokenizers show 16/16 conditions complete in the token audit
  6. all 4 gates recorded inside the token audit report are True

All hashes/paths in the written sidecar are recomputed from the real
on-disk files at generation time -- never copied from a prior message or
cached value.

Usage:
  python scripts/audits/generate_context_multiturn_pilot_readiness_attestation.py
"""
import argparse
import datetime
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts'))

from audit_context_multiturn_templates_dry_run import load_templates, TEMPLATE_PATH_V2  # noqa: E402
from audit_context_multiturn_token_lengths_cluster import (  # noqa: E402
    sha256_of_file, compute_template_content_sha256, get_git_commit,
)

CHECKLIST_V3_PATH = os.path.join(REPO_ROOT, 'output', 'audits', 'context',
                                  'context_multiturn_templates_human_review_checklist_v3.json')
TOKEN_AUDIT_V2_PATH = os.path.join(REPO_ROOT, 'output', 'audits', 'context',
                                    'context_multiturn_token_length_audit_v2.json')
APPROVED_STATUS_PREFIX = 'APPROVED_FOR_PILOT'

# Verbatim, human-authored limitations (2026-09-10) -- never paraphrased,
# never auto-derived from data.
LIMITATIONS = [
    "ctx_persona retains conceptual overlap with persona_roleplay",
    "ctx_fictional uses a family-specific acknowledgement",
    "ctx_continuation retains a shared artificial response-position cue",
    "single-versus-multi interaction includes the structural effect of an assistant history turn",
]


class AttestationGateFailure(Exception):
    """Raised when any of the 6 required pre-write verifications fails.
    The caller must not write a sidecar (partial or otherwise) if this
    is raised."""
    pass


def load_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def checklist_all_approved(checklist):
    entries = checklist.get('entries', [])
    if len(entries) != 16:
        return False, f"checklist has {len(entries)} entries, expected 16"
    not_approved = [e['template_id'] for e in entries
                     if not str(e.get('reviewer_status', '')).startswith(APPROVED_STATUS_PREFIX)]
    if not_approved:
        return False, f"not all 16 approved: {not_approved}"
    return True, "all 16 template_id carry an APPROVED_FOR_PILOT* reviewer_status"


def run_verifications():
    def require(cond, msg):
        if not cond:
            raise AttestationGateFailure(msg)

    require(os.path.exists(TEMPLATE_PATH_V2), f"template file missing: {TEMPLATE_PATH_V2}")
    require(os.path.exists(CHECKLIST_V3_PATH), f"checklist v3 missing: {CHECKLIST_V3_PATH}")
    require(os.path.exists(TOKEN_AUDIT_V2_PATH), f"token audit v2 missing: {TOKEN_AUDIT_V2_PATH}")

    data = load_templates(TEMPLATE_PATH_V2)
    checklist = load_json(CHECKLIST_V3_PATH)
    token_audit = load_json(TOKEN_AUDIT_V2_PATH)

    current_file_hash = sha256_of_file(TEMPLATE_PATH_V2)
    current_content_hash = compute_template_content_sha256(data)

    # 1. template file bytes unchanged since the token audit measured it
    require(current_file_hash == token_audit.get('source_template_sha256'),
            f"template file hash changed since token audit: current={current_file_hash}, "
            f"token_audit={token_audit.get('source_template_sha256')}")
    # 2. template normalized content unchanged since the token audit measured it
    require(current_content_hash == token_audit.get('template_content_sha256'),
            f"template content hash changed since token audit: current={current_content_hash}, "
            f"token_audit={token_audit.get('template_content_sha256')}")
    # 3. human review complete for all 16
    checklist_ok, checklist_detail = checklist_all_approved(checklist)
    require(checklist_ok, f"checklist gate failed: {checklist_detail}")
    # 4. token audit's own overall verdict
    require(token_audit.get('result_status') == 'HUMAN_AND_TOKEN_AUDITED_READY_FOR_PILOT',
            f"token audit result_status is {token_audit.get('result_status')!r}, "
            f"expected HUMAN_AND_TOKEN_AUDITED_READY_FOR_PILOT")
    # 5. all 3 tokenizers at 16/16
    per_model_counts = token_audit.get('gates_detail', {}).get('sixteen_conditions_complete_all_models', {})
    require(len(per_model_counts) == 3 and all(v == 16 for v in per_model_counts.values()),
            f"expected 3 models all at 16/16, got {per_model_counts}")
    # 6. all 4 gates inside the token audit report are true
    ta_gates = token_audit.get('gates', {})
    require(len(ta_gates) == 4 and all(ta_gates.values()),
            f"token audit gates not all true: {ta_gates}")

    return data, checklist, token_audit, current_file_hash, current_content_hash, checklist_detail


def main(args):
    (data, checklist, token_audit, current_file_hash, current_content_hash,
     checklist_detail) = run_verifications()

    sidecar = {
        'attestation_version': 'v1',
        'result_status': 'HUMAN_AND_TOKEN_AUDITED_READY_FOR_PILOT',
        'source_template_path': os.path.relpath(TEMPLATE_PATH_V2, REPO_ROOT),
        'source_template_status_at_audit': data.get('status'),
        'source_template_sha256': current_file_hash,
        'template_content_sha256': current_content_hash,
        'human_review_checklist_path': os.path.relpath(CHECKLIST_V3_PATH, REPO_ROOT),
        'token_audit_path': os.path.relpath(TOKEN_AUDIT_V2_PATH, REPO_ROOT),
        'gates': {
            'human_review_complete': True,
            'real_tokenizer_audit_complete': True,
            'three_tokenizers_successful': True,
            'sixteen_conditions_complete': True,
            'length_balance_accepted': True,
            'ready_for_pilot': True,
        },
        'gates_provenance': {
            'human_review_complete': checklist_detail,
            'real_tokenizer_audit_complete': f"token_audit.result_status == "
                                              f"{token_audit.get('result_status')!r}",
            'three_tokenizers_successful': token_audit.get('gates_detail', {}).get(
                'three_real_tokenizers_succeeded'),
            'sixteen_conditions_complete': token_audit.get('gates_detail', {}).get(
                'sixteen_conditions_complete_all_models'),
            'length_balance_accepted': "Recorded human decision (2026-09-10): length differences "
                                        "(single-digit tokens per family, no severe cross-tokenizer "
                                        "disagreement) were explicitly accepted without modifying "
                                        "template text -- this is a recorded human judgment, NOT an "
                                        "automatically-computed threshold determination.",
            'ready_for_pilot': "Logical conclusion of the other 5 gates plus all 6 pre-write "
                                "verifications in this script (see run_verifications()).",
        },
        'study_status': data.get('study_status'),
        'limitations': LIMITATIONS,
        'attested_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'git_commit': get_git_commit(),
    }

    if os.path.exists(args.write_report):
        print(f"\nRefusing to overwrite existing file: {args.write_report}")
        sys.exit(1)
    os.makedirs(os.path.dirname(os.path.abspath(args.write_report)), exist_ok=True)
    tmp_path = args.write_report + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(sidecar, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, args.write_report)
    print(f"Wrote (new file, non-overwriting): {args.write_report}")
    print(f"source_template_sha256: {current_file_hash}")
    print(f"template_content_sha256: {current_content_hash}")
    print(f"result_status: {sidecar['result_status']}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write_report', type=str,
                         default=os.path.join(REPO_ROOT, 'output', 'audits', 'context',
                                               'context_multiturn_pilot_readiness_attestation.json'))
    args = parser.parse_args()
    try:
        main(args)
    except AttestationGateFailure as e:
        print(f"\nATTESTATION GATE FAILURE -- no sidecar written: {e}")
        sys.exit(1)
