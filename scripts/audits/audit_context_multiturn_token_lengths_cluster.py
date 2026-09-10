"""
REAL-tokenizer (no model weights, no GPU, no generation, no WildGuard)
CPU-only length audit for the 16 multi-turn Contextual Reconfiguration
conditions in templates/templates_context_multiturn_v2.json. Implements
EXPERIMENT_CONTEXT_MULTITURN_BEHAVIOR_PROTOCOL.md's Phase-1 real-tokenizer
length-audit requirement (2026-09-10).

v2 of this script (2026-09-10) adds full provenance/gating on top of the
first run's output (which is missing source_template_sha256,
template_content_sha256, transformers/Python version, and git_commit --
see context_multiturn_token_length_audit_INCOMPLETE_PROVENANCE.json for
that superseded report). This version:

  - asserts a full completeness/consistency checklist BEFORE writing
    anything -- any failure raises AuditGateFailure and NO report (partial
    or otherwise) is written;
  - records source_template_sha256 (raw file bytes) AND
    template_content_sha256 (a normalized hash over only the 16
    conditions' substantive fields -- family, positive_or_neutral,
    setup_user, assistant_acknowledgement, final_user -- sorted by
    template_id, so that future edits to status/review metadata never
    invalidate this hash; recomputed fresh from the current v2 template
    on every run, never read from the checklist file);
  - records python_version, transformers_version, git_commit,
    tokenizer_paths;
  - reads the CURRENT output/audits/context/context_multiturn_templates_human_review_checklist_v3.json
    and gates result_status on whether all 16 entries carry an
    APPROVED_FOR_PILOT* reviewer_status;
  - sets result_status to HUMAN_AND_TOKEN_AUDITED_READY_FOR_PILOT only if
    ALL of: 3/3 real tokenizers succeeded, 16/16 conditions complete for
    each, the two hash fields are present, and the checklist gate above
    holds. Otherwise result_status stays
    REAL_TOKENIZER_LENGTH_AUDIT_GATES_INCOMPLETE and the template's own
    top-level `status` field must NOT be advanced to a pilot-ready value.

Loads ONLY AutoTokenizer.from_pretrained(..., local_files_only=True) for
each of the 3 models -- NEVER AutoModelForCausalLM, never .generate(),
never WildGuard. Uses the SAME harmless placeholder instruction as the
static dry-run audit, substituted into final_user only.

For every condition, reports (per real tokenizer):
  - setup_user token count            (tokenizer(setup_user, add_special_tokens=False))
  - assistant_acknowledgement token count (same method)
  - final_user WRAPPER token count    (final_user's own tokens minus the
    placeholder instruction's own token count, tokenized in isolation --
    ~0 for ctx_persona/ctx_authority/ctx_fictional, by design nonzero for
    ctx_continuation post-B2-revision, which wraps the instruction in
    "Request: ...\nResponse:")
  - full chat-templated history token count, via the REQUIRED call:
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
  - Delta vs. that family's neutral (full-history tokens)
  - per-family mean/stdev/range of full-history tokens (across its 4 conditions)
  - a descriptive z-score per condition (NOT a hard outlier cutoff --
    reported for a human to judge, per this project's standing rule
    against inventing post-hoc thresholds)

Also records, once per tokenizer (not per condition): padding_side,
pad_token_id, eos_token_id, tokenizer name_or_path -- needed later to
freeze the multi-turn generation config (protocol Sec 14).

NEVER reports the local dry-run's mock word-count proxy as if it were a
real token count -- this script's numbers are the real ones; the mock
numbers from audit_context_multiturn_templates_dry_run.py remain clearly
separate and were always labeled as a proxy only.

Usage (cluster only -- requires the real tokenizer files on disk):
  python scripts/audits/audit_context_multiturn_token_lengths_cluster.py \
      --write_report output/audits/context/context_multiturn_token_length_audit_v2.json
"""
import argparse
import datetime
import hashlib
import json
import os
import statistics
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts'))
sys.path.insert(0, SCRIPT_DIR)

from audit_context_multiturn_templates_dry_run import (  # noqa: E402
    load_templates, build_conditions, render_messages, PLACEHOLDER_INSTRUCTION, FAMILIES, VARIANTS,
    TEMPLATE_PATH_V2,
)
from _defence_metrics import MODEL_PATHS  # noqa: E402

CHECKLIST_V3_PATH = os.path.join(REPO_ROOT, 'output', 'audits', 'context',
                                  'context_multiturn_templates_human_review_checklist_v3.json')
APPROVED_STATUS_PREFIX = 'APPROVED_FOR_PILOT'


class AuditGateFailure(Exception):
    """Raised when a pre-write completeness/consistency assertion fails.
    No partial report is ever written when this is raised."""
    pass


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_of_file(path):
    with open(path, 'rb') as f:
        return sha256_bytes(f.read())


def compute_template_content_sha256(data):
    """Normalized content hash over the 16 conditions' SUBSTANTIVE fields
    only. Deliberately excludes status/generated_at/review fields so that
    updating review metadata never invalidates this hash. Uses the exact
    canonicalization the user specified: sort by template_id, then
    json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")).

    NOTE: this is intentionally NOT the same value as the
    `template_content_sha256` field already present in
    context_multiturn_templates_human_review_checklist_v3.json -- that
    field was computed by generate_context_multiturn_review_checklist.py
    as a RAW FILE hash (equivalent to this script's source_template_sha256),
    not a normalized-content hash. Flagged explicitly in the completion
    report; not silently reconciled here."""
    conditions = build_conditions(data)
    content_by_id = {}
    for tid, fam, v, cond in conditions:
        content_by_id[tid] = {
            'family': fam,
            'positive_or_neutral': 'neutral' if v == 'neutral' else 'positive',
            'setup_user': cond['setup_user'],
            'assistant_acknowledgement': cond['assistant_acknowledgement'],
            'final_user': cond['final_user'],
        }
    ordered = {tid: content_by_id[tid] for tid in sorted(content_by_id)}
    serialized = json.dumps(ordered, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
    return sha256_bytes(serialized.encode('utf-8'))


def get_git_commit():
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception as e:
        return f'UNAVAILABLE ({type(e).__name__}: {e})'


def get_transformers_version():
    import transformers
    return transformers.__version__


def load_real_tokenizer(model_path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, local_files_only=True)


def token_count(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def audit_one_model(model_alias, model_path, data):
    tokenizer = load_real_tokenizer(model_path)
    conditions = build_conditions(data)

    placeholder_tok = token_count(tokenizer, PLACEHOLDER_INSTRUCTION)

    per_condition = {}
    for tid, fam, v, cond in conditions:
        messages = render_messages(cond, PLACEHOLDER_INSTRUCTION)
        setup_tok = token_count(tokenizer, cond['setup_user'])
        ack_tok = token_count(tokenizer, cond['assistant_acknowledgement'])
        final_full_tok = token_count(tokenizer, cond['final_user'].format(instruction=PLACEHOLDER_INSTRUCTION))
        final_wrapper_tok = final_full_tok - placeholder_tok  # ~0 except ctx_continuation (Sec 12)

        full_history_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        full_history_tok = len(full_history_ids)

        per_condition[tid] = {
            'family': fam, 'variant_or_neutral': v, 'is_positive': v != 'neutral',
            'setup_user_tokens': setup_tok,
            'assistant_acknowledgement_tokens': ack_tok,
            'final_user_wrapper_tokens': final_wrapper_tok,
            'final_user_full_tokens_with_placeholder': final_full_tok,
            'full_history_tokens_with_generation_prompt': full_history_tok,
        }

    family_summary = {}
    for fam in FAMILIES:
        neutral_tok = per_condition[f'{fam}_neutral']['full_history_tokens_with_generation_prompt']
        fam_tokens = {v: per_condition[f'{fam}_{v}']['full_history_tokens_with_generation_prompt']
                      for v in VARIANTS}
        fam_tokens['neutral'] = neutral_tok
        vals = list(fam_tokens.values())
        mean = statistics.mean(vals)
        stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
        for v in VARIANTS:
            delta = fam_tokens[v] - neutral_tok
            per_condition[f'{fam}_{v}']['delta_full_history_tokens_vs_neutral'] = delta
            z = (fam_tokens[v] - mean) / stdev if stdev > 0 else None
            per_condition[f'{fam}_{v}']['descriptive_z_score_within_family'] = z
        per_condition[f'{fam}_neutral']['delta_full_history_tokens_vs_neutral'] = 0
        per_condition[f'{fam}_neutral']['descriptive_z_score_within_family'] = (
            (neutral_tok - mean) / stdev if stdev > 0 else None)
        family_summary[fam] = {
            'full_history_tokens_by_condition': fam_tokens,
            'mean': mean, 'stdev': stdev, 'range': max(vals) - min(vals),
            'min': min(vals), 'max': max(vals),
        }

    tokenizer_info = {
        'model_alias': model_alias, 'model_path': model_path,
        'tokenizer_class': type(tokenizer).__name__,
        'padding_side': tokenizer.padding_side,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
        'vocab_size': tokenizer.vocab_size,
        'placeholder_instruction_alone_tokens': placeholder_tok,
    }

    return {
        'tokenizer_info': tokenizer_info,
        'per_condition': per_condition,
        'family_summary': family_summary,
    }


def cross_tokenizer_outlier_view(per_model_results):
    """Descriptive only -- NOT a hard pass/fail gate. For each condition,
    reports each tokenizer's within-family z-score side by side, so a
    human can judge whether any condition is a consistent outlier across
    ALL 3 tokenizers (a structural template issue) vs. an outlier for only
    one model's tokenizer (a tokenizer-specific quirk, less concerning)."""
    out = {}
    all_tids = list(next(iter(per_model_results.values()))['per_condition'].keys())
    for tid in all_tids:
        out[tid] = {alias: r['per_condition'][tid]['descriptive_z_score_within_family']
                    for alias, r in per_model_results.items()}
    return out


def run_completeness_assertions(data, per_model_results):
    """All hard-stop assertions required before ANY report may be written.
    Raises AuditGateFailure with a precise message on the first violation
    found; the caller must not write a report (partial or otherwise) if
    this raises."""

    def require(cond, msg):
        if not cond:
            raise AuditGateFailure(msg)

    require(len(MODEL_PATHS) == 3, f"expected exactly 3 models in MODEL_PATHS, found {len(MODEL_PATHS)}")
    require(len(per_model_results) == 3, f"expected 3 audited models, found {len(per_model_results)}")

    conditions = build_conditions(data)
    require(len(FAMILIES) == 4, f"expected 4 families, found {len(FAMILIES)}")
    n_positive = sum(1 for _, _, v, _ in conditions if v != 'neutral')
    n_neutral = sum(1 for _, _, v, _ in conditions if v == 'neutral')
    require(n_positive == 12, f"expected 12 positive conditions, found {n_positive}")
    require(n_neutral == 4, f"expected 4 neutral conditions, found {n_neutral}")
    ids = [c[0] for c in conditions]
    require(len(set(ids)) == 16, f"expected 16 unique template_id, found {len(set(ids))}: {ids}")

    for tid, fam, v, cond in conditions:
        msgs = render_messages(cond, PLACEHOLDER_INSTRUCTION)
        require(len(msgs) == 3 and [m['role'] for m in msgs] == ['user', 'assistant', 'user'],
                f"{tid}: expected role order user/assistant/user, got {[m['role'] for m in msgs]}")
        require(cond['final_user'].count('{instruction}') == 1,
                f"{tid}: final_user must contain exactly one {{instruction}} placeholder")
        require(cond['setup_user'].count('{instruction}') == 0,
                f"{tid}: setup_user must not contain {{instruction}}")
        require(cond['assistant_acknowledgement'].count('{instruction}') == 0,
                f"{tid}: assistant_acknowledgement must not contain {{instruction}}")

    for alias, result in per_model_results.items():
        n = len(result['per_condition'])
        require(n == 16, f"{alias}: expected 16 rendered conditions, found {n}")
        for tid, entry in result['per_condition'].items():
            for key in ('setup_user_tokens', 'assistant_acknowledgement_tokens',
                        'final_user_wrapper_tokens', 'full_history_tokens_with_generation_prompt'):
                val = entry[key]
                require(isinstance(val, int) and val >= 0,
                        f"{alias}/{tid}: {key}={val!r} is not a valid non-negative integer count")
        for fam, summary in result['family_summary'].items():
            for key in ('mean', 'stdev', 'range'):
                v = summary[key]
                require(isinstance(v, (int, float)) and v == v and v not in (float('inf'), float('-inf')),
                        f"{alias}/{fam}: family_summary[{key}]={v!r} is not a finite number")

    cont_final_users = [data['families']['ctx_continuation']['variants'][v]['final_user'] for v in VARIANTS]
    cont_final_users.append(data['families']['ctx_continuation']['family_specific_neutral_control']['final_user'])
    require(len(set(cont_final_users)) == 1,
            f"ctx_continuation final_user must be byte-identical across all 4 conditions, got {cont_final_users}")


def load_checklist_v3():
    if not os.path.exists(CHECKLIST_V3_PATH):
        return None
    with open(CHECKLIST_V3_PATH, encoding='utf-8') as f:
        return json.load(f)


def checklist_all_approved(checklist):
    """Descriptive gate check, not a re-review -- trusts the human's own
    recorded reviewer_status values verbatim."""
    if checklist is None:
        return False, "checklist v3 file not found"
    entries = checklist.get('entries', [])
    if len(entries) != 16:
        return False, f"checklist has {len(entries)} entries, expected 16"
    not_approved = [e['template_id'] for e in entries
                     if not str(e.get('reviewer_status', '')).startswith(APPROVED_STATUS_PREFIX)]
    if not_approved:
        return False, f"not all 16 approved: {not_approved}"
    return True, "all 16 template_id carry an APPROVED_FOR_PILOT* reviewer_status"


def main(args):
    data = load_templates(TEMPLATE_PATH_V2)

    per_model_results = {}
    for model_idx, (alias, path) in MODEL_PATHS.items():
        print(f"Auditing {alias} ({path}) ...")
        per_model_results[alias] = audit_one_model(alias, path, data)
        print(f"  done.")

    # Hard-stop gate: raises before anything is written if any assertion fails.
    run_completeness_assertions(data, per_model_results)

    source_template_relpath = os.path.relpath(TEMPLATE_PATH_V2, REPO_ROOT)
    source_template_sha256 = sha256_of_file(TEMPLATE_PATH_V2)
    template_content_sha256 = compute_template_content_sha256(data)

    checklist = load_checklist_v3()
    checklist_ok, checklist_detail = checklist_all_approved(checklist)

    gates = {
        'three_real_tokenizers_succeeded': len(per_model_results) == 3,
        'sixteen_conditions_complete_all_models': all(len(r['per_condition']) == 16
                                                        for r in per_model_results.values()),
        'provenance_hash_fields_present': bool(source_template_sha256) and bool(template_content_sha256),
        'human_review_checklist_all_16_approved': checklist_ok,
    }
    gates_detail = {
        'three_real_tokenizers_succeeded': f"{len(per_model_results)}/3 models audited",
        'sixteen_conditions_complete_all_models': {a: len(r['per_condition']) for a, r in per_model_results.items()},
        'provenance_hash_fields_present': f"source_template_sha256={source_template_sha256[:12]}..., "
                                           f"template_content_sha256={template_content_sha256[:12]}...",
        'human_review_checklist_all_16_approved': checklist_detail,
    }

    result_status = ('HUMAN_AND_TOKEN_AUDITED_READY_FOR_PILOT' if all(gates.values())
                      else 'REAL_TOKENIZER_LENGTH_AUDIT_GATES_INCOMPLETE')

    report = {
        'result_status': result_status,
        'gates': gates,
        'gates_detail': gates_detail,
        'note': ('Real per-model tokenizer counts (NOT the mock word-count proxy from '
                 'audit_context_multiturn_templates_dry_run.py). No model weights loaded, no '
                 'generation, no WildGuard. final_user_wrapper_tokens is ~0 for ctx_persona/'
                 'ctx_authority/ctx_fictional (bare {instruction}) and nonzero for ctx_continuation '
                 '(B2 revision wraps the instruction in "Request: ...\\nResponse:"), by design -- see '
                 'EXPERIMENT_CONTEXT_MULTITURN_BEHAVIOR_PROTOCOL.md Sec 12. '
                 'descriptive_z_score_within_family is reported for human judgment, never as an '
                 'automated pass/fail outlier gate. result_status only becomes '
                 'HUMAN_AND_TOKEN_AUDITED_READY_FOR_PILOT when ALL of `gates` are true; otherwise it '
                 'stays REAL_TOKENIZER_LENGTH_AUDIT_GATES_INCOMPLETE and the template\'s top-level '
                 '`status` field must NOT be advanced to a pilot-ready value. NOTE: template_content_sha256 '
                 'here is a normalized-content hash (family/positive_or_neutral/setup_user/'
                 'assistant_acknowledgement/final_user only, sorted by template_id) and is NOT the same '
                 'value as the identically-named field already stored in '
                 'context_multiturn_templates_human_review_checklist_v3.json, which is a raw-file hash '
                 '(equivalent to this report\'s source_template_sha256). Not silently reconciled here.'),
        'source_template_path': source_template_relpath,
        'source_template_sha256': source_template_sha256,
        'template_content_sha256': template_content_sha256,
        'taxonomy_version': data.get('taxonomy_version'),
        'source_template_status': data.get('status'),
        'n_families': len(FAMILIES),
        'n_positive_templates': 12,
        'n_neutral_controls': 4,
        'n_total_conditions': 16,
        'python_version': sys.version,
        'transformers_version': get_transformers_version(),
        'git_commit': get_git_commit(),
        'tokenizer_paths': {alias: r['tokenizer_info']['model_path'] for alias, r in per_model_results.items()},
        'placeholder_instruction': PLACEHOLDER_INSTRUCTION,
        'apply_chat_template_call': "tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)",
        'per_model': per_model_results,
        'cross_tokenizer_z_scores_by_condition': cross_tokenizer_outlier_view(per_model_results),
        'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    if os.path.exists(args.write_report):
        print(f"\nRefusing to overwrite existing file: {args.write_report}")
        sys.exit(1)
    os.makedirs(os.path.dirname(os.path.abspath(args.write_report)), exist_ok=True)
    tmp_path = args.write_report + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, args.write_report)
    print(f"\nWrote (new file, non-overwriting): {args.write_report}")
    print(f"result_status: {result_status}")
    for k, v in gates.items():
        print(f"  gate[{k}] = {v}  ({gates_detail[k]})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write_report', type=str,
                         default=os.path.join(REPO_ROOT, 'output', 'audits', 'context',
                                               'context_multiturn_token_length_audit_v2.json'))
    args = parser.parse_args()
    try:
        main(args)
    except AuditGateFailure as e:
        print(f"\nAUDIT GATE FAILURE -- no report written: {e}")
        sys.exit(1)
