"""
REAL-tokenizer (no model weights, no GPU, no generation, no WildGuard)
CPU-only length audit for the 16 multi-turn Contextual Reconfiguration
conditions in templates/templates_context_multiturn_v1.json. Implements
EXPERIMENT_CONTEXT_MULTITURN_BEHAVIOR_PROTOCOL.md's Phase-1 real-tokenizer
length-audit requirement (2026-09-10).

Loads ONLY AutoTokenizer.from_pretrained(..., local_files_only=True) for
each of the 3 models -- NEVER AutoModelForCausalLM, never .generate(),
never WildGuard. Uses the SAME harmless placeholder instruction as the
static dry-run audit, substituted into final_user only.

For every condition, reports (per real tokenizer):
  - setup_user token count            (tokenizer(setup_user, add_special_tokens=False))
  - assistant_acknowledgement token count (same method)
  - final_user WRAPPER token count    (final_user's own tokens minus the
    placeholder instruction's own token count, tokenized in isolation --
    by the frozen Sec 4 design this is expected to be ~0 for every
    condition, since final_user is ALWAYS exactly "{instruction}" with no
    additional wrapper text; computed explicitly rather than assumed, in
    case of a tokenization boundary surprise)
  - full chat-templated history token count, via the REQUIRED call:
        tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
  - Delta vs. that family's neutral (full-history tokens)
  - per-family mean/stdev/range of full-history tokens (across its 4 conditions)
  - a descriptive z-score per condition (NOT a hard outlier cutoff --
    reported for a human to judge, per this project's standing rule
    against inventing post-hoc thresholds)

Also records, once per tokenizer (not per condition): padding_side,
pad_token_id, eos_token_id, tokenizer name_or_path, and (best-effort)
a version/revision string -- needed later to freeze the multi-turn
generation config (protocol Sec "future frozen decisions").

NEVER reports the local dry-run's mock word-count proxy as if it were a
real token count -- this script's numbers are the real ones; the mock
numbers from audit_context_multiturn_templates_dry_run.py remain clearly
separate and were always labeled as a proxy only.

Usage (cluster only -- requires the real tokenizer files on disk):
  python scripts/audits/audit_context_multiturn_token_lengths_cluster.py \
      --write_report output/audits/context/context_multiturn_token_length_audit.json
"""
import argparse
import datetime
import json
import os
import statistics
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts'))

from audit_context_multiturn_templates_dry_run import (  # noqa: E402
    load_templates, build_conditions, render_messages, PLACEHOLDER_INSTRUCTION, FAMILIES, VARIANTS,
)
from _defence_metrics import MODEL_PATHS  # noqa: E402


def load_real_tokenizer(model_path):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, local_files_only=True)


def token_count(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def audit_one_model(model_alias, model_path):
    tokenizer = load_real_tokenizer(model_path)
    data = load_templates()
    conditions = build_conditions(data)

    placeholder_tok = token_count(tokenizer, PLACEHOLDER_INSTRUCTION)

    per_condition = {}
    for tid, fam, v, cond in conditions:
        messages = render_messages(cond, PLACEHOLDER_INSTRUCTION)
        setup_tok = token_count(tokenizer, cond['setup_user'])
        ack_tok = token_count(tokenizer, cond['assistant_acknowledgement'])
        final_full_tok = token_count(tokenizer, cond['final_user'].format(instruction=PLACEHOLDER_INSTRUCTION))
        final_wrapper_tok = final_full_tok - placeholder_tok  # expected ~0 by design (Sec 4)

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

    # per-family delta vs neutral, mean/stdev/range, descriptive z-scores
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


def main(args):
    per_model_results = {}
    for model_idx, (alias, path) in MODEL_PATHS.items():
        print(f"Auditing {alias} ({path}) ...")
        per_model_results[alias] = audit_one_model(alias, path)
        print(f"  done.")

    report = {
        'result_status': 'REAL_TOKENIZER_LENGTH_AUDIT',
        'note': ('Real per-model tokenizer counts (NOT the mock word-count proxy from '
                 'audit_context_multiturn_templates_dry_run.py). No model weights loaded, no '
                 'generation, no WildGuard. final_user_wrapper_tokens is expected to be ~0 for every '
                 'condition by the frozen Sec 4 design (final_user is always bare {instruction}); a '
                 'nonzero value here would indicate a tokenization boundary effect worth investigating, '
                 'not assumed away. descriptive_z_score_within_family is reported for human judgment, '
                 'never as an automated pass/fail outlier gate.'),
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write_report', type=str,
                         default=os.path.join(REPO_ROOT, 'output', 'audits', 'context',
                                               'context_multiturn_token_length_audit.json'))
    args = parser.parse_args()
    main(args)
