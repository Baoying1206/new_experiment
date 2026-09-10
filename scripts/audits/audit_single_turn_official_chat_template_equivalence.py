"""
Real-tokenizer (no model weights, no GPU) audit comparing the single-turn
study's ACTUAL hand-rolled prompt construction against each model's
OFFICIAL tokenizer.apply_chat_template(...). Implements protocol Sec
'审计single-turn与官方chat template的token等价性' (2026-09-10) --
EXPERIMENT_CONTEXT_MULTITURN_BEHAVIOR_PROTOCOL.md.

Uses the REAL pipeline.model_utils.{qwen2,llama3,gemma2}_model functions
(tokenize_instructions_*_chat) and each model's REAL _load_tokenizer
kwargs, imported directly from the cluster's pipeline package -- never
hand-copied or reimplemented, so this script is byte-for-byte faithful to
what the single-turn study actually ran, including its known quirks
(e.g. LLAMA3_CHAT_TEMPLATE's literal 4-quote typo -- see the driver's
LLAMA3_LEADING_QUOTE_FINDING constant, and QWEN's use_fast=False slow
tokenizer -- see QWEN_TOKENIZER_FAST_VS_SLOW_OPEN_QUESTION).

For each of the 3 models: loads ONE tokenizer using single-turn's REAL
_load_tokenizer kwargs (never a "clean" substitute), then for >=4 harmless
placeholder instructions x (1 positive + 1 neutral per family, from
templates/templates_context_v1.json -- the SINGLE-TURN template, read-
only, NEVER modified) = 32 rendered prompts per model:

  A = tokenize_instructions_*_chat(tokenizer, instructions=[rendered])['input_ids'][0]
  B = tokenizer.apply_chat_template([{"role": "user", "content": rendered}],
                                     tokenize=True, add_generation_prompt=True)

Compares A vs B token-for-token using the SAME tokenizer instance for
both calls, so the ONLY variable is the serialization method itself:
exact match, first divergence index, length difference, and decoded text
with skip_special_tokens=False (so any BOS/EOS/turn-delimiter difference
is directly visible).

NEVER reruns or modifies the completed single-turn study's own results;
NEVER changes the multi-turn driver's own path based on what this finds.
If ANY model shows a mismatch, result_status is
SERIALIZATION_EQUIVALENCE_DIVERGENT and, per the protocol: the multi-turn
path is not auto-changed, Interaction_f's interpretation must be
downgraded to a joint presentation-mode + serialization-implementation
difference, and any official-chat-template single-turn sensitivity re-run
is a human decision, never automatic.

Requires (cluster only): PYTHONPATH set to reach `pipeline` (see
slurm/multiturn_behavioral_test_pilot.sh for the exact export), and each
model's tokenizer files locally available (local_files_only=True -- no
model WEIGHTS are ever loaded, no GPU used, CPU-only).

Usage:
  python scripts/audits/audit_single_turn_official_chat_template_equivalence.py \
      --write_report output/audits/context/multiturn_single_turn_serialization_equivalence.json
"""
import argparse
import datetime
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts'))

from _defence_metrics import MODEL_PATHS, sha256_of_file, git_commit_hash  # noqa: E402

SINGLE_TURN_TEMPLATE_PATH = os.path.join(REPO_ROOT, 'templates', 'templates_context_v1.json')

# >= 4 harmless placeholder instructions (protocol requirement), varied
# content to guard against any single-instruction-specific token quirk.
PLACEHOLDER_INSTRUCTIONS = [
    "Explain how a widget factory schedules its weekly production run.",
    "Describe the water cycle in simple terms for a classroom presentation.",
    "Summarize the plot of a mystery novel in three sentences.",
    "List five tips for organizing a home office.",
]

FAMILIES = ['ctx_persona', 'ctx_authority', 'ctx_fictional', 'ctx_continuation']


def load_single_turn_conditions():
    """1 positive variant + neutral per family = 8 (family, variant_or_neutral,
    template_text) tuples, from the FROZEN single-turn template (read-only,
    never modified, never touched by this script)."""
    with open(SINGLE_TURN_TEMPLATE_PATH, encoding='utf-8') as f:
        data = json.load(f)
    conditions = []
    for fam in FAMILIES:
        fam_data = data['families'][fam]
        first_variant_key = sorted(fam_data['variants'].keys())[0]
        conditions.append((fam, first_variant_key, fam_data['variants'][first_variant_key]))
        conditions.append((fam, 'neutral', fam_data['family_specific_neutral_control']))
    return conditions


def render(template_text, instruction):
    return template_text.format(instruction=instruction)


def load_single_turn_tokenizer(model_alias, model_path):
    """Reproduces pipeline/model_utils/{qwen2,llama3,gemma2}_model.py's
    REAL _load_tokenizer exactly (cited by file+line in
    EXPERIMENT_CONTEXT_MULTITURN_BEHAVIOR_PROTOCOL.md). Tokenizer only, no
    model weights."""
    from transformers import AutoTokenizer
    if model_alias == 'Qwen2.5-7B-Instruct':
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False,
                                             local_files_only=True)
        tok.padding_side = 'left'
    elif model_alias == 'Meta-Llama-3.1-8B-Instruct':
        tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        tok.padding_side = 'left'
        tok.pad_token = tok.eos_token
    elif model_alias == 'gemma-2-9b-it':
        tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        tok.padding_side = 'left'
    else:
        raise ValueError(f"unknown model_alias {model_alias!r}")
    return tok


def get_single_turn_tokenize_fn(model_alias):
    if model_alias == 'Qwen2.5-7B-Instruct':
        from pipeline.model_utils.qwen2_model import tokenize_instructions_qwen_chat
        return tokenize_instructions_qwen_chat
    if model_alias == 'Meta-Llama-3.1-8B-Instruct':
        from pipeline.model_utils.llama3_model import tokenize_instructions_llama3_chat
        return tokenize_instructions_llama3_chat
    if model_alias == 'gemma-2-9b-it':
        from pipeline.model_utils.gemma2_model import tokenize_instructions_gemma_chat
        return tokenize_instructions_gemma_chat
    raise ValueError(f"unknown model_alias {model_alias!r}")


def compare_one(tokenizer, tokenize_fn, rendered_text):
    single_turn_result = tokenize_fn(tokenizer=tokenizer, instructions=[rendered_text], system=None)
    single_turn_ids = single_turn_result.input_ids[0].tolist()
    official_ids = list(tokenizer.apply_chat_template(
        [{"role": "user", "content": rendered_text}], tokenize=True, add_generation_prompt=True))

    exact_match = single_turn_ids == official_ids
    first_divergence = None
    for i in range(min(len(single_turn_ids), len(official_ids))):
        if single_turn_ids[i] != official_ids[i]:
            first_divergence = i
            break
    if first_divergence is None and len(single_turn_ids) != len(official_ids):
        first_divergence = min(len(single_turn_ids), len(official_ids))

    return {
        'exact_match': exact_match,
        'single_turn_length': len(single_turn_ids), 'official_length': len(official_ids),
        'first_divergence_index': first_divergence,
        'single_turn_ids': single_turn_ids, 'official_ids': official_ids,
        'single_turn_decoded_text': tokenizer.decode(single_turn_ids, skip_special_tokens=False),
        'official_decoded_text': tokenizer.decode(official_ids, skip_special_tokens=False),
    }


def audit_one_model(model_alias, model_path, conditions):
    tokenizer = load_single_turn_tokenizer(model_alias, model_path)
    tokenize_fn = get_single_turn_tokenize_fn(model_alias)

    cases = []
    for fam, vkey, template_text in conditions:
        for instr in PLACEHOLDER_INSTRUCTIONS:
            rendered = render(template_text, instr)
            result = compare_one(tokenizer, tokenize_fn, rendered)
            result.update({'family': fam, 'variant_or_neutral': vkey, 'placeholder_instruction': instr})
            cases.append(result)

    n_total = len(cases)
    n_exact_match = sum(1 for c in cases if c['exact_match'])
    mismatches = [c for c in cases if not c['exact_match']]

    return {
        'model_alias': model_alias, 'model_path': model_path,
        'tokenizer_class': type(tokenizer).__name__, 'padding_side': tokenizer.padding_side,
        'n_total_cases': n_total, 'n_exact_match': n_exact_match,
        'all_match': n_exact_match == n_total,
        'mismatches': mismatches,
        'cases': cases,
    }


def main(args):
    conditions = load_single_turn_conditions()
    assert len(conditions) == 8, f"expected 8 conditions (1 positive + neutral x 4 families), got {len(conditions)}"
    assert len(PLACEHOLDER_INSTRUCTIONS) >= 4

    per_model = {}
    for idx, (alias, path) in MODEL_PATHS.items():
        print(f"Auditing {alias} ({path}) ...")
        per_model[alias] = audit_one_model(alias, path, conditions)
        m = per_model[alias]
        status = "ALL MATCH" if m['all_match'] else f"{len(m['mismatches'])} MISMATCH(ES)"
        print(f"  {alias}: {m['n_exact_match']}/{m['n_total_cases']} exact match -- {status}")

    all_models_match = all(m['all_match'] for m in per_model.values())
    result_status = ('SERIALIZATION_EQUIVALENCE_CONFIRMED' if all_models_match
                      else 'SERIALIZATION_EQUIVALENCE_DIVERGENT')

    report = {
        'result_status': result_status,
        'note': ("Compares the single-turn study's REAL hand-rolled prompt construction "
                 "(pipeline.model_utils.*_model.tokenize_instructions_*_chat, imported directly, never "
                 "hand-copied) against each model's official tokenizer.apply_chat_template(...), using the "
                 "SAME tokenizer instance for both (loaded with single-turn's real _load_tokenizer kwargs) "
                 "so the ONLY variable is the serialization method itself. Never reruns or modifies the "
                 "completed single-turn study's own results. If DIVERGENT: the multi-turn path is NOT "
                 "automatically changed; Interaction_f must be interpreted as a joint presentation-mode + "
                 "serialization-implementation difference; any official-chat-template single-turn "
                 "sensitivity re-run requires a human decision, never automatic."),
        'placeholder_instructions': PLACEHOLDER_INSTRUCTIONS,
        'n_placeholder_instructions': len(PLACEHOLDER_INSTRUCTIONS),
        'n_conditions': len(conditions),
        'conditions': [{'family': f, 'variant_or_neutral': v} for f, v, _ in conditions],
        'source_template_path': os.path.relpath(SINGLE_TURN_TEMPLATE_PATH, REPO_ROOT),
        'source_template_sha256': sha256_of_file(SINGLE_TURN_TEMPLATE_PATH),
        'git_commit': git_commit_hash(),
        'per_model_summary': {
            alias: {'n_total_cases': m['n_total_cases'], 'n_exact_match': m['n_exact_match'],
                    'all_match': m['all_match'], 'n_mismatches': len(m['mismatches']),
                    'tokenizer_class': m['tokenizer_class'], 'padding_side': m['padding_side']}
            for alias, m in per_model.items()
        },
        'per_model_detail': per_model,
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
    print(f"\nWrote: {args.write_report}")
    print(f"result_status: {result_status}")
    for alias, m in per_model.items():
        print(f"  {alias}: {m['n_exact_match']}/{m['n_total_cases']} exact match")
        if m['mismatches']:
            first = m['mismatches'][0]
            print(f"    first mismatch example: family={first['family']} variant={first['variant_or_neutral']} "
                  f"first_divergence_index={first['first_divergence_index']} "
                  f"lengths=({first['single_turn_length']},{first['official_length']})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--write_report', type=str,
                         default=os.path.join(REPO_ROOT, 'output', 'audits', 'context',
                                               'multiturn_single_turn_serialization_equivalence.json'))
    args = parser.parse_args()
    main(args)
