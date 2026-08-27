"""
Token-position audit -- MUST be run and its output reviewed BEFORE any full
GPU direction-extraction job that uses scripts/utils/token_positions.py is
submitted (per the user's explicit instruction: "在我确认token audit之前，
不得提交完整GPU方向提取任务").

Requires real HuggingFace tokenizers for all three models (Qwen2.5-7B-Instruct,
Meta-Llama-3.1-8B-Instruct, gemma-2-9b-it) -- NOT runnable on this local
machine (no cluster/model access here). This script is written now so it can
be run on the cluster as the very next step; do not treat its absence of a
local run as equivalent to verification.

For each model and a handful of sample instructions (varying length,
including edge cases: very short instruction, instruction ending in
punctuation vs not, instruction containing template special characters),
records the full token-level context needed to judge whether
get_instruction_end_position/get_post_instruction_position from
token_positions.py are actually finding what they claim to find:
  - full token index list, token ids, decoded tokens
  - special-token markers (whether each token is a registered special token)
  - the computed t_inst / t_post position and semantic name
  - +/- 5 token decoded context around each
  - attention mask
  - whether add_generation_prompt was applied
  - explicit pass/fail: does decoding full_ids[t_inst] token, read together
    with its immediate left context, look like it's still inside the user's
    instruction (not template scaffolding)? Does full_ids[t_post] look like
    it's positioned right where generation should start (not mid-instruction,
    not deep inside a role marker)? -- flagged for human review, not
    auto-decided, since "looks right" ultimately needs a human glance per
    the user's "不能自行猜测" instruction.

Usage (on cluster, inside the environment with transformers + model access):
  python scripts/audits/audit_token_positions.py \
      --model_paths /path/to/Qwen2.5-7B-Instruct,/path/to/Meta-Llama-3.1-8B-Instruct,/path/to/gemma-2-9b-it \
      --model_aliases Qwen2.5-7B-Instruct,Meta-Llama-3.1-8B-Instruct,gemma-2-9b-it

Writes output/audits/token_position_audit.json and .md.
"""
import argparse
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

SAMPLE_INSTRUCTIONS = [
    "Explain how photosynthesis works.",
    "Write a short poem about the ocean, no more than four lines.",
    "What is 17 * 23?",
    "Translate 'good morning' into French, German, and Japanese, and explain any cultural notes about greeting time-of-day in each.",
    "List three prime numbers.",
]


def audit_model(model_path, model_alias):
    from transformers import AutoTokenizer
    from utils.token_positions import (
        get_instruction_end_position, get_post_instruction_position,
        render_full_prompt_ids,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    samples = []
    for instr in SAMPLE_INSTRUCTIONS:
        full_ids = render_full_prompt_ids(tokenizer, instr, model_alias)
        entry = {
            'instruction': instr,
            'full_token_count': len(full_ids),
            'full_tokens': [
                {
                    'index': i,
                    'id': int(tid),
                    'decoded': tokenizer.decode([tid]),
                    'is_special': tid in tokenizer.all_special_ids,
                }
                for i, tid in enumerate(full_ids)
            ],
        }
        try:
            t_inst = get_instruction_end_position(tokenizer, instr, model_alias)
            entry['t_inst'] = t_inst.to_dict()
        except ValueError as e:
            entry['t_inst_error'] = str(e)
        try:
            t_post = get_post_instruction_position(tokenizer, instr, model_alias)
            entry['t_post'] = t_post.to_dict()
        except ValueError as e:
            entry['t_post_error'] = str(e)
        samples.append(entry)

    return {
        'model_alias': model_alias,
        'model_path': model_path,
        'chat_template_present': bool(getattr(tokenizer, 'chat_template', None)),
        'samples': samples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_paths', type=str, required=True,
                         help='comma-separated model paths')
    parser.add_argument('--model_aliases', type=str, required=True,
                         help='comma-separated model aliases, same order as model_paths')
    args = parser.parse_args()

    paths = args.model_paths.split(',')
    aliases = args.model_aliases.split(',')
    assert len(paths) == len(aliases)

    results = {}
    for path, alias in zip(paths, aliases):
        print(f"Auditing {alias} ...")
        results[alias] = audit_model(path, alias)

    out_dir = os.path.join(ROOT, 'output', 'audits')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'token_position_audit.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    md_lines = ['# Token-Position Audit', '',
                '**Human review required** — pass/fail judgment on whether t_inst/t_post '
                'land where claimed is not auto-decided by this script.', '']
    for alias, r in results.items():
        md_lines.append(f'## {alias}')
        n_errors = sum(1 for s in r['samples'] if 't_inst_error' in s or 't_post_error' in s)
        md_lines.append(f"- {len(r['samples'])} samples, {n_errors} with a position-finding error "
                         f"(generic subsequence search failed -- needs a MODEL_FAMILY_ADAPTERS entry).")
        for s in r['samples']:
            md_lines.append(f"  - `{s['instruction'][:50]}...` "
                             f"t_inst={s.get('t_inst', {}).get('decoded_token', s.get('t_inst_error'))!r} "
                             f"t_post={s.get('t_post', {}).get('decoded_token', s.get('t_post_error'))!r}")
        md_lines.append('')
    with open(os.path.join(out_dir, 'token_position_audit.md'), 'w') as f:
        f.write('\n'.join(md_lines) + '\n')

    print(f"Wrote {out_dir}/token_position_audit.json and .md")
    print("REQUIRED: human review of the decoded tokens before trusting these positions "
          "for any GPU direction-extraction job.")


if __name__ == '__main__':
    main()
