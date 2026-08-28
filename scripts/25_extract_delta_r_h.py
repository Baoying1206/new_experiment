"""
Experiment 2 core data: extracts delta_R (refusal-axis projection) and
delta_H (harmfulness-axis projection) for each of the 6 mechanisms + placebo,
per instruction, per layer -- computed via projection of the template-induced
activation shift onto EACH axis AT ITS OWN position (t_post for
refusal_direction, t_inst for harmfulness_direction), NOT averaged/conflated
onto a single shared position. This is the literal Experiment 2 requirement.

  delta_R_i,mech = dot(act(mech, x_i, t_post) - act(plain, x_i, t_post_of_plain),
                        refusal_direction_hat)   [per layer]
  delta_H_i,mech = dot(act(mech, x_i, t_inst) - act(plain, x_i, t_inst_of_plain),
                        harmfulness_direction_hat)   [per layer]

refusal_direction is the v3 refused-vs-accepted behavioral direction
(scripts/26_rebuild_refusal_direction_behavioral.py, output_v3_behavioral_refusal/),
matching arXiv 2507.11878's actual contrast -- NOT the v2 harmful-vs-harmless
simplification (output_v2_dual_position/), superseded once v3 validated well
(large held-out Cohen's d) and showed lower collinearity with
harmfulness_direction than v2 did. harmfulness_direction is still v2
(harmful-vs-harmless at t_inst already matches the paper).

t_post is the templated/plain prompt's own last token (always well-defined,
same method as before). t_inst for a TEMPLATED condition is located
structurally (scripts/utils/token_positions.py's get_user_turn_end_position:
the last token before the user-turn's closing special token), NOT via
subsequence search against the raw instruction text -- the raw instruction
text is not recoverable as a literal substring for every mechanism
(encoding_obfuscation transforms it), so subsequence search (used for the
plain condition and for building the reference directions from raw
harmful/harmless instructions) is not usable here in general.

Also computes the placebo-calibrated versions (delta_R_PC = delta_R_mech -
delta_R_placebo, matched by instruction id) -- same logic as
19_taxonomy_robustness.py's placebo-calibrated re-test, extended to delta_R/delta_H.

Requires:
  - output_v2_dual_position/{model_alias}/refusal_dir_v2_{lang}.pt,
    harmfulness_dir_v2_{lang}.pt (scripts/23_extract_reference_directions.py --confirmed)
  - completions_{lang}{suffix}.json filtered to direction_ids (the same 300-id
    scope Experiment 1 used)

Usage:
  python scripts/25_extract_delta_r_h.py \
      --model_path /path/to/Qwen2.5-7B-Instruct --model_alias Qwen2.5-7B-Instruct \
      --output_dir output --lang en --suffix _full572 --ids_key direction_ids
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)
from utils.token_positions import get_post_instruction_position, get_user_turn_end_position
from pipeline.utils.hook_utils import add_hooks

SPLITS_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'splits.json')
REAL_MECHS = ['prefix_injection', 'refusal_suppression', 'instruction_hierarchy',
              'persona_roleplay', 'fictional_framing', 'encoding_obfuscation']
ALL_CONDS = REAL_MECHS + ['placebo']


def get_activations_pre_hook(layer, cache):
    def hook_fn(module, input):
        cache[layer] = input[0][0].detach().float().cpu()  # [seq_len, d_model]
    return hook_fn


def extract_sequence_and_positions(model_base, text, model_alias, n_layers):
    """batch_size=1 forward pass. Returns (t_inst_idx, t_post_idx, acts: dict[layer] -> tensor[seq_len, d_model])."""
    tokenized = model_base.tokenize_instructions_fn(instructions=[text])
    full_ids = tokenized.input_ids[0].tolist()
    cache = {}
    fwd_pre_hooks = [
        (model_base.model_block_modules[layer], get_activations_pre_hook(layer, cache))
        for layer in range(n_layers)
    ]
    with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
        with torch.no_grad():
            model_base.model(input_ids=tokenized.input_ids.to(model_base.model.device),
                              attention_mask=tokenized.attention_mask.to(model_base.model.device))
    t_inst = get_user_turn_end_position(model_base.tokenizer, full_ids, model_alias)
    t_post = get_post_instruction_position(model_base.tokenizer, text, model_alias, full_ids=full_ids)
    return t_inst.position_index, t_post.position_index, cache


def main(args):
    from pipeline.model_utils.model_factory import construct_model_base

    # refusal_direction: v3 (refused-vs-accepted, matching arXiv 2507.11878's actual
    # contrast, validated via held-out Cohen's d in
    # scripts/26_rebuild_refusal_direction_behavioral.py) -- NOT v2 (harmful-vs-harmless,
    # the earlier documented simplification, superseded once the behavioral version
    # was confirmed to validate well and to be less collinear with harmfulness_direction).
    v3_dir = os.path.join(args.output_dir, 'output_v3_behavioral_refusal', args.model_alias)
    v2_dir = os.path.join(args.output_dir, 'output_v2_dual_position', args.model_alias)
    refusal_dir = torch.load(os.path.join(v3_dir, f'refusal_dir_v3_{args.lang}.pt'), map_location='cpu').float()
    harmfulness_dir = torch.load(os.path.join(v2_dir, f'harmfulness_dir_v2_{args.lang}.pt'), map_location='cpu').float()
    refusal_hat = F.normalize(refusal_dir, dim=-1)        # [n_layers, d_model]
    harmfulness_hat = F.normalize(harmfulness_dir, dim=-1)  # [n_layers, d_model]

    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    keep_ids = set(splits[args.ids_key])
    print(f"Filtering to splits.json['{args.ids_key}']: {len(keep_ids)} ids\n")

    completions_path = os.path.join(args.output_dir, args.model_alias, f'completions_{args.lang}{args.suffix}.json')
    with open(completions_path, encoding='utf-8') as f:
        completions = json.load(f)['completions']
    by_id = {}
    for c in completions:
        if c['id'] not in keep_ids:
            continue
        by_id.setdefault(c['id'], {})[c['condition']] = c['instruction']
    ids = sorted(pid for pid in by_id if 'plain' in by_id[pid] and all(m in by_id[pid] for m in ALL_CONDS))
    print(f"{len(ids)} instructions with plain + all 7 conditions present\n")

    print("Loading model...")
    model_base = construct_model_base(args.model_path, lang=args.lang)
    n_layers = model_base.model.config.num_hidden_layers
    print(f"  Loaded: {args.model_alias}  n_layers={n_layers}\n")

    assert refusal_dir.shape[0] == n_layers, (
        f"refusal_direction has {refusal_dir.shape[0]} layers, model has {n_layers} -- "
        f"these v2 directions were not built for this model, refusing to proceed."
    )

    delta_R = {mech: torch.zeros(len(ids), n_layers) for mech in ALL_CONDS}
    delta_H = {mech: torch.zeros(len(ids), n_layers) for mech in ALL_CONDS}
    valid_mask = {mech: torch.zeros(len(ids), dtype=torch.bool) for mech in ALL_CONDS}
    failures = []

    for i, pid in enumerate(ids):
        try:
            plain_text = by_id[pid]['plain']
            t_inst_p, t_post_p, acts_p = extract_sequence_and_positions(
                model_base, plain_text, args.model_alias, n_layers)
            plain_at_inst = torch.stack([acts_p[l][t_inst_p] for l in range(n_layers)])  # [n_layers, d]
            plain_at_post = torch.stack([acts_p[l][t_post_p] for l in range(n_layers)])  # [n_layers, d]
        except ValueError as e:
            failures.append({'id': pid, 'condition': 'plain', 'error': str(e)})
            continue  # can't compute any mechanism's diff without plain -- skip this instruction entirely

        for mech in ALL_CONDS:
            try:
                mech_text = by_id[pid][mech]
                t_inst_m, t_post_m, acts_m = extract_sequence_and_positions(
                    model_base, mech_text, args.model_alias, n_layers)
                mech_at_inst = torch.stack([acts_m[l][t_inst_m] for l in range(n_layers)])
                mech_at_post = torch.stack([acts_m[l][t_post_m] for l in range(n_layers)])

                diff_at_inst = mech_at_inst - plain_at_inst  # [n_layers, d]
                diff_at_post = mech_at_post - plain_at_post  # [n_layers, d]

                delta_R[mech][i] = (diff_at_post * refusal_hat).sum(-1)
                delta_H[mech][i] = (diff_at_inst * harmfulness_hat).sum(-1)
                valid_mask[mech][i] = True
            except ValueError as e:
                failures.append({'id': pid, 'condition': mech, 'error': str(e)})

        if (i + 1) % 32 == 0 or i == len(ids) - 1:
            print(f"  {i + 1}/{len(ids)}  ({len(failures)} failures so far)")

    if failures:
        print(f"\n{len(failures)} position-finding failures out of {len(ids) * len(ALL_CONDS)} "
              f"attempts -- these instructions/conditions are excluded from delta_R/delta_H "
              f"(marked invalid in valid_mask, zeros in the tensor, NOT silently treated as 0-effect):")
        for f in failures[:20]:
            print(f"  [{f['id']}][{f['condition']}] {f['error'][:150]}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more (see saved .pt file's 'failures' field)")
        failure_rate = len(failures) / (len(ids) * len(ALL_CONDS))
        assert failure_rate < 0.05, (
            f"failure rate {failure_rate:.1%} is too high to proceed silently -- "
            f"investigate the position-finding errors above before trusting this data."
        )

    # placebo-calibrated versions
    delta_R_pc = {mech: delta_R[mech] - delta_R['placebo'] for mech in REAL_MECHS}
    delta_H_pc = {mech: delta_H[mech] - delta_H['placebo'] for mech in REAL_MECHS}

    out_dir = os.path.join(args.output_dir, args.model_alias)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'delta_r_h_{args.lang}{args.suffix}_{args.ids_key}.pt')
    torch.save({
        'instruction_ids': ids, 'n_layers': n_layers, 'ids_key': args.ids_key,
        'delta_R': delta_R, 'delta_H': delta_H,
        'delta_R_placebo_calibrated': delta_R_pc, 'delta_H_placebo_calibrated': delta_H_pc,
        'valid_mask': valid_mask, 'failures': failures,
    }, out_path)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',  type=str, required=True)
    parser.add_argument('--model_alias', type=str, required=True)
    parser.add_argument('--output_dir',  type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    parser.add_argument('--lang',        type=str, default='en')
    parser.add_argument('--suffix',      type=str, default='_full572')
    parser.add_argument('--ids_key',     type=str, default='direction_ids')
    args = parser.parse_args()
    main(args)
