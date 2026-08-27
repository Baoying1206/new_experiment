"""
Stage-1 prerequisite for the taxonomy-robustness diagnostics: persists raw,
per-instruction, all-layer paired activation diffs

    delta_h[i, t] = act(t(x_i)) - act(x_i)     for t in {6 real mechanisms, placebo}

to disk, instead of only the aggregated mean directions/cosine stats that
04/09/12/13 already saved. Every downstream Stage-1 analysis (exact
permutation test on the 3+3 category split, bootstrap-by-instruction,
split-half, leave-one-template-out, placebo calibration, surface-vs-activation
distance comparison) needs these raw per-instruction vectors, not just the
mean template_direction -- none of the existing scripts kept them.

No generation, no WildGuard -- just a forward pass to extract activations,
so this is much cheaper than Phase 1 (10b) or even the earlier extraction
scripts that also handled generation-conditioned completions.

Output: {output_dir}/{model_alias}/paired_diffs_{lang}.pt, a dict:
  {
    'instruction_ids': {mechanism: [ids in tensor order]},
    'diffs': {mechanism: tensor [n_matched, n_layers, d_model]},
  }
for mechanism in REAL_MECHS + ['placebo'].

Usage:
  python scripts/18_extract_paired_diffs.py \
      --model_path /path/to/Qwen2.5-7B-Instruct \
      --model_alias Qwen2.5-7B-Instruct \
      --output_dir output \
      --langs en \
      --batch_size 8
"""
import argparse
import json
import os

import torch
from tqdm import tqdm

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import add_hooks

SCRIPT_DIR = os.path.dirname(__file__)
REAL_MECHS = ['prefix_injection', 'refusal_suppression', 'instruction_hierarchy',
              'persona_roleplay', 'fictional_framing', 'encoding_obfuscation']
ALL_CONDS = REAL_MECHS + ['placebo']


def get_activations_pre_hook(layer, cache, position, offset):
    def hook_fn(module, input):
        activation = input[0]
        cache[offset:offset + activation.shape[0], layer, :] = (
            activation[:, position, :].detach().to(cache.dtype).cpu()
        )
    return hook_fn


def get_all_activations(model_base, instructions, batch_size, position=-1):
    model = model_base.model
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    n_samples = len(instructions)
    all_acts = torch.zeros((n_samples, n_layers, d_model), dtype=torch.float32)
    for i in tqdm(range(0, n_samples, batch_size), desc="extracting activations"):
        batch = instructions[i:i + batch_size]
        fwd_pre_hooks = [
            (model_base.model_block_modules[layer],
             get_activations_pre_hook(layer=layer, cache=all_acts, position=position, offset=i))
            for layer in range(n_layers)
        ]
        inputs = model_base.tokenize_instructions_fn(instructions=batch)
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            with torch.no_grad():
                model(input_ids=inputs.input_ids.to(model.device),
                      attention_mask=inputs.attention_mask.to(model.device))
    return all_acts


def main(args):
    model_alias = args.model_alias or os.path.basename(args.model_path)
    out_dir = os.path.join(args.output_dir, model_alias)
    langs = args.langs.split(',')

    print("Loading model...")
    model_base = construct_model_base(args.model_path, lang='en')
    n_layers = model_base.model.config.num_hidden_layers
    d_model = model_base.model.config.hidden_size
    print(f"  Loaded: {model_alias}  n_layers={n_layers}  d_model={d_model}\n")

    for lang in langs:
        completions_path = os.path.join(out_dir, f'completions_{lang}.json')
        if not os.path.exists(completions_path):
            print(f"[{lang}] Missing completions, skipping.")
            continue
        with open(completions_path, encoding='utf-8') as f:
            completions = json.load(f)['completions']

        by_id = {}
        for c in completions:
            by_id.setdefault(c['id'], {})[c['condition']] = c['instruction']

        print(f"\n[{lang}] Extracting paired diffs for {ALL_CONDS}...")
        diffs_by_mech, ids_by_mech = {}, {}
        for mech in ALL_CONDS:
            pids = [pid for pid in by_id if 'plain' in by_id[pid] and mech in by_id[pid]]
            plain_instrs = [by_id[pid]['plain'] for pid in pids]
            mech_instrs = [by_id[pid][mech] for pid in pids]

            all_instrs = plain_instrs + mech_instrs
            acts = get_all_activations(model_base, all_instrs, args.batch_size)  # [2n, n_layers, d_model]
            n = len(pids)
            plain_acts = acts[:n]
            mech_acts = acts[n:]
            diffs = mech_acts - plain_acts  # [n, n_layers, d_model]

            diffs_by_mech[mech] = diffs
            ids_by_mech[mech] = pids
            print(f"  [{mech}] n={n}  diffs.shape={tuple(diffs.shape)}")
            torch.cuda.empty_cache()

        out_path = os.path.join(out_dir, f'paired_diffs_{lang}.pt')
        torch.save({'instruction_ids': ids_by_mech, 'diffs': diffs_by_mech,
                    'n_layers': n_layers, 'd_model': d_model}, out_path)
        print(f"  Saved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',  type=str, required=True)
    parser.add_argument('--model_alias', type=str, default=None)
    parser.add_argument('--output_dir',  type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    parser.add_argument('--langs',       type=str, default='en')
    parser.add_argument('--batch_size',  type=int, default=8)
    args = parser.parse_args()
    if args.model_alias is None:
        args.model_alias = os.path.basename(args.model_path)
    main(args)
