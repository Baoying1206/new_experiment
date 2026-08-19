"""
Split-half reliability: for each (language, mechanism), randomly split the 75
paired prompts into two halves, compute template_direction from each half
independently, and take their cosine similarity. Repeat with many random
splits to get a stable estimate.

This answers a calibration question the raw cross-language/cross-mechanism
numbers can't answer on their own: given ~75-prompt sampling noise, how high
would cosine similarity be even for "the same underlying direction measured
twice"? If a model's split-half ceiling is itself high (e.g. Gemma), its
raw cross-language similarities need to be read relative to that ceiling,
not at face value -- a model with a more anisotropic activation space will
show inflated similarity between literally anything, including two random
halves of the identical condition.

Requires the same pipeline/ + GPU environment as 03/04. Re-extracts
activations (the per-prompt paired diffs aren't persisted by 04, only the
final aggregated stats are), so this is a second GPU pass over the same data.

Usage:
  python scripts/06_split_half_reliability.py \
      --model_path /path/to/Qwen2.5-7B-Instruct \
      --model_alias Qwen2.5-7B-Instruct \
      --output_dir output \
      --batch_size 8 \
      --n_splits 200
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import add_hooks  # verify this import path on the cluster

SCRIPT_DIR = os.path.dirname(__file__)
PILOT_LANGS = ['en', 'zh', 'de', 'ko', 'ar', 'th', 'yo', 'sw', 'am']
MECHANISMS = ['prefix_injection', 'refusal_suppression', 'instruction_hierarchy',
              'persona_roleplay', 'fictional_framing', 'encoding_obfuscation', 'placebo']


# ── Activation extraction (identical pattern to 04_extract_directions_and_analyze.py) ──

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


def extract_lang_data(model_base, completions_path, batch_size):
    with open(completions_path, encoding='utf-8') as f:
        completions = json.load(f)['completions']
    instructions = [c['instruction'] for c in completions]
    all_acts = get_all_activations(model_base, instructions, batch_size)
    acts_by_id_cond = {}
    for i, c in enumerate(completions):
        acts_by_id_cond.setdefault(c['id'], {})[c['condition']] = all_acts[i]
    return acts_by_id_cond


def paired_diffs(acts_by_id_cond, mechanism):
    diffs = []
    for pid, by_cond in acts_by_id_cond.items():
        if mechanism in by_cond and 'plain' in by_cond:
            diffs.append(by_cond[mechanism] - by_cond['plain'])
    if not diffs:
        return None
    return torch.stack(diffs, dim=0)  # [n_prompts, n_layers, d_model]


# ── Split-half reliability ──────────────────────────────────────────────────────

def split_half_once(diffs, seed):
    n = diffs.shape[0]
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    half = n // 2
    idx1, idx2 = idx[:half], idx[half:]
    mean1 = diffs[idx1].mean(0).float()
    mean2 = diffs[idx2].mean(0).float()
    return F.cosine_similarity(mean1, mean2, dim=-1).mean().item()  # avg over layers


def split_half_reliability(diffs, n_splits=200, base_seed=0):
    vals = [split_half_once(diffs, base_seed + s) for s in range(n_splits)]
    return {
        'mean': float(np.mean(vals)),
        'ci_low': float(np.percentile(vals, 2.5)),
        'ci_high': float(np.percentile(vals, 97.5)),
        'n_splits': n_splits,
        'n_prompts': int(diffs.shape[0]),
    }


# ── Main ────────────────────────────────────────────────────────────────────────

def main(args):
    model_alias = args.model_alias or os.path.basename(args.model_path)
    out_dir = os.path.join(args.output_dir, model_alias)

    print("Loading model...")
    model_base = construct_model_base(args.model_path, lang='en')
    print(f"  Loaded: {model_alias}\n")

    results = {'model': model_alias, 'n_splits': args.n_splits, 'per_lang_mechanism': {}}

    for lang in PILOT_LANGS:
        completions_path = os.path.join(out_dir, f'completions_{lang}.json')
        if not os.path.exists(completions_path):
            print(f"[{lang}] Missing {completions_path}, skipping.")
            continue
        print(f"[{lang}] Extracting activations...")
        acts_by_id_cond = extract_lang_data(model_base, completions_path, args.batch_size)

        for mech in MECHANISMS:
            diffs = paired_diffs(acts_by_id_cond, mech)
            if diffs is None:
                continue
            rel = split_half_reliability(diffs, args.n_splits)
            results['per_lang_mechanism'][f'{lang}__{mech}'] = rel
            print(f"  [{lang}][{mech}] split-half reliability = {rel['mean']:.3f} "
                  f"[{rel['ci_low']:.3f}, {rel['ci_high']:.3f}] (n={rel['n_prompts']})")
        torch.cuda.empty_cache()

    # Summary: overall mean reliability per language and per mechanism
    by_lang, by_mech = {}, {}
    for key, rel in results['per_lang_mechanism'].items():
        lang, mech = key.split('__')
        by_lang.setdefault(lang, []).append(rel['mean'])
        by_mech.setdefault(mech, []).append(rel['mean'])
    results['summary_by_lang'] = {l: float(np.mean(v)) for l, v in by_lang.items()}
    results['summary_by_mechanism'] = {m: float(np.mean(v)) for m, v in by_mech.items()}
    results['overall_mean'] = float(np.mean([r['mean'] for r in results['per_lang_mechanism'].values()]))

    print(f"\nOverall split-half reliability ceiling: {results['overall_mean']:.3f}")

    out_path = os.path.join(out_dir, 'split_half_reliability.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',  type=str, required=True)
    parser.add_argument('--model_alias', type=str, default=None)
    parser.add_argument('--output_dir',  type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    parser.add_argument('--batch_size',  type=int, default=8)
    parser.add_argument('--n_splits',    type=int, default=200)
    args = parser.parse_args()
    if args.model_alias is None:
        args.model_alias = os.path.basename(args.model_path)
    main(args)
