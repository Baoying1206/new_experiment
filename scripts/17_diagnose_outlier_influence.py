"""
Diagnoses whether template_direction's raw (unweighted) mean-of-differences
construction is dominated by a small number of high-magnitude per-instruction
diff vectors, as opposed to reflecting a broad consensus across all sampled
instructions.

For each mechanism, computes the per-instruction diff vector
  d_i = act(templated)_i - act(plain)_i
at the layer used for injection/geometry analysis (default: n_layers // 2,
override with --layer to match whatever layer 14/16 identified), and reports
the distribution of ||d_i|| across instructions -- if a handful of samples
have norms much larger than the rest, the unweighted mean direction is likely
dominated by them rather than reflecting the full sample.

Also reports what the direction would look like under unit-mean construction
(normalize each d_i to unit length before averaging) versus the current raw
mean, and their cosine similarity -- a low cosine between the two indicates
the raw mean is meaningfully different from an outlier-robust estimate.

Scoped to English only by default (--langs en) as a first, cheap check
before deciding whether to run this across all 9 languages.

Usage:
  python scripts/17_diagnose_outlier_influence.py \
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
import torch.nn.functional as F
from tqdm import tqdm

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import add_hooks

SCRIPT_DIR = os.path.dirname(__file__)
REAL_MECHS = ['prefix_injection', 'refusal_suppression', 'instruction_hierarchy',
              'persona_roleplay', 'fictional_framing', 'encoding_obfuscation']


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


def per_instruction_diffs(model_base, completions, mechanism, batch_size, layer):
    by_id = {}
    for c in completions:
        by_id.setdefault(c['id'], {})[c['condition']] = c['instruction']

    plain_instrs, mech_instrs, ids = [], [], []
    for pid, by_cond in by_id.items():
        if 'plain' in by_cond and mechanism in by_cond:
            plain_instrs.append(by_cond['plain'])
            mech_instrs.append(by_cond[mechanism])
            ids.append(pid)

    all_instrs = plain_instrs + mech_instrs
    acts = get_all_activations(model_base, all_instrs, batch_size)  # [2n, n_layers, d_model]
    n = len(plain_instrs)
    plain_acts = acts[:n, layer, :].float()
    mech_acts = acts[n:, layer, :].float()
    diffs = mech_acts - plain_acts  # [n, d_model]
    return diffs, ids


def main(args):
    model_alias = args.model_alias or os.path.basename(args.model_path)
    out_dir = os.path.join(args.output_dir, model_alias)
    langs = args.langs.split(',')

    print("Loading model...")
    model_base = construct_model_base(args.model_path, lang='en')
    n_layers = model_base.model.config.num_hidden_layers
    layer = args.layer if args.layer is not None else n_layers // 2
    print(f"  Loaded: {model_alias}  n_layers={n_layers}  checking layer={layer}\n")

    results = {}
    for lang in langs:
        completions_path = os.path.join(out_dir, f'completions_{lang}.json')
        with open(completions_path, encoding='utf-8') as f:
            completions = json.load(f)['completions']

        results[lang] = {}
        print(f"\n=== {lang} ===")
        for mech in REAL_MECHS:
            diffs, ids = per_instruction_diffs(model_base, completions, mech, args.batch_size, layer)
            norms = diffs.norm(dim=-1)  # [n]

            raw_mean = diffs.mean(0)  # unweighted mean, what template_direction currently uses
            unit_diffs = F.normalize(diffs, dim=-1)
            unit_mean = unit_diffs.mean(0)  # each sample weighted equally regardless of magnitude
            agreement = F.cosine_similarity(raw_mean.unsqueeze(0), unit_mean.unsqueeze(0), dim=-1).item()

            sorted_norms = norms.sort(descending=True).values
            top1_share = (sorted_norms[0] / sorted_norms.sum()).item()
            top5_share = (sorted_norms[:5].sum() / sorted_norms.sum()).item()

            stats = {
                'n': len(norms), 'norm_mean': norms.mean().item(), 'norm_std': norms.std().item(),
                'norm_min': norms.min().item(), 'norm_max': norms.max().item(),
                'norm_max_over_mean': (norms.max() / norms.mean()).item(),
                'top1_share_of_total_norm': top1_share, 'top5_share_of_total_norm': top5_share,
                'raw_vs_unit_mean_cosine': agreement,
            }
            results[lang][mech] = stats
            print(f"  [{mech}] n={stats['n']}  norm mean={stats['norm_mean']:.2f} std={stats['norm_std']:.2f}  "
                  f"max/mean={stats['norm_max_over_mean']:.2f}  top1_share={top1_share:.3f}  "
                  f"top5_share={top5_share:.3f}  raw_vs_unit_cos={agreement:.3f}")
        torch.cuda.empty_cache()

    out_path = os.path.join(out_dir, f'outlier_diagnosis_layer{layer}.json')
    with open(out_path, 'w') as f:
        json.dump({'model': model_alias, 'layer': layer, 'results': results}, f, indent=2)
    print(f"\nSaved: {out_path}")
    print("\nInterpretation:")
    print("  - norm_max_over_mean >> 1 (e.g. >5) and top1_share >> 1/n suggests one sample dominates.")
    print("  - raw_vs_unit_cos close to 1.0 means outliers aren't meaningfully distorting the direction")
    print("    (raw mean and outlier-robust unit-mean point almost the same way) -- no fix needed.")
    print("  - raw_vs_unit_cos noticeably below 1.0 means the current raw-mean template_direction")
    print("    differs from an outlier-robust estimate -- worth switching to unit-mean construction.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',      type=str, required=True)
    parser.add_argument('--model_alias',     type=str, default=None)
    parser.add_argument('--output_dir',      type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    parser.add_argument('--langs',           type=str, default='en')
    parser.add_argument('--layer',           type=int, default=None,
                         help='Layer to check. Default: n_layers // 2. Pass the same layer '
                              'used elsewhere (e.g. from 14_find_safety_layer.py) for consistency.')
    parser.add_argument('--batch_size',      type=int, default=8)
    args = parser.parse_args()
    if args.model_alias is None:
        args.model_alias = os.path.basename(args.model_path)
    main(args)
