"""
Re-checks 09_refusal_geometry.py's cos(refusal)/cos(harmfulness)/frac_along_refusal
at a SINGLE, data-driven layer (from 14_find_safety_layer.py's peak-norm
criterion) instead of averaging across all layers -- tests whether the
09-section findings (low frac_along_refusal, Wei et al. taxonomy not
separating geometrically) were diluted by averaging in layers where the
refusal-relevant signal is weak or absent.

The layer is passed in explicitly (from 14's output), not re-derived here,
so this script can't silently cherry-pick a layer after seeing results.

Usage:
  python scripts/16_single_layer_geometry.py \
      --model_path /path/to/Qwen2.5-7B-Instruct \
      --model_alias Qwen2.5-7B-Instruct \
      --output_dir output \
      --refusal_dir_root /path/to/experiment_thesis/output/jailbreak_analysis \
      --layer 18 \
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
PILOT_LANGS = ['en', 'zh', 'de', 'ko', 'ar', 'th', 'yo', 'sw', 'am']
REAL_MECHS = ['prefix_injection', 'refusal_suppression', 'instruction_hierarchy',
              'persona_roleplay', 'fictional_framing', 'encoding_obfuscation']
TIERS = {'en': 'H', 'zh': 'H', 'de': 'H', 'ko': 'M', 'ar': 'M', 'th': 'M',
         'yo': 'L', 'sw': 'L', 'am': 'L'}


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


def template_direction(acts_by_id_cond, mechanism):
    diffs = []
    for pid, by_cond in acts_by_id_cond.items():
        if mechanism in by_cond and 'plain' in by_cond:
            diffs.append(by_cond[mechanism] - by_cond['plain'])
    if not diffs:
        return None
    return torch.stack(diffs, dim=0).mean(0)  # [n_layers, d_model]


def geometry_at_layer(direction_full, reference_full, layer):
    """direction_full, reference_full: [n_layers, d_model]. Single-layer version
    of 09's geometry_vs_reference -- no averaging across layers."""
    direction = direction_full[layer].float()
    reference = reference_full[layer].float()
    cos = F.cosine_similarity(direction.unsqueeze(0), reference.unsqueeze(0), dim=-1).item()
    ref_unit = F.normalize(reference, dim=-1)
    parallel_mag = abs((direction * ref_unit).sum().item())
    total_mag = direction.norm().item()
    frac_parallel = parallel_mag / max(total_mag, 1e-8)
    return {'cos': cos, 'frac_parallel': frac_parallel}


def main(args):
    model_alias = args.model_alias or os.path.basename(args.model_path)
    out_dir = os.path.join(args.output_dir, model_alias)

    mech_cat_path = os.path.join(SCRIPT_DIR, '..', 'templates', 'templates_en.json')
    with open(mech_cat_path, encoding='utf-8') as f:
        mech_categories = json.load(f)['mechanism_categories']
    mech_to_cat = {m: cat for cat, mechs in mech_categories.items() for m in mechs}

    print("Loading model...")
    model_base = construct_model_base(args.model_path, lang='en')
    print(f"  Loaded: {model_alias}  fixed layer={args.layer} (pre-selected by 14_find_safety_layer.py, "
          f"not chosen after seeing this script's results)\n")

    records = []
    skipped_langs = []

    for lang in PILOT_LANGS:
        completions_path = os.path.join(out_dir, f'completions_{lang}.json')
        refusal_path = os.path.join(args.refusal_dir_root, model_alias, f'refusal_dir_{lang}.pt')
        harm_path = os.path.join(args.refusal_dir_root, model_alias, f'harmfulness_dir_{lang}.pt')

        if not os.path.exists(completions_path):
            skipped_langs.append((lang, 'no completions'))
            continue
        has_refusal = os.path.exists(refusal_path)
        has_harm = os.path.exists(harm_path)
        if not has_refusal and not has_harm:
            skipped_langs.append((lang, 'no refusal/harmfulness direction'))
            continue

        print(f"[{lang}] Extracting activations...")
        acts = extract_lang_data(model_base, completions_path, args.batch_size)
        refusal_dir = torch.load(refusal_path, map_location='cpu') if has_refusal else None
        harm_dir = torch.load(harm_path, map_location='cpu') if has_harm else None

        for mech in REAL_MECHS:
            d = template_direction(acts, mech)
            if d is None:
                continue
            g_ref = geometry_at_layer(d, refusal_dir, args.layer) if refusal_dir is not None else None
            g_harm = geometry_at_layer(d, harm_dir, args.layer) if harm_dir is not None else None
            records.append({
                'lang': lang, 'mechanism': mech,
                'category': mech_to_cat.get(mech, 'unknown'), 'tier': TIERS[lang],
                'cos_refusal': g_ref['cos'] if g_ref else None,
                'frac_refusal': g_ref['frac_parallel'] if g_ref else None,
                'cos_harmfulness': g_harm['cos'] if g_harm else None,
                'frac_harmfulness': g_harm['frac_parallel'] if g_harm else None,
            })
            ref_str = f"{g_ref['cos']:+.3f} (frac={g_ref['frac_parallel']:.3f})" if g_ref else "n/a"
            print(f"  [{lang}][{mech}] cos(refusal)={ref_str}")
        torch.cuda.empty_cache()

    def group_mean(key, filter_fn):
        vals = [r[key] for r in records if filter_fn(r) and r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    by_category = {}
    for cat in ['competing_objectives', 'mismatched_generalization']:
        by_category[cat] = {
            'cos_refusal_mean': group_mean('cos_refusal', lambda r, c=cat: r['category'] == c),
            'cos_harmfulness_mean': group_mean('cos_harmfulness', lambda r, c=cat: r['category'] == c),
            'frac_refusal_mean': group_mean('frac_refusal', lambda r, c=cat: r['category'] == c),
        }

    def fmt(v):
        return f"{v:.3f}" if v is not None else "n/a"

    print(f"\n=== By mechanism category (single layer {args.layer}) ===")
    for cat, stats in by_category.items():
        print(f"  {cat}: cos(refusal)={fmt(stats['cos_refusal_mean'])}  "
              f"cos(harmfulness)={fmt(stats['cos_harmfulness_mean'])}  "
              f"frac_along_refusal={fmt(stats['frac_refusal_mean'])}")

    out_path = os.path.join(out_dir, f'single_layer_geometry_L{args.layer}.json')
    with open(out_path, 'w') as f:
        json.dump({
            'model': model_alias, 'layer': args.layer,
            'records': records, 'by_category': by_category,
            'skipped_langs': skipped_langs,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")
    print("\nCompare 'frac_along_refusal' and by-category cos values above against the "
          "all-layer-averaged numbers already in refusal_geometry.json (Section 09) -- "
          "if these are substantially higher, the all-layer average was diluting a real, "
          "layer-concentrated signal.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',       type=str, required=True)
    parser.add_argument('--model_alias',      type=str, default=None)
    parser.add_argument('--output_dir',       type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    parser.add_argument('--refusal_dir_root', type=str, required=True)
    parser.add_argument('--layer',            type=int, required=True,
                         help='Fixed layer to use, e.g. from 14_find_safety_layer.py output '
                              '(mode_layer or a specific per-language peak_layer). Must be '
                              'chosen BEFORE looking at this script\'s results.')
    parser.add_argument('--batch_size',       type=int, default=8)
    args = parser.parse_args()
    if args.model_alias is None:
        args.model_alias = os.path.basename(args.model_path)
    main(args)
