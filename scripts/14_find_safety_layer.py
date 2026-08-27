"""
Identifies the "most safety-relevant" layer per language/model, as a
pre-registered (not cherry-picked after seeing template_direction results)
criterion for a follow-up, layer-specific version of the frac_along_refusal
analysis in 09_refusal_geometry.py.

Criterion: the layer where refusal_direction has the largest L2 norm --
larger DiM magnitude indicates cleaner separability between harmful and
harmless activations at that layer, i.e. where the refusal-relevant signal
is most concentrated. This mirrors the 'best_detect_layer' logic already
used for jailbreak_vector in experiment_thesis/scripts/extract_jailbreak_vectors.py,
applied here to refusal_direction instead.

CPU-only -- reads the already-saved refusal_dir_{lang}.pt files directly, no
model loading or GPU required. Run this BEFORE looking at any
frac_along_refusal numbers, so the layer choice can't be influenced by
wanting a particular result.

Usage:
  python scripts/14_find_safety_layer.py --refusal_dir_root /path/to/experiment_thesis/output/jailbreak_analysis
"""
import argparse
import json
import os

import torch

SCRIPT_DIR = os.path.dirname(__file__)
PILOT_LANGS = ['en', 'zh', 'de', 'ko', 'ar', 'th', 'yo', 'sw', 'am']
MODEL_ALIASES = ['Qwen2.5-7B-Instruct', 'Meta-Llama-3.1-8B-Instruct', 'gemma-2-9b-it']


def main(args):
    results = {}
    for model_alias in MODEL_ALIASES:
        model_dir = os.path.join(args.refusal_dir_root, model_alias)
        if not os.path.isdir(model_dir):
            print(f"[{model_alias}] directory not found, skipping.")
            continue

        print(f"\n=== {model_alias} ===")
        per_lang_argmax = {}
        per_lang_norms = {}
        for lang in PILOT_LANGS:
            path = os.path.join(model_dir, f'refusal_dir_{lang}.pt')
            if not os.path.exists(path):
                print(f"  [{lang}] refusal_dir missing, skipping.")
                continue
            refusal_dir = torch.load(path, map_location='cpu').float()  # [n_layers, d_model]
            norms = refusal_dir.norm(dim=-1)  # [n_layers]
            n_layers_total = len(norms)
            # Exclude the final ~20% of layers, matching Arditi et al.'s l < 0.8L
            # criterion -- residual stream norms grow systematically toward the
            # unembedding layer regardless of refusal-relevance, so an unguarded
            # argmax over all layers picks the last layer almost by construction
            # (confirmed empirically: all 3 models, all 9 languages picked the
            # literal final layer before this cutoff was added).
            cutoff = int(0.8 * n_layers_total)
            usable_norms = norms[:cutoff]
            argmax_layer = int(usable_norms.argmax().item())
            per_lang_argmax[lang] = argmax_layer
            per_lang_norms[lang] = norms.tolist()
            print(f"  [{lang}] peak layer = {argmax_layer}  (norm={norms[argmax_layer]:.2f}, "
                  f"n_layers={n_layers_total}, searched [0,{cutoff}))")

        if per_lang_argmax:
            layers = list(per_lang_argmax.values())
            n_layers = len(next(iter(per_lang_norms.values())))
            mode_layer = max(set(layers), key=layers.count)
            mean_layer = sum(layers) / len(layers)
            print(f"  --> across {len(layers)} languages: mode={mode_layer}  "
                  f"mean={mean_layer:.1f}  n_layers={n_layers}  "
                  f"range=[{min(layers)},{max(layers)}]")
            results[model_alias] = {
                'per_lang_peak_layer': per_lang_argmax,
                'per_lang_norms': per_lang_norms,
                'mode_layer': mode_layer,
                'mean_layer': mean_layer,
                'n_layers': n_layers,
            }

    out_path = os.path.join(SCRIPT_DIR, '..', 'output', 'safety_layer_identification.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")
    print("\nNext step: rerun the geometry comparison restricted to each model's identified layer "
          "(mode_layer, or per-language peak_layer) instead of averaging across all layers, and "
          "compare frac_along_refusal at that layer against the all-layer average already computed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--refusal_dir_root', type=str, required=True,
                         help='experiment_thesis output dir containing {model_alias}/refusal_dir_{lang}.pt')
    args = parser.parse_args()
    main(args)
