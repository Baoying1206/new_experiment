"""
Per-layer profile of same_mechanism_cross_language cosine similarity (the
counterpart to 12_layerwise_profile.py's same_language_cross_mechanism view):
at which layer does cross-lingual sharing of a jailbreak template's induced
shift start to break down, especially for the HL (high-vs-low resource) pair
that anchors the "resource-tier continuum" finding in section 01?

Not a language-identity probe -- template_direction is already a difference
(templated minus plain), so most of "which language is this" has been
subtracted out. This measures whether the MARGINAL EFFECT of a given
template looks the same across two languages, layer by layer.

Reads the 'point' field already saved in pilot_results.json -- no model, no
GPU, just re-reading existing results with a per-layer lens.

Usage:
  python scripts/13_layerwise_cross_language.py
"""
import json
import os

SCRIPT_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'output')

MODELS = ['Qwen2.5-7B-Instruct', 'Meta-Llama-3.1-8B-Instruct', 'gemma-2-9b-it']
TIERS = {'en': 'H', 'zh': 'H', 'de': 'H', 'ko': 'M', 'ar': 'M', 'th': 'M',
         'yo': 'L', 'sw': 'L', 'am': 'L'}
REAL_MECHS = ['prefix_injection', 'refusal_suppression', 'instruction_hierarchy',
              'persona_roleplay', 'fictional_framing', 'encoding_obfuscation']
TIER_PAIRS = ['HH', 'MM', 'LL', 'HM', 'HL', 'LM']


def tier_pair_of(l1, l2):
    return ''.join(sorted([TIERS[l1], TIERS[l2]]))


def main():
    results = {}
    for model in MODELS:
        path = os.path.join(OUTPUT_DIR, model, 'pilot_results.json')
        if not os.path.exists(path):
            print(f"Missing {path}, skipping {model}")
            continue
        with open(path) as f:
            d = json.load(f)
        smcl = d['same_mechanism_cross_language']

        n_layers = None
        tp_curves_accum = {tp: [] for tp in TIER_PAIRS}
        for key, val in smcl.items():
            mech, langs = key.split('__')
            if mech not in REAL_MECHS:
                continue
            l1, l2 = langs.split('_vs_')
            tp = tier_pair_of(l1, l2)
            tp_curves_accum[tp].append(val['point'])
            if n_layers is None:
                n_layers = len(val['point'])

        tp_curves = {}
        for tp in TIER_PAIRS:
            curves = tp_curves_accum[tp]
            tp_curves[tp] = [sum(c[i] for c in curves) / len(curves) for i in range(n_layers)]

        results[model] = {'n_layers': n_layers, 'tier_pair_curves': tp_curves}

        print(f"\n=== {model} (n_layers={n_layers}) ===")
        print("layer  " + "  ".join(f"{tp:>6}" for tp in TIER_PAIRS))
        for i in range(n_layers):
            row = "  ".join(f"{tp_curves[tp][i]:6.3f}" for tp in TIER_PAIRS)
            print(f"{i:5d}  {row}")

    out_path = os.path.join(OUTPUT_DIR, 'layerwise_cross_language.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
