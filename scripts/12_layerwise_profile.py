"""
Per-layer profile of same_language_cross_mechanism cosine similarity, to see
WHERE (which layers) the low-resource "mechanism collapse" pattern (H<M<L in
the layer-averaged number) actually comes from -- is it uniform across all
layers, or does it emerge/strengthen at specific depths?

Reads the 'point' field already saved in pilot_results.json (per-layer cosine
list, one per mechanism-pair) -- no model, no GPU, just re-reading existing
results with a per-layer lens instead of the layer-averaged one used
everywhere else in this project.

Usage:
  python scripts/12_layerwise_profile.py
"""
import json
import os

SCRIPT_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'output')

MODELS = ['Qwen2.5-7B-Instruct', 'Meta-Llama-3.1-8B-Instruct', 'gemma-2-9b-it']
TIERS = {'en': 'H', 'zh': 'H', 'de': 'H', 'ko': 'M', 'ar': 'M', 'th': 'M',
         'yo': 'L', 'sw': 'L', 'am': 'L'}
PILOT_LANGS = list(TIERS.keys())


def per_language_curve(slcm, lang):
    """Average the 15 mechanism-pair 'point' lists for this language, layer by layer."""
    pairs = [v['point'] for k, v in slcm.items() if k.startswith(lang + '__')]
    n_layers = len(pairs[0])
    curve = [sum(p[l] for p in pairs) / len(pairs) for l in range(n_layers)]
    return curve


def main():
    results = {}
    for model in MODELS:
        path = os.path.join(OUTPUT_DIR, model, 'pilot_results.json')
        if not os.path.exists(path):
            print(f"Missing {path}, skipping {model}")
            continue
        with open(path) as f:
            d = json.load(f)
        slcm = d['same_language_cross_mechanism']

        lang_curves = {lang: per_language_curve(slcm, lang) for lang in PILOT_LANGS}
        n_layers = len(next(iter(lang_curves.values())))

        tier_curves = {}
        for tier in ['H', 'M', 'L']:
            langs = [l for l in PILOT_LANGS if TIERS[l] == tier]
            tier_curves[tier] = [
                sum(lang_curves[l][i] for l in langs) / len(langs)
                for i in range(n_layers)
            ]

        results[model] = {'n_layers': n_layers, 'lang_curves': lang_curves, 'tier_curves': tier_curves}

        print(f"\n=== {model} (n_layers={n_layers}) ===")
        print("layer  " + "  ".join(f"{t:>6}" for t in ['H', 'M', 'L']))
        for i in range(n_layers):
            row = "  ".join(f"{tier_curves[t][i]:6.3f}" for t in ['H', 'M', 'L'])
            print(f"{i:5d}  {row}")

    out_path = os.path.join(OUTPUT_DIR, 'layerwise_profile.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
