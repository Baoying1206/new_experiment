"""
Rules out (or implicates) a confound in the "low-resource mechanism collapse"
finding from 04_extract_directions_and_analyze.py: same_language_cross_mechanism
cosine similarity rises as resource tier drops for Qwen/Llama. That could be a
genuine model-internal effect (the model's representations become less
differentiated for low-resource languages), or it could just be that machine
translation quality is worse for low-resource languages and happens to make
the SIX TEMPLATES' WRAPPER TEXT ITSELF more similar to each other -- i.e. the
templates literally read more alike in yo/am post-translation, independent of
anything the model does internally.

This script never loads a model. It only compares the raw translated template
wrapper text (the scaffolding around {instruction}/{instruction_b64}, with the
instruction placeholder stripped out) across the 6 real mechanisms, per
language, using character-trigram Jaccard similarity -- then checks whether
this SURFACE-TEXT similarity also rises from high- to low-resource tier. If it
does, that's at least a partial confound for the activation-level finding; if
it stays flat, the activation-level collapse is not explained by the
templates simply reading more alike after translation.

Usage:
  python scripts/11_translation_surface_similarity.py
"""
import itertools
import json
import os

SCRIPT_DIR = os.path.dirname(__file__)
TEMPLATES_DIR = os.path.join(SCRIPT_DIR, '..', 'templates')

REAL_MECHS = ['prefix_injection', 'refusal_suppression', 'instruction_hierarchy',
              'persona_roleplay', 'fictional_framing', 'encoding_obfuscation']
TIERS = {'en': 'H', 'zh': 'H', 'de': 'H', 'ko': 'M', 'ar': 'M', 'th': 'M',
         'yo': 'L', 'sw': 'L', 'am': 'L'}
PILOT_LANGS = list(TIERS.keys())


def load_wrapper(lang, mech):
    fname = 'templates_en.json' if lang == 'en' else f'templates_{lang}.json'
    with open(os.path.join(TEMPLATES_DIR, fname), encoding='utf-8') as f:
        data = json.load(f)
    text = data['templates'][mech]['text']
    return text.replace('{instruction_b64}', '').replace('{instruction}', '')


def char_trigrams(text):
    text = text.strip()
    if len(text) < 3:
        return set()
    return {text[i:i + 3] for i in range(len(text) - 2)}


def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    per_lang = {}
    for lang in PILOT_LANGS:
        trigrams = {mech: char_trigrams(load_wrapper(lang, mech)) for mech in REAL_MECHS}
        sims = [jaccard(trigrams[m1], trigrams[m2])
                for m1, m2 in itertools.combinations(REAL_MECHS, 2)]
        per_lang[lang] = sum(sims) / len(sims)
        print(f"[{lang}] ({TIERS[lang]})  surface_text_jaccard_mean={per_lang[lang]:.3f}  n_pairs={len(sims)}")

    print("\n=== By resource tier ===")
    by_tier = {}
    for tier in ['H', 'M', 'L']:
        vals = [per_lang[l] for l in PILOT_LANGS if TIERS[l] == tier]
        by_tier[tier] = sum(vals) / len(vals)
        print(f"  {tier}: {by_tier[tier]:.3f}")

    out_path = os.path.join(SCRIPT_DIR, '..', 'output', 'translation_surface_similarity.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'per_language': per_lang, 'by_tier': by_tier}, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
