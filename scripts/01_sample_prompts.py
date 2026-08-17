"""
Stratified sample of ~N harmful prompts for the jailbreak-template pilot, matched
across en/ko/yo (same underlying prompts, just their per-language translations) so
cross-lingual comparisons are apples-to-apples rather than comparing different
random subsets per language.

Sampling is stratified by the same coarse category mapping used in
experiment_thesis/scripts/extract_jailbreak_vectors.py, proportional to each
category's size in the corpus, so this pilot doesn't reintroduce the category
imbalance problem the rest of the project is correcting for.

Output: data/sampled_prompts.json -- list of
  {id, category, instructions: {en: ..., ko: ..., yo: ...}}

No model/GPU needed -- pure data wrangling, safe to run locally.
"""
import json
import math
import os
import random

ENHANCED_DIR = os.path.join(os.path.dirname(__file__), '..', 'ployrefuse_Enhanced')
OUT_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'sampled_prompts.json')

PILOT_LANGS = ['en', 'zh', 'de', 'ko', 'ar', 'th', 'yo', 'sw', 'am']
TARGET_N = 75
SEED = 0

# Same coarse mapping as extract_jailbreak_vectors.py -- keep in sync if that one changes.
CATEGORY_MAP = {
    'Violence': 'violence', 'Physical harm': 'violence',
    'Non-violent crimes': 'crime', 'illegal': 'crime', 'harmful': 'crime',
    'Illegal goods and services': 'crime',
    'Disinformation and deception': 'misinfo', 'misinformation_disinformation': 'misinfo',
    'Disinformation': 'misinfo',
    'Hate, harassment and discrimination': 'harassment', 'harassment_bullying': 'harassment',
    'Harassment/Discrimination': 'harassment',
    'cybercrime_intrusion': 'cybercrime', 'Malware/Hacking': 'cybercrime',
    'Sexual content': 'sexual', 'Sexual/Adult content': 'sexual',
    'chemical_biological': 'cbrn',
    'Privacy': 'other', 'Economic harm': 'other', 'Fraud/Deception': 'other',
    'Expert advice': 'other', 'Government decision-making': 'other',
}


def load_lang_file(lang):
    path = os.path.join(ENHANCED_DIR, f'harmful_test_translated_{lang}.json')
    with open(path) as f:
        return json.load(f)


def main():
    random.seed(SEED)

    # English file is canonical: 'instruction' field == original English text,
    # used as the join key across all language files.
    en_items = load_lang_file('en')
    for item in en_items:
        item['_coarse_cat'] = CATEGORY_MAP.get(item.get('category') or 'other', 'other')

    by_cat = {}
    for item in en_items:
        by_cat.setdefault(item['_coarse_cat'], []).append(item)

    total = len(en_items)
    print(f"Total harmful_test items: {total}, categories: "
          f"{ {c: len(v) for c, v in by_cat.items()} }")

    # Proportional stratified sample, capped so every non-empty category gets >=1.
    sampled_en = []
    for cat, items in by_cat.items():
        n_cat = max(1, round(TARGET_N * len(items) / total))
        n_cat = min(n_cat, len(items))
        sampled_en.extend(random.sample(items, n_cat))

    # Trim/pad to land close to TARGET_N if rounding overshot.
    random.shuffle(sampled_en)
    sampled_en = sampled_en[:TARGET_N]
    print(f"Sampled {len(sampled_en)} prompts "
          f"(target {TARGET_N}), category counts: "
          f"{ {c: sum(1 for x in sampled_en if x['_coarse_cat']==c) for c in by_cat} }")

    sampled_keys = {item['instruction'] for item in sampled_en}  # English text = join key

    # Build lookup for ko/yo translations keyed by the same English text.
    lookups = {}
    for lang in PILOT_LANGS:
        if lang == 'en':
            continue
        items = load_lang_file(lang)
        lookups[lang] = {it['instruction']: it.get('instruction_translated', it['instruction'])
                          for it in items}

    out = []
    missing = 0
    for i, item in enumerate(sampled_en):
        key = item['instruction']
        row = {
            'id': f'p{i:03d}',
            'category': item['_coarse_cat'],
            'instruction_en': key,
            'instructions': {'en': key},
        }
        ok = True
        for lang in PILOT_LANGS:
            if lang == 'en':
                continue
            translated = lookups[lang].get(key)
            if translated is None:
                ok = False
                missing += 1
                break
            row['instructions'][lang] = translated
        if ok:
            out.append(row)

    if missing:
        print(f"WARNING: {missing} sampled prompts missing a translation in one of {PILOT_LANGS}, dropped.")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nFinal sample: {len(out)} prompts, saved to {OUT_PATH}")
    final_cats = {}
    for row in out:
        final_cats[row['category']] = final_cats.get(row['category'], 0) + 1
    print(f"Final category distribution: {final_cats}")


if __name__ == '__main__':
    main()
