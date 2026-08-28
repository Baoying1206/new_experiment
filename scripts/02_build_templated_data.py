"""
Combine sampled_prompts.json + per-language templates into the actual generation
inputs: for each language, for each of the 75 prompts, 8 conditions (plain +
6 attack templates + 1 placebo control).

encoding_obfuscation is base64-encoded from the TRANSLATED instruction (per
language), never from the English original -- encoding the English text and
reusing it across languages would mean non-English runs are secretly partly in
English, contaminating the cross-lingual comparison.

Output: data/generation_input_{lang}.json -- list of
  {id, condition, mechanism, category, instruction, instruction_en}
where `instruction` is the final text to feed the model (matches the
{"instruction": ...} contract expected by generate_completions / run_baseline.py).

No model/GPU needed -- pure text templating, safe to run locally.

Optionally restrict to a named id subset from data/splits.json (e.g.
cross_lingual_ids) via --ids_key, and/or a subset of languages via --langs,
writing to generation_input_{lang}{suffix}.json so this doesn't overwrite
the full-pool file for languages that also have a full-scale version (e.g.
English keeps generation_input_en.json at 572; the other 8 languages get
generation_input_{lang}_xling.json at 200 via --ids_key cross_lingual_ids
--langs zh,de,ko,ar,th,yo,sw,am --suffix _xling).
"""
import argparse
import base64
import json
import os

SCRIPT_DIR = os.path.dirname(__file__)
SAMPLED_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'sampled_prompts.json')
SPLITS_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'splits.json')
TEMPLATE_DIR = os.path.join(SCRIPT_DIR, '..', 'templates')
OUT_DIR = os.path.join(SCRIPT_DIR, '..', 'data')

PILOT_LANGS = ['en', 'zh', 'de', 'ko', 'ar', 'th', 'yo', 'sw', 'am']


def load_templates(lang):
    en_path = os.path.join(TEMPLATE_DIR, 'templates_en.json')
    with open(en_path) as f:
        en_data = json.load(f)
    mechanism_of = {name: spec['mechanism'] for name, spec in en_data['templates'].items()}

    if lang == 'en':
        texts = {name: spec['text'] for name, spec in en_data['templates'].items()}
    else:
        lang_path = os.path.join(TEMPLATE_DIR, f'templates_{lang}.json')
        with open(lang_path) as f:
            lang_data = json.load(f)
        texts = {name: spec['text'] for name, spec in lang_data['templates'].items()}
        missing = set(mechanism_of) - set(texts)
        if missing:
            raise ValueError(f"[{lang}] missing template translations for: {missing}")

    return texts, mechanism_of


def split_payload(instruction):
    """Splits instruction into two halves at the word boundary nearest the
    character midpoint, so a+b reassembles to exactly the original string
    (word-preserving, unlike a raw character-index split which could cut a
    word in half and make the SmartGPT concatenation trick read as garbled
    text rather than the original request)."""
    words = instruction.split(' ')
    if len(words) < 2:
        # single-word instruction (rare) -- fall back to a literal character split
        mid = max(1, len(instruction) // 2)
        return instruction[:mid], instruction[mid:]
    # find the split point among word boundaries closest to the character midpoint
    target = len(instruction) / 2
    best_i, best_dist = 1, float('inf')
    running_len = 0
    for i in range(1, len(words)):
        running_len += len(words[i - 1]) + 1  # +1 for the space
        dist = abs(running_len - target)
        if dist < best_dist:
            best_dist, best_i = dist, i
    a = ' '.join(words[:best_i])
    b = ' ' + ' '.join(words[best_i:])  # leading space so a+b == instruction exactly
    return a, b


def build_condition(template_name, template_text, mechanism, instruction):
    if template_name == 'encoding_obfuscation':
        b64 = base64.b64encode(instruction.encode('utf-8')).decode('ascii')
        return template_text.format(instruction_b64=b64)
    if template_name == 'payload_splitting':
        payload_a, payload_b = split_payload(instruction)
        return template_text.format(payload_a=payload_a, payload_b=payload_b)
    return template_text.format(instruction=instruction)


def main(args):
    with open(SAMPLED_PATH, encoding='utf-8') as f:
        sampled = json.load(f)

    if args.ids_key:
        with open(SPLITS_PATH) as f:
            splits = json.load(f)
        keep_ids = set(splits[args.ids_key])
        sampled = [item for item in sampled if item['id'] in keep_ids]
        print(f"Loaded {len(sampled)} prompts (filtered to splits.json['{args.ids_key}'], "
              f"{len(keep_ids)} requested).")
    else:
        print(f"Loaded {len(sampled)} sampled prompts (no id filter).")

    only_mechanisms = set(args.only_mechanisms.split(',')) if args.only_mechanisms else None
    if only_mechanisms:
        print(f"Restricting to mechanisms: {sorted(only_mechanisms)}  "
              f"(skip_plain={args.skip_plain}) -- for regenerating only NEW/changed "
              f"mechanisms without re-generating ones already covered by an existing "
              f"completions file.\n")

    langs = args.langs.split(',') if args.langs else PILOT_LANGS
    for lang in langs:
        texts, mechanism_of = load_templates(lang)
        if only_mechanisms:
            missing = only_mechanisms - set(texts.keys())
            if missing:
                raise ValueError(f"--only_mechanisms names {missing} not found in "
                                  f"templates_{lang}.json's templates: {sorted(texts.keys())}")
            texts = {name: text for name, text in texts.items() if name in only_mechanisms}
        rows = []
        needs_translation_flagged = False

        for item in sampled:
            base_instruction = item['instructions'][lang]
            instruction_en = item['instruction_en']

            # condition: plain (no template)
            if not args.skip_plain:
                rows.append({
                    'id': item['id'], 'condition': 'plain', 'mechanism': 'none',
                    'category': item['category'],
                    'instruction': base_instruction, 'instruction_en': instruction_en,
                })

            # condition: each template
            for name, text in texts.items():
                if '[NEEDS REAL TRANSLATION]' in text:
                    needs_translation_flagged = True
                rendered = build_condition(name, text, mechanism_of[name], base_instruction)
                rows.append({
                    'id': item['id'], 'condition': name, 'mechanism': mechanism_of[name],
                    'category': item['category'],
                    'instruction': rendered, 'instruction_en': instruction_en,
                })

        out_path = os.path.join(OUT_DIR, f'generation_input_{lang}{args.suffix}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

        n_conditions = len(texts) + (0 if args.skip_plain else 1)
        print(f"[{lang}] {len(rows)} rows ({len(sampled)} prompts x {n_conditions} conditions) "
              f"-> {out_path}"
              + ("  *** contains untranslated placeholders, DO NOT run on cluster yet ***"
                 if needs_translation_flagged else ""))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ids_key', type=str, default=None,
                         help="Key into data/splits.json to restrict prompts to "
                              "(e.g. 'cross_lingual_ids'). Default: use all sampled prompts.")
    parser.add_argument('--langs', type=str, default=None,
                         help="Comma-separated languages. Default: all 9 pilot languages.")
    parser.add_argument('--suffix', type=str, default='',
                         help="Output filename suffix, e.g. '_xling' -> generation_input_{lang}_xling.json")
    parser.add_argument('--only_mechanisms', type=str, default=None,
                         help="Comma-separated template names to restrict to (e.g. "
                              "'payload_splitting,distractor_instructions') -- for building a "
                              "minimal generation_input covering only NEW mechanisms not already "
                              "present in an existing completions file, instead of regenerating "
                              "everything. Default: all templates in templates_{lang}.json.")
    parser.add_argument('--skip_plain', action='store_true',
                         help="Omit the 'plain' (no-template) condition row -- use when plain's "
                              "completions already exist and don't need regenerating.")
    main(parser.parse_args())
