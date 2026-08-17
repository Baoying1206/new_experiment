"""
Auto-translate templates_en.json into a target language via Google Translate,
with automatic back-translation for a lightweight quality check.

THIS DOES NOT REPLACE NATIVE-SPEAKER REVIEW. It only catches gross failures
(placeholder lost, meaning flipped/garbled). Read every back_translation_check
field it prints before trusting the output, especially for persona_roleplay,
refusal_suppression, and instruction_hierarchy -- these rely on tone/force
that MT is most likely to flatten or lose, particularly for Yoruba.

Uses `deep-translator` (unofficial, scrapes the public Google Translate
endpoint -- no API key needed, but can break if Google changes their page,
and is rate-limited). Fine for this one-off job (~7 short strings x 2 calls
each); not meant as a production translation pipeline. If it breaks, either
retry later or switch to the official `google-cloud-translate` package with
a GCP API key.

Needs live internet access -- run this on your laptop (or the cluster's login
node if it has internet), NOT as a Slurm batch job on a compute node.

Usage:
  pip install deep-translator
  python scripts/00_translate_templates.py --target yo
  python scripts/00_translate_templates.py --target ko
"""
import argparse
import json
import os
import time

SCRIPT_DIR = os.path.dirname(__file__)
EN_PATH = os.path.join(SCRIPT_DIR, '..', 'templates', 'templates_en.json')

# Alphabetic, no punctuation -- least likely to get mangled or "translated" itself.
PLACEHOLDER_TOKENS = {
    '{instruction}': 'ZZINSTRUCTIONPLACEHOLDERZZ',
    '{instruction_b64}': 'ZZINSTRUCTIONB64PLACEHOLDERZZ',
}


def translate(text, source, target, retries=3):
    from deep_translator import GoogleTranslator
    for attempt in range(retries):
        try:
            return GoogleTranslator(source=source, target=target).translate(text)
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  (retry {attempt+1}/{retries} after error: {e})")
            time.sleep(2)


def main(args):
    with open(EN_PATH, encoding='utf-8') as f:
        en_data = json.load(f)

    out_templates = {}
    any_warning = False
    print(f"Translating templates_en.json -> {args.target}, with back-translation QC.\n")

    for name, spec in en_data['templates'].items():
        en_text = spec['text']

        placeholder = next(p for p in PLACEHOLDER_TOKENS if p in en_text)
        token = PLACEHOLDER_TOKENS[placeholder]
        protected = en_text.replace(placeholder, token)

        translated_protected = translate(protected, 'en', args.target)
        placeholder_survived = token in translated_protected
        translated = translated_protected.replace(token, placeholder)

        # Back-translate for QC (translate the token-protected version back, not the
        # final one, so the placeholder doesn't confuse the back-translation model).
        back_protected = translate(translated_protected, args.target, 'en')
        back_text = back_protected.replace(token, placeholder)

        out_templates[name] = {
            'mechanism': spec['mechanism'],
            'text': translated,
            'translation_confidence': 'unverified_mt_draft',
            'back_translation_check': back_text,
        }

        print(f"[{name}] ({spec['mechanism']})")
        print(f"  EN original     : {en_text}")
        print(f"  {args.target} draft{' '*(8-len(args.target))}: {translated}")
        print(f"  back-translated : {back_text}")
        if not placeholder_survived:
            any_warning = True
            print(f"  *** WARNING: placeholder lost in translation -- fix this entry manually ***")
        print()

    out_path = os.path.join(SCRIPT_DIR, '..', 'templates', f'templates_{args.target}_draft.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({
            '_note': (
                f'MT draft ({args.target}) via Google Translate (deep-translator), with automatic '
                f'back-translation for QC. NOT reviewed by a native speaker -- read every '
                f'back_translation_check field against the English original before trusting this. '
                f'If tone/force drifts (esp. persona_roleplay, refusal_suppression, '
                f'instruction_hierarchy) or meaning is garbled, fix that entry by hand or flag it '
                f'low-confidence and interpret its pilot results cautiously. '
                f'Rename to templates_{args.target}.json only after this review.'
            ),
            'templates': out_templates,
        }, f, indent=2, ensure_ascii=False)

    print(f"{'='*60}")
    print(f"Draft saved to: {out_path}")
    print(f"Review the back-translations above, then rename to templates_{args.target}.json "
          f"(overwriting the placeholder version) once you're satisfied.")
    if any_warning:
        print("*** At least one placeholder was lost in translation -- check warnings above. ***")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, required=True, choices=['ko', 'yo'])
    args = parser.parse_args()
    main(args)
