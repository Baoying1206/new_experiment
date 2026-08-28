"""
Merges the corrected 6-mechanism taxonomy's completions from two sources,
instead of paying for a full 8-condition regeneration:
  - completions_{lang}{old_suffix}.json: reused as-is for the KEPT conditions
    (plain, placebo, prefix_injection, refusal_suppression, persona_roleplay,
    encoding_obfuscation) -- these templates/completions are unchanged by the
    taxonomy correction (see templates/templates_en.json's
    _taxonomy_correction_note -- only persona_roleplay's CO/MG *label*
    changed, not its template text or generated completions).
  - completions_{lang}{new_suffix}.json: the freshly-generated NEW mechanisms
    (payload_splitting, distractor_instructions), built from
    generation_input_{lang}{new_suffix}.json (02_build_templated_data.py
    --only_mechanisms payload_splitting,distractor_instructions --skip_plain).

Output: completions_{lang}{corrected_suffix}.json, with exactly 8 conditions
per instruction id (KEPT_CONDITIONS + NEW_MECHANISMS), same schema as any
other completions file so 18_extract_paired_diffs.py / 03's format all read
it unchanged. Refuses to write if any instruction id is missing a condition
in either source, or if any id present in one source is absent from the
other, rather than silently producing a partial merge.

Usage:
  python scripts/31_merge_corrected_completions.py \
      --output_dir output --model_alias Qwen2.5-7B-Instruct --lang en \
      --old_suffix _full572 --new_suffix _full572_newmechs --corrected_suffix _full572_corrected
"""
import argparse
import json
import os

SCRIPT_DIR = os.path.dirname(__file__)
KEPT_CONDITIONS = ['plain', 'placebo', 'prefix_injection', 'refusal_suppression',
                    'persona_roleplay', 'encoding_obfuscation']
NEW_MECHANISMS = ['payload_splitting', 'distractor_instructions']
ALL_CORRECTED_CONDITIONS = KEPT_CONDITIONS + NEW_MECHANISMS


def load_completions(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)['completions']


def main(args):
    out_dir = os.path.join(args.output_dir, args.model_alias)
    old_path = os.path.join(out_dir, f'completions_{args.lang}{args.old_suffix}.json')
    new_path = os.path.join(out_dir, f'completions_{args.lang}{args.new_suffix}.json')
    out_path = os.path.join(out_dir, f'completions_{args.lang}{args.corrected_suffix}.json')

    old_completions = load_completions(old_path)
    new_completions = load_completions(new_path)
    print(f"Loaded {len(old_completions)} from {old_path}")
    print(f"Loaded {len(new_completions)} from {new_path}")

    kept = [c for c in old_completions if c['condition'] in KEPT_CONDITIONS]
    print(f"  {len(kept)} of the old file's rows are in KEPT_CONDITIONS={KEPT_CONDITIONS}")

    by_id = {}
    for c in kept + new_completions:
        by_id.setdefault(c['id'], {})[c['condition']] = c

    ids_with_all_conditions = [pid for pid, conds in by_id.items()
                                if set(conds.keys()) == set(ALL_CORRECTED_CONDITIONS)]
    ids_incomplete = {pid: sorted(set(ALL_CORRECTED_CONDITIONS) - set(conds.keys()))
                       for pid, conds in by_id.items()
                       if set(conds.keys()) != set(ALL_CORRECTED_CONDITIONS)}

    print(f"\n{len(by_id)} unique ids seen total")
    print(f"{len(ids_with_all_conditions)} ids have all {len(ALL_CORRECTED_CONDITIONS)} corrected conditions")
    if ids_incomplete:
        print(f"{len(ids_incomplete)} ids are INCOMPLETE -- refusing to write a partial merge. "
              f"Examples: {dict(list(ids_incomplete.items())[:5])}")
        raise SystemExit(1)

    merged = []
    for pid in sorted(by_id.keys()):
        for cond in ALL_CORRECTED_CONDITIONS:
            merged.append(by_id[pid][cond])

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'completions': merged}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(merged)} rows ({len(ids_with_all_conditions)} ids x "
          f"{len(ALL_CORRECTED_CONDITIONS)} conditions) -> {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',        type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    parser.add_argument('--model_alias',       type=str, required=True)
    parser.add_argument('--lang',              type=str, default='en')
    parser.add_argument('--old_suffix',        type=str, default='_full572')
    parser.add_argument('--new_suffix',        type=str, default='_full572_newmechs')
    parser.add_argument('--corrected_suffix',  type=str, default='_full572_corrected')
    args = parser.parse_args()
    main(args)
