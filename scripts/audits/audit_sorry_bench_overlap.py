"""
Checks whether SORRY-Bench's base (prompt_style="base", unmutated) 450
instructions overlap with the 572-instruction pool (data/sampled_prompts.json)
by normalised text, and reports the per-category count of the 45 safety
categories -- part of evaluating SORRY-Bench as an independent R-axis
candidate source (EXPERIMENT2_RH_REBUILD_PROTOCOL.md Sec 13).

This script contains NO SORRY-Bench data itself (only generic comparison
logic) and is safe to commit -- the actual dataset lives outside this repo
(SORRY-Bench's license prohibits redistribution) and is passed in via
--sorry_bench_question_jsonl, a path this script never hardcodes or bundles.

Read-only: does not modify data/sampled_prompts.json, does not modify the
SORRY-Bench files, does not write anything except the two output files
below (containing only counts/hashes/category labels, never raw instruction
text from SORRY-Bench, consistent with the no-redistribution license term).

Usage:
  python scripts/audits/audit_sorry_bench_overlap.py \
      --sorry_bench_question_jsonl ~/sorry_bench_local/sorry-bench-202406/question.jsonl \
      --sorry_bench_meta_info ~/sorry_bench_local/sorry-bench-202406/meta_info.py
"""
import argparse
import importlib.util
import json
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'data')
OUT_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'output', 'audits')


def _norm(s):
    return re.sub(r'\s+', ' ', s.strip().lower())


def load_category_descriptions(meta_info_path):
    spec = importlib.util.spec_from_file_location('sorry_bench_meta_info', meta_info_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.category_descriptions  # list, index 0 = category "1"


def main(args):
    with open(os.path.join(DATA_DIR, 'sampled_prompts.json'), encoding='utf-8') as f:
        pool = json.load(f)
    pool_texts = set(_norm(x['instruction_en']) for x in pool)
    print(f"572-pool: {len(pool)} instructions, {len(pool_texts)} unique normalised texts")

    rows = []
    with open(args.sorry_bench_question_jsonl, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    base_rows = [r for r in rows if r.get('prompt_style') == 'base']
    print(f"SORRY-Bench question.jsonl: {len(rows)} total rows, {len(base_rows)} with prompt_style=='base'")

    cats = load_category_descriptions(args.sorry_bench_meta_info)
    print(f"Loaded {len(cats)} category descriptions from meta_info.py")

    overlap_ids = []
    per_category_count = {}
    for r in base_rows:
        text = r['turns'][0] if r.get('turns') else ''
        norm_text = _norm(text)
        if norm_text in pool_texts:
            overlap_ids.append(r['question_id'])
        cat_idx = int(r['category']) - 1  # category "1" -> index 0
        cat_name = cats[cat_idx] if 0 <= cat_idx < len(cats) else f'UNKNOWN_CATEGORY_{r["category"]}'
        per_category_count[cat_name] = per_category_count.get(cat_name, 0) + 1

    print(f"\nOverlap with 572-pool (normalised text): {len(overlap_ids)}/{len(base_rows)}")
    if overlap_ids:
        print(f"  Overlapping question_ids: {overlap_ids}")
    else:
        print("  ZERO overlap -- SORRY-Bench base set is confirmed independent of the 572-pool.")

    print(f"\nPer-category counts ({len(per_category_count)} categories present):")
    for cat, count in sorted(per_category_count.items(), key=lambda kv: -kv[1]):
        print(f"  {count:3d}  {cat}")

    # Output contains ONLY counts/ids/category-names -- never SORRY-Bench's
    # raw instruction text, consistent with the no-redistribution license term.
    result = {
        'result_status': 'AUDIT_NON_RESULT',
        'note': 'Overlap/category audit only. Contains NO raw SORRY-Bench instruction text '
                '(license prohibits redistribution) -- only counts, question_ids, and category names.',
        'pool_size': len(pool),
        'sorry_bench_total_rows': len(rows),
        'sorry_bench_base_rows': len(base_rows),
        'overlap_count': len(overlap_ids),
        'overlap_question_ids': overlap_ids,
        'per_category_count': per_category_count,
        'n_categories_present': len(per_category_count),
        'n_categories_total_in_meta_info': len(cats),
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'sorry_bench_overlap_audit.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved (counts/ids only, no raw text): {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sorry_bench_question_jsonl', type=str, required=True)
    parser.add_argument('--sorry_bench_meta_info', type=str, required=True)
    args = parser.parse_args()
    main(args)
