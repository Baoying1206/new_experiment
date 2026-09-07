"""
Builds an independent R-axis manifest (EXPERIMENT2_RH_REBUILD_PROTOCOL.md
Sec 15 / scripts/utils/axis_manifest.py schema) from
28b_sorry_bench_feasibility_pilot.py's saved output (ids/categories/
refused labels only) plus the external SORRY-Bench question.jsonl (read
again here ONLY to compute per-row normalised-text hashes and the source
file's sha256 -- never to copy raw text into the manifest).

Frozen construction formula (EXPERIMENT2_RH_REBUILD_PROTOCOL.md Sec 2.1):
v_R = mean(h(x)|refused) - mean(h(x)|accepted), simple pooled mean, NO
category/mechanism stratification -- this manifest records each row's
SORRY-Bench category as `prompt_family` (so category-stratified robustness
analysis remains possible later, if separately authorized), but the split
here is stratified ONLY by refused/accepted label, not by category. This
script does not implement, and is not, "stratified construction as the
primary method" -- see the 2026-09-04 correction (LOMO/stratification are
secondary-robustness-only until separately decided).

Does NOT read or embed any raw SORRY-Bench instruction/response text --
output contains only ids, hashes, category names, and labels. Consistent
with SORRY-Bench's no-redistribution license term.

Usage:
  python scripts/28c_build_sorry_bench_axis_manifest.py \
      --pilot_json output/sorry_bench_feasibility_pilot_Meta-Llama-3.1-8B-Instruct_en_n448.json \
      --sorry_bench_question_jsonl ~/sorry_bench_local/sorry-bench-202406/question.jsonl \
      --model_alias Meta-Llama-3.1-8B-Instruct \
      --val_fraction 0.2 --seed 0 \
      --out_path output/axis_manifests/sorry_bench_refusal_axis_Meta-Llama-3.1-8B-Instruct.json
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)
from utils.axis_manifest import (
    normalized_text_hash, sha256_of_file, MANIFEST_ROW_REQUIRED_FIELDS,
)


def _norm(s):
    return re.sub(r'\s+', ' ', s.strip().lower())


def sorry_bench_repo_commit(question_jsonl_path):
    """Best-effort: the cloned SORRY-Bench HF dataset repo's own git commit,
    for precise provenance (falls back to None if not a git checkout)."""
    repo_dir = os.path.dirname(os.path.abspath(question_jsonl_path))
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo_dir,
                              capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return None


def main(args):
    with open(args.pilot_json, encoding='utf-8') as f:
        pilot = json.load(f)
    assert pilot.get('result_status') == 'FEASIBILITY_CHECK_NON_RESULT', (
        f"expected a feasibility-pilot output file, got result_status={pilot.get('result_status')!r}"
    )
    labels = pilot['per_instruction_labels']  # [{'id': 'sorry_bench_N', 'category': ..., 'refused': 0|1}]
    print(f"Loaded {len(labels)} labeled instances from {args.pilot_json}")

    # Re-read the ORIGINAL SORRY-Bench file only to compute per-row text
    # hashes (never to copy the text itself into the manifest).
    id_to_text = {}
    with open(args.sorry_bench_question_jsonl, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get('prompt_style') == 'base':
                id_to_text[f"sorry_bench_{row['question_id']}"] = row['turns'][0]
    source_hash = sha256_of_file(args.sorry_bench_question_jsonl)
    repo_commit = sorry_bench_repo_commit(args.sorry_bench_question_jsonl)
    dataset_version = repo_commit if repo_commit else '202406'
    print(f"SORRY-Bench source file sha256: {source_hash}  (dataset_version={dataset_version})")

    # Stratified split by refused/accepted label ONLY (not by category --
    # see module docstring). Deterministic: shuffle each label group with
    # the same seed, slice off val_fraction.
    rng = random.Random(args.seed)
    refused = [r for r in labels if r['refused'] == 1]
    accepted = [r for r in labels if r['refused'] == 0]
    rng.shuffle(refused)
    rng.shuffle(accepted)
    n_val_refused = max(1, int(len(refused) * args.val_fraction))
    n_val_accepted = max(1, int(len(accepted) * args.val_fraction))
    val_set = set(id(r) for r in refused[:n_val_refused]) | set(id(r) for r in accepted[:n_val_accepted])
    print(f"Split: axis={len(refused) - n_val_refused} refused + {len(accepted) - n_val_accepted} accepted, "
          f"val={n_val_refused} refused + {n_val_accepted} accepted")

    rows = []
    for r in labels:
        text = id_to_text.get(r['id'])
        assert text is not None, f"{r['id']} not found in {args.sorry_bench_question_jsonl} (prompt_style=='base')"
        rows.append({
            'dataset_name': 'sorry-bench-202406',
            'dataset_version': dataset_version,
            'source_path': os.path.abspath(args.sorry_bench_question_jsonl),
            'source_file_sha256': source_hash,
            'stable_source_id': r['id'],  # native question_id -- SORRY-Bench HAS a real id, no synthetic scheme needed
            'normalized_text_hash': normalized_text_hash(text),
            'prompt_family': r['category'],  # SORRY-Bench's own 45-category label, NOT a canonical mechanism
            'condition': 'plain',
            'model_alias': args.model_alias,
            'response_id': f'{os.path.basename(args.pilot_json)}:{r["id"]}',  # pointer only, no response text stored anywhere in git
            'refusal_label': r['refused'],
            'label_source': 'wildguard',
            'split': 'val' if id(r) in val_set else 'axis',
            'overlaps_572_pool': False,  # already excluded by 28b before the pilot ran
            'contains_canonical_template': False,  # prompt_style=='base', no template of any kind
        })

    for row in rows:
        missing = [f for f in MANIFEST_ROW_REQUIRED_FIELDS if f not in row]
        assert not missing, f"row for {row['stable_source_id']} missing fields: {missing}"

    manifest = {
        'schema_version': '1.0',
        'built_from_pilot_json': args.pilot_json,
        'built_by': 'scripts/28c_build_sorry_bench_axis_manifest.py',
        'val_fraction': args.val_fraction,
        'seed': args.seed,
        'rows': rows,
    }
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    with open(args.out_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved manifest ({len(rows)} rows, no raw text): {args.out_path}")
    print("Next: validate it with scripts/26_rebuild_refusal_direction_behavioral.py --axis_manifest "
          f"{args.out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot_json', type=str, required=True)
    parser.add_argument('--sorry_bench_question_jsonl', type=str, required=True)
    parser.add_argument('--model_alias', type=str, required=True)
    parser.add_argument('--val_fraction', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out_path', type=str, required=True)
    args = parser.parse_args()
    main(args)
