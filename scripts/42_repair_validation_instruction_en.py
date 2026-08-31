"""
Deterministic repair for the instruction_en bug fixed in
40_defence_generation_driver.py (commit d4ed608): existing
experiment3_validation_generations_{model}_{method}.jsonl files have every
row's instruction_en set to the WRAPPED/rendered prompt (a duplicate of
`instruction`) instead of the true plain original -- pipeline's
generate_completions() always returns instructions_en=[x['instruction'] for
x in dataset], never reading our own instruction_en field.

This is metadata-only: `response`/`generation_tokens`/every other field is
untouched and valid (the target model DID receive the correct wrapped
prompt as `instruction`). Only `instruction_en` (the text WildGuard judges
against) is corrected here.

The ground-truth plain text is re-derived per record from
(instruction_id, benign_or_harmful) alone (not from `template`, since the
plain original is the same across all 6 templates for a given instruction):
  harmful -> data/sampled_prompts.json's instruction_en, keyed by id
  benign  -> data/benign_validation_80.json's instruction, keyed by benign_id

Because judge_cache_key = f(instruction_en, response, ...), repairing
instruction_en changes every record's judge_cache_key -- the existing
judgement file for this (model, method) becomes entirely stale after this
repair and must be re-judged from scratch (this script does not do that
itself; see the reported next-step command).

Default is DRY RUN (no write). Only writes with --apply, via a temp-file +
os.replace atomic swap. Refuses to run (raises, writes nothing) if any
record's instruction_id can't be resolved to a ground-truth plain text.

Usage:
  python scripts/42_repair_validation_instruction_en.py \
      --model_alias Meta-Llama-3.1-8B-Instruct --method global --output_path output
  python scripts/42_repair_validation_instruction_en.py \
      --model_alias Meta-Llama-3.1-8B-Instruct --method global --output_path output --apply
"""
import argparse
import copy
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)
import _defence_metrics as dm

POOL_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'sampled_prompts.json')
BENIGN_VAL_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'benign_validation_80.json')
BENIGN_TEST_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'benign_test_100.json')  # not read; existence-checked only


def sha256_of_file(path):
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_ground_truth():
    with open(POOL_PATH, encoding='utf-8') as f:
        pool = json.load(f)
    harmful_truth = {p['id']: p['instruction_en'] for p in pool}
    with open(BENIGN_VAL_PATH, encoding='utf-8') as f:
        benign = json.load(f)
    benign_truth = {b['benign_id']: b['instruction'] for b in benign}
    return harmful_truth, benign_truth


def resolve_truth(record, harmful_truth, benign_truth):
    if record['benign_or_harmful'] == 'harmful':
        return harmful_truth.get(record['instruction_id'])
    elif record['benign_or_harmful'] == 'benign':
        return benign_truth.get(record['instruction_id'])
    raise ValueError(f"unrecognized benign_or_harmful: {record['benign_or_harmful']!r}")


def atomic_write_jsonl(path, rows):
    tmp_path = path + '.tmp_repair'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    os.replace(tmp_path, path)


def main(args):
    gen_path = os.path.join(args.output_path, 'canonical_v2',
                             f'experiment3_validation_generations_{args.model_alias}_{args.method}.jsonl')
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN (no files will be written)'}")
    print(f"Target: {gen_path}")

    rows = dm.load_jsonl(gen_path)
    assert rows, f"no rows loaded from {gen_path} -- refusing to run on an empty/missing file"
    sha_before = sha256_of_file(gen_path)
    before_snapshot = copy.deepcopy(rows)

    harmful_truth, benign_truth = load_ground_truth()

    unresolved = []
    already_correct = []
    to_fix = []
    for r in rows:
        truth = resolve_truth(r, harmful_truth, benign_truth)
        if truth is None:
            unresolved.append(r['instruction_id'])
            continue
        if r.get('instruction_en') == truth:
            already_correct.append(r['record_key'])
        else:
            to_fix.append((r, truth))

    assert not unresolved, (
        f"{len(unresolved)} record(s) have an instruction_id with no ground-truth plain text "
        f"(sample: {unresolved[:5]}) -- refusing to run, this is out of scope for this repair."
    )

    print(f"Total rows: {len(rows)}")
    print(f"Already correct: {len(already_correct)}")
    print(f"To fix: {len(to_fix)}")

    for r, truth in to_fix:
        r['instruction_en'] = truth

    # Semantic check: every field EXCEPT instruction_en must be byte-identical before/after.
    for before, after in zip(before_snapshot, rows):
        b2 = {k: v for k, v in before.items() if k != 'instruction_en'}
        a2 = {k: v for k, v in after.items() if k != 'instruction_en'}
        assert b2 == a2, f"non-instruction_en field changed for record_key={before.get('record_key')}"

    result = {
        'model_alias': args.model_alias, 'method': args.method, 'input_path': gen_path,
        'sha256_before': sha_before, 'total_rows': len(rows),
        'already_correct_rows': len(already_correct), 'changed_rows': len(to_fix),
        'unchanged_field_semantic_validation': 'passed (all non-instruction_en fields identical, verified per-record)',
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'applied': False, 'sha256_after': None,
    }

    if args.apply and to_fix:
        atomic_write_jsonl(gen_path, rows)
        result['applied'] = True
        result['sha256_after'] = sha256_of_file(gen_path)
    elif args.apply:
        result['note'] = 'nothing to change -- file left untouched, no write performed'
    else:
        result['note'] = 'dry run -- no file written'

    out_dir = os.path.join(args.output_path, 'canonical_v2')
    report_path = os.path.join(out_dir, f'experiment3_instruction_en_repair_{args.model_alias}_{args.method}.json')
    with open(report_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {report_path}")

    if args.apply and to_fix:
        judge_path = os.path.join(out_dir, f'experiment3_validation_judgements_{args.model_alias}_{args.method}.jsonl')
        print(f"\n*** instruction_en repaired for {len(to_fix)} rows. Every judge_cache_key for this "
              f"(model, method) has now changed -- the existing judgement file is entirely stale: ***")
        print(f"    {judge_path}")
        print("Recommended next steps:")
        print(f"  mv {judge_path} {judge_path}.stale_pre_instruction_en_fix")
        print(f"  (re-run the STAGE=judge job for this model x method to rejudge from scratch)")
        print(f"  (re-run 41_join_and_summarize_defence_validation.py afterward)")
    elif not args.apply:
        print("This was a DRY RUN. Re-run with --apply to actually write the repaired file.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_alias', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--output_path', type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    main(args)
