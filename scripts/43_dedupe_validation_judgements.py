"""
Deterministic dedup for experiment3_validation_judgements_{model}_{method}.jsonl
files affected by the run_judge same-batch-duplicate bug fixed in
40_defence_generation_driver.py (run_judge): to_judge was not deduplicated by
judge_cache_key WITHIN a single call, so two records sharing a key that landed
in the same WILDGUARD_JUDGE_BATCH_SIZE chunk were both judged and both written
by on_new_batch (the caller's existing_cache_keys filter only updates BETWEEN
batches, not within one). Confirmed on the real Llama x global rejudge (job
4985): 54 duplicated judge_cache_key values, 84 extra rows, with 54/54
duplicate groups having byte-identical judgement content (request_harmful,
refusal, response_harmful, parse_success, raw_judge_output) -- i.e. this is
pure redundant writing, not WildGuard non-determinism.

This script refuses (raises, writes nothing) if any duplicate group's
non-judge_cache_key judgement content differs -- that would indicate genuine
non-determinism, which is out of scope for a mechanical dedup and must be
investigated separately.

Default is DRY RUN (no write). Only writes with --apply, via a temp-file +
os.replace atomic swap. Keeps the FIRST occurrence of each judge_cache_key
(arbitrary among identical duplicates, since content is verified identical).

Usage:
  python scripts/43_dedupe_validation_judgements.py \
      --model_alias Meta-Llama-3.1-8B-Instruct --method global --output_path output
  python scripts/43_dedupe_validation_judgements.py \
      --model_alias Meta-Llama-3.1-8B-Instruct --method global --output_path output --apply
"""
import argparse
import collections
import copy
import json
import os
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)
import _defence_metrics as dm

CONTENT_FIELDS = ('request_harmful', 'refusal', 'response_harmful', 'parse_success', 'raw_judge_output')


class JudgementContentMismatchError(Exception):
    pass


def find_duplicate_groups(rows):
    by_key = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_key[r['judge_cache_key']].append(i)
    return {k: idxs for k, idxs in by_key.items() if len(idxs) > 1}


def verify_identical_content(rows, duplicate_groups):
    mismatches = []
    for key, idxs in duplicate_groups.items():
        core = [{f: rows[i].get(f) for f in CONTENT_FIELDS} for i in idxs]
        if not all(c == core[0] for c in core):
            mismatches.append(key)
    if mismatches:
        raise JudgementContentMismatchError(
            f"{len(mismatches)} duplicate judge_cache_key group(s) have DIFFERING judgement content "
            f"(sample: {mismatches[:5]}) -- this indicates non-deterministic judging, not the known "
            f"same-batch-write bug. Refusing to dedup; investigate separately."
        )


def dedupe_rows(rows, duplicate_groups):
    drop_indices = set()
    for idxs in duplicate_groups.values():
        drop_indices.update(idxs[1:])  # keep the first occurrence
    return [r for i, r in enumerate(rows) if i not in drop_indices], len(drop_indices)


def atomic_write_jsonl(path, rows):
    tmp_path = path + '.tmp_dedupe'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    os.replace(tmp_path, path)


def main(args):
    judge_path = os.path.join(args.output_path, 'canonical_v2',
                               f'experiment3_validation_judgements_{args.model_alias}_{args.method}.jsonl')
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN (no files will be written)'}")
    print(f"Target: {judge_path}")

    rows = dm.load_jsonl(judge_path)
    assert rows, f"no rows loaded from {judge_path} -- refusing to run on an empty/missing file"
    sha_before = dm.sha256_of_file(judge_path)
    before_snapshot = copy.deepcopy(rows)

    duplicate_groups = find_duplicate_groups(rows)
    print(f"Total rows: {len(rows)}")
    print(f"Unique judge_cache_key values: {len(rows) - sum(len(v) - 1 for v in duplicate_groups.values())}")
    print(f"Duplicated keys: {len(duplicate_groups)}")
    print(f"Extra rows to remove: {sum(len(v) - 1 for v in duplicate_groups.values())}")

    if not duplicate_groups:
        print("No duplicate judge_cache_key values found -- nothing to do.")
        result = {
            'model_alias': args.model_alias, 'method': args.method, 'input_path': judge_path,
            'sha256_before': sha_before, 'total_rows': len(rows), 'duplicate_groups': 0,
            'rows_removed': 0, 'applied': False, 'sha256_after': None,
            'note': 'no duplicates found -- file left untouched',
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        }
    else:
        verify_identical_content(rows, duplicate_groups)
        print("All duplicate groups verified to have byte-identical judgement content "
              f"({tuple(CONTENT_FIELDS)}).")

        deduped_rows, n_removed = dedupe_rows(rows, duplicate_groups)

        # Semantic check: every row that survives must be byte-identical to some row
        # in the original file (we only ever drop rows, never mutate one).
        before_by_id = {id(r): r for r in before_snapshot}  # not usable across copy; verify by content instead
        original_as_json = [json.dumps(r, sort_keys=True, ensure_ascii=False) for r in before_snapshot]
        for r in deduped_rows:
            assert json.dumps(r, sort_keys=True, ensure_ascii=False) in original_as_json, \
                "a surviving row does not byte-match any original row -- dedup must only drop rows, never alter them"

        result = {
            'model_alias': args.model_alias, 'method': args.method, 'input_path': judge_path,
            'sha256_before': sha_before, 'total_rows_before': len(rows), 'total_rows_after': len(deduped_rows),
            'duplicate_groups': len(duplicate_groups), 'rows_removed': n_removed,
            'content_verification': 'passed (all duplicate groups byte-identical on '
                                     f'{tuple(CONTENT_FIELDS)})',
            'applied': False, 'sha256_after': None,
            'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        }

        if args.apply:
            atomic_write_jsonl(judge_path, deduped_rows)
            result['applied'] = True
            result['sha256_after'] = dm.sha256_of_file(judge_path)
            print(f"Wrote {len(deduped_rows)} rows (removed {n_removed}).")
        else:
            result['note'] = 'dry run -- no file written'
            print("This was a DRY RUN. Re-run with --apply to actually write the deduped file.")

    out_dir = os.path.join(args.output_path, 'canonical_v2')
    report_path = os.path.join(out_dir, f'experiment3_judgement_dedupe_{args.model_alias}_{args.method}.json')
    with open(report_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {report_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_alias', type=str, required=True)
    parser.add_argument('--method', type=str, required=True)
    parser.add_argument('--output_path', type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    main(args)
