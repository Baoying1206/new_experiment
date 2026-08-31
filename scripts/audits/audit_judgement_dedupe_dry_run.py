"""
Synthetic-data tests for scripts/43_dedupe_validation_judgements.py.
Reproduces the real Llama x global bug (54 duplicate judge_cache_key groups,
84 extra rows, all byte-identical content) at small scale, plus the
content-mismatch refusal path and the no-duplicates/idempotent no-op path.

No GPU, no torch.

Usage:
  python scripts/audits/audit_judgement_dedupe_dry_run.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(__file__)
DEDUPE_SCRIPT = os.path.join(SCRIPT_DIR, '..', '43_dedupe_validation_judgements.py')


def run_dedupe(output_path, model_alias, method, apply=False):
    cmd = [sys.executable, DEDUPE_SCRIPT, '--model_alias', model_alias, '--method', method,
           '--output_path', output_path]
    if apply:
        cmd.append('--apply')
    return subprocess.run(cmd, capture_output=True, text=True)


def write_rows(path, rows):
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def make_row(key, iid, template, alpha, **overrides):
    r = {
        'judge_cache_key': key, 'raw_judge_output': 'Harmful request: yes\nResponse refusal: no\nHarmful response: yes',
        'request_harmful': 1, 'refusal': 0, 'response_harmful': 1, 'parse_success': True,
        'parse_error_reason': None, 'instruction_id': iid, 'template': template, 'method': 'global',
        'benign_or_harmful': 'harmful', 'alpha': alpha, 'model': 'FakeModel',
    }
    r.update(overrides)
    return r


def main():
    tmpdir = tempfile.mkdtemp()
    try:
        out_dir = os.path.join(tmpdir, 'canonical_v2')
        os.makedirs(out_dir, exist_ok=True)
        judge_path = os.path.join(out_dir, 'experiment3_validation_judgements_FakeModel_global.jsonl')

        # ---- Test 1: no duplicates -> no-op ----
        rows_no_dup = [
            make_row('k1', 'p1', 'persona_roleplay', 1.0),
            make_row('k2', 'p2', 'prefix_injection', 0.5),
        ]
        write_rows(judge_path, rows_no_dup)
        r1 = run_dedupe(tmpdir, 'FakeModel', 'global', apply=True)
        assert r1.returncode == 0, r1.stderr
        assert 'Duplicated keys: 0' in r1.stdout, r1.stdout
        assert 'nothing to do' in r1.stdout, r1.stdout
        with open(judge_path, encoding='utf-8') as f:
            assert len([l for l in f if l.strip()]) == 2
        print("Test 1 PASSED: no duplicates -> file untouched, reported as no-op.")

        # ---- Test 2: dry run with duplicates writes nothing ----
        rows_dup = [
            make_row('kA', 'p1', 'persona_roleplay', 1.0),
            make_row('kA', 'p2', 'prefix_injection', 0.5),  # same key, identical content, different template/alpha
            make_row('kB', 'p3', 'payload_splitting', 1.5),
        ]
        write_rows(judge_path, rows_dup)
        with open(judge_path, 'rb') as f:
            original_bytes = f.read()
        r2 = run_dedupe(tmpdir, 'FakeModel', 'global', apply=False)
        assert r2.returncode == 0, r2.stderr
        with open(judge_path, 'rb') as f:
            assert f.read() == original_bytes, "dry run modified the file!"
        assert 'Duplicated keys: 1' in r2.stdout, r2.stdout
        assert 'Extra rows to remove: 1' in r2.stdout, r2.stdout
        assert 'byte-identical' in r2.stdout, r2.stdout
        print("Test 2 PASSED: dry run wrote nothing, correctly identified 1 duplicated key / 1 extra row.")

        # ---- Test 3: apply removes exactly the extra row, keeps content ----
        r3 = run_dedupe(tmpdir, 'FakeModel', 'global', apply=True)
        assert r3.returncode == 0, r3.stderr
        with open(judge_path, encoding='utf-8') as f:
            deduped = [json.loads(l) for l in f]
        assert len(deduped) == 2, deduped
        keys = sorted(r['judge_cache_key'] for r in deduped)
        assert keys == ['kA', 'kB'], keys
        print("Test 3 PASSED: apply removed exactly the 1 duplicate row, kept 1 row per unique key.")

        # ---- Test 4: second apply is idempotent (0 duplicates left) ----
        r4 = run_dedupe(tmpdir, 'FakeModel', 'global', apply=True)
        assert r4.returncode == 0, r4.stderr
        assert 'Duplicated keys: 0' in r4.stdout, r4.stdout
        print("Test 4 PASSED: second apply is idempotent (no duplicates left).")

        # ---- Test 5: differing content among a duplicate key -> refuses, writes nothing ----
        rows_mismatch = [
            make_row('kC', 'p4', 'persona_roleplay', 1.0, response_harmful=1),
            make_row('kC', 'p5', 'prefix_injection', 0.5, response_harmful=0),  # same key, DIFFERENT content
        ]
        write_rows(judge_path, rows_mismatch)
        with open(judge_path, 'rb') as f:
            original_bytes5 = f.read()
        r5 = run_dedupe(tmpdir, 'FakeModel', 'global', apply=True)
        assert r5.returncode != 0, "expected non-zero exit on content mismatch"
        assert 'DIFFERING judgement content' in (r5.stdout + r5.stderr), r5.stdout + r5.stderr
        with open(judge_path, 'rb') as f:
            assert f.read() == original_bytes5, "file was modified despite a content mismatch -- must refuse cleanly"
        print("Test 5 PASSED: a duplicate key with differing judgement content is refused (no write), "
              "distinguishing the known same-batch-write bug from genuine non-determinism.")

        print()
        print("ALL JUDGEMENT DEDUPE TESTS PASSED.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
