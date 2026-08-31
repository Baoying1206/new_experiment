"""
Synthetic-data tests for scripts/42_repair_validation_instruction_en.py.
Builds fake sampled_prompts.json/benign_validation_80.json and a fake
generation JSONL with instruction_en deliberately set to the WRAPPED text
(reproducing the real bug), then runs the repair script exactly as it will
be invoked. No GPU.

Usage:
  python scripts/audits/audit_instruction_en_repair_dry_run.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(__file__)
REPAIR_SCRIPT = os.path.join(SCRIPT_DIR, '..', '42_repair_validation_instruction_en.py')


def run_repair(output_path, model_alias, method, apply=False):
    cmd = [sys.executable, REPAIR_SCRIPT, '--model_alias', model_alias, '--method', method,
           '--output_path', output_path]
    if apply:
        cmd.append('--apply')
    return subprocess.run(cmd, capture_output=True, text=True)


def main():
    tmpdir = tempfile.mkdtemp()
    try:
        data_dir = os.path.join(tmpdir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        # Monkeypatch-free approach: the repair script hardcodes paths relative to
        # its own SCRIPT_DIR/../data -- so instead we symlink a fake output_path
        # AND point the repair script's ground-truth files at the real repo's
        # actual data/sampled_prompts.json + benign_validation_80.json, using
        # REAL instruction_ids so the test exercises the real lookup path.
        import importlib.util
        spec = importlib.util.spec_from_file_location('repair42', REPAIR_SCRIPT)
        repair_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(repair_mod)

        harmful_truth, benign_truth = repair_mod.load_ground_truth()
        real_harmful_id = next(iter(harmful_truth))
        real_benign_id = next(iter(benign_truth))
        true_harmful_text = harmful_truth[real_harmful_id]
        true_benign_text = benign_truth[real_benign_id]

        out_dir = os.path.join(tmpdir, 'canonical_v2')
        os.makedirs(out_dir, exist_ok=True)
        gen_path = os.path.join(out_dir, 'experiment3_validation_generations_FakeModel_global.jsonl')

        def make_row(rid, instruction_id, boh, wrong_instruction_en):
            return {
                'record_key': rid, 'model': 'FakeModel', 'split': 'validation',
                'instruction_id': instruction_id, 'benign_or_harmful': boh,
                'template': 'persona_roleplay', 'method': 'global', 'alpha': 1.0,
                'direction_config_hash': 'DCH', 'generation_config_hash': 'GCH',
                'response': f'resp for {rid}', 'generation_tokens': '1 2 3',
                'generation_length': 3, 'stop_reason': 'eos', 'prompt_token_count': 10,
                'instruction_en': wrong_instruction_en,  # WRAPPED text -- the bug
            }

        rows = [
            make_row('k1', real_harmful_id, 'harmful', 'You are now an AI... ' + true_harmful_text),  # wrong
            make_row('k2', real_benign_id, 'benign', 'You are now an AI... ' + true_benign_text),      # wrong
            make_row('k3', real_harmful_id, 'harmful', true_harmful_text),  # ALREADY correct (different template/alpha row)
        ]
        with open(gen_path, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')

        # ---- Test 1: dry run writes nothing ----
        with open(gen_path, 'rb') as f:
            original_bytes = f.read()
        r = run_repair(tmpdir, 'FakeModel', 'global', apply=False)
        assert r.returncode == 0, f"dry run should succeed: {r.stderr}"
        with open(gen_path, 'rb') as f:
            assert f.read() == original_bytes, "dry run modified the file!"
        assert 'To fix: 2' in r.stdout, r.stdout
        assert 'Already correct: 1' in r.stdout, r.stdout
        print("Test 1 PASSED: dry run wrote nothing, correctly identified 2 wrong / 1 already-correct.")

        # ---- Test 2: apply fixes exactly the 2 wrong rows ----
        r2 = run_repair(tmpdir, 'FakeModel', 'global', apply=True)
        assert r2.returncode == 0, f"apply should succeed: {r2.stderr}"
        with open(gen_path, encoding='utf-8') as f:
            fixed_rows = [json.loads(l) for l in f]
        by_key = {row['record_key']: row for row in fixed_rows}
        assert by_key['k1']['instruction_en'] == true_harmful_text
        assert by_key['k2']['instruction_en'] == true_benign_text
        assert by_key['k3']['instruction_en'] == true_harmful_text  # unchanged, was already correct
        print("Test 2 PASSED: apply corrected exactly the 2 wrong rows to the true plain text; "
              "the already-correct row is untouched.")

        # ---- Test 3: non-instruction_en fields are 100% unchanged ----
        original_by_key = {row['record_key']: row for row in rows}
        for key, fixed in by_key.items():
            orig = original_by_key[key]
            o2 = {k: v for k, v in orig.items() if k != 'instruction_en'}
            f2 = {k: v for k, v in fixed.items() if k != 'instruction_en'}
            assert o2 == f2, f"non-instruction_en field changed for {key}"
        print("Test 3 PASSED: all non-instruction_en fields verified identical before/after.")

        # ---- Test 4: second apply is idempotent (0 changes) ----
        r3 = run_repair(tmpdir, 'FakeModel', 'global', apply=True)
        assert r3.returncode == 0
        assert 'To fix: 0' in r3.stdout, r3.stdout
        assert 'Already correct: 3' in r3.stdout, r3.stdout
        print("Test 4 PASSED: second apply is idempotent (0 rows need fixing, all 3 already correct).")

        print()
        print("ALL INSTRUCTION_EN REPAIR TESTS PASSED.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
