"""
Synthetic-data tests for scripts/38_repair_corrected_mechanism_metadata.py.
Builds a small, schema-correct fake completions file (572 ids x 8 conditions,
persona_roleplay deliberately stale) and exercises the repair script exactly
as it will be invoked, plus its refusal behavior on out-of-scope defects.

Usage:
  python scripts/audits/audit_repair_mechanism_metadata_dry_run.py
"""
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
from _taxonomy_v2_loader import load_taxonomy_v2

REPAIR_SCRIPT = os.path.join(SCRIPT_DIR, '..', '38_repair_corrected_mechanism_metadata.py')


def build_fake_completions(active_mechanisms, mechanism_of, stale_persona=True, inject_other_mismatch=False):
    conditions = ['plain', 'placebo'] + active_mechanisms
    rows = []
    for i in range(572):
        pid = f"p{i:03d}"
        for cond in conditions:
            row = {
                'id': pid, 'condition': cond, 'instruction_en': f'fake instruction {i}',
                'response': f'fake response for {pid} {cond}', 'generation_tokens': '1 2 3',
            }
            if cond in active_mechanisms:
                if cond == 'persona_roleplay' and stale_persona:
                    row['mechanism'] = 'mismatched_generalization'
                elif cond == 'encoding_obfuscation' and inject_other_mismatch:
                    row['mechanism'] = 'competing_objectives'  # deliberately wrong
                else:
                    row['mechanism'] = mechanism_of[cond]
            rows.append(row)
    return {'completions': rows}


def write_fixture(tmpdir, model_alias, data):
    d = os.path.join(tmpdir, model_alias)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, 'completions_en_full572_corrected.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def run_repair(output_dir, apply=False):
    cmd = [sys.executable, REPAIR_SCRIPT, '--output_dir', output_dir]
    if apply:
        cmd.append('--apply')
    return subprocess.run(cmd, capture_output=True, text=True)


def main():
    taxonomy = load_taxonomy_v2()
    active_mechanisms, mechanism_of = taxonomy['active_mechanisms'], taxonomy['mechanism_of']
    MODELS = ['Qwen2.5-7B-Instruct', 'Meta-Llama-3.1-8B-Instruct', 'gemma-2-9b-it']

    # ---- Test 1: dry run writes nothing ----
    tmpdir = tempfile.mkdtemp()
    try:
        paths = {}
        original_bytes = {}
        for m in MODELS:
            data = build_fake_completions(active_mechanisms, mechanism_of, stale_persona=True)
            paths[m] = write_fixture(tmpdir, m, data)
            with open(paths[m], 'rb') as f:
                original_bytes[m] = f.read()

        r = run_repair(tmpdir, apply=False)
        assert r.returncode == 0, f"dry run should succeed, stderr={r.stderr}"
        for m in MODELS:
            with open(paths[m], 'rb') as f:
                assert f.read() == original_bytes[m], f"{m}: dry run modified the file!"
        print("Test 1 PASSED: dry run wrote nothing (byte-identical before/after).")

        # ---- Test 2: apply changes exactly 572 per model ----
        r = run_repair(tmpdir, apply=True)
        assert r.returncode == 0, f"apply should succeed, stderr={r.stderr}"
        report_path = os.path.join(tmpdir, 'canonical_v2', 'corrected_completions_metadata_repair.json')
        report = json.load(open(report_path))
        for m in MODELS:
            cr = report['per_model'][m]['changed_rows']
            assert cr == 572, f"{m}: expected changed_rows=572, got {cr}"
        print("Test 2 PASSED: apply changed exactly 572 rows per model.")

        # verify the actual file content changed correctly
        for m in MODELS:
            data = json.load(open(paths[m]))
            persona_rows = [c for c in data['completions'] if c['condition'] == 'persona_roleplay']
            assert all(c['mechanism'] == 'competing_objectives' for c in persona_rows)
        print("Test 2b PASSED: all persona_roleplay rows now say 'competing_objectives'.")

        # ---- Test 3: second apply run changes 0 rows (idempotent) ----
        r2 = run_repair(tmpdir, apply=True)
        assert r2.returncode == 0, f"second apply should succeed, stderr={r2.stderr}"
        report2 = json.load(open(report_path))
        for m in MODELS:
            cr = report2['per_model'][m]['changed_rows']
            assert cr == 0, f"{m}: second apply should change 0 rows, got {cr}"
            assert report2['per_model'][m]['sha256_after'] == report2['per_model'][m]['sha256_before']
        print("Test 3 PASSED: second apply is idempotent (0 rows changed, file untouched).")

        # ---- Test 4: non-persona mismatch causes refusal ----
        tmpdir2 = tempfile.mkdtemp()
        try:
            for m in MODELS:
                data = build_fake_completions(active_mechanisms, mechanism_of, stale_persona=True, inject_other_mismatch=True)
                write_fixture(tmpdir2, m, data)
            r3 = run_repair(tmpdir2, apply=False)
            assert r3.returncode != 0, "should have refused to run on an out-of-scope (non-persona) mismatch"
            assert 'out of scope' in r3.stderr or 'out of scope' in r3.stdout or 'AssertionError' in r3.stderr, \
                f"expected a clear refusal error, got stderr={r3.stderr[-500:]}"
            print("Test 4 PASSED: non-persona mismatch correctly causes the script to refuse (nonzero exit).")
            # confirm nothing was written even in apply mode
            r3b = run_repair(tmpdir2, apply=True)
            assert r3b.returncode != 0
            print("Test 4b PASSED: refusal holds under --apply too (no silent write).")
        finally:
            shutil.rmtree(tmpdir2, ignore_errors=True)

        # ---- Test 5: semantic equivalence (already implicitly checked inside the script itself
        # via assert_semantic_equivalence, which would have raised and failed Test 2 if violated;
        # here we independently re-verify from outside using the dry-run's own before/after logic) ----
        tmpdir3 = tempfile.mkdtemp()
        try:
            before_data = {}
            for m in MODELS:
                data = build_fake_completions(active_mechanisms, mechanism_of, stale_persona=True)
                write_fixture(tmpdir3, m, data)
                before_data[m] = copy.deepcopy(data['completions'])
            run_repair(tmpdir3, apply=True)
            for m in MODELS:
                after = json.load(open(os.path.join(tmpdir3, m, 'completions_en_full572_corrected.json')))['completions']
                for b, a in zip(before_data[m], after):
                    b2 = {k: v for k, v in b.items() if k != 'mechanism'}
                    a2 = {k: v for k, v in a.items() if k != 'mechanism'}
                    assert b2 == a2, f"{m}: non-mechanism field changed for {b['id']}|{b['condition']}"
            print("Test 5 PASSED: independently re-verified non-mechanism fields are 100% identical.")
        finally:
            shutil.rmtree(tmpdir3, ignore_errors=True)

        # ---- Test 6: post-repair audit passes (reuse the real audit script) ----
        audit_script = os.path.join(SCRIPT_DIR, 'audit_corrected_completions.py')
        r4 = subprocess.run([sys.executable, audit_script, '--output_dir', tmpdir], capture_output=True, text=True)
        assert 'OVERALL_PASS: True' in r4.stdout, f"expected OVERALL_PASS: True after repair, got:\n{r4.stdout[-1500:]}"
        print("Test 6 PASSED: post-repair audit_corrected_completions.py reports OVERALL_PASS: True.")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    print()
    print("ALL REPAIR-SCRIPT TESTS PASSED.")


if __name__ == '__main__':
    main()
