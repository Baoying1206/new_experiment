"""
CPU-only smoke test: runs audit_source_overlap.py end-to-end against the
real local ployrefuse_Enhanced/ data and checks its outputs have the
expected shape and known findings -- catches silent regressions in the
audit logic itself (e.g. if the local data changes, or normalisation logic
breaks). Does NOT require a model, tokenizer, or GPU.

Does not check audit_token_positions.py or layer_selection_leakage.md's
content (the former needs a real tokenizer, the latter is static
documentation, not a script) -- only confirms audit_token_positions.py
imports cleanly (transformers may or may not be installed locally; if not,
this is skipped rather than failed).

Run: python scripts/audits/test_smoke.py
"""
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))


def test_source_overlap_audit_runs_and_produces_expected_shape():
    subprocess.run([sys.executable, os.path.join(SCRIPT_DIR, 'audit_source_overlap.py')],
                    cwd=ROOT, check=True)
    out_path = os.path.join(ROOT, 'output', 'audits', 'axis_source_overlap.json')
    assert os.path.exists(out_path)
    with open(out_path) as f:
        data = json.load(f)

    assert 'per_language' in data and 'en' in data['per_language']
    en = data['per_language']['en']
    assert en['provenance_status'] == 'unknown_missing_local_data', (
        "expected English to still be missing local train/val data -- if this "
        "now says 'checked_locally', the missing files have been added; update "
        "layer_selection_leakage.md / EXPERIMENT_REDUCTION_PLAN.md / the audit "
        "report accordingly instead of just letting this test go stale"
    )

    zh = data['per_language']['zh']
    assert zh['provenance_status'] == 'checked_locally'
    assert zh['overlap_by_normalised_text']['harmful_train_vs_harmful_test'] == 0

    prov = data['sampled_prompts_provenance']
    assert prov['sampled_prompts_is_subset_of_harmful_test_en'] is True
    assert prov['overlap_count'] == 562

    md_path = os.path.join(ROOT, 'output', 'audits', 'axis_source_overlap.md')
    assert os.path.exists(md_path)


def test_token_positions_module_importable():
    sys.path.insert(0, os.path.join(ROOT, 'scripts', 'utils'))
    import token_positions  # noqa: F401
    assert hasattr(token_positions, 'get_instruction_end_position')
    assert hasattr(token_positions, 'get_post_instruction_position')


def test_audit_token_positions_script_importable_or_skips_cleanly():
    # transformers may not be installed locally -- that's fine, this only
    # checks the script's own logic (arg parsing, sample list) is not broken,
    # by importing it in isolation and checking SAMPLE_INSTRUCTIONS exists.
    sys.path.insert(0, SCRIPT_DIR)
    try:
        import audit_token_positions
    except ImportError as e:
        print(f"SKIP (expected without cluster env): {e}")
        return
    assert len(audit_token_positions.SAMPLE_INSTRUCTIONS) >= 3


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_')]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
        except subprocess.CalledProcessError as e:
            failed += 1
            print(f"FAIL {t.__name__}: subprocess exited {e.returncode}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
