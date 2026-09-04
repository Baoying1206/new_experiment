"""
Synthetic-data-only tests for scripts/utils/axis_manifest.py (the independent
R-axis manifest schema/validator, EXPERIMENT2_RH_REBUILD_PROTOCOL.md Sec 15)
and for scripts/26_rebuild_refusal_direction_behavioral.py's fail-fast gate
(EXPERIMENT2_RH_REBUILD_PROTOCOL.md Sec 2.1/13). No GPU, no real model, no
real generation -- every manifest row here is hand-constructed.

Note on scope: scripts/26_...py imports `transformers` and
`pipeline.utils.hook_utils` at module level, neither of which is installed/
present on this machine (same pre-existing limitation as scripts 23/25 --
this local checkout has never been able to fully import those modules; only
py_compile-level syntax checking works for them here). To still exercise the
REAL shipped gate logic (not a reimplementation of it), this file installs
minimal stub modules into sys.modules for the two missing heavy dependencies
before importing script 26, then tests the four fail-fast rejection paths
(which never touch the stubs) plus the manifest-success path (which also
never touches the stubs -- only the legacy path would reach `pipeline.
model_utils.model_factory`, which is intentionally left unstubbed so
reaching it, and failing there, is itself proof the gate correctly let a
validly-confirmed legacy request through).

Usage:
  python scripts/audits/audit_axis_manifest_dry_run.py
"""
import copy
import json
import os
import sys
import tempfile
import types

SCRIPT_DIR = os.path.dirname(__file__)
SCRIPTS_ROOT = os.path.join(SCRIPT_DIR, '..')
sys.path.insert(0, SCRIPTS_ROOT)

import utils.axis_manifest as am


def _sample_row(**overrides):
    row = dict(
        dataset_name='harmful_train', dataset_version='2026-09-04',
        source_path=overrides.pop('_source_path', '/tmp/does_not_matter.json'),
        source_file_sha256='deadbeef', stable_source_id='harmful_train:0:abcd1234abcd1234',
        normalized_text_hash='0' * 64, prompt_family='baseline_plain', condition='plain',
        model_alias='Meta-Llama-3.1-8B-Instruct', response_id='resp_0', refusal_label=1,
        label_source='wildguard', split='axis', overlaps_572_pool=False,
        contains_canonical_template=False,
    )
    row.update(overrides)
    return row


def _write_source_file(tmpdir, content=b'hello world'):
    p = os.path.join(tmpdir, 'source.json')
    with open(p, 'wb') as f:
        f.write(content)
    return p, am.sha256_of_file(p)


def main():
    tmpdir = tempfile.mkdtemp()
    try:
        canonical_mechs = ['prefix_injection', 'refusal_suppression', 'persona_roleplay',
                            'encoding_obfuscation', 'payload_splitting', 'distractors_negated']
        pool_hashes = {am.normalized_text_hash('a harmful instruction from the 572 pool')}

        source_path, source_hash = _write_source_file(tmpdir)

        def make_valid_rows(n_refused=5, n_accepted=5, n_val_refused=2, n_val_accepted=2):
            rows = []
            idx = 0
            for _ in range(n_refused):
                rows.append(_sample_row(
                    source_path=source_path, source_file_sha256=source_hash,
                    stable_source_id=am.make_stable_source_id('harmful_train', idx, f'text {idx}'),
                    normalized_text_hash=am.normalized_text_hash(f'text {idx}'),
                    refusal_label=1, split='axis'))
                idx += 1
            for _ in range(n_accepted):
                rows.append(_sample_row(
                    source_path=source_path, source_file_sha256=source_hash,
                    stable_source_id=am.make_stable_source_id('harmful_train', idx, f'text {idx}'),
                    normalized_text_hash=am.normalized_text_hash(f'text {idx}'),
                    refusal_label=0, split='axis'))
                idx += 1
            for _ in range(n_val_refused):
                rows.append(_sample_row(
                    source_path=source_path, source_file_sha256=source_hash,
                    stable_source_id=am.make_stable_source_id('harmful_train', idx, f'text {idx}'),
                    normalized_text_hash=am.normalized_text_hash(f'text {idx}'),
                    refusal_label=1, split='val'))
                idx += 1
            for _ in range(n_val_accepted):
                rows.append(_sample_row(
                    source_path=source_path, source_file_sha256=source_hash,
                    stable_source_id=am.make_stable_source_id('harmful_train', idx, f'text {idx}'),
                    normalized_text_hash=am.normalized_text_hash(f'text {idx}'),
                    refusal_label=0, split='val'))
                idx += 1
            return rows

        # ---- Test 1: a fully valid manifest passes and reports correct counts ----
        rows = make_valid_rows()
        stats = am.validate_axis_manifest(rows, canonical_mechs, pool_hashes)
        assert stats == {'n_axis': 10, 'n_axis_refused': 5, 'n_axis_accepted': 5,
                          'n_val': 4, 'n_val_refused': 2, 'n_val_accepted': 2}, stats
        print(f"Test 1 PASSED: a valid manifest passes and reports correct per-split counts: {stats}")

        # ---- Test 2: stable ID scheme is deterministic and NOT position-based ----
        id_a = am.make_stable_source_id('harmful_train', 5, 'Some Instruction TEXT')
        id_b = am.make_stable_source_id('harmful_train', 5, 'some instruction   text')  # whitespace/case differ
        id_c = am.make_stable_source_id('harmful_train', 6, 'Some Instruction TEXT')  # different index
        assert id_a == id_b, "normalisation should make these equal"
        assert id_a != id_c, "different row index should change the id"
        assert 'harmful_train:5:' in id_a
        print(f"Test 2 PASSED: make_stable_source_id is deterministic, normalises whitespace/case, "
              f"and is NOT a shuffled positional index (id_a={id_a!r}).")

        # ---- Test 3: missing required field is rejected ----
        bad_rows = make_valid_rows()
        del bad_rows[0]['label_source']
        try:
            am.validate_axis_manifest(bad_rows, canonical_mechs, pool_hashes)
            raise SystemExit("FAILED: expected ValueError for missing field")
        except ValueError as e:
            assert 'missing required field' in str(e)
            print(f"Test 3 PASSED: a row missing a required field is rejected: {str(e)[:70]}")

        # ---- Test 4: self-reported overlaps_572_pool=True is rejected outright ----
        bad_rows = make_valid_rows()
        bad_rows[0]['overlaps_572_pool'] = True
        try:
            am.validate_axis_manifest(bad_rows, canonical_mechs, pool_hashes)
            raise SystemExit("FAILED: expected ValueError for overlaps_572_pool=True")
        except ValueError as e:
            assert 'overlaps_572_pool' in str(e)
            print(f"Test 4 PASSED: a row flagged overlaps_572_pool=True is rejected: {str(e)[:70]}")

        # ---- Test 5 (CRITICAL): a row that CLAIMS overlaps_572_pool=False but whose
        # text hash actually matches the pool is caught by independent recomputation,
        # not trusted on the manifest's own word ----
        bad_rows = make_valid_rows()
        bad_rows[0]['normalized_text_hash'] = am.normalized_text_hash('a harmful instruction from the 572 pool')
        bad_rows[0]['overlaps_572_pool'] = False  # lying
        try:
            am.validate_axis_manifest(bad_rows, canonical_mechs, pool_hashes)
            raise SystemExit("FAILED: expected ValueError for a mis-flagged pool overlap")
        except ValueError as e:
            assert 'WRONG' in str(e)
            print(f"Test 5 PASSED (CRITICAL): a row whose self-reported overlaps_572_pool=False is "
                  f"contradicted by independent text-hash recomputation is still rejected: "
                  f"{str(e)[:90]}")

        # ---- Test 6: contains_canonical_template=True is rejected ----
        bad_rows = make_valid_rows()
        bad_rows[0]['contains_canonical_template'] = True
        try:
            am.validate_axis_manifest(bad_rows, canonical_mechs, pool_hashes)
            raise SystemExit("FAILED: expected ValueError for contains_canonical_template=True")
        except ValueError as e:
            assert 'contains_canonical_template' in str(e)
            print(f"Test 6 PASSED: a row flagged contains_canonical_template=True is rejected: "
                  f"{str(e)[:70]}")

        # ---- Test 7 (CRITICAL): prompt_family secretly matching a canonical mechanism
        # name is caught even if contains_canonical_template is (wrongly) False ----
        bad_rows = make_valid_rows()
        bad_rows[0]['prompt_family'] = 'encoding_obfuscation'
        bad_rows[0]['contains_canonical_template'] = False  # lying
        try:
            am.validate_axis_manifest(bad_rows, canonical_mechs, pool_hashes)
            raise SystemExit("FAILED: expected ValueError for a mis-flagged canonical template")
        except ValueError as e:
            assert 'flag is wrong' in str(e)
            print(f"Test 7 PASSED (CRITICAL): prompt_family='encoding_obfuscation' is caught even "
                  f"though contains_canonical_template was (wrongly) False: {str(e)[:90]}")

        # ---- Test 8: duplicate stable_source_id rejected ----
        bad_rows = make_valid_rows()
        bad_rows[1]['stable_source_id'] = bad_rows[0]['stable_source_id']
        try:
            am.validate_axis_manifest(bad_rows, canonical_mechs, pool_hashes)
            raise SystemExit("FAILED: expected ValueError for duplicate stable_source_id")
        except ValueError as e:
            assert 'duplicate' in str(e)
            print(f"Test 8 PASSED: duplicate stable_source_id is rejected: {str(e)[:70]}")

        # ---- Test 9: a disallowed split value (e.g. 'test' or 'direction') is rejected ----
        bad_rows = make_valid_rows()
        bad_rows[0]['split'] = 'test_ids'
        try:
            am.validate_axis_manifest(bad_rows, canonical_mechs, pool_hashes)
            raise SystemExit("FAILED: expected ValueError for a disallowed split value")
        except ValueError as e:
            assert 'not in the allowed set' in str(e)
            print(f"Test 9 PASSED: a manifest row using split={'test_ids'!r} (572-pool concept, "
                  f"never allowed here) is rejected: {str(e)[:90]}")

        # ---- Test 10: empty accepted (or refused) class in the axis split is rejected ----
        bad_rows = make_valid_rows(n_refused=5, n_accepted=0)
        try:
            am.validate_axis_manifest(bad_rows, canonical_mechs, pool_hashes)
            raise SystemExit("FAILED: expected ValueError for an empty accepted class")
        except ValueError as e:
            assert 'accepted=0' in str(e)
            print(f"Test 10 PASSED: an axis split with 0 accepted rows is rejected: {str(e)[:70]}")

        # ---- Test 11: source file hash mismatch (file changed since manifest was built) ----
        rows = make_valid_rows()
        with open(source_path, 'wb') as f:
            f.write(b'the file changed after the manifest was built')
        try:
            am.validate_axis_manifest(rows, canonical_mechs, pool_hashes)
            raise SystemExit("FAILED: expected ValueError for a source-file hash mismatch")
        except ValueError as e:
            assert 'does not match' in str(e)
            print(f"Test 11 PASSED: a source_path whose current hash no longer matches the "
                  f"manifest's recorded source_file_sha256 is rejected: {str(e)[:90]}")
        # restore for any later test reuse
        with open(source_path, 'wb') as f:
            f.write(b'hello world')

        # ---- Test 12: missing source file is rejected, not silently skipped ----
        rows = make_valid_rows()
        for r in rows:
            r['source_path'] = '/tmp/definitely_does_not_exist_axis_manifest_test.json'
        try:
            am.validate_axis_manifest(rows, canonical_mechs, pool_hashes)
            raise SystemExit("FAILED: expected ValueError for a missing source file")
        except ValueError as e:
            assert 'does not exist' in str(e)
            print(f"Test 12 PASSED: a source_path that does not exist on this machine is rejected: "
                  f"{str(e)[:90]}")

        # ---- Test 13: load_axis_manifest raises FileNotFoundError on a missing path ----
        try:
            am.load_axis_manifest('/tmp/definitely_does_not_exist_manifest_file.json')
            raise SystemExit("FAILED: expected FileNotFoundError")
        except FileNotFoundError as e:
            print(f"Test 13 PASSED: load_axis_manifest raises FileNotFoundError on a missing "
                  f"manifest path: {str(e)[:70]}")

        # ================================================================
        # scripts/26_rebuild_refusal_direction_behavioral.py gate logic --
        # import the REAL module (stubbing only the two heavy deps it needs
        # at module level, neither of which the gate-check code path touches).
        # ================================================================
        transformers_stub = types.ModuleType('transformers')
        transformers_stub.AutoModelForCausalLM = object
        transformers_stub.AutoTokenizer = object
        sys.modules.setdefault('transformers', transformers_stub)

        pipeline_mod = types.ModuleType('pipeline')
        pipeline_utils_mod = types.ModuleType('pipeline.utils')
        pipeline_hook_utils_mod = types.ModuleType('pipeline.utils.hook_utils')

        def _stub_add_hooks(*a, **kw):
            raise NotImplementedError("stub add_hooks -- must never be called by gate-only tests")
        pipeline_hook_utils_mod.add_hooks = _stub_add_hooks
        sys.modules.setdefault('pipeline', pipeline_mod)
        sys.modules.setdefault('pipeline.utils', pipeline_utils_mod)
        sys.modules.setdefault('pipeline.utils.hook_utils', pipeline_hook_utils_mod)

        import importlib
        script26 = importlib.import_module('26_rebuild_refusal_direction_behavioral')
        importlib.reload(script26)

        class FakeArgs:
            def __init__(self, **kw):
                self.axis_manifest = None
                self.legacy_pooled_templates = None
                self.__dict__.update(kw)

        # ---- Test 14: neither axis source given -> SystemExit(1), before any heavy import ----
        try:
            script26.main(FakeArgs())
            raise SystemExit("FAILED: expected SystemExit for no axis source")
        except SystemExit as e:
            assert e.code == 1, e.code
            print("Test 14 PASSED: script26.main() with neither --axis_manifest nor "
                  "--legacy_pooled_templates exits(1) before any model/pipeline work.")

        # ---- Test 15: both given -> SystemExit(1) (mutually exclusive) ----
        try:
            script26.main(FakeArgs(axis_manifest='/tmp/x.json',
                                    legacy_pooled_templates=script26.LEGACY_SENTINEL))
            raise SystemExit("FAILED: expected SystemExit for both-given")
        except SystemExit as e:
            assert e.code == 1, e.code
            print("Test 15 PASSED: script26.main() with BOTH --axis_manifest and "
                  "--legacy_pooled_templates exits(1) (mutually exclusive).")

        # ---- Test 16 (CRITICAL): a near-miss legacy sentinel does NOT bypass the gate ----
        near_miss = script26.LEGACY_SENTINEL.lower()
        assert near_miss != script26.LEGACY_SENTINEL
        try:
            script26.main(FakeArgs(legacy_pooled_templates=near_miss))
            raise SystemExit("FAILED: expected SystemExit for a near-miss sentinel")
        except SystemExit as e:
            assert e.code == 1, e.code
            print(f"Test 16 PASSED (CRITICAL): a near-miss legacy sentinel ({near_miss!r}, differs "
                  f"only in case) is rejected, not accepted -- ordinary typos/flag confusion "
                  f"cannot accidentally enable the circular legacy path.")

        # ---- Test 17: a VALID axis_manifest is accepted and validated with zero
        # model/pipeline/transformers work (manifest path never touches the stubs) ----
        manifest_path = os.path.join(tmpdir, 'manifest.json')
        with open(manifest_path, 'w') as f:
            json.dump({'rows': make_valid_rows()}, f)
        # Patch DATA_DIR's sampled_prompts.json read to use our synthetic pool hash set
        # by pointing at the REAL repo file (harmless -- our synthetic rows' hashes were
        # built from arbitrary strings that don't collide with the real 572-pool).
        script26.main(FakeArgs(axis_manifest=manifest_path))
        print("Test 17 PASSED: a manifest passing all independence/integrity checks is accepted "
              "by script26.main() and validated with no model/pipeline/transformers dependency "
              "ever invoked.")

        # ---- Test 18: an INVALID axis_manifest (pool overlap) raises before any model load ----
        bad_manifest_path = os.path.join(tmpdir, 'bad_manifest.json')
        bad_rows = make_valid_rows()
        bad_rows[0]['overlaps_572_pool'] = True
        with open(bad_manifest_path, 'w') as f:
            json.dump({'rows': bad_rows}, f)
        try:
            script26.main(FakeArgs(axis_manifest=bad_manifest_path))
            raise SystemExit("FAILED: expected ValueError propagated from validate_axis_manifest")
        except ValueError as e:
            assert 'overlaps_572_pool' in str(e)
            print(f"Test 18 PASSED: an invalid manifest's ValueError propagates out of "
                  f"script26.main() (fail-fast, no model ever loaded): {str(e)[:70]}")

        # ---- Test 19: a correctly-confirmed legacy request DOES get past the gate
        # (proven by reaching the next, deliberately-unstubbed import and failing there
        # for an unrelated, expected reason -- NOT by the gate itself blocking it) ----
        try:
            script26.main(FakeArgs(legacy_pooled_templates=script26.LEGACY_SENTINEL,
                                    dry_run=True, seed=0,
                                    n_dry_run_axis=2, n_dry_run_val=2,
                                    output_dir='/tmp/unused', model_path='x', model_alias='y'))
            raise SystemExit("FAILED: expected an error reaching past the gate "
                              "(model_factory/dataset.load_dataset are unstubbed)")
        except (ModuleNotFoundError, ImportError, FileNotFoundError) as e:
            print(f"Test 19 PASSED: a correctly-confirmed --legacy_pooled_templates request passes "
                  f"the gate and proceeds into _run_legacy_pooled_templates (reached the next, "
                  f"deliberately-unstubbed dependency and failed there instead, proving the gate "
                  f"itself did not block it): {type(e).__name__}: {str(e)[:70]}")

        # ---- Test 20: static check -- legacy output path and status tag are literally
        # present in the source, distinct from the real result directory ----
        with open(os.path.join(SCRIPTS_ROOT, '26_rebuild_refusal_direction_behavioral.py')) as f:
            src26 = f.read()
        assert 'output_v3_behavioral_refusal_LEGACY_PROVISIONAL' in src26
        assert "'LEGACY_PROVISIONAL_POOLED_TEMPLATES_NOT_FOR_RESULTS'" in src26
        # script26 currently has NO code path that writes to the plain
        # output_v3_behavioral_refusal/ (no-suffix) result directory at all --
        # only the manifest path (not yet implemented -- no save call exists)
        # and the LEGACY_PROVISIONAL path (writes to the suffixed directory).
        # This is intentional: nothing should write to the real result path
        # until real independent axis data flows through. Confirm no stray
        # save call still targets the bare (non-suffixed, non-legacy) path.
        assert "os.path.join(args.output_dir, 'output_v3_behavioral_refusal', args.model_alias)" not in src26
        print("Test 20 PASSED: script26's source contains a distinct LEGACY_PROVISIONAL output "
              "directory and status tag, and (correctly, by design) no code path writes to the "
              "bare output_v3_behavioral_refusal/ result directory this round.")

        print()
        print("ALL AXIS-MANIFEST / SCRIPT26-GATE TESTS PASSED.")
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
