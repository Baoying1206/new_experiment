"""CPU-only unit tests for direction_metadata.py's schema enforcement.

Updated 2026-09-04 (R/H rebuild, artifact-lineage audit fallout):
build_direction_metadata() deliberately no longer returns a directly-savable
dict -- it only builds the LOGICAL fields (knowable before any tensor
exists); the three TENSOR_FIELDS (sha256/shape/dtype) are filled in only by
save_direction_atomic(), which is now the sole sanctioned write path. See
scripts/audits/audit_rh_rebuild_dry_run.py for the fuller synthetic-data
test suite covering save_direction_atomic/verify_direction_file/
save_delta_atomic/verify_delta_file -- this file keeps the original,
narrower schema-enforcement tests, updated for the new contract."""
import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(__file__))
from direction_metadata import (
    build_direction_metadata, save_direction_metadata, load_direction_metadata,
    save_direction_atomic, verify_direction_file,
    LOGICAL_FIELDS, REQUIRED_FIELDS,
)


def _sample_meta(**overrides):
    kwargs = dict(
        direction_type='refusal_direction', model='Qwen2.5-7B-Instruct',
        model_revision='abc123', tokenizer_revision='abc123',
        chat_template_hash='deadbeef0000', semantic_position='t_post', layer=18,
        source_partition='direction_ids', source_ids=['p002', 'p003', 'p004'],
        construction_contrast='harmful_mean_minus_harmless_mean', random_seed=0,
        git_commit='testcommit',
    )
    kwargs.update(overrides)
    return build_direction_metadata(**kwargs)


def test_all_required_fields_present():
    # build_direction_metadata() only produces the LOGICAL fields now (tensor
    # fields aren't knowable until save_direction_atomic() computes them from
    # a real tensor) -- this is intentional, not a regression. See the module
    # docstring update above.
    meta = _sample_meta()
    for f in LOGICAL_FIELDS:
        assert f in meta, f"missing {f}"
    for f in ('tensor_sha256', 'tensor_shape', 'tensor_dtype'):
        assert f not in meta, (
            f"build_direction_metadata() should NOT produce {f} -- that would mean it's "
            f"claiming tensor properties before any tensor was ever given to it"
        )


def test_rejects_bad_direction_type():
    try:
        _sample_meta(direction_type='not_a_real_type')
        assert False
    except AssertionError:
        pass


def test_rejects_bad_semantic_position():
    try:
        _sample_meta(semantic_position='last_token')
        assert False
    except AssertionError:
        pass


def test_source_ids_hash_deterministic_regardless_of_order():
    m1 = _sample_meta(source_ids=['p002', 'p003', 'p004'])
    m2 = _sample_meta(source_ids=['p004', 'p002', 'p003'])
    assert m1['source_ids_hash'] == m2['source_ids_hash']


def test_source_ids_hash_differs_for_different_sets():
    m1 = _sample_meta(source_ids=['p002', 'p003'])
    m2 = _sample_meta(source_ids=['p002', 'p003', 'p004'])
    assert m1['source_ids_hash'] != m2['source_ids_hash']


def test_save_and_load_roundtrip():
    # Now goes through the sanctioned atomic path: a real tensor is required,
    # since tensor_sha256/shape/dtype are computed from it, not hand-supplied.
    meta = _sample_meta()
    tensor = torch.randn(3, 5)
    with tempfile.TemporaryDirectory() as d:
        pt_path = os.path.join(d, 'refusal_dir_en.pt')
        written = save_direction_atomic(tensor, meta, pt_path)
        loaded_tensor, loaded_meta = verify_direction_file(pt_path)
        assert torch.allclose(loaded_tensor, tensor)
        assert loaded_meta == written


def test_save_refuses_incomplete_metadata():
    incomplete = {'direction_type': 'refusal_direction'}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'bad.json')
        try:
            save_direction_metadata(incomplete, path)
            assert False
        except ValueError:
            pass
        assert not os.path.exists(path)


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
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
