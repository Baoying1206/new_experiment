"""CPU-only unit tests for direction_metadata.py's schema enforcement."""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
from direction_metadata import (
    build_direction_metadata, save_direction_metadata, load_direction_metadata,
    REQUIRED_FIELDS,
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
    meta = _sample_meta()
    for f in REQUIRED_FIELDS:
        assert f in meta, f"missing {f}"


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
    meta = _sample_meta()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'refusal_dir_en.json')
        save_direction_metadata(meta, path)
        loaded = load_direction_metadata(path)
        assert loaded == meta


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
