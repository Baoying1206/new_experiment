"""
Metadata schema for direction files going forward (Decision 1/2/3 rebuild).
Every new direction .pt must be saved alongside a same-named .json built by
build_direction_metadata() below -- a direction tensor with no accompanying
metadata is not acceptable for any new extraction from this point on.

Usage:
    meta = build_direction_metadata(
        direction_type='refusal_direction', model='Qwen2.5-7B-Instruct',
        model_revision=..., tokenizer_revision=..., chat_template_hash=...,
        semantic_position='t_post', layer=18, source_partition='direction_ids',
        source_ids=[...], construction_contrast='harmful_mean_minus_harmless_mean',
        git_commit=..., random_seed=0,
    )
    torch.save(direction.cpu(), 'refusal_dir_en.pt')
    save_direction_metadata(meta, 'refusal_dir_en.json')
"""
import hashlib
import json
import subprocess


REQUIRED_FIELDS = [
    'direction_type', 'model', 'model_revision', 'tokenizer_revision',
    'chat_template_hash', 'semantic_position', 'layer', 'source_partition',
    'source_ids_hash', 'sample_count', 'construction_contrast', 'git_commit',
    'random_seed',
]


def _ids_hash(source_ids):
    joined = ','.join(sorted(str(i) for i in source_ids))
    return hashlib.sha256(joined.encode('utf-8')).hexdigest()[:16]


def current_git_commit(repo_dir='.'):
    try:
        out = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=repo_dir,
                              capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return 'unknown'


def build_direction_metadata(direction_type, model, model_revision, tokenizer_revision,
                              chat_template_hash, semantic_position, layer,
                              source_partition, source_ids, construction_contrast,
                              random_seed, git_commit=None, extra=None):
    assert direction_type in ('refusal_direction', 'harmfulness_direction', 'template_direction')
    assert semantic_position in ('t_inst', 't_post')
    assert source_partition in (
        'direction_ids', 'validation_ids', 'test_ids',
        'independent_train', 'crossfit_fold',
    )
    meta = {
        'direction_type': direction_type,
        'model': model,
        'model_revision': model_revision,
        'tokenizer_revision': tokenizer_revision,
        'chat_template_hash': chat_template_hash,
        'semantic_position': semantic_position,
        'layer': layer,
        'source_partition': source_partition,
        'source_ids_hash': _ids_hash(source_ids),
        'sample_count': len(source_ids),
        'construction_contrast': construction_contrast,
        'git_commit': git_commit or current_git_commit(),
        'random_seed': random_seed,
    }
    if extra:
        meta.update(extra)
    for f in REQUIRED_FIELDS:
        assert f in meta, f"missing required metadata field: {f}"
    return meta


def save_direction_metadata(meta, json_path):
    for f in REQUIRED_FIELDS:
        if f not in meta:
            raise ValueError(f"refusing to save direction metadata missing required field: {f}")
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2)


def load_direction_metadata(json_path):
    with open(json_path) as f:
        meta = json.load(f)
    missing = [f for f in REQUIRED_FIELDS if f not in meta]
    if missing:
        raise ValueError(f"{json_path} is missing required fields: {missing}")
    return meta
