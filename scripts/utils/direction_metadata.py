"""
Metadata schema for direction files (Decision 1/2/3 rebuild, extended for the
EXPERIMENT2_RH_REBUILD_PROTOCOL.md fail-fast/atomicity requirements). Every
new direction .pt must be saved alongside a same-named .json built by
build_direction_metadata() below -- a direction tensor with no accompanying
metadata is not acceptable for any new extraction from this point on, and
(per the rebuild protocol) the reverse is now also enforced: metadata may
never exist without its tensor being verifiably present and hash-matched --
see save_direction_atomic()/verify_direction_file() below, which are the
ONLY sanctioned way to write/read a direction file from this point forward.

Why this file was extended (2026-09-04, artifact-lineage audit): the
previous read/write path (torch.save(...) then save_direction_metadata(...)
as two independent, non-atomic steps) could leave a metadata JSON on disk
whose paired .pt either never existed, was truncated by a killed job, or
silently diverged from what the metadata claims -- there was no mechanism
to detect any of these. save_direction_atomic() closes this: the tensor is
written to a temp path and atomically renamed into place FIRST; only if
that succeeds does the metadata (now including the tensor's own real
SHA-256/shape/dtype) get written, also via temp+atomic-rename. A metadata
file existing is now proof its tensor exists and matches -- never a
placeholder.

Usage (new code path):
    meta = build_direction_metadata(
        direction_type='refusal_direction', model='Qwen2.5-7B-Instruct',
        model_revision=..., tokenizer_revision=..., chat_template_hash=...,
        semantic_position='t_post', layer=18, source_partition='direction_ids',
        source_ids=[...], construction_contrast='harmful_mean_minus_harmless_mean',
        git_commit=..., random_seed=0,
    )
    save_direction_atomic(direction_tensor, meta, 'refusal_dir_en.pt')

    # later, anywhere that needs to trust this file:
    tensor, meta = verify_direction_file('refusal_dir_en.pt')  # raises on any mismatch
"""
import hashlib
import json
import os
import subprocess

import torch


# Fields buildable BEFORE the tensor exists (from experiment config alone).
LOGICAL_FIELDS = [
    'direction_type', 'model', 'model_revision', 'tokenizer_revision',
    'chat_template_hash', 'semantic_position', 'layer', 'source_partition',
    'source_ids_hash', 'sample_count', 'construction_contrast', 'git_commit',
    'random_seed',
]
# Fields only knowable once the tensor has actually been computed -- filled
# in exclusively by save_direction_atomic(), never hand-supplied.
TENSOR_FIELDS = ['tensor_sha256', 'tensor_shape', 'tensor_dtype']
REQUIRED_FIELDS = LOGICAL_FIELDS + TENSOR_FIELDS


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


def sha256_of_tensor(tensor):
    """Content hash of a tensor's actual values (not its Python object identity
    or its file's mtime) -- moved to CPU/contiguous first so the hash is
    device- and stride-independent."""
    arr = tensor.detach().cpu().contiguous()
    return hashlib.sha256(arr.numpy().tobytes()).hexdigest()


def sha256_of_file(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def atomic_torch_save(tensor, path):
    """Writes to a temp file in the SAME directory (so os.replace is an atomic
    rename, not a cross-filesystem copy) and only replaces the real path once
    the write is fully flushed -- a process killed mid-write leaves only the
    .tmp file behind, never a truncated file at `path`."""
    tmp_path = path + '.tmp'
    torch.save(tensor, tmp_path)
    os.replace(tmp_path, path)


def atomic_json_save(obj, path):
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp_path, path)


def build_direction_metadata(direction_type, model, model_revision, tokenizer_revision,
                              chat_template_hash, semantic_position, layer,
                              source_partition, source_ids, construction_contrast,
                              random_seed, git_commit=None, extra=None):
    """Builds the LOGICAL portion of the metadata only -- deliberately does NOT
    accept or require tensor_sha256/tensor_shape/tensor_dtype, since those
    aren't knowable until the tensor is actually computed. This return value
    is not a complete, savable metadata dict by itself -- pass it to
    save_direction_atomic() together with the real tensor, which fills in the
    tensor fields and performs the actual (atomic) write. Calling
    save_direction_metadata() directly on this return value will correctly
    refuse (missing tensor fields) -- that is intentional, not a bug."""
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
    for f in LOGICAL_FIELDS:
        assert f in meta, f"missing required logical metadata field: {f}"
    return meta


def save_direction_metadata(meta, json_path):
    """Low-level: given an ALREADY-COMPLETE meta dict (including the three
    tensor fields), writes it atomically. Prefer save_direction_atomic() in
    new code -- this is kept for the metadata-only rewrite path (e.g. a
    verified re-save) where the tensor itself isn't being touched."""
    missing = [f for f in REQUIRED_FIELDS if f not in meta]
    if missing:
        raise ValueError(f"refusing to save direction metadata missing required field(s): {missing}")
    atomic_json_save(meta, json_path)


def save_direction_atomic(direction, logical_meta, pt_path):
    """THE sanctioned way to persist a direction from this point forward.
    Order matters and is the whole point: the tensor is written+renamed into
    place FIRST; the metadata (with the tensor's own real sha256/shape/dtype
    now filled in) is only written after that succeeds. If this function
    raises partway through, at most a stray .tmp file is left behind --
    never a metadata JSON whose tensor is missing, truncated, or mismatched.
    pt_path must end in '.pt'; the sidecar is written to the same path with
    '.json' instead."""
    assert pt_path.endswith('.pt'), f"pt_path must end in .pt, got {pt_path!r}"
    json_path = pt_path[:-3] + '.json'
    meta = dict(logical_meta)
    meta['tensor_sha256'] = sha256_of_tensor(direction)
    meta['tensor_shape'] = list(direction.shape)
    meta['tensor_dtype'] = str(direction.dtype)
    missing = [f for f in REQUIRED_FIELDS if f not in meta]
    if missing:
        raise ValueError(f"refusing to save: metadata missing required field(s) {missing}")
    atomic_torch_save(direction.cpu(), pt_path)
    atomic_json_save(meta, json_path)
    return meta


def load_direction_metadata(json_path):
    with open(json_path) as f:
        meta = json.load(f)
    missing = [f for f in REQUIRED_FIELDS if f not in meta]
    if missing:
        raise ValueError(f"{json_path} is missing required fields: {missing}")
    return meta


def verify_direction_file(pt_path):
    """Fail-fast loader: the ONLY sanctioned way to load a direction file
    from this point forward. Raises immediately (never returns a partial or
    unverified result) if: the tensor is missing, the metadata sidecar is
    missing, the tensor's actual SHA-256/shape/dtype don't match what the
    metadata records. Returns (tensor, meta) only once all of these pass."""
    assert pt_path.endswith('.pt'), f"pt_path must end in .pt, got {pt_path!r}"
    json_path = pt_path[:-3] + '.json'
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"direction tensor missing: {pt_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"direction metadata sidecar missing: {json_path} -- a tensor with no verifiable "
            f"metadata is not trusted, even if the .pt file itself is readable."
        )
    meta = load_direction_metadata(json_path)
    tensor = torch.load(pt_path, map_location='cpu')
    actual_hash = sha256_of_tensor(tensor)
    if actual_hash != meta['tensor_sha256']:
        raise ValueError(
            f"{pt_path}: tensor SHA-256 ({actual_hash}) does not match metadata's recorded hash "
            f"({meta['tensor_sha256']}) -- the file was modified, corrupted, or replaced after "
            f"its metadata was written. Refusing to use it."
        )
    if list(tensor.shape) != list(meta['tensor_shape']):
        raise ValueError(f"{pt_path}: tensor shape {list(tensor.shape)} != metadata's recorded "
                          f"shape {meta['tensor_shape']}")
    if str(tensor.dtype) != meta['tensor_dtype']:
        raise ValueError(f"{pt_path}: tensor dtype {tensor.dtype} != metadata's recorded dtype "
                          f"{meta['tensor_dtype']}")
    return tensor, meta


# ── delta_R/delta_H artifact schema (25_extract_delta_r_h.py's output is a
# dict of per-mechanism tensors, not a single [n_layers, d_model] direction,
# so it gets its own parallel schema rather than being forced into the
# direction one above) ──────────────────────────────────────────────────

DELTA_REQUIRED_FIELDS = [
    'model', 'lang', 'suffix', 'ids_key', 'active_mechanisms',
    'n_instructions', 'n_layers', 'refusal_direction_path', 'refusal_direction_sha256',
    'harmfulness_direction_path', 'harmfulness_direction_sha256',
    'token_position_R', 'token_position_H', 'estimator', 'git_commit',
    'payload_sha256',
]


def sha256_of_nested_tensors(obj):
    """Deterministic content hash of a possibly-nested dict/list of tensors
    (e.g. {mech: tensor} or {mech: {sub: tensor}}) -- sorts keys at every
    level so the hash never depends on dict insertion order, and prefixes
    each leaf with its path so two structures with the same tensor values at
    different keys don't collide. Non-tensor leaves (e.g. the `failures`
    list of dicts) are NOT covered by this hash -- only tensor content is;
    non-tensor fields belong in metadata verbatim instead."""
    h = hashlib.sha256()

    def _walk(o, path):
        if torch.is_tensor(o):
            h.update(path.encode('utf-8'))
            h.update(o.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(o, dict):
            for k in sorted(o.keys(), key=str):
                _walk(o[k], f'{path}/{k}')
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                _walk(v, f'{path}[{i}]')

    _walk(obj, '')
    return h.hexdigest()


def build_delta_metadata(model, lang, suffix, ids_key, active_mechanisms,
                          n_instructions, n_layers, refusal_direction_path,
                          harmfulness_direction_path, token_position_R, token_position_H,
                          estimator, git_commit=None, extra=None):
    assert token_position_R == 't_post', f"refusal-axis projection must be at t_post, got {token_position_R!r}"
    assert token_position_H == 't_inst', f"harmfulness-axis projection must be at t_inst, got {token_position_H!r}"
    meta = {
        'model': model, 'lang': lang, 'suffix': suffix, 'ids_key': ids_key,
        'active_mechanisms': list(active_mechanisms),
        'n_instructions': n_instructions, 'n_layers': n_layers,
        'refusal_direction_path': refusal_direction_path,
        'refusal_direction_sha256': sha256_of_file(refusal_direction_path),
        'harmfulness_direction_path': harmfulness_direction_path,
        'harmfulness_direction_sha256': sha256_of_file(harmfulness_direction_path),
        'token_position_R': token_position_R, 'token_position_H': token_position_H,
        'estimator': estimator, 'git_commit': git_commit or current_git_commit(),
    }
    if extra:
        meta.update(extra)
    for f in DELTA_REQUIRED_FIELDS:
        if f == 'payload_sha256':
            continue  # filled in by save_delta_atomic once the payload dict exists
        assert f in meta, f"missing required delta metadata field: {f}"
    return meta


def save_delta_atomic(payload, logical_meta, pt_path):
    """payload: the dict saved by 25_extract_delta_r_h.py (delta_R/delta_H/
    delta_R_placebo_calibrated/delta_H_placebo_calibrated/valid_mask/failures/
    instruction_ids/n_layers/ids_key). Tensor write is atomic and happens
    BEFORE metadata, same discipline as save_direction_atomic()."""
    assert pt_path.endswith('.pt'), f"pt_path must end in .pt, got {pt_path!r}"
    json_path = pt_path[:-3] + '.json'
    meta = dict(logical_meta)
    meta['payload_sha256'] = sha256_of_nested_tensors(payload)
    missing = [f for f in DELTA_REQUIRED_FIELDS if f not in meta]
    if missing:
        raise ValueError(f"refusing to save: delta metadata missing required field(s) {missing}")
    atomic_torch_save(payload, pt_path)
    atomic_json_save(meta, json_path)
    return meta


def verify_delta_file(pt_path, expected_active_mechanisms=None):
    """Fail-fast loader for delta_R/delta_H payloads. Raises immediately if:
    the tensor is missing, the metadata sidecar is missing, the payload's
    hash doesn't match metadata, or (if expected_active_mechanisms is given)
    the payload's mechanism set doesn't match the CURRENT canonical taxonomy
    -- this specific check exists because the pre-rebuild version of this
    pipeline computed delta_R/delta_H (if it ran at all) against a stale,
    pre-correction mechanism list; a payload keyed by the wrong mechanism
    names must never be silently accepted as current-taxonomy data."""
    assert pt_path.endswith('.pt'), f"pt_path must end in .pt, got {pt_path!r}"
    json_path = pt_path[:-3] + '.json'
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"delta_R/delta_H tensor missing: {pt_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"delta_R/delta_H metadata sidecar missing: {json_path}")
    with open(json_path) as f:
        meta = json.load(f)
    missing = [f for f in DELTA_REQUIRED_FIELDS if f not in meta]
    if missing:
        raise ValueError(f"{json_path} is missing required fields: {missing}")
    payload = torch.load(pt_path, map_location='cpu')
    actual_hash = sha256_of_nested_tensors(payload)
    if actual_hash != meta['payload_sha256']:
        raise ValueError(
            f"{pt_path}: payload SHA-256 ({actual_hash}) does not match metadata's recorded hash "
            f"({meta['payload_sha256']}) -- refusing to use it."
        )
    if expected_active_mechanisms is not None:
        payload_mechs = set(payload['delta_R'].keys()) - {'placebo'}
        expected = set(expected_active_mechanisms)
        if payload_mechs != expected:
            raise ValueError(
                f"{pt_path}: payload's mechanism set {sorted(payload_mechs)} does not match the "
                f"current canonical taxonomy's active_mechanisms {sorted(expected)} -- this file "
                f"was computed against a different (likely stale/pre-correction) mechanism list "
                f"and must not be used as current-taxonomy delta_R/delta_H data."
            )
    return payload, meta
