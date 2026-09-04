"""
Schema and validator for independent R-axis manifests
(EXPERIMENT2_RH_REBUILD_PROTOCOL.md Sec 15). A manifest is a JSON file
listing already-generated, already-WildGuard-labeled rows for building
refusal_direction_v3 from a source confirmed independent of the
572-instruction pool AND of the 6 canonical jailbreak templates.

Pure Python/JSON -- no torch, no model, no `pipeline`/`dataset` cluster-only
package dependency -- so this is fully testable on any machine, including
this local checkout which has neither of those packages available.

Added 2026-09-04 (circularity-correction round). No manifest currently
exists anywhere in this project; this module defines the contract any
future manifest-building step must satisfy, and is the validator
scripts/26_rebuild_refusal_direction_behavioral.py runs BEFORE loading any
model. See EXPERIMENT2_RH_REBUILD_PROTOCOL.md Sec 13 for the current
candidate-data survey (none confirmed usable yet).
"""
import hashlib
import json
import os
import re

MANIFEST_ROW_REQUIRED_FIELDS = [
    'dataset_name', 'dataset_version', 'source_path', 'source_file_sha256',
    'stable_source_id', 'normalized_text_hash', 'prompt_family', 'condition',
    'model_alias', 'response_id', 'refusal_label', 'label_source', 'split',
    'overlaps_572_pool', 'contains_canonical_template',
]

ALLOWED_SPLITS = ('axis', 'val')


def normalize_text(s):
    return re.sub(r'\s+', ' ', s.strip().lower())


def normalized_text_hash(s):
    return hashlib.sha256(normalize_text(s).encode('utf-8')).hexdigest()


def make_stable_source_id(dataset_name, row_index, text):
    """Deterministic ID for a source row that has no native ID field:
    dataset_name + ORIGINAL (pre-shuffle) row index + normalised-text hash
    (first 16 hex chars). Never a post-shuffle positional index -- that was
    the harmful_train_axis_{i} scheme this replaces, which couldn't even
    reproducibly name 'the same row' across two runs with different shuffles."""
    h = normalized_text_hash(text)[:16]
    return f"{dataset_name}:{row_index}:{h}"


def sha256_of_file(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_axis_manifest(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"axis manifest not found: {path}")
    with open(path, encoding='utf-8') as f:
        manifest = json.load(f)
    if 'rows' not in manifest:
        raise ValueError(f"{path}: manifest missing top-level 'rows' field")
    return manifest['rows']


def load_pool_text_hashes(sampled_prompts_path):
    """The 572-pool's own normalised-text hashes -- used to INDEPENDENTLY
    recheck each manifest row's self-reported overlaps_572_pool flag, never
    to trust that flag on its own."""
    with open(sampled_prompts_path, encoding='utf-8') as f:
        pool = json.load(f)
    return set(normalized_text_hash(x['instruction_en']) for x in pool)


def validate_axis_manifest(rows, canonical_mechanisms, pool_text_hashes,
                            required_splits=ALLOWED_SPLITS):
    """Raises ValueError with a specific, actionable message on the first
    failed check (fail-fast, no partial/best-effort acceptance). Returns a
    dict of per-split refused/accepted counts on success. Called BEFORE any
    model load."""
    if not rows:
        raise ValueError("manifest has zero rows")

    for i, row in enumerate(rows):
        missing = [f for f in MANIFEST_ROW_REQUIRED_FIELDS if f not in row]
        if missing:
            raise ValueError(f"manifest row {i} missing required field(s): {missing}")

    # 1. overlaps_572_pool must be False for every row -- AND independently
    #    recomputed, never taken on the manifest's own word.
    bad_overlap_flag = [r['stable_source_id'] for r in rows if r['overlaps_572_pool']]
    if bad_overlap_flag:
        raise ValueError(
            f"{len(bad_overlap_flag)} row(s) flagged overlaps_572_pool=True -- an axis manifest "
            f"must be entirely disjoint from the 572-pool: {bad_overlap_flag[:5]}"
            f"{'...' if len(bad_overlap_flag) > 5 else ''}"
        )
    recomputed_overlap = [r['stable_source_id'] for r in rows
                           if r['normalized_text_hash'] in pool_text_hashes]
    if recomputed_overlap:
        raise ValueError(
            f"{len(recomputed_overlap)} row(s) have overlaps_572_pool=False in the manifest but "
            f"their normalized_text_hash actually matches the 572-pool -- the manifest's "
            f"self-reported flag is WRONG (not just missing): {recomputed_overlap[:5]}"
            f"{'...' if len(recomputed_overlap) > 5 else ''}"
        )

    # 2. contains_canonical_template must be False for every row -- AND
    #    independently rechecked against prompt_family.
    bad_template_flag = [r['stable_source_id'] for r in rows if r['contains_canonical_template']]
    if bad_template_flag:
        raise ValueError(
            f"{len(bad_template_flag)} row(s) flagged contains_canonical_template=True -- an axis "
            f"manifest must not include any of the 6 canonical mechanisms: {bad_template_flag[:5]}"
            f"{'...' if len(bad_template_flag) > 5 else ''}"
        )
    bad_family = [r['stable_source_id'] for r in rows if r['prompt_family'] in canonical_mechanisms]
    if bad_family:
        raise ValueError(
            f"{len(bad_family)} row(s) have prompt_family matching a canonical mechanism name "
            f"despite contains_canonical_template=False -- the flag is wrong: {bad_family[:5]}"
        )

    # 3. stable IDs must be unique
    ids = [r['stable_source_id'] for r in rows]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"duplicate stable_source_id(s): {dupes[:5]}{'...' if len(dupes) > 5 else ''}")

    # 4. split values restricted -- test_ids/direction_ids/validation_ids
    #    (572-pool concepts) must never appear in an independent-source manifest
    bad_splits = sorted(set(r['split'] for r in rows) - set(required_splits))
    if bad_splits:
        raise ValueError(
            f"manifest rows use split value(s) {bad_splits}, not in the allowed set "
            f"{required_splits} -- this manifest must never reference test_ids/direction_ids/"
            f"validation_ids (those name 572-pool partitions, not an independent axis source)"
        )

    # 5. axis split has both classes non-empty
    axis_rows = [r for r in rows if r['split'] == 'axis']
    if not axis_rows:
        raise ValueError("manifest has no rows with split='axis'")
    n_ref = sum(1 for r in axis_rows if r['refusal_label'] == 1)
    n_acc = sum(1 for r in axis_rows if r['refusal_label'] == 0)
    if n_ref == 0 or n_acc == 0:
        raise ValueError(
            f"axis split has refused={n_ref}, accepted={n_acc} -- both classes must be non-empty "
            f"to build a direction"
        )

    # 6. every unique source_path's recorded hash must match the real file on disk
    by_path = {}
    for r in rows:
        by_path.setdefault(r['source_path'], r['source_file_sha256'])
    for source_path, claimed_hash in by_path.items():
        if not os.path.exists(source_path):
            raise ValueError(
                f"manifest references source_path {source_path!r} which does not exist on this "
                f"machine -- cannot verify its hash, refusing to proceed"
            )
        actual_hash = sha256_of_file(source_path)
        if actual_hash != claimed_hash:
            raise ValueError(
                f"{source_path}: file hash {actual_hash} does not match the manifest's recorded "
                f"hash {claimed_hash} -- the source file has changed since the manifest was built; "
                f"re-derive the manifest before trusting it"
            )

    val_rows = [r for r in rows if r['split'] == 'val']
    n_ref_val = sum(1 for r in val_rows if r['refusal_label'] == 1)
    n_acc_val = sum(1 for r in val_rows if r['refusal_label'] == 0)

    return {
        'n_axis': len(axis_rows), 'n_axis_refused': n_ref, 'n_axis_accepted': n_acc,
        'n_val': len(val_rows), 'n_val_refused': n_ref_val, 'n_val_accepted': n_acc_val,
    }
