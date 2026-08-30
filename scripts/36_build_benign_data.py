"""
Builds the disjoint benign validation/test sets for the Exp3 defence protocol.
CPU-only, read-only against the existing English independent-train-split
harmless data confirmed in output/audits/english_axis_data_followup.json:
  related_work/Multilingual-Refusal/dataset/splits/harmless_val.json  (6264, disjoint from harmless_test/train and the 572 harmful pool)
  related_work/Multilingual-Refusal/dataset/splits/harmless_test.json (6266, disjoint from harmless_val/train and the 572 harmful pool)

validation-benign (80) sampled from harmless_val.json; test-benign (100)
sampled from harmless_test.json -- two disjoint source pools, so the two
samples cannot overlap by construction; still verified explicitly below
(by normalized instruction text, matching this project's existing overlap-
check convention) rather than assumed.

Outputs (committed):
  data/benign_validation_80.json   -- [{'benign_id':..., 'instruction':...}, ...]
  data/benign_test_100.json
  data/benign_data_manifest.json   -- source paths, seeds, counts, sha256 of each
                                       output file's content, disjointness checks

Usage:
  python scripts/36_build_benign_data.py
"""
import hashlib
import json
import os

SCRIPT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
POOL_PATH = os.path.join(DATA_DIR, 'sampled_prompts.json')

RELATED_WORK_CANDIDATES = [
    os.path.join(SCRIPT_DIR, '..', '..', 'related_work', 'Multilingual-Refusal'),
    os.path.expanduser('~/Downloads/related_work/Multilingual-Refusal'),
    os.path.expanduser('~/thesis_experiment/Multilingual-Refusal'),
    os.path.expanduser('~/Multilingual-Refusal'),
]

N_VAL = 80
N_TEST = 100
SEED_VAL = 20260830
SEED_TEST = 20260831


def find_related_work_dir():
    for c in RELATED_WORK_CANDIDATES:
        p = os.path.join(c, 'dataset', 'splits', 'harmless_val.json')
        if os.path.exists(p):
            return c
    raise FileNotFoundError(
        f"Could not find related_work/Multilingual-Refusal/dataset/splits/harmless_val.json "
        f"in any of: {RELATED_WORK_CANDIDATES}"
    )


def normalize(text):
    return text.strip().lower()


def sha256_of(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def sample_deterministic(pool, n, seed):
    """Fisher-Yates via a seeded RNG, stdlib-only (no numpy dependency)."""
    import random
    rng = random.Random(seed)
    idxs = list(range(len(pool)))
    rng.shuffle(idxs)
    chosen = sorted(idxs[:n])  # sort for stable, reviewable output order
    return [pool[i] for i in chosen]


def main():
    related_work_dir = find_related_work_dir()
    val_path = os.path.join(related_work_dir, 'dataset', 'splits', 'harmless_val.json')
    test_path = os.path.join(related_work_dir, 'dataset', 'splits', 'harmless_test.json')

    with open(val_path, encoding='utf-8') as f:
        harmless_val_pool = json.load(f)
    with open(test_path, encoding='utf-8') as f:
        harmless_test_pool = json.load(f)
    with open(POOL_PATH, encoding='utf-8') as f:
        harmful_572_pool = json.load(f)

    print(f"harmless_val pool: {len(harmless_val_pool)}  (source: {val_path})")
    print(f"harmless_test pool: {len(harmless_test_pool)}  (source: {test_path})")
    print(f"harmful 572 pool: {len(harmful_572_pool)}  (source: {POOL_PATH})")

    val_sample = sample_deterministic(harmless_val_pool, N_VAL, SEED_VAL)
    test_sample = sample_deterministic(harmless_test_pool, N_TEST, SEED_TEST)

    val_texts = set(normalize(x['instruction']) for x in val_sample)
    test_texts = set(normalize(x['instruction']) for x in test_sample)
    pool_texts = set(normalize(x['instruction_en']) for x in harmful_572_pool)

    overlap_val_test = val_texts & test_texts
    overlap_val_pool = val_texts & pool_texts
    overlap_test_pool = test_texts & pool_texts

    assert len(overlap_val_test) == 0, f"benign val/test overlap: {overlap_val_test}"
    assert len(overlap_val_pool) == 0, f"benign val overlaps harmful 572 pool: {overlap_val_pool}"
    assert len(overlap_test_pool) == 0, f"benign test overlaps harmful 572 pool: {overlap_test_pool}"
    print("Disjointness checks passed: val/test/harmful-572-pool are pairwise disjoint by normalized text.")

    val_out = [{'benign_id': f'bv{i:03d}', 'instruction': x['instruction']} for i, x in enumerate(val_sample)]
    test_out = [{'benign_id': f'bt{i:03d}', 'instruction': x['instruction']} for i, x in enumerate(test_sample)]

    val_out_path = os.path.join(DATA_DIR, 'benign_validation_80.json')
    test_out_path = os.path.join(DATA_DIR, 'benign_test_100.json')
    with open(val_out_path, 'w', encoding='utf-8') as f:
        json.dump(val_out, f, indent=2, ensure_ascii=False)
    with open(test_out_path, 'w', encoding='utf-8') as f:
        json.dump(test_out, f, indent=2, ensure_ascii=False)

    manifest = {
        'source_related_work_dir': related_work_dir,
        'source_files': {'harmless_val': val_path, 'harmless_test': test_path},
        'source_pool_sizes': {'harmless_val': len(harmless_val_pool), 'harmless_test': len(harmless_test_pool)},
        'n_validation_benign': N_VAL, 'n_test_benign': N_TEST,
        'seed_validation': SEED_VAL, 'seed_test': SEED_TEST,
        'disjointness_checks': {
            'val_vs_test_overlap': len(overlap_val_test),
            'val_vs_harmful572_overlap': len(overlap_val_pool),
            'test_vs_harmful572_overlap': len(overlap_test_pool),
        },
        'output_files': {
            'benign_validation_80.json': {'path': val_out_path, 'sha256': sha256_of(val_out), 'n': len(val_out)},
            'benign_test_100.json': {'path': test_out_path, 'sha256': sha256_of(test_out), 'n': len(test_out)},
        },
    }
    manifest_path = os.path.join(DATA_DIR, 'benign_data_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Saved: {val_out_path}  sha256={manifest['output_files']['benign_validation_80.json']['sha256'][:16]}...")
    print(f"Saved: {test_out_path}  sha256={manifest['output_files']['benign_test_100.json']['sha256'][:16]}...")
    print(f"Saved: {manifest_path}")


if __name__ == '__main__':
    main()
