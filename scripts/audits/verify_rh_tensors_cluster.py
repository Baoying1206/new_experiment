"""
Read-only R/H tensor field-verification for the 6 pre-existing direction
files this experiment depends on (EXPERIMENT2_CONTEXT_REPRESENTATION_PROTOCOL.md
Sec 11). Loads (never writes) refusal_dir_v3_en.pt / harmfulness_dir_v2_en.pt
for all 3 models via utils.direction_metadata.verify_direction_file(),
which already raises on any hash/shape/dtype mismatch against the file's
own metadata sidecar. This script adds the remaining required checks
(finiteness, layer-index range, semantic_position vs the dual-position
design) and HALTS IMMEDIATELY (does not continue to the next file) on the
first violation of any of the 5 stop conditions:
  1. tensor SHA-256 != metadata's recorded hash
  2. any non-finite (NaN/Inf) value in the tensor
  3. shape or dtype mismatch against metadata
  4. primary_layer index out of range for the tensor's layer count
  5. semantic_position != the expected value for that file
     (t_post for refusal_dir_v3_en.pt, t_inst for harmfulness_dir_v2_en.pt
      -- EXPERIMENT2_CONTEXT_REPRESENTATION_PROTOCOL.md Sec 5's
      dual-position definition)

Read-only: no model, no tokenizer, no GPU, no writes anywhere.

Usage (on the cluster, from the repo root):
  python3 scripts/audits/verify_rh_tensors_cluster.py
"""
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts'))

# (relative .pt path, expected primary_layer, expected semantic_position)
FILES = [
    ('output/output_v3_behavioral_refusal/Qwen2.5-7B-Instruct/refusal_dir_v3_en.pt', 16, 't_post'),
    ('output/output_v3_behavioral_refusal/Meta-Llama-3.1-8B-Instruct/refusal_dir_v3_en.pt', 19, 't_post'),
    ('output/output_v3_behavioral_refusal/gemma-2-9b-it/refusal_dir_v3_en.pt', 25, 't_post'),
    ('output/output_v2_dual_position/Qwen2.5-7B-Instruct/harmfulness_dir_v2_en.pt', 16, 't_inst'),
    ('output/output_v2_dual_position/Meta-Llama-3.1-8B-Instruct/harmfulness_dir_v2_en.pt', 19, 't_inst'),
    ('output/output_v2_dual_position/gemma-2-9b-it/harmfulness_dir_v2_en.pt', 25, 't_inst'),
]


def main():
    import torch
    from utils.direction_metadata import verify_direction_file, sha256_of_tensor

    results = []
    for rel_path, primary_layer, expected_pos in FILES:
        pt_path = os.path.join(REPO_ROOT, rel_path)
        print(f"Verifying {rel_path} ...")

        # verify_direction_file() raises immediately on missing file, missing
        # sidecar, hash mismatch, shape mismatch, or dtype mismatch (stop
        # conditions 1 and 3) -- no try/except here; a raise here is
        # intentionally fatal for the whole script.
        tensor, meta = verify_direction_file(pt_path)
        actual_hash = sha256_of_tensor(tensor)

        # stop condition 2: non-finite values
        if not torch.isfinite(tensor).all():
            print(f"\nSTOP: non-finite (NaN/Inf) value(s) found in {rel_path}")
            sys.exit(1)

        # stop condition 4: primary_layer out of range
        if primary_layer >= tensor.shape[0]:
            print(f"\nSTOP: primary_layer={primary_layer} out of range for {rel_path} "
                  f"with shape={list(tensor.shape)}")
            sys.exit(1)

        # stop condition 5: semantic_position mismatch (dual-position design)
        actual_pos = meta.get('semantic_position')
        if actual_pos != expected_pos:
            print(f"\nSTOP: {rel_path} semantic_position={actual_pos!r}, expected {expected_pos!r} "
                  f"(dual-position design: R@t_post, H@t_inst)")
            sys.exit(1)

        results.append({
            'file': rel_path,
            'actual_sha256': actual_hash,
            'metadata_sha256': meta['tensor_sha256'],
            'match': actual_hash == meta['tensor_sha256'],
            'shape': list(tensor.shape),
            'dtype': str(tensor.dtype),
            'finite': True,
            'layer_count': tensor.shape[0],
            'hidden_dim': tensor.shape[1],
            'primary_layer_index_used_downstream': primary_layer,
            'primary_layer_in_range': True,
            'semantic_position': actual_pos,
            'status_or_backfill_note': meta.get('status', meta.get('tensor_hash_backfilled_note', 'n/a')),
        })
        print(f"  OK: shape={results[-1]['shape']} dtype={results[-1]['dtype']} "
              f"semantic_position={actual_pos!r}")

    print()
    print(json.dumps(results, indent=2))
    print(f"\nALL {len(results)} R/H TENSORS VERIFIED OK (hash-matched, finite, "
          f"in-range, correct dual-position semantic_position).")


if __name__ == '__main__':
    main()
