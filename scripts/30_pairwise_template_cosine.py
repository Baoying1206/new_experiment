"""
Computes and saves the full 6x6 pairwise cosine similarity matrix between
the 6 real mechanisms' mean directions -- this is the raw input that
09/19/27's within/between-group T-statistics are built from, but none of
them save the matrix itself. Useful for reading off which SPECIFIC pairs
are close/far, not just the CO/MG group-level summary.

CPU-only, reads the existing paired_diffs_{lang}{suffix}.pt (18's output,
already used for Exp1) -- no new GPU extraction.

Usage:
  python scripts/30_pairwise_template_cosine.py \
      --output_dir output --lang en --suffix _full572 \
      --models Qwen2.5-7B-Instruct,Meta-Llama-3.1-8B-Instruct,gemma-2-9b-it
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)
import _taxonomy_config as tc

OLD_REAL_MECHS = tc.CANONICAL_REAL_MECHS
OLD_CO = tc.CANONICAL_CO_MECHS
OLD_MG = tc.CANONICAL_MG_MECHS
CORRECTED_REAL_MECHS = tc.CORRECTED_REAL_MECHS
CORRECTED_CO = tc.CORRECTED_CO_MECHS
CORRECTED_MG = tc.CORRECTED_MG_MECHS
FIXED_LAYER_FRACTION = 0.6


def main(args):
    if args.taxonomy == 'corrected':
        REAL_MECHS, CO, MG = CORRECTED_REAL_MECHS, CORRECTED_CO, CORRECTED_MG
    else:
        REAL_MECHS, CO, MG = OLD_REAL_MECHS, OLD_CO, OLD_MG
    print(f"Taxonomy: {args.taxonomy}  CO={CO}  MG={MG}\n")

    models = args.models.split(',')
    results = {'lang': args.lang, 'suffix': args.suffix, 'taxonomy': args.taxonomy, 'per_model': {}}

    for model_alias in models:
        print(f"\n=== {model_alias} ===")
        path = os.path.join(args.output_dir, model_alias, f'paired_diffs_{args.lang}{args.suffix}.pt')
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            results['per_model'][model_alias] = {'error': 'missing paired_diffs file', 'path': path}
            continue
        data = torch.load(path, map_location='cpu')
        missing = set(REAL_MECHS) - set(data['diffs'].keys())
        if missing:
            print(f"  MISSING mechanisms {missing} in {path} -- was it extracted with "
                  f"--mechanisms matching taxonomy={args.taxonomy}? Got keys: {sorted(data['diffs'].keys())}")
            results['per_model'][model_alias] = {'error': f'missing mechanisms {sorted(missing)}', 'path': path}
            continue
        n_layers = data['n_layers']
        fixed_layer = int(FIXED_LAYER_FRACTION * n_layers)

        # mean direction per mechanism, all layers: [n_layers, d]
        mean_dirs = {mech: data['diffs'][mech].float().mean(0) for mech in REAL_MECHS}

        # full pairwise cosine matrix, per layer
        cos_per_layer = []
        for l in range(n_layers):
            row = {}
            for i, m1 in enumerate(REAL_MECHS):
                row[m1] = {}
                for m2 in REAL_MECHS:
                    row[m1][m2] = F.cosine_similarity(
                        mean_dirs[m1][l].unsqueeze(0), mean_dirs[m2][l].unsqueeze(0), dim=-1
                    ).item()
            cos_per_layer.append(row)

        results['per_model'][model_alias] = {
            'n_layers': n_layers, 'fixed_layer': fixed_layer,
            'cosine_matrix_per_layer': cos_per_layer,
        }

        print(f"  Loaded {path}  n_layers={n_layers}  fixed_layer={fixed_layer}\n")
        print(f"  --- Pairwise cosine at fixed layer {fixed_layer} ---")
        header = "  " + " " * 24 + "".join(f"{m[:10]:>12s}" for m in REAL_MECHS)
        print(header)
        mat = cos_per_layer[fixed_layer]
        for m1 in REAL_MECHS:
            row_str = f"  {m1:24s}" + "".join(f"{mat[m1][m2]:12.3f}" for m2 in REAL_MECHS)
            print(row_str)

        # within-group vs between-group breakdown, explicit pairs, at fixed layer
        print(f"\n  --- CO-internal pairs ---")
        for i, m1 in enumerate(CO):
            for m2 in CO[i + 1:]:
                print(f"    cos({m1}, {m2}) = {mat[m1][m2]:.4f}")
        print(f"  --- MG-internal pairs ---")
        for i, m1 in enumerate(MG):
            for m2 in MG[i + 1:]:
                print(f"    cos({m1}, {m2}) = {mat[m1][m2]:.4f}")
        print(f"  --- CO-MG cross pairs ---")
        for m1 in CO:
            for m2 in MG:
                print(f"    cos({m1}, {m2}) = {mat[m1][m2]:.4f}")

        # closest/farthest pair overall (excluding self-pairs)
        all_pairs = [(m1, m2, mat[m1][m2]) for i, m1 in enumerate(REAL_MECHS)
                     for m2 in REAL_MECHS[i + 1:]]
        closest = max(all_pairs, key=lambda x: x[2])
        farthest = min(all_pairs, key=lambda x: x[2])
        print(f"\n  Closest pair: {closest[0]} & {closest[1]}  (cos={closest[2]:.4f})")
        print(f"  Farthest pair: {farthest[0]} & {farthest[1]}  (cos={farthest[2]:.4f})")

    taxonomy_suffix = '' if args.taxonomy == 'old' else f'_{args.taxonomy}'
    out_path = os.path.join(args.output_dir, f'pairwise_template_cosine_{args.lang}{args.suffix}{taxonomy_suffix}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    parser.add_argument('--lang',       type=str, default='en')
    parser.add_argument('--suffix',     type=str, default='_full572')
    parser.add_argument('--models',     type=str,
                         default='Qwen2.5-7B-Instruct,Meta-Llama-3.1-8B-Instruct,gemma-2-9b-it')
    parser.add_argument('--taxonomy',   type=str, default='old', choices=['old', 'corrected'],
                         help="'old' (default): original 6-mechanism set (instruction_hierarchy/ "
                              "fictional_framing included, persona_roleplay as MG). 'corrected': "
                              "_taxonomy_config.py's CORRECTED_* set (persona_roleplay as CO, "
                              "payload_splitting/distractors_negated instead of the two dropped "
                              "mechanisms) -- requires a paired_diffs file extracted with "
                              "18_extract_paired_diffs.py --mechanisms matching that set (e.g. "
                              "--suffix _full572_corrected).")
    args = parser.parse_args()
    main(args)
