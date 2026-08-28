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

import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(__file__)
REAL_MECHS = ['prefix_injection', 'refusal_suppression', 'instruction_hierarchy',
              'persona_roleplay', 'fictional_framing', 'encoding_obfuscation']
CO = ['prefix_injection', 'refusal_suppression', 'instruction_hierarchy']
MG = ['persona_roleplay', 'fictional_framing', 'encoding_obfuscation']
FIXED_LAYER_FRACTION = 0.6


def main(args):
    models = args.models.split(',')
    results = {'lang': args.lang, 'suffix': args.suffix, 'per_model': {}}

    for model_alias in models:
        print(f"\n=== {model_alias} ===")
        path = os.path.join(args.output_dir, model_alias, f'paired_diffs_{args.lang}{args.suffix}.pt')
        if not os.path.exists(path):
            print(f"  MISSING: {path}")
            results['per_model'][model_alias] = {'error': 'missing paired_diffs file', 'path': path}
            continue
        data = torch.load(path, map_location='cpu')
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

    out_path = os.path.join(args.output_dir, f'pairwise_template_cosine_{args.lang}{args.suffix}.json')
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
    args = parser.parse_args()
    main(args)
