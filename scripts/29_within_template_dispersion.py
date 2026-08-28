"""
Extends Experiment 1: 09/19 only compared templates' MEAN directions against
each other. This asks the question one level down -- for a given template,
do the 300 individual instructions it was applied to actually point the
same way, or does the mean direction paper over real internal spread
(sub-groups, or just diffuse noise)?

CPU-only -- reads paired_diffs_{lang}{suffix}.pt (18_extract_paired_diffs.py's
existing output, already used for Exp1), no new GPU extraction needed.

For each mechanism (6 real + placebo, for reference), at every layer:

  1. R2_mean = fraction of total sum-of-squared-norms explained by each
     instruction's projection onto the mechanism's OWN mean direction:
         v_i = proj_i (along mean_hat) + residual_i
         R2_mean = sum(||proj_i||^2) / sum(||v_i||^2)
     High R2_mean = the single mean direction is a good summary of the 300
     instructions. Low R2_mean = individual instructions diverge from the
     mean substantially -- the mean is averaging over real internal spread.

  2. cosine_to_mean distribution (mean/std/min/25%/50%/75%/max) across the
     300 instructions -- a second, more interpretable view of the same
     question (R2_mean is an aggregate energy ratio; this shows the actual
     spread, including whether a few instructions are strongly anti-aligned
     with the mean rather than just weakly aligned).

  3. Residual PCA: after removing each instruction's along-mean component,
     runs PCA (via SVD) on the 300 residual vectors and reports the top-3
     components' share of residual variance. If residual variance is spread
     roughly evenly across many components, that's consistent with diffuse
     noise (no further structure). If a small number of components capture
     a disproportionate share, that's evidence of a genuine second (or
     third) direction within the template -- i.e. sub-groups, not just
     noise around one mean.

Also directly compares all 6 real mechanisms' R2_mean and mean-cosine-to-mean
side by side at the fixed floor(0.6*n_layers) layer, per model, to check
whether persona_roleplay (flagged in Exp1's leave-one-out analysis as the
one mechanism whose removal consistently improved the CO/MG clustering
metric, in all 3 models) is also an outlier on internal coherence -- i.e.
whether it drags down clustering because it is itself unusually internally
scattered, rather than for some other reason.

Usage:
  python scripts/29_within_template_dispersion.py \
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
ALL_MECHS_FOR_REFERENCE = REAL_MECHS + ['placebo']
FIXED_LAYER_FRACTION = 0.6


def analyze_mechanism_at_layer(vectors):
    """vectors: [n, d] tensor for one mechanism at one layer. Returns dict of metrics."""
    mean_vec = vectors.mean(0)  # [d]
    mean_norm = mean_vec.norm()
    if mean_norm < 1e-8:
        return None  # mean direction is ~zero, projection/cosine undefined -- caller should skip/flag
    mean_hat = mean_vec / mean_norm

    proj_scalars = vectors @ mean_hat  # [n] -- signed length along mean_hat
    proj_vecs = proj_scalars.unsqueeze(-1) * mean_hat.unsqueeze(0)  # [n, d]
    residuals = vectors - proj_vecs  # [n, d]

    total_ss = (vectors ** 2).sum().item()
    proj_ss = (proj_vecs ** 2).sum().item()
    r2_mean = proj_ss / total_ss if total_ss > 0 else None

    cos_to_mean = F.cosine_similarity(vectors, mean_vec.unsqueeze(0).expand_as(vectors), dim=-1)

    # residual PCA via SVD (residuals are NOT re-centered -- we want the
    # structure of what's left after removing the along-mean component, not
    # variance around the residuals' own mean, which would remove yet
    # another degree of freedom and is not what "is there a second
    # direction within the template" is asking)
    try:
        _, S, _ = torch.linalg.svd(residuals, full_matrices=False)
        residual_var = S ** 2
        residual_total = residual_var.sum().item()
        if residual_total > 0:
            top_k = min(3, len(residual_var))
            top_fracs = (residual_var[:top_k] / residual_total).tolist()
        else:
            top_fracs = [0.0] * min(3, residual_var.numel())
    except Exception:
        top_fracs = None

    return {
        'r2_mean': r2_mean,
        'cosine_to_mean': {
            'mean': cos_to_mean.mean().item(), 'std': cos_to_mean.std().item(),
            'min': cos_to_mean.min().item(), 'p25': cos_to_mean.quantile(0.25).item(),
            'median': cos_to_mean.median().item(), 'p75': cos_to_mean.quantile(0.75).item(),
            'max': cos_to_mean.max().item(),
        },
        'residual_pca_top3_variance_fraction': top_fracs,
    }


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
        print(f"  Loaded {path}  n_layers={n_layers}  fixed_layer={fixed_layer}")

        model_results = {'n_layers': n_layers, 'fixed_layer': fixed_layer, 'per_mechanism': {}}
        for mech in ALL_MECHS_FOR_REFERENCE:
            if mech not in data['diffs']:
                print(f"  [{mech}] MISSING from paired_diffs -- skipping")
                continue
            diffs = data['diffs'][mech].float()  # [n, n_layers, d]
            per_layer = []
            for l in range(n_layers):
                m = analyze_mechanism_at_layer(diffs[:, l, :])
                per_layer.append(m)
            model_results['per_mechanism'][mech] = {'per_layer': per_layer}

        results['per_model'][model_alias] = model_results

        print(f"\n  --- At fixed layer {fixed_layer} (floor(0.6*n_layers)), 6 real mechanisms compared ---")
        print(f"  {'mechanism':24s}  {'R2_mean':>8s}  {'mean_cos_to_mean':>16s}  {'min_cos':>8s}  "
              f"{'top1_resid_frac':>15s}")
        r2_by_mech = {}
        for mech in REAL_MECHS:
            fl = model_results['per_mechanism'].get(mech, {}).get('per_layer', [None] * n_layers)[fixed_layer]
            if fl is None:
                print(f"  {mech:24s}  (undefined -- mean direction ~zero at this layer)")
                continue
            r2_by_mech[mech] = fl['r2_mean']
            top1 = fl['residual_pca_top3_variance_fraction'][0] if fl['residual_pca_top3_variance_fraction'] else None
            top1_str = f"{top1:15.4f}" if top1 is not None else f"{'n/a':>15s}"
            print(f"  {mech:24s}  {fl['r2_mean']:8.4f}  {fl['cosine_to_mean']['mean']:16.4f}  "
                  f"{fl['cosine_to_mean']['min']:8.4f}  {top1_str}")

        if r2_by_mech:
            lowest_r2_mech = min(r2_by_mech, key=r2_by_mech.get)
            print(f"\n  Lowest R2_mean (most internally scattered): {lowest_r2_mech} "
                  f"(R2_mean={r2_by_mech[lowest_r2_mech]:.4f})")
            is_persona_lowest = lowest_r2_mech == 'persona_roleplay'
            model_results['persona_roleplay_is_most_scattered_at_fixed_layer'] = is_persona_lowest
            print(f"  persona_roleplay is the most scattered: {is_persona_lowest}")

    out_path = os.path.join(args.output_dir, f'within_template_dispersion_{args.lang}{args.suffix}.json')
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
