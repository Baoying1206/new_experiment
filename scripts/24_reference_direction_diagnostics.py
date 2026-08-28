"""
Sanity-checks the newly-rebuilt v2 dual-position reference directions
(scripts/23_extract_reference_directions.py) before they're used to build
δR/δH -- CPU-only, no model/GPU needed here.

Per Decision 2/3's requirement, must report cosine(refusal_direction,
harmfulness_direction) per layer -- treating these as orthogonal when
they're actually correlated (Zhao et al. 2025's whole point) would bias any
downstream projection-based δR/δH computation.

Also checks for basic extraction sanity: near-zero norm (would suggest the
harmful/harmless contrast produced no signal at that layer -- possible but
worth flagging, not necessarily wrong) and NaN/Inf (would indicate a real
extraction bug).

Usage:
  python scripts/24_reference_direction_diagnostics.py \
      --output_dir output --model_alias Qwen2.5-7B-Instruct --lang en
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(__file__)


def main(args):
    v2_dir = os.path.join(args.output_dir, 'output_v2_dual_position', args.model_alias)
    refusal_dir = torch.load(os.path.join(v2_dir, f'refusal_dir_v2_{args.lang}.pt'), map_location='cpu').float()
    harmfulness_dir = torch.load(os.path.join(v2_dir, f'harmfulness_dir_v2_{args.lang}.pt'), map_location='cpu').float()

    assert refusal_dir.shape == harmfulness_dir.shape, (
        f"shape mismatch: refusal_direction {refusal_dir.shape} vs "
        f"harmfulness_direction {harmfulness_dir.shape}"
    )
    n_layers, d_model = refusal_dir.shape
    print(f"Model: {args.model_alias}  n_layers={n_layers}  d_model={d_model}\n")

    cos_per_layer = F.cosine_similarity(refusal_dir, harmfulness_dir, dim=-1)  # [n_layers]
    refusal_norm = refusal_dir.norm(dim=-1)
    harmfulness_norm = harmfulness_dir.norm(dim=-1)

    n_nan_inf = int((~torch.isfinite(cos_per_layer)).sum().item())
    near_zero_refusal = (refusal_norm < 1e-4).nonzero().flatten().tolist()
    near_zero_harmfulness = (harmfulness_norm < 1e-4).nonzero().flatten().tolist()

    print("=== cos(refusal_direction, harmfulness_direction) per layer ===")
    for l in range(n_layers):
        flag = ''
        if not torch.isfinite(cos_per_layer[l]):
            flag = '  <-- NaN/Inf'
        print(f"  layer {l:2d}: cos={cos_per_layer[l]:+.4f}  "
              f"|refusal|={refusal_norm[l]:.3f}  |harmfulness|={harmfulness_norm[l]:.3f}{flag}")

    print(f"\nMean cos across layers: {cos_per_layer[torch.isfinite(cos_per_layer)].mean():.4f}")
    print(f"Range: [{cos_per_layer[torch.isfinite(cos_per_layer)].min():.4f}, "
          f"{cos_per_layer[torch.isfinite(cos_per_layer)].max():.4f}]")
    print(f"NaN/Inf layers: {n_nan_inf}")
    print(f"Near-zero-norm refusal_direction layers: {near_zero_refusal}")
    print(f"Near-zero-norm harmfulness_direction layers: {near_zero_harmfulness}")

    if n_nan_inf > 0:
        print("\nWARNING: NaN/Inf in cosine similarity -- likely a real extraction bug, investigate "
              "before using these directions for anything.")
    if near_zero_refusal or near_zero_harmfulness:
        print("\nNOTE: near-zero-norm layers found -- not necessarily a bug (early/late layers can "
              "legitimately have weak harmful-vs-harmless signal) but worth checking those specific "
              "layers aren't ones you plan to use as the primary analysis layer.")
    if abs(cos_per_layer[torch.isfinite(cos_per_layer)].mean().item()) > 0.9:
        print("\nNOTE: mean cosine is very high in magnitude -- refusal_direction and "
              "harmfulness_direction are nearly collinear on average. This doesn't invalidate the "
              "dual-axis framing (Zhao et al. 2025's point is exactly that these are correlated, not "
              "orthogonal) but means δR/δH will carry substantial shared variance -- report this "
              "cosine explicitly wherever δR/δH results are presented, per Decision 2/3.")

    results = {
        'model': args.model_alias, 'lang': args.lang, 'n_layers': n_layers, 'd_model': d_model,
        'cos_per_layer': cos_per_layer.tolist(),
        'refusal_norm_per_layer': refusal_norm.tolist(),
        'harmfulness_norm_per_layer': harmfulness_norm.tolist(),
        'mean_cos': cos_per_layer[torch.isfinite(cos_per_layer)].mean().item(),
        'n_nan_inf_layers': n_nan_inf,
        'near_zero_norm_refusal_layers': near_zero_refusal,
        'near_zero_norm_harmfulness_layers': near_zero_harmfulness,
    }
    out_path = os.path.join(v2_dir, f'reference_direction_diagnostics_{args.lang}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',  type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    parser.add_argument('--model_alias', type=str, required=True)
    parser.add_argument('--lang',        type=str, default='en')
    args = parser.parse_args()
    main(args)
