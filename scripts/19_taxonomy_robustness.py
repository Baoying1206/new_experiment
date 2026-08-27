"""
Stage-1 core diagnostics: is "Wei et al.'s taxonomy doesn't cluster
geometrically" (Section 09) robust, or an artifact of layer choice, a few
outlier instructions, or one dominant template?

Requires paired_diffs_{lang}{suffix}.pt from 18_extract_paired_diffs.py --
CPU-only, no model/GPU needed here. Pass the same --suffix used when 18 was
run (e.g. '_full572' for English's confirmatory direction-set run,
'_xling' for the 5 confirmatory non-English languages) -- omitting it reads
the original unsuffixed 75-instruction pilot's paired_diffs_{lang}.pt.

Implements, at EVERY layer (not a single chosen layer, sidestepping the
layer-selection problem entirely by reporting consistency across all of
them):

  1. Exact enumeration test: with 6 templates split 3-3, there are exactly
     C(6,3)/2 = 10 distinct partitions. T(partition) = mean(within-group
     cosine) - mean(between-group cosine). Reports where Wei et al.'s
     specific partition (competing_objectives vs mismatched_generalization)
     ranks among all 10, per layer -- not just a p-value, per the plan's
     own caution that p-values are uninformative with only 10 possible
     partitions.
  2. Bootstrap (1000 reps, resampling instruction ids with replacement,
     paired across mechanisms): CI on Wei's T at every layer.
  3. Split-half (200 reps): per-mechanism direction stability R_t = cos of
     the two half-sample directions, at every layer -- distinguishes "the
     categories don't cluster" from "the individual template directions
     are themselves too noisy to mean anything."
  4. Leave-one-template-out: recompute Wei's T with each of the 6 templates
     removed in turn, to check no single template is driving the result.
  5. Placebo-calibrated re-test: delta_h^PC_i,t = delta_h_i,t - delta_h_i,placebo,
     then repeat (1) on the calibrated vectors -- tests whether generic
     template-wrapper effects (not mechanism-specific ones) explain the
     non-clustering.

NOT implemented here (deferred, see conversation): median/trimmed-mean
variants of the bootstrap, surface-feature-vs-activation-distance
comparison (separate script, text-only), and the model/language mixed
regression (needs paired_diffs extracted across multiple languages first).

Usage:
  python scripts/19_taxonomy_robustness.py \
      --output_dir output \
      --model_alias Qwen2.5-7B-Instruct \
      --lang en \
      --n_bootstrap 1000 \
      --n_splithalf 200
"""
import argparse
import json
import os
import random
from itertools import combinations

import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(__file__)
REAL_MECHS = ['prefix_injection', 'refusal_suppression', 'instruction_hierarchy',
              'persona_roleplay', 'fictional_framing', 'encoding_obfuscation']
CO = ['prefix_injection', 'refusal_suppression', 'instruction_hierarchy']
MG = ['persona_roleplay', 'fictional_framing', 'encoding_obfuscation']


def all_3v3_partitions(mechs):
    mechs_set = set(mechs)
    seen, partitions = set(), []
    for combo in combinations(mechs, 3):
        a = frozenset(combo)
        b = frozenset(mechs_set - a)
        key = frozenset([a, b])
        if key in seen:
            continue
        seen.add(key)
        partitions.append((sorted(a), sorted(b)))
    return partitions


def T_statistic_all_layers(vecs_all_layers, group_a, group_b):
    """vecs_all_layers: {mech: tensor[n_layers, d]}. Returns tensor[n_layers]."""
    def pair_cos(m1, m2):
        return F.cosine_similarity(vecs_all_layers[m1], vecs_all_layers[m2], dim=-1)  # [n_layers]
    within = [pair_cos(m1, m2) for i, m1 in enumerate(group_a) for m2 in group_a[i + 1:]]
    within += [pair_cos(m1, m2) for i, m1 in enumerate(group_b) for m2 in group_b[i + 1:]]
    between = [pair_cos(m1, m2) for m1 in group_a for m2 in group_b]
    return torch.stack(within, 0).mean(0) - torch.stack(between, 0).mean(0)


def build_id_index(diffs_data, mechs):
    return {m: {pid: i for i, pid in enumerate(diffs_data['instruction_ids'][m])} for m in mechs}


def common_ids_for(diffs_data, mechs):
    id_lists = [set(diffs_data['instruction_ids'][m]) for m in mechs]
    return sorted(set.intersection(*id_lists))


def mean_vecs_all_layers(diffs_data, id_index, mechs, ids):
    """Returns {mech: tensor[n_layers, d]} -- mean over `ids` at every layer."""
    out = {}
    for m in mechs:
        idxs = [id_index[m][pid] for pid in ids]
        out[m] = diffs_data['diffs'][m][idxs].mean(0)  # [n_layers, d]
    return out


def main(args):
    out_dir = os.path.join(args.output_dir, args.model_alias)
    diffs_path = os.path.join(out_dir, f'paired_diffs_{args.lang}{args.suffix}.pt')
    diffs_data = torch.load(diffs_path, map_location='cpu')
    diffs_data['diffs'] = {m: v.float() for m, v in diffs_data['diffs'].items()}
    n_layers = diffs_data['n_layers']
    ids_key = diffs_data.get('ids_key')
    scope_desc = f"built from splits.json['{ids_key}']" if ids_key else \
        "unfiltered -- no --ids_key was used when this .pt was built"
    print(f"Loaded {diffs_path}  n_layers={n_layers}  ({scope_desc})")

    id_index = build_id_index(diffs_data, REAL_MECHS + ['placebo'])
    common_ids = common_ids_for(diffs_data, REAL_MECHS)
    common_ids_pc = common_ids_for(diffs_data, REAL_MECHS + ['placebo'])
    print(f"common_ids (6 mechs) = {len(common_ids)}  common_ids_with_placebo = {len(common_ids_pc)}\n")

    partitions = all_3v3_partitions(REAL_MECHS)
    wei_idx = next(i for i, (a, b) in enumerate(partitions)
                    if set(a) == set(CO) or set(a) == set(MG))
    print(f"10 possible 3-3 partitions enumerated; Wei et al.'s partition is #{wei_idx}\n")

    results = {'model': args.model_alias, 'lang': args.lang, 'suffix': args.suffix,
               'ids_key': ids_key, 'n_layers': n_layers, 'n_common_ids': len(common_ids)}

    # ── 1+2: exact enumeration + bootstrap ──────────────────────────────────
    print("=== 1. Exact enumeration test (observed data) ===")
    vecs = mean_vecs_all_layers(diffs_data, id_index, REAL_MECHS, common_ids)
    all_T = torch.stack([T_statistic_all_layers(vecs, a, b) for a, b in partitions], 0)  # [10, n_layers]
    wei_T = all_T[wei_idx]  # [n_layers]
    ranks = (all_T >= wei_T.unsqueeze(0)).sum(0)  # [n_layers], 1=highest

    n_layers_top1 = int((ranks == 1).sum().item())
    n_layers_positive = int((wei_T > 0).sum().item())
    print(f"  Wei partition is rank #1 (out of 10) in {n_layers_top1}/{n_layers} layers")
    print(f"  Wei T > 0 in {n_layers_positive}/{n_layers} layers")
    print(f"  Wei T range: [{wei_T.min():.4f}, {wei_T.max():.4f}]  mean={wei_T.mean():.4f}")
    results['exact_permutation'] = {
        'wei_T_per_layer': wei_T.tolist(), 'rank_per_layer': ranks.tolist(),
        'n_layers_top1': n_layers_top1, 'n_layers_positive': n_layers_positive,
        'mean_wei_T': wei_T.mean().item(),
    }

    print("\n=== 2. Bootstrap (n={}) ===".format(args.n_bootstrap))
    boot_T = torch.zeros(args.n_bootstrap, n_layers)
    rng = random.Random(0)
    for b in range(args.n_bootstrap):
        resampled = [common_ids[rng.randrange(len(common_ids))] for _ in range(len(common_ids))]
        v = mean_vecs_all_layers(diffs_data, id_index, REAL_MECHS, resampled)
        boot_T[b] = T_statistic_all_layers(v, CO, MG)
    ci_low = boot_T.quantile(0.025, dim=0)
    ci_high = boot_T.quantile(0.975, dim=0)
    frac_positive = (boot_T > 0).float().mean(0)
    n_layers_ci_excludes_zero_positive = int(((ci_low > 0)).sum().item())
    print(f"  Layers where 95% CI is entirely > 0 (stable positive effect): "
          f"{n_layers_ci_excludes_zero_positive}/{n_layers}")
    print(f"  Mean frac_positive across layers: {frac_positive.mean():.3f}")
    results['bootstrap'] = {
        'ci_low_per_layer': ci_low.tolist(), 'ci_high_per_layer': ci_high.tolist(),
        'frac_positive_per_layer': frac_positive.tolist(),
        'n_layers_ci_excludes_zero_positive': n_layers_ci_excludes_zero_positive,
    }

    # ── 3: split-half per-mechanism stability ───────────────────────────────
    print(f"\n=== 3. Split-half stability (n={args.n_splithalf}) ===")
    results['split_half'] = {}
    for m in REAL_MECHS:
        cos_reps = torch.zeros(args.n_splithalf, n_layers)
        for r in range(args.n_splithalf):
            shuffled = common_ids.copy()
            rng.shuffle(shuffled)
            half = len(shuffled) // 2
            idxA = [id_index[m][pid] for pid in shuffled[:half]]
            idxB = [id_index[m][pid] for pid in shuffled[half:]]
            vecA = diffs_data['diffs'][m][idxA].mean(0)
            vecB = diffs_data['diffs'][m][idxB].mean(0)
            cos_reps[r] = F.cosine_similarity(vecA, vecB, dim=-1)
        mean_R = cos_reps.mean(0)
        print(f"  [{m}] split-half R: mean={mean_R.mean():.3f}  "
              f"min={mean_R.min():.3f}  max={mean_R.max():.3f}")
        results['split_half'][m] = {'mean_R_per_layer': mean_R.tolist()}

    # ── 4: leave-one-template-out ────────────────────────────────────────────
    print("\n=== 4. Leave-one-template-out ===")
    results['leave_one_out'] = {}
    for removed in REAL_MECHS:
        group_a = [m for m in CO if m != removed]
        group_b = [m for m in MG if m != removed]
        v = mean_vecs_all_layers(diffs_data, id_index, [m for m in REAL_MECHS if m != removed], common_ids)
        T = T_statistic_all_layers(v, group_a, group_b)
        n_pos = int((T > 0).sum().item())
        print(f"  [remove {removed}] T>0 in {n_pos}/{n_layers} layers  mean_T={T.mean():.4f}")
        results['leave_one_out'][removed] = {'T_per_layer': T.tolist(), 'n_layers_positive': n_pos}

    # ── 5: placebo-calibrated re-test ────────────────────────────────────────
    print("\n=== 5. Placebo-calibrated re-test ===")
    pc_diffs = {}
    for m in REAL_MECHS:
        idx_m = [id_index[m][pid] for pid in common_ids_pc]
        idx_p = [id_index['placebo'][pid] for pid in common_ids_pc]
        pc_diffs[m] = diffs_data['diffs'][m][idx_m] - diffs_data['diffs']['placebo'][idx_p]  # [n, n_layers, d]
    pc_vecs = {m: pc_diffs[m].mean(0) for m in REAL_MECHS}  # [n_layers, d]
    pc_all_T = torch.stack([T_statistic_all_layers(pc_vecs, a, b) for a, b in partitions], 0)
    pc_wei_T = pc_all_T[wei_idx]
    pc_ranks = (pc_all_T >= pc_wei_T.unsqueeze(0)).sum(0)
    pc_n_top1 = int((pc_ranks == 1).sum().item())
    pc_n_pos = int((pc_wei_T > 0).sum().item())
    print(f"  [placebo-calibrated] Wei partition rank #1 in {pc_n_top1}/{n_layers} layers, "
          f"T>0 in {pc_n_pos}/{n_layers} layers, mean_T={pc_wei_T.mean():.4f}")
    print(f"  [uncalibrated, for comparison] rank #1 in {n_layers_top1}/{n_layers}, "
          f"T>0 in {n_layers_positive}/{n_layers}, mean_T={wei_T.mean():.4f}")
    results['placebo_calibrated'] = {
        'wei_T_per_layer': pc_wei_T.tolist(), 'n_layers_top1': pc_n_top1,
        'n_layers_positive': pc_n_pos, 'mean_wei_T': pc_wei_T.mean().item(),
    }

    out_path = os.path.join(out_dir, f'taxonomy_robustness_{args.lang}{args.suffix}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',    type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    parser.add_argument('--model_alias',   type=str, required=True)
    parser.add_argument('--lang',          type=str, default='en')
    parser.add_argument('--suffix',        type=str, default='',
                         help="Matches the --suffix used to build paired_diffs_{lang}{suffix}.pt "
                              "in 18_extract_paired_diffs.py, e.g. '_full572' or '_xling'.")
    parser.add_argument('--n_bootstrap',   type=int, default=1000)
    parser.add_argument('--n_splithalf',   type=int, default=200)
    args = parser.parse_args()
    main(args)
