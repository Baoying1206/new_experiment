"""
Re-runs 19_taxonomy_robustness.py's exact battery of tests on the CORRECTED
Wei et al. taxonomy (scripts/_taxonomy_config.py's CORRECTED_CO_MECHS/
CORRECTED_MG_MECHS: CO=[prefix_injection, refusal_suppression,
persona_roleplay], MG=[encoding_obfuscation, payload_splitting,
distractors_negated] -- verified against arXiv 2307.02483 fulltext, see
_taxonomy_config.py and templates/templates_en.json for the full citation
trail). 19 itself is intentionally left unmodified (still tests the old,
now-known-mislabeled mapping) -- this is a separate script per instruction,
not a refactor of 19.

Same statistical logic as 19 verbatim (exact 3v3 enumeration test, ordinary-
mean vs robust-estimator comparison, bootstrap, split-half, leave-one-out,
placebo-calibrated re-test) -- only the mechanism set and default input
suffix differ. Reads paired_diffs_{lang}{suffix}.pt with
--suffix _full572_corrected by default (18_extract_paired_diffs.py
--mechanisms prefix_injection,refusal_suppression,persona_roleplay,
encoding_obfuscation,payload_splitting,distractors_negated).

CPU-only, no GPU needed.

Usage:
  python scripts/32_taxonomy_robustness_corrected.py \
      --output_dir output --model_alias Qwen2.5-7B-Instruct --lang en
"""
import argparse
import json
import os
import random
import sys
from itertools import combinations

import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)
import _taxonomy_config as tc

REAL_MECHS = tc.CORRECTED_REAL_MECHS
CO = tc.CORRECTED_CO_MECHS
MG = tc.CORRECTED_MG_MECHS


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
    def pair_cos(m1, m2):
        return F.cosine_similarity(vecs_all_layers[m1], vecs_all_layers[m2], dim=-1)
    within = [pair_cos(m1, m2) for i, m1 in enumerate(group_a) for m2 in group_a[i + 1:]]
    within += [pair_cos(m1, m2) for i, m1 in enumerate(group_b) for m2 in group_b[i + 1:]]
    between = [pair_cos(m1, m2) for m1 in group_a for m2 in group_b]
    return torch.stack(within, 0).mean(0) - torch.stack(between, 0).mean(0)


def build_id_index(diffs_data, mechs):
    return {m: {pid: i for i, pid in enumerate(diffs_data['instruction_ids'][m])} for m in mechs}


def common_ids_for(diffs_data, mechs):
    id_lists = [set(diffs_data['instruction_ids'][m]) for m in mechs]
    return sorted(set.intersection(*id_lists))


ESTIMATORS = ['mean', 'median', 'trimmed_mean']


def aggregate(x, method, trim_frac=0.1):
    if method == 'mean':
        return x.mean(0)
    if method == 'median':
        return x.median(0).values
    if method == 'trimmed_mean':
        n = x.shape[0]
        k = int(n * trim_frac)
        if k == 0:
            return x.mean(0)
        sorted_x, _ = torch.sort(x, dim=0)
        return sorted_x[k:n - k].mean(0)
    raise ValueError(f"unknown estimator: {method}")


def mean_vecs_all_layers(diffs_data, id_index, mechs, ids, method='mean', trim_frac=0.1):
    out = {}
    for m in mechs:
        idxs = [id_index[m][pid] for pid in ids]
        out[m] = aggregate(diffs_data['diffs'][m][idxs], method, trim_frac)
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
    print(f"Corrected mechanism set: CO={CO}  MG={MG}\n")

    missing = set(REAL_MECHS + ['placebo']) - set(diffs_data['diffs'].keys())
    if missing:
        raise ValueError(f"paired_diffs file is missing mechanisms {missing} -- was it extracted "
                          f"with --mechanisms matching CORRECTED_REAL_MECHS? Got keys: "
                          f"{sorted(diffs_data['diffs'].keys())}")

    id_index = build_id_index(diffs_data, REAL_MECHS + ['placebo'])
    common_ids = common_ids_for(diffs_data, REAL_MECHS)
    common_ids_pc = common_ids_for(diffs_data, REAL_MECHS + ['placebo'])
    print(f"common_ids (6 mechs) = {len(common_ids)}  common_ids_with_placebo = {len(common_ids_pc)}\n")

    partitions = all_3v3_partitions(REAL_MECHS)
    wei_idx = next(i for i, (a, b) in enumerate(partitions)
                    if set(a) == set(CO) or set(a) == set(MG))
    print(f"10 possible 3-3 partitions enumerated; Wei et al.'s (corrected) partition is #{wei_idx}\n")

    results = {'model': args.model_alias, 'lang': args.lang, 'suffix': args.suffix,
               'ids_key': ids_key, 'n_layers': n_layers, 'n_common_ids': len(common_ids),
               'taxonomy_version': 'wei_canonical_v2', 'CO': CO, 'MG': MG}

    # ── 1+1b: exact enumeration + robust estimators ─────────────────────────
    print("=== 1. Exact enumeration test (observed data) ===")
    vecs = mean_vecs_all_layers(diffs_data, id_index, REAL_MECHS, common_ids)
    all_T = torch.stack([T_statistic_all_layers(vecs, a, b) for a, b in partitions], 0)
    wei_T = all_T[wei_idx]
    ranks = (all_T >= wei_T.unsqueeze(0)).sum(0)

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

    print("\n=== 1b. Ordinary mean vs robust estimators (exact enumeration test) ===")
    results['estimator_comparison'] = {}
    for method in ESTIMATORS:
        if method == 'mean':
            T_est, ranks_est = wei_T, ranks
        else:
            v_est = mean_vecs_all_layers(diffs_data, id_index, REAL_MECHS, common_ids,
                                          method=method, trim_frac=args.trim_frac)
            all_T_est = torch.stack([T_statistic_all_layers(v_est, a, b) for a, b in partitions], 0)
            T_est = all_T_est[wei_idx]
            ranks_est = (all_T_est >= T_est.unsqueeze(0)).sum(0)
        n_top1_est = int((ranks_est == 1).sum().item())
        n_pos_est = int((T_est > 0).sum().item())
        print(f"  [{method}] Wei rank #1 in {n_top1_est}/{n_layers} layers, "
              f"T>0 in {n_pos_est}/{n_layers} layers, mean_T={T_est.mean():.4f}")
        results['estimator_comparison'][method] = {
            'wei_T_per_layer': T_est.tolist(), 'n_layers_top1': n_top1_est,
            'n_layers_positive': n_pos_est, 'mean_wei_T': T_est.mean().item(),
        }
    agree = len(set(results['estimator_comparison'][m]['n_layers_top1'] > 0 for m in ESTIMATORS)) == 1
    print(f"  --> estimators agree on whether Wei's partition is ever rank #1: {agree}")
    results['estimator_comparison']['estimators_agree_qualitatively'] = agree

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

    print("\n=== 5. Placebo-calibrated re-test ===")
    pc_diffs = {}
    for m in REAL_MECHS:
        idx_m = [id_index[m][pid] for pid in common_ids_pc]
        idx_p = [id_index['placebo'][pid] for pid in common_ids_pc]
        pc_diffs[m] = diffs_data['diffs'][m][idx_m] - diffs_data['diffs']['placebo'][idx_p]
    pc_vecs = {m: pc_diffs[m].mean(0) for m in REAL_MECHS}
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

    out_path = os.path.join(out_dir, f'taxonomy_robustness_corrected_{args.lang}{args.suffix}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',    type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    parser.add_argument('--model_alias',   type=str, required=True)
    parser.add_argument('--lang',          type=str, default='en')
    parser.add_argument('--suffix',        type=str, default='_full572_corrected')
    parser.add_argument('--n_bootstrap',   type=int, default=1000)
    parser.add_argument('--n_splithalf',   type=int, default=200)
    parser.add_argument('--trim_frac',     type=float, default=0.1)
    args = parser.parse_args()
    main(args)
