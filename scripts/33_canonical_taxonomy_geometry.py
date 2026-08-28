"""
Experiment 1, canonical V2 taxonomy: tests whether the literature-grounded
competing_objectives (CO) / mismatched_generalization (MG) partition of the
6 active mechanisms (scripts/_taxonomy_v2_loader.py, read dynamically from
templates/templates_en.json -- no hardcoded CO/MG list in this file) shows
within-group cohesion and between-group separation in paired activation-
shift geometry.

CORE MEASUREMENT: cosine similarity between MECHANISMS' MEAN PAIRED-DIFFERENCE
DIRECTIONS, never raw activations of different prompts directly (those share
source instruction / chat template / common residual-stream content and
would be spuriously similar for reasons having nothing to do with the
mechanism itself):
    delta_h[i,m,l] = activation(template_prompt[i,m], l) - activation(plain_prompt[i], l)
    d_m[l] = aggregate_i( delta_h[i,m,l] )   (mean, or median/trimmed_mean as sensitivity)
    s(a,b,l) = cos(d_a[l], d_b[l])

CPU-only. Reads paired_diffs_{lang}_full572_corrected.pt (18_extract_paired_diffs.py
--mechanisms matching the V2 active_mechanisms) for each model -- no GPU
needed, no new extraction.

Fixed pre-registered layers (0-based tensor indices, confirmed against this
project's existing floor(0.6*n_layers) convention -- NOT re-selected from
this script's own results): Qwen=16, Llama=19, Gemma=25. All layers are also
swept as a point-estimate-only sensitivity analysis (no bootstrap at every
layer -- 2000-rep bootstrap runs ONLY at each model's fixed layer, for all
3 estimators x {raw, placebo_calibrated} = 6 combinations).

Primary combination: mean estimator, placebo_calibrated. The other 5
combinations are computed identically (same code path) as robustness checks.

Outputs:
  output/canonical_v2/experiment1_taxonomy_geometry.json
  output/canonical_v2/experiment1_pairwise_cosine.csv
  output/canonical_v2/experiment1_partition_scores.csv
  output/canonical_v2/experiment1_bootstrap.json
  output/canonical_v2/figures/experiment1_cosine_heatmap.{png,pdf}
  output/canonical_v2/figures/experiment1_layerwise_taxonomy.{png,pdf}
  output/canonical_v2/figures/experiment1_partition_ranking.{png,pdf}

Usage:
  python scripts/33_canonical_taxonomy_geometry.py --output_dir output
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sys
from itertools import combinations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)
from _taxonomy_v2_loader import load_taxonomy_v2

SPLITS_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'splits.json')
MODELS = ['Qwen2.5-7B-Instruct', 'Meta-Llama-3.1-8B-Instruct', 'gemma-2-9b-it']
FIXED_LAYERS = {'Qwen2.5-7B-Instruct': 16, 'Meta-Llama-3.1-8B-Instruct': 19, 'gemma-2-9b-it': 25}
ESTIMATORS = ['mean', 'median', 'trimmed_mean']
TRIM_FRAC = 0.1  # matches 19_taxonomy_robustness.py / 32_taxonomy_robustness_corrected.py's existing default
RAW_PC = ['raw', 'placebo_calibrated']
PRIMARY_ESTIMATOR = 'mean'
PRIMARY_RAW_PC = 'placebo_calibrated'
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260828
SPLIT_HALF_REPS = 500
SPLIT_HALF_SEED = 20260828
TIE_TOL = 1e-12


# ── core math ────────────────────────────────────────────────────────────

def aggregate(x, method, trim_frac=TRIM_FRAC):
    """x: [n, d]. Returns [d]."""
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


def cos(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item()


def all_3v3_partitions(mechs):
    """A|B and B|A are the same partition -- deduplicated via a frozenset-of-frozensets key."""
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


def compute_partition_stats(vecs, group_a, group_b):
    """vecs: {mech: [d]}. Returns S_a, S_b, S_between, delta_a, delta_b, T."""
    within_a = [cos(vecs[group_a[i]], vecs[group_a[j]]) for i in range(3) for j in range(i + 1, 3)]
    within_b = [cos(vecs[group_b[i]], vecs[group_b[j]]) for i in range(3) for j in range(i + 1, 3)]
    between = [cos(vecs[a], vecs[b]) for a in group_a for b in group_b]
    S_a = sum(within_a) / len(within_a)
    S_b = sum(within_b) / len(within_b)
    S_between = sum(between) / len(between)
    delta_a = S_a - S_between
    delta_b = S_b - S_between
    T = (S_a + S_b) / 2 - S_between
    return S_a, S_b, S_between, delta_a, delta_b, T


def rank_partitions(T_values, tol=TIE_TOL):
    """Minimum-rank method with a tie tolerance: ties within `tol` of each other share the
    best (lowest) rank number. Returns a list of ranks, 1 = best, same length as T_values."""
    n = len(T_values)
    ranks = []
    for i in range(n):
        r = 1 + sum(1 for j in range(n) if T_values[j] > T_values[i] + tol)
        ranks.append(r)
    return ranks


def mech_vecs_at_layer(diffs_data, id_index, mechs, ids, layer, method, raw_or_pc, trim_frac=TRIM_FRAC):
    out = {}
    for m in mechs:
        idxs = [id_index[m][pid] for pid in ids]
        vecs = diffs_data['diffs'][m][idxs, layer, :]
        if raw_or_pc == 'placebo_calibrated':
            idxs_p = [id_index['placebo'][pid] for pid in ids]
            vecs = vecs - diffs_data['diffs']['placebo'][idxs_p, layer, :]
        out[m] = aggregate(vecs, method, trim_frac)
    return out


# ── per-model pipeline ──────────────────────────────────────────────────

def load_model_diffs(output_dir, model_alias, lang, suffix, active_mechanisms):
    path = os.path.join(output_dir, model_alias, f'paired_diffs_{lang}{suffix}.pt')
    diffs_data = torch.load(path, map_location='cpu')
    diffs_data['diffs'] = {m: v.float() for m, v in diffs_data['diffs'].items()}
    missing = set(active_mechanisms + ['placebo']) - set(diffs_data['diffs'].keys())
    if missing:
        raise ValueError(f"{path} is missing mechanisms {missing} -- was it extracted with "
                          f"--mechanisms matching the V2 active_mechanisms? Got keys: "
                          f"{sorted(diffs_data['diffs'].keys())}")
    id_index = {m: {pid: i for i, pid in enumerate(diffs_data['instruction_ids'][m])}
                for m in active_mechanisms + ['placebo']}
    id_sets = [set(diffs_data['instruction_ids'][m]) for m in active_mechanisms + ['placebo']]
    common_ids = sorted(set.intersection(*id_sets))
    return path, diffs_data, id_index, common_ids


def point_estimate_sweep(diffs_data, id_index, common_ids, n_layers, CO, MG, active_mechanisms,
                          partitions, wei_idx, model_alias):
    """Returns (pairwise_rows, partition_rows) -- long-format lists of dicts, all layers,
    all estimators, both raw/pc."""
    pairwise_rows, partition_rows = [], []
    fixed_layer = FIXED_LAYERS[model_alias]
    for layer in range(n_layers):
        layer_ordinal = layer + 1
        relative_depth = layer / (n_layers - 1) if n_layers > 1 else 0.0
        for estimator in ESTIMATORS:
            for raw_or_pc in RAW_PC:
                vecs = mech_vecs_at_layer(diffs_data, id_index, active_mechanisms, common_ids,
                                           layer, estimator, raw_or_pc)
                for i, m1 in enumerate(active_mechanisms):
                    for m2 in active_mechanisms[i + 1:]:
                        pairwise_rows.append({
                            'model': model_alias, 'layer_index': layer, 'layer_ordinal': layer_ordinal,
                            'relative_depth': relative_depth, 'estimator': estimator,
                            'raw_or_pc': raw_or_pc, 'mechanism_a': m1, 'mechanism_b': m2,
                            'cosine': cos(vecs[m1], vecs[m2]),
                        })
                all_T = []
                for p_idx, (a, b) in enumerate(partitions):
                    S_a, S_b, S_between, delta_a, delta_b, T = compute_partition_stats(vecs, a, b)
                    all_T.append(T)
                    partition_rows.append({
                        'model': model_alias, 'layer_index': layer, 'layer_ordinal': layer_ordinal,
                        'relative_depth': relative_depth, 'estimator': estimator,
                        'raw_or_pc': raw_or_pc, 'partition_idx': p_idx,
                        'group_a': '|'.join(a), 'group_b': '|'.join(b),
                        'is_canonical_wei': p_idx == wei_idx,
                        'S_group_a': S_a, 'S_group_b': S_b, 'S_between': S_between,
                        'delta_group_a': delta_a, 'delta_group_b': delta_b, 'T': T,
                        'rank': None,  # filled in below
                    })
                ranks = rank_partitions(all_T)
                for offset, r in enumerate(ranks):
                    partition_rows[-len(partitions) + offset]['rank'] = r
    return pairwise_rows, partition_rows


def reliability_analysis(diffs_data, id_index, common_ids, active_mechanisms, layer, model_alias):
    rng = random.Random(SPLIT_HALF_SEED)
    results = {}
    for raw_or_pc in RAW_PC:
        results[raw_or_pc] = {}
        for m in active_mechanisms:
            idxs = [id_index[m][pid] for pid in common_ids]
            vecs = diffs_data['diffs'][m][idxs, layer, :]
            if raw_or_pc == 'placebo_calibrated':
                idxs_p = [id_index['placebo'][pid] for pid in common_ids]
                vecs = vecs - diffs_data['diffs']['placebo'][idxs_p, layer, :]
            mean_dir = vecs.mean(0)
            direction_norm = mean_dir.norm().item()

            cos_to_mean = F.cosine_similarity(vecs, mean_dir.unsqueeze(0).expand_as(vecs), dim=-1)
            mean_cos_to_mean = cos_to_mean.mean().item()
            min_cos_to_mean = cos_to_mean.min().item()

            proj = (vecs @ mean_dir) / (mean_dir.norm() + 1e-12)
            proj_vecs = proj.unsqueeze(-1) * (mean_dir / (mean_dir.norm() + 1e-12)).unsqueeze(0)
            total_ss = (vecs ** 2).sum().item()
            proj_ss = (proj_vecs ** 2).sum().item()
            mdef = proj_ss / total_ss if total_ss > 0 else None

            sh_cos = torch.zeros(SPLIT_HALF_REPS)
            n = len(common_ids)
            half = n // 2
            for r in range(SPLIT_HALF_REPS):
                shuffled = common_ids.copy()
                rng.shuffle(shuffled)
                idxsA = [id_index[m][pid] for pid in shuffled[:half]]
                idxsB = [id_index[m][pid] for pid in shuffled[half:2 * half]]
                vecsA = diffs_data['diffs'][m][idxsA, layer, :]
                vecsB = diffs_data['diffs'][m][idxsB, layer, :]
                if raw_or_pc == 'placebo_calibrated':
                    idxsA_p = [id_index['placebo'][pid] for pid in shuffled[:half]]
                    idxsB_p = [id_index['placebo'][pid] for pid in shuffled[half:2 * half]]
                    vecsA = vecsA - diffs_data['diffs']['placebo'][idxsA_p, layer, :]
                    vecsB = vecsB - diffs_data['diffs']['placebo'][idxsB_p, layer, :]
                dirA, dirB = vecsA.mean(0), vecsB.mean(0)
                sh_cos[r] = F.cosine_similarity(dirA.unsqueeze(0), dirB.unsqueeze(0), dim=-1)

            results[raw_or_pc][m] = {
                'direction_norm': direction_norm,
                'mean_cosine_to_mean_direction': mean_cos_to_mean,
                'min_cosine_to_mean_direction': min_cos_to_mean,
                'mean_direction_energy_fraction': mdef,
                'mdef_note': "Fraction of total sum-of-squared-norms explained by each "
                             "instruction's projection onto the mechanism's own mean direction "
                             "-- NOT a regression R-squared (no model is fit; this is a fixed, "
                             "pre-specified direction, not one chosen to maximize explained variance).",
                'split_half_cosine_mean': sh_cos.mean().item(),
                'split_half_cosine_median': sh_cos.median().item(),
                'split_half_cosine_ci95': [sh_cos.quantile(0.025).item(), sh_cos.quantile(0.975).item()],
                'split_half_n_reps': SPLIT_HALF_REPS,
            }
    return results


def bootstrap_analysis(diffs_data, id_index, common_ids, CO, MG, active_mechanisms,
                        partitions, wei_idx, layer):
    rng = random.Random(BOOTSTRAP_SEED)
    resampled_id_lists = [
        [common_ids[rng.randrange(len(common_ids))] for _ in range(len(common_ids))]
        for _ in range(N_BOOTSTRAP)
    ]

    results = {}
    for estimator in ESTIMATORS:
        results[estimator] = {}
        for raw_or_pc in RAW_PC:
            S_CO_l, S_MG_l, S_between_l = [], [], []
            Delta_CO_l, Delta_MG_l, T_l = [], [], []
            wei_ranks = []
            for ids in resampled_id_lists:
                vecs = mech_vecs_at_layer(diffs_data, id_index, active_mechanisms, ids,
                                           layer, estimator, raw_or_pc)
                S_CO, S_MG, S_between, Delta_CO, Delta_MG, T = compute_partition_stats(vecs, CO, MG)
                S_CO_l.append(S_CO); S_MG_l.append(S_MG); S_between_l.append(S_between)
                Delta_CO_l.append(Delta_CO); Delta_MG_l.append(Delta_MG); T_l.append(T)

                all_T = [compute_partition_stats(vecs, a, b)[5] for a, b in partitions]
                ranks = rank_partitions(all_T)
                wei_ranks.append(ranks[wei_idx])

            def summarize(values):
                t = torch.tensor(values)
                return {
                    'ci95': [t.quantile(0.025).item(), t.quantile(0.975).item()],
                    'mean': t.mean().item(), 'median': t.median().item(),
                }

            rank_dist = {r: wei_ranks.count(r) for r in range(1, len(partitions) + 1)}
            results[estimator][raw_or_pc] = {
                'S_CO': summarize(S_CO_l), 'S_MG': summarize(S_MG_l), 'S_between': summarize(S_between_l),
                'Delta_CO': summarize(Delta_CO_l), 'Delta_MG': summarize(Delta_MG_l),
                'T_taxonomy': summarize(T_l),
                'P_Delta_CO_gt_0': sum(1 for x in Delta_CO_l if x > 0) / N_BOOTSTRAP,
                'P_Delta_MG_gt_0': sum(1 for x in Delta_MG_l if x > 0) / N_BOOTSTRAP,
                'P_T_gt_0': sum(1 for x in T_l if x > 0) / N_BOOTSTRAP,
                'P_wei_rank_1': sum(1 for r in wei_ranks if r == 1) / N_BOOTSTRAP,
                'wei_rank_distribution': rank_dist,
                'n_bootstrap': N_BOOTSTRAP, 'seed': BOOTSTRAP_SEED,
                'note': "P(...) values are bootstrap resampling proportions, not formal p-values.",
            }
    return results


# ── figures ──────────────────────────────────────────────────────────────

def plot_cosine_heatmap(all_model_data, active_mechanisms, CO, MG, out_path_base):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(6.5 * len(MODELS), 5.5))
    fig.subplots_adjust(wspace=0.55)
    order = CO + MG
    for ax, model_alias in zip(axes, MODELS):
        data = all_model_data[model_alias]
        mat = data['primary_cosine_matrix_at_fixed_layer']
        grid = [[mat[a][b] for b in order] for a in order]
        im = ax.imshow(grid, vmin=-1, vmax=1, cmap='RdBu_r')
        ax.set_xticks(range(6)); ax.set_xticklabels(order, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(6)); ax.set_yticklabels(order, fontsize=8)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{grid[i][j]:.2f}", ha='center', va='center', fontsize=7,
                         color='white' if abs(grid[i][j]) > 0.5 else 'black')
        ax.axhline(2.5, color='black', linewidth=2)
        ax.axvline(2.5, color='black', linewidth=2)
        ax.set_title(f"{model_alias}\n(layer {FIXED_LAYERS[model_alias]}, "
                      f"{PRIMARY_ESTIMATOR}/{PRIMARY_RAW_PC})", fontsize=9)
    fig.colorbar(im, ax=axes, shrink=0.7, label='cosine similarity')
    fig.savefig(out_path_base + '.png', dpi=150, bbox_inches='tight')
    fig.savefig(out_path_base + '.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_layerwise_taxonomy(all_model_data, out_path_base):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(6 * len(MODELS), 4.5))
    for ax, model_alias in zip(axes, MODELS):
        data = all_model_data[model_alias]
        layers = data['layerwise_primary']['layer_index']
        ax.plot(layers, data['layerwise_primary']['Delta_CO'], label='Delta_CO')
        ax.plot(layers, data['layerwise_primary']['Delta_MG'], label='Delta_MG')
        ax.plot(layers, data['layerwise_primary']['T'], label='T_taxonomy', linewidth=2, color='black')
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.axvline(FIXED_LAYERS[model_alias], color='red', linestyle=':', linewidth=1,
                   label=f'fixed layer {FIXED_LAYERS[model_alias]}')
        ax.set_title(model_alias, fontsize=9)
        ax.set_xlabel('layer index (0-based)')
        ax.legend(fontsize=7)
    fig.savefig(out_path_base + '.png', dpi=150, bbox_inches='tight')
    fig.savefig(out_path_base + '.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_partition_ranking(all_model_data, wei_idx, out_path_base):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(6 * len(MODELS), 4.5))
    for ax, model_alias in zip(axes, MODELS):
        data = all_model_data[model_alias]
        Ts = data['primary_partition_T_at_fixed_layer']
        colors = ['crimson' if i == wei_idx else 'steelblue' for i in range(len(Ts))]
        ax.bar(range(len(Ts)), Ts, color=colors)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_title(model_alias, fontsize=9)
        ax.set_xlabel('partition index')
        ax.set_ylabel('T')
    fig.savefig(out_path_base + '.png', dpi=150, bbox_inches='tight')
    fig.savefig(out_path_base + '.pdf', bbox_inches='tight')
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────

def main(args):
    taxonomy = load_taxonomy_v2()
    active_mechanisms = taxonomy['active_mechanisms']
    CO, MG = taxonomy['CO_mechs'], taxonomy['MG_mechs']
    print(f"Taxonomy v2: CO={CO}  MG={MG}\n")

    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    direction_ids = set(splits['direction_ids'])

    partitions = all_3v3_partitions(active_mechanisms)
    wei_idx = next(i for i, (a, b) in enumerate(partitions) if set(a) == set(CO) or set(a) == set(MG))
    print(f"10 partitions enumerated; canonical Wei V2 partition is #{wei_idx}\n")

    try:
        git_commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=SCRIPT_DIR,
                                     capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        git_commit = 'unknown'

    all_model_data = {}
    all_pairwise_rows, all_partition_rows = [], []
    warnings = []

    for model_alias in MODELS:
        print(f"=== {model_alias} ===")
        path, diffs_data, id_index, common_ids = load_model_diffs(
            args.output_dir, model_alias, args.lang, args.suffix, active_mechanisms)
        n_layers = diffs_data['n_layers']
        fixed_layer = FIXED_LAYERS[model_alias]
        print(f"  Loaded {path}  n_layers={n_layers}  fixed_layer={fixed_layer}  "
              f"common_ids={len(common_ids)}")

        if set(common_ids) != direction_ids:
            w = (f"[{model_alias}] common_ids ({len(common_ids)}) != direction_ids "
                 f"({len(direction_ids)}) -- missing={sorted(direction_ids - set(common_ids))[:5]}...")
            print(f"  WARNING: {w}")
            warnings.append(w)

        pairwise_rows, partition_rows = point_estimate_sweep(
            diffs_data, id_index, common_ids, n_layers, CO, MG, active_mechanisms,
            partitions, wei_idx, model_alias)
        all_pairwise_rows.extend(pairwise_rows)
        all_partition_rows.extend(partition_rows)

        primary_vecs = mech_vecs_at_layer(diffs_data, id_index, active_mechanisms, common_ids,
                                           fixed_layer, PRIMARY_ESTIMATOR, PRIMARY_RAW_PC)
        primary_matrix = {a: {b: cos(primary_vecs[a], primary_vecs[b]) for b in active_mechanisms}
                           for a in active_mechanisms}
        primary_T_all = [compute_partition_stats(primary_vecs, a, b)[5] for a, b in partitions]
        primary_ranks = rank_partitions(primary_T_all)

        layerwise_primary = {'layer_index': [], 'Delta_CO': [], 'Delta_MG': [], 'T': [], 'wei_rank': []}
        for row in partition_rows:
            if row['estimator'] == PRIMARY_ESTIMATOR and row['raw_or_pc'] == PRIMARY_RAW_PC \
                    and row['is_canonical_wei']:
                layerwise_primary['layer_index'].append(row['layer_index'])
                layerwise_primary['Delta_CO'].append(row['delta_group_a'])
                layerwise_primary['Delta_MG'].append(row['delta_group_b'])
                layerwise_primary['T'].append(row['T'])
                layerwise_primary['wei_rank'].append(row['rank'])

        print(f"  Running reliability analysis at fixed layer...")
        reliability = reliability_analysis(diffs_data, id_index, common_ids, active_mechanisms,
                                            fixed_layer, model_alias)

        print(f"  Running bootstrap ({N_BOOTSTRAP} reps x 6 combos) at fixed layer -- this may take a while...")
        bootstrap = bootstrap_analysis(diffs_data, id_index, common_ids, CO, MG, active_mechanisms,
                                        partitions, wei_idx, fixed_layer)

        primary_wei_rank = primary_ranks[wei_idx]
        primary_S_CO, primary_S_MG, primary_S_between, primary_Delta_CO, primary_Delta_MG, primary_T = \
            compute_partition_stats(primary_vecs, CO, MG)
        print(f"  [primary: {PRIMARY_ESTIMATOR}/{PRIMARY_RAW_PC}] S_CO={primary_S_CO:.4f} "
              f"S_MG={primary_S_MG:.4f} S_between={primary_S_between:.4f} "
              f"Delta_CO={primary_Delta_CO:.4f} Delta_MG={primary_Delta_MG:.4f} T={primary_T:.4f} "
              f"Wei_rank={primary_wei_rank}/10")

        all_model_data[model_alias] = {
            'n_layers': n_layers, 'fixed_layer_index': fixed_layer,
            'fixed_layer_ordinal': fixed_layer + 1,
            'relative_depth_of_fixed_layer': fixed_layer / (n_layers - 1) if n_layers > 1 else 0.0,
            'n_common_ids': len(common_ids),
            'primary_S_CO': primary_S_CO, 'primary_S_MG': primary_S_MG,
            'primary_S_between': primary_S_between, 'primary_Delta_CO': primary_Delta_CO,
            'primary_Delta_MG': primary_Delta_MG, 'primary_T': primary_T,
            'primary_wei_rank': primary_wei_rank,
            'primary_cosine_matrix_at_fixed_layer': primary_matrix,
            'primary_partition_T_at_fixed_layer': primary_T_all,
            'layerwise_primary': layerwise_primary,
            'reliability_at_fixed_layer': reliability,
            'bootstrap_at_fixed_layer': bootstrap,
        }

    # ── write outputs ────────────────────────────────────────────────────
    out_dir = os.path.join(args.output_dir, 'canonical_v2')
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    metadata = {
        'taxonomy_version': taxonomy['taxonomy_version'], 'config_path': taxonomy['config_path'],
        'active_mechanisms': active_mechanisms, 'CO_mechs': CO, 'MG_mechs': MG,
        'canonical_wei_partition_index': wei_idx,
        'input_files': {m: os.path.join(args.output_dir, m, f'paired_diffs_{args.lang}{args.suffix}.pt')
                         for m in MODELS},
        'git_commit': git_commit, 'direction_ids_count': len(direction_ids),
        'split_file': SPLITS_PATH,
        'layer_indexing': '0-based tensor index; layer_ordinal = layer_index + 1 (Nth transformer block)',
        'fixed_layers_0based': FIXED_LAYERS,
        'estimators': ESTIMATORS, 'trim_frac': TRIM_FRAC,
        'raw_pc_versions': RAW_PC,
        'primary_combination': {'estimator': PRIMARY_ESTIMATOR, 'raw_or_pc': PRIMARY_RAW_PC},
        'bootstrap_n': N_BOOTSTRAP, 'bootstrap_seed': BOOTSTRAP_SEED,
        'split_half_reps': SPLIT_HALF_REPS, 'split_half_seed': SPLIT_HALF_SEED,
        'tie_tolerance': TIE_TOL, 'ranking_method': 'minimum_rank',
        'partition_dedup': 'A|B and B|A treated as the same partition',
        'torch_version': torch.__version__,
        'warnings': warnings,
    }

    results_json = {'metadata': metadata, 'per_model': all_model_data}
    with open(os.path.join(out_dir, 'experiment1_taxonomy_geometry.json'), 'w') as f:
        json.dump(results_json, f, indent=2)

    with open(os.path.join(out_dir, 'experiment1_pairwise_cosine.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_pairwise_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_pairwise_rows)

    with open(os.path.join(out_dir, 'experiment1_partition_scores.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_partition_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_partition_rows)

    bootstrap_json = {m: all_model_data[m]['bootstrap_at_fixed_layer'] for m in MODELS}
    with open(os.path.join(out_dir, 'experiment1_bootstrap.json'), 'w') as f:
        json.dump({'metadata': {'n_bootstrap': N_BOOTSTRAP, 'seed': BOOTSTRAP_SEED,
                                 'fixed_layers': FIXED_LAYERS}, 'per_model': bootstrap_json}, f, indent=2)

    plot_cosine_heatmap(all_model_data, active_mechanisms, CO, MG,
                         os.path.join(fig_dir, 'experiment1_cosine_heatmap'))
    plot_layerwise_taxonomy(all_model_data, os.path.join(fig_dir, 'experiment1_layerwise_taxonomy'))
    plot_partition_ranking(all_model_data, wei_idx, os.path.join(fig_dir, 'experiment1_partition_ranking'))

    print(f"\nSaved outputs to {out_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    parser.add_argument('--lang',       type=str, default='en')
    parser.add_argument('--suffix',     type=str, default='_full572_corrected')
    args = parser.parse_args()
    main(args)
