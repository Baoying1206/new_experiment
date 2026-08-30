"""
Experiment 3, common-direction coverage audit: for each of Qwen2.5-7B-Instruct,
Meta-Llama-3.1-8B-Instruct and gemma-2-9b-it, asks whether a single "category
common direction" (one for competing_objectives, one for mismatched_
generalization) adequately and evenly represents its 3 member templates'
individual paired-difference directions -- NOT whether such a direction would
defend against anything. Purely descriptive/geometric, CPU-only, read-only
against the existing corrected .pt files -- no GPU, no new completions, no
steering/defence, no modification of scripts 33/34 or their outputs.

CORE QUANTITIES (per model, at a layer):
  d_k      = aggregate_i[ h_l(template_k(x_i)) - h_l(plain(x_i)) ]   (over direction_ids only)
  d_placebo= aggregate_i[ h_l(placebo(x_i))    - h_l(plain(x_i)) ]
  dtilde_k = d_k - d_placebo                                         (placebo calibration)
Placebo calibration is done by aggregating template_k and placebo SEPARATELY
and then subtracting the two aggregated vectors (matches the exact formula
given for this experiment) -- this is mathematically identical to per-
instruction subtract-then-aggregate ONLY for the mean estimator; for
median/trimmed_mean (sensitivity only) it is a deliberate, different choice
from scripts 33/34's per-instruction convention, not a bug.

  u_k   = dtilde_k / ||dtilde_k||
  g_CO  = normalize(mean(u_prefix, u_refusal, u_persona))      -- PRIMARY g-definition
  g_MG  = normalize(mean(u_encoding, u_payload, u_distractors))
  g_CO_raw/g_MG_raw = normalize(mean(dtilde_... , not unit-normalized first))  -- SENSITIVITY

  C_k   = cos(dtilde_k, g_category(k))                          -- member coverage
  g_{G,-k} = normalize(mean_{j in G, j!=k}(u_j))                -- leave-one-out prototype
  A_k   = cos(dtilde_k, g_{G,-k}) - cos(dtilde_k, g_other_category)
  parallel_energy_fraction_k = C_k^2 ; residual = 1 - parallel

Fixed pre-registered layers (0-based, inherited unchanged from scripts
33/34, NOT re-selected here): Qwen=16, Llama=19, Gemma=25. Only
validation_ids/test_ids are EXCLUDED from this script entirely -- only
direction_ids (300) are read.

Bootstrap: 2000 reps, seed=20260830, resample direction_ids WITH
replacement; the SAME resampled id list is used for all 6 mechanisms +
placebo in a given rep, and d_k/dtilde_k/g/C_k/A_k/etc. are recomputed
FROM SCRATCH each rep (never resampling already-aggregated directions or
cosines). Bootstrap runs only at each model's fixed layer, for the PRIMARY
estimator+raw_pc combo (mean, placebo_calibrated), under BOTH g-definitions
(primary unit_normalized + sensitivity raw_magnitude) -- covering the
estimator x raw/pc sensitivity grid with 2000 reps each (3x2=6 combos) was
judged too costly for the marginal value; report this scope choice to the
user before running on real data.

Layerwise point-estimate sweep (no bootstrap) covers ALL layers, primary
estimator+raw_pc, both g-definitions -- for visualizing structure across
depth only; never used to reselect the fixed layer.

Outputs:
  output/canonical_v2/experiment3_common_direction_coverage.json   (fixed-layer, full estimator x raw_pc x gdef sweep)
  output/canonical_v2/experiment3_common_direction_coverage.csv    (layerwise, primary estimator+raw_pc, both gdefs)
  output/canonical_v2/experiment3_common_direction_bootstrap.json  (2000-rep bootstrap, primary combo, both gdefs)
  output/canonical_v2/figures/experiment3_member_coverage.{png,pdf}
  output/canonical_v2/figures/experiment3_layerwise_coverage.{png,pdf}
  output/canonical_v2/figures/experiment3_residual_energy.{png,pdf}

Usage:
  python scripts/35_common_direction_coverage_audit.py --output_dir output
"""
import argparse
import csv
import json
import os
import random
import subprocess
import sys

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
TRIM_FRAC = 0.1  # trims 10% from each tail => 20% trimmed mean overall
RAW_PC = ['raw', 'placebo_calibrated']
PRIMARY_ESTIMATOR = 'mean'
PRIMARY_RAW_PC = 'placebo_calibrated'
PRIMARY_GDEF = 'unit_normalized'
SENSITIVITY_GDEF = 'raw_magnitude'
GDEFS = [PRIMARY_GDEF, SENSITIVITY_GDEF]

N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 20260830


# ── core math ────────────────────────────────────────────────────────────

def aggregate(x, method, trim_frac=TRIM_FRAC):
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
    raise ValueError(method)


def cos(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=-1).item()


def normalize(v):
    n = v.norm()
    return v / n if n > 0 else v


def direction_at_layer(diffs_data, id_index, mech, ids, layer, method, raw_or_pc, trim_frac=TRIM_FRAC):
    """dtilde_k at one layer: aggregate template_k's diffs, and (if calibrated)
    separately aggregate placebo's diffs, then subtract the two aggregates."""
    idxs = [id_index[mech][pid] for pid in ids]
    d = aggregate(diffs_data['diffs'][mech][idxs, layer, :], method, trim_frac)
    if raw_or_pc == 'placebo_calibrated':
        idxs_p = [id_index['placebo'][pid] for pid in ids]
        d_placebo = aggregate(diffs_data['diffs']['placebo'][idxs_p, layer, :], method, trim_frac)
        d = d - d_placebo
    return d


def direction_all_layers(diffs_data, id_index, mech, ids, method, raw_or_pc, trim_frac=TRIM_FRAC):
    """Vectorized across all layers -- [n_layers, d]. Same aggregate-then-subtract order."""
    idxs = [id_index[mech][pid] for pid in ids]
    vecs = diffs_data['diffs'][mech][idxs]  # [n, n_layers, d]
    d = _aggregate_layers(vecs, method, trim_frac)
    if raw_or_pc == 'placebo_calibrated':
        idxs_p = [id_index['placebo'][pid] for pid in ids]
        vecs_p = diffs_data['diffs']['placebo'][idxs_p]
        d_placebo = _aggregate_layers(vecs_p, method, trim_frac)
        d = d - d_placebo
    return d


def _aggregate_layers(vecs, method, trim_frac=TRIM_FRAC):
    if method == 'mean':
        return vecs.mean(0)
    if method == 'median':
        return vecs.median(0).values
    if method == 'trimmed_mean':
        n = vecs.shape[0]
        k = int(n * trim_frac)
        if k == 0:
            return vecs.mean(0)
        sorted_x, _ = torch.sort(vecs, dim=0)
        return sorted_x[k:n - k].mean(0)
    raise ValueError(method)


def build_dtilde_at_layer(diffs_data, id_index, ids, active_mechs, layer, method, raw_or_pc):
    return {m: direction_at_layer(diffs_data, id_index, m, ids, layer, method, raw_or_pc) for m in active_mechs}


def build_dtilde_all_layers(diffs_data, id_index, ids, active_mechs, method, raw_or_pc):
    return {m: direction_all_layers(diffs_data, id_index, m, ids, method, raw_or_pc) for m in active_mechs}


def build_g(dtilde, mechs, gdef):
    if gdef == 'unit_normalized':
        us = [normalize(dtilde[m]) for m in mechs]
        g = torch.stack(us, 0).mean(0)
    elif gdef == 'raw_magnitude':
        g = torch.stack([dtilde[m] for m in mechs], 0).mean(0)
    else:
        raise ValueError(gdef)
    return normalize(g)


def leave_one_out_g(dtilde, group, exclude, gdef):
    members = [m for m in group if m != exclude]
    return build_g(dtilde, members, gdef)


def resultant_length(dtilde, group):
    """R_G = ||mean_k(u_k)||, always via unit-normalized members regardless of gdef
    (this quantity is only defined this one way per the spec)."""
    us = [normalize(dtilde[m]) for m in group]
    return torch.stack(us, 0).mean(0).norm().item()


def compute_metrics(dtilde, active_mechs, CO, MG, gdef):
    g = {'CO': build_g(dtilde, CO, gdef), 'MG': build_g(dtilde, MG, gdef)}
    cat_of = {m: 'CO' for m in CO}
    cat_of.update({m: 'MG' for m in MG})
    other_cat = {'CO': 'MG', 'MG': 'CO'}
    group_of = {'CO': CO, 'MG': MG}

    per_mech = {}
    for m in active_mechs:
        own_cat = cat_of[m]
        C_k = cos(dtilde[m], g[own_cat])
        g_loo = leave_one_out_g(dtilde, group_of[own_cat], m, gdef)
        cos_own_loo = cos(dtilde[m], g_loo)
        cos_other_full = cos(dtilde[m], g[other_cat[own_cat]])
        A_k = cos_own_loo - cos_other_full
        parallel = C_k ** 2
        per_mech[m] = {
            'category': own_cat,
            'C_k': C_k,
            'cosine_to_own_leave_one_out_prototype': cos_own_loo,
            'cosine_to_other_category_prototype': cos_other_full,
            'A_k': A_k,
            'parallel_energy_fraction': parallel,
            'residual_energy_fraction': 1.0 - parallel,
        }

    per_cat = {}
    for cat, group in [('CO', CO), ('MG', MG)]:
        cks = torch.tensor([per_mech[m]['C_k'] for m in group])
        per_cat[cat] = {
            'mean_coverage': cks.mean().item(),
            'median_coverage': cks.median().item(),
            'minimum_coverage': cks.min().item(),
            'coverage_range': (cks.max() - cks.min()).item(),
            'coverage_standard_deviation': cks.std().item(),
            'resultant_length': resultant_length(dtilde, group),
        }
    return per_mech, per_cat


def summarize(values):
    t = torch.tensor(values)
    return {
        'mean': t.mean().item(), 'median': t.median().item(),
        'ci_lo': t.quantile(0.025).item(), 'ci_hi': t.quantile(0.975).item(),
        'n': len(values),
    }


# ── fixed-layer full combo sweep (point estimates) ──────────────────────

def fixed_layer_combo_sweep(diffs_data, id_index, direction_ids, active_mechs, CO, MG, fixed_layer):
    results = {}
    for method in ESTIMATORS:
        results[method] = {}
        for raw_or_pc in RAW_PC:
            dtilde = build_dtilde_at_layer(diffs_data, id_index, direction_ids, active_mechs,
                                            fixed_layer, method, raw_or_pc)
            results[method][raw_or_pc] = {}
            for gdef in GDEFS:
                per_mech, per_cat = compute_metrics(dtilde, active_mechs, CO, MG, gdef)
                results[method][raw_or_pc][gdef] = {'per_mechanism': per_mech, 'per_category': per_cat}
    return results


# ── layerwise point-estimate sweep (primary combo, both gdefs, all layers) ─

def layerwise_sweep(diffs_data, id_index, direction_ids, active_mechs, CO, MG, n_layers, model_alias, fixed_layer):
    dtilde_all = build_dtilde_all_layers(diffs_data, id_index, direction_ids, active_mechs,
                                          PRIMARY_ESTIMATOR, PRIMARY_RAW_PC)
    rows = []
    for l in range(n_layers):
        dtilde_l = {m: dtilde_all[m][l] for m in active_mechs}
        relative_depth = l / (n_layers - 1) if n_layers > 1 else 0.0
        for gdef in GDEFS:
            per_mech, per_cat = compute_metrics(dtilde_l, active_mechs, CO, MG, gdef)
            for m in active_mechs:
                for metric in ['C_k', 'parallel_energy_fraction', 'residual_energy_fraction']:
                    rows.append({
                        'model': model_alias, 'layer_index': l, 'layer_ordinal': l + 1,
                        'relative_depth': relative_depth, 'is_fixed_layer': l == fixed_layer,
                        'gdef': gdef, 'level': 'mechanism', 'name': m,
                        'metric': metric, 'value': per_mech[m][metric],
                    })
            for cat in ['CO', 'MG']:
                for metric in ['mean_coverage', 'minimum_coverage', 'coverage_standard_deviation', 'resultant_length']:
                    rows.append({
                        'model': model_alias, 'layer_index': l, 'layer_ordinal': l + 1,
                        'relative_depth': relative_depth, 'is_fixed_layer': l == fixed_layer,
                        'gdef': gdef, 'level': 'category', 'name': cat,
                        'metric': metric, 'value': per_cat[cat][metric],
                    })
    return rows


# ── bootstrap (primary estimator+raw_pc, both gdefs, fixed layer only) ────

def bootstrap_analysis(diffs_data, id_index, direction_ids, active_mechs, CO, MG, fixed_layer):
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(direction_ids)
    raw = {gdef: {
        'per_mechanism': {m: {k: [] for k in
                               ['C_k', 'A_k', 'cosine_to_own_leave_one_out_prototype',
                                'cosine_to_other_category_prototype', 'parallel_energy_fraction']}
                          for m in active_mechs},
        'per_category': {cat: {k: [] for k in
                                ['mean_coverage', 'minimum_coverage', 'coverage_standard_deviation', 'resultant_length']}
                          for cat in ['CO', 'MG']},
    } for gdef in GDEFS}

    for _ in range(N_BOOTSTRAP):
        resampled = [direction_ids[rng.randrange(n)] for _ in range(n)]
        dtilde = build_dtilde_at_layer(diffs_data, id_index, resampled, active_mechs,
                                        fixed_layer, PRIMARY_ESTIMATOR, PRIMARY_RAW_PC)
        for gdef in GDEFS:
            per_mech, per_cat = compute_metrics(dtilde, active_mechs, CO, MG, gdef)
            for m in active_mechs:
                for k in raw[gdef]['per_mechanism'][m]:
                    raw[gdef]['per_mechanism'][m][k].append(per_mech[m][k])
            for cat in ['CO', 'MG']:
                for k in raw[gdef]['per_category'][cat]:
                    raw[gdef]['per_category'][cat][k].append(per_cat[cat][k])

    summary = {}
    for gdef in GDEFS:
        summary[gdef] = {
            'per_mechanism': {m: {k: summarize(v) for k, v in raw[gdef]['per_mechanism'][m].items()}
                              for m in active_mechs},
            'per_category': {cat: {k: summarize(v) for k, v in raw[gdef]['per_category'][cat].items()}
                             for cat in ['CO', 'MG']},
        }
    return summary


# ── figures ──────────────────────────────────────────────────────────────

def plot_member_coverage(fixed_results, bootstrap_results, active_mechs, CO, out_path_base):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(15, 5), sharey=True)
    for ax, model_alias in zip(axes, MODELS):
        boot = bootstrap_results[model_alias][PRIMARY_GDEF]['per_mechanism']
        means = [boot[m]['C_k']['mean'] for m in active_mechs]
        los = [boot[m]['C_k']['ci_lo'] for m in active_mechs]
        his = [boot[m]['C_k']['ci_hi'] for m in active_mechs]
        errs = [[m - l for m, l in zip(means, los)], [h - m for m, h in zip(means, his)]]
        colors = ['#2f5d8a' if m in CO else '#a4550a' for m in active_mechs]
        ax.bar(range(len(active_mechs)), means, yerr=errs, color=colors, capsize=4)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_xticks(range(len(active_mechs)))
        ax.set_xticklabels(active_mechs, rotation=45, ha='right', fontsize=8)
        ax.set_title(model_alias, fontsize=10)
    axes[0].set_ylabel('C_k (coverage, primary combo, bootstrap CI)')
    fig.suptitle('Experiment 3: member coverage C_k, primary combination (mean, placebo_calibrated, unit_normalized g)')
    fig.tight_layout()
    fig.savefig(out_path_base + '.png', dpi=150, bbox_inches='tight')
    fig.savefig(out_path_base + '.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_layerwise_coverage(layerwise_rows_by_model, active_mechs, out_path_base):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(16, 5), sharey=True)
    for ax, model_alias in zip(axes, MODELS):
        rows = [r for r in layerwise_rows_by_model[model_alias]
                if r['gdef'] == PRIMARY_GDEF and r['level'] == 'mechanism' and r['metric'] == 'C_k']
        for m in active_mechs:
            sub = sorted([r for r in rows if r['name'] == m], key=lambda r: r['layer_index'])
            ax.plot([r['layer_index'] for r in sub], [r['value'] for r in sub], label=m, linewidth=1.2)
        fixed_layer = FIXED_LAYERS[model_alias]
        ax.axvline(fixed_layer, color='red', linestyle=':', label=f'fixed layer {fixed_layer}')
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_title(model_alias, fontsize=10)
        ax.set_xlabel('layer index (0-based)')
    axes[0].set_ylabel('C_k (coverage)')
    axes[-1].legend(fontsize=7, loc='lower right')
    fig.suptitle('Experiment 3: layerwise member coverage, primary combination (point estimates, no bootstrap)')
    fig.tight_layout()
    fig.savefig(out_path_base + '.png', dpi=150, bbox_inches='tight')
    fig.savefig(out_path_base + '.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_residual_energy(fixed_results, active_mechs, CO, out_path_base):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(15, 5), sharey=True)
    for ax, model_alias in zip(axes, MODELS):
        per_mech = fixed_results[model_alias][PRIMARY_ESTIMATOR][PRIMARY_RAW_PC][PRIMARY_GDEF]['per_mechanism']
        parallel = [per_mech[m]['parallel_energy_fraction'] for m in active_mechs]
        residual = [per_mech[m]['residual_energy_fraction'] for m in active_mechs]
        x = range(len(active_mechs))
        ax.bar(x, parallel, label='parallel (on g)', color='#2f5d8a')
        ax.bar(x, residual, bottom=parallel, label='residual (off g)', color='#dcd8cd')
        ax.set_xticks(list(x))
        ax.set_xticklabels(active_mechs, rotation=45, ha='right', fontsize=8)
        ax.set_title(model_alias, fontsize=10)
    axes[0].set_ylabel('energy fraction')
    axes[-1].legend(fontsize=8)
    fig.suptitle('Experiment 3: parallel vs residual energy fraction at fixed layer, primary combination')
    fig.tight_layout()
    fig.savefig(out_path_base + '.png', dpi=150, bbox_inches='tight')
    fig.savefig(out_path_base + '.pdf', bbox_inches='tight')
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────

def main(args):
    taxonomy = load_taxonomy_v2()
    active_mechs = taxonomy['active_mechanisms']
    CO, MG = taxonomy['CO_mechs'], taxonomy['MG_mechs']
    print(f"Taxonomy v2: CO={CO}  MG={MG}\n")

    with open(SPLITS_PATH) as f:
        direction_ids = sorted(json.load(f)['direction_ids'])
    print(f"direction_ids: {len(direction_ids)} (validation_ids/test_ids not read by this script)")

    try:
        git_commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=SCRIPT_DIR,
                                     capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        git_commit = 'unknown'

    out_dir = os.path.join(args.output_dir, 'canonical_v2')
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    fixed_results = {}
    bootstrap_results = {}
    all_layerwise_rows = []
    layerwise_rows_by_model = {}
    per_model_meta = {}
    warnings = []

    for model_alias in MODELS:
        print(f"=== {model_alias} ===")
        path = os.path.join(args.output_dir, model_alias, f'paired_diffs_{args.lang}{args.suffix}.pt')
        diffs_data = torch.load(path, map_location='cpu')
        diffs_data['diffs'] = {m: v.float() for m, v in diffs_data['diffs'].items()}
        n_layers = diffs_data['n_layers']
        fixed_layer = FIXED_LAYERS[model_alias]
        id_index = {m: {pid: i for i, pid in enumerate(diffs_data['instruction_ids'][m])}
                    for m in active_mechs + ['placebo']}
        id_sets = [set(diffs_data['instruction_ids'][m]) for m in active_mechs + ['placebo']]
        common_ids = sorted(set.intersection(*id_sets))
        if set(common_ids) != set(direction_ids):
            msg = f"{model_alias}: common_ids != direction_ids"
            print(f"  WARNING: {msg}")
            warnings.append(msg)
        ids_to_use = [i for i in direction_ids if i in id_index[active_mechs[0]]]
        print(f"  n_layers={n_layers}  fixed_layer={fixed_layer}  n_ids_used={len(ids_to_use)}")

        print("  Fixed-layer combo sweep (point estimates)...")
        fr = fixed_layer_combo_sweep(diffs_data, id_index, ids_to_use, active_mechs, CO, MG, fixed_layer)
        fixed_results[model_alias] = fr

        # lightweight finiteness check on the primary combo's dtilde/g vectors
        dtilde_primary = build_dtilde_at_layer(diffs_data, id_index, ids_to_use, active_mechs,
                                                fixed_layer, PRIMARY_ESTIMATOR, PRIMARY_RAW_PC)
        n_nonfinite = sum(int((~torch.isfinite(v)).sum().item()) for v in dtilde_primary.values())
        if n_nonfinite > 0:
            msg = f"{model_alias}: {n_nonfinite} non-finite values in primary-combo dtilde vectors"
            warnings.append(msg)
            print(f"  WARNING: {msg}")

        print("  Layerwise sweep (point estimates, all layers, primary combo, both gdefs)...")
        rows = layerwise_sweep(diffs_data, id_index, ids_to_use, active_mechs, CO, MG,
                                n_layers, model_alias, fixed_layer)
        layerwise_rows_by_model[model_alias] = rows
        all_layerwise_rows.extend(rows)

        print(f"  Bootstrap ({N_BOOTSTRAP} reps, seed={BOOTSTRAP_SEED}, primary combo, both gdefs)...")
        br = bootstrap_analysis(diffs_data, id_index, ids_to_use, active_mechs, CO, MG, fixed_layer)
        bootstrap_results[model_alias] = br

        per_model_meta[model_alias] = {
            'input_file': path, 'n_layers': n_layers, 'fixed_layer_0based': fixed_layer,
            'fixed_layer_ordinal': fixed_layer + 1,
            'relative_depth_of_fixed_layer': fixed_layer / (n_layers - 1) if n_layers > 1 else 0.0,
            'n_ids_used': len(ids_to_use),
        }
        print()

    metadata = {
        'taxonomy_version': taxonomy['taxonomy_version'], 'config_path': taxonomy['config_path'],
        'CO_mechs': CO, 'MG_mechs': MG, 'active_mechanisms': active_mechs,
        'direction_ids_count': len(direction_ids),
        'validation_ids_used': False, 'test_ids_used': False,
        'fixed_layers_0based': FIXED_LAYERS,
        'estimators': ESTIMATORS, 'trim_frac': TRIM_FRAC, 'raw_pc_versions': RAW_PC,
        'primary_combination': {'estimator': PRIMARY_ESTIMATOR, 'raw_or_pc': PRIMARY_RAW_PC, 'gdef': PRIMARY_GDEF},
        'sensitivity_gdef': SENSITIVITY_GDEF,
        'placebo_calibration': 'aggregate template_k and placebo separately (per estimator), then subtract '
                                'the two aggregated vectors: dtilde_k = d_k - d_placebo',
        'common_direction_definitions': {
            'unit_normalized (primary)': 'g = normalize(mean_k(dtilde_k / ||dtilde_k||))',
            'raw_magnitude (sensitivity)': 'g = normalize(mean_k(dtilde_k))',
        },
        'bootstrap_n': N_BOOTSTRAP, 'bootstrap_seed': BOOTSTRAP_SEED,
        'bootstrap_scope': 'primary estimator+raw_pc combo (mean, placebo_calibrated) only; '
                            'both g-definitions (unit_normalized, raw_magnitude) bootstrapped',
        'layerwise_scope': 'point estimates only, primary estimator+raw_pc combo, both g-definitions, all layers; '
                            'not used to reselect the fixed layer',
        'git_commit': git_commit, 'torch_version': torch.__version__,
        'per_model': per_model_meta,
        'warnings': warnings,
    }

    with open(os.path.join(out_dir, 'experiment3_common_direction_coverage.json'), 'w') as f:
        json.dump({'metadata': metadata, 'per_model_fixed_layer_sweep': fixed_results}, f, indent=2)

    with open(os.path.join(out_dir, 'experiment3_common_direction_coverage.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_layerwise_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_layerwise_rows)

    with open(os.path.join(out_dir, 'experiment3_common_direction_bootstrap.json'), 'w') as f:
        json.dump({'metadata': metadata, 'per_model': bootstrap_results}, f, indent=2)

    print("Plotting figures...")
    plot_member_coverage(fixed_results, bootstrap_results, active_mechs, CO,
                          os.path.join(fig_dir, 'experiment3_member_coverage'))
    plot_layerwise_coverage(layerwise_rows_by_model, active_mechs,
                             os.path.join(fig_dir, 'experiment3_layerwise_coverage'))
    plot_residual_energy(fixed_results, active_mechs, CO,
                          os.path.join(fig_dir, 'experiment3_residual_energy'))

    print(f"\nSaved outputs to {out_dir}/")
    if warnings:
        print(f"WARNINGS: {warnings}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    parser.add_argument('--lang',       type=str, default='en')
    parser.add_argument('--suffix',     type=str, default='_full572_corrected')
    args = parser.parse_args()
    main(args)
