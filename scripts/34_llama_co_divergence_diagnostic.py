"""
Explains (not just describes) why Meta-Llama-3.1-8B-Instruct's competing_objectives
(CO) templates -- prefix_injection, refusal_suppression, persona_roleplay --
show Delta_CO < 0 at the fixed layer (index 19, 0-based) in
33_canonical_taxonomy_geometry.py's canonical V2 result, unlike Qwen/Gemma.
CPU-only, reads the existing paired_diffs_en_full572_corrected.pt for Llama
only -- no new GPU extraction, no steering, no dual-axis, no other models.

Tests 3 competing, non-exclusive explanations via 4 analyses (H1-H4):

  H1 (measurement instability): are the 3 CO mechanisms' own directions
     reliably estimated at the fixed layer, or is refusal_suppression's
     direction itself noisy/small (which would make Delta_CO<0 an artifact
     of unreliable measurement rather than a real geometric fact)?
  H2 (layerwise angular vs magnitude): does refusal_suppression's cosine to
     the other two CO members decline specifically in later layers (angular
     divergence), while its own vector magnitude (norm) stays comparable to
     theirs (ruling out "it just goes to zero" as the explanation)?
  H3 (shared-component masking vs template-specific divergence): after
     cross-fitted removal of whatever direction is COMMON to all 6 active
     mechanisms (estimated on one random half of instructions, removed from
     the other half -- never estimated and tested on the same half), does
     Delta_CO become positive (support for "a shared template-wrapper
     direction was masking real CO cohesion") or stay negative (support for
     "the divergence is intrinsic to how these specific templates differ,
     not an artifact of a shared confound")?
  H4 (leave-one-template-out prototype affinity): for each of the 6
     mechanisms (not just refusal_suppression, though it's the pre-registered
     primary target), is it more aligned with a prototype built from the
     OTHER 2 mechanisms in its own assigned category, or with a prototype
     built from the 3 mechanisms in the opposite category?

Fixed layer 19 (0-based) is never re-selected based on this script's own
results -- matches 33_canonical_taxonomy_geometry.py's FIXED_LAYERS.

Terminology (do not conflate):
  - H1/H4 use REPEATED RESAMPLING WITH REPLACEMENT of the 300 direction_ids
    ("source-level bootstrap"), reported as bootstrap CIs.
  - H3 uses REPEATED DISJOINT 150/150 SPLITS (no replacement, each instruction
    appears in exactly one half per split) -- its percentile interval is
    called a "repeated-split stability interval", NOT a bootstrap CI, since
    the resampling scheme is different (partition, not resample-with-replacement).

Outputs:
  output/canonical_v2/experiment2_llama_co_divergence.json
  output/canonical_v2/experiment2_llama_co_reliability.csv
  output/canonical_v2/experiment2_llama_co_layerwise.csv
  output/canonical_v2/experiment2_llama_prototype_affinity.csv
  output/canonical_v2/experiment2_llama_common_component.csv
  output/canonical_v2/figures/experiment2_llama_co_layerwise.{png,pdf}
  output/canonical_v2/figures/experiment2_llama_prototype_affinity.{png,pdf}

Usage:
  python scripts/34_llama_co_divergence_diagnostic.py --output_dir output
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
MODEL_ALIAS = 'Meta-Llama-3.1-8B-Instruct'
FIXED_LAYER = 19  # 0-based; matches 33_canonical_taxonomy_geometry.py's FIXED_LAYERS[MODEL_ALIAS]
PRIMARY_ESTIMATOR = 'mean'
PRIMARY_RAW_PC = 'placebo_calibrated'
TRIM_FRAC = 0.1

H1_SPLIT_HALF_REPS = 500
H1_SPLIT_HALF_SEED = 20260828  # matches 33's reliability-analysis convention (no new seed specified for H1)
H3_N_SPLITS = 500
H3_SEED = 20260829
H4_N_BOOTSTRAP = 2000
H4_SEED = 20260829
DANGLE_N_BOOTSTRAP = 2000
DANGLE_SEED = 20260829

TARGET_MECH = 'refusal_suppression'
RELIABILITY_LOW_THRESHOLDS = [0.6, 0.7, 0.8]  # sensitivity sweep, none declared "the" standard


# ── core math (small, self-contained -- consistent with, but not imported
# from, 33_canonical_taxonomy_geometry.py, per this repo's convention of not
# cross-importing numbered scripts) ─────────────────────────────────────

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


def mech_vec_at_layer(diffs_data, id_index, mech, ids, layer, method, raw_or_pc, trim_frac=TRIM_FRAC):
    idxs = [id_index[mech][pid] for pid in ids]
    vecs = diffs_data['diffs'][mech][idxs, layer, :]
    if raw_or_pc == 'placebo_calibrated':
        idxs_p = [id_index['placebo'][pid] for pid in ids]
        vecs = vecs - diffs_data['diffs']['placebo'][idxs_p, layer, :]
    return aggregate(vecs, method, trim_frac)


def mech_vecs_all_layers(diffs_data, id_index, mech, ids, method, raw_or_pc, trim_frac=TRIM_FRAC):
    """Vectorized across ALL layers in one call -- returns [n_layers, d]. Used so
    layer-subset bootstraps (D_angle) don't pay per-layer Python-loop cost."""
    idxs = [id_index[mech][pid] for pid in ids]
    vecs = diffs_data['diffs'][mech][idxs]  # [n, n_layers, d]
    if raw_or_pc == 'placebo_calibrated':
        idxs_p = [id_index['placebo'][pid] for pid in ids]
        vecs = vecs - diffs_data['diffs']['placebo'][idxs_p]
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


def compute_partition_stats(vecs, group_a, group_b):
    within_a = [cos(vecs[group_a[i]], vecs[group_a[j]]) for i in range(3) for j in range(i + 1, 3)]
    within_b = [cos(vecs[group_b[i]], vecs[group_b[j]]) for i in range(3) for j in range(i + 1, 3)]
    between = [cos(vecs[a], vecs[b]) for a in group_a for b in group_b]
    S_a = sum(within_a) / len(within_a)
    S_b = sum(within_b) / len(within_b)
    S_between = sum(between) / len(between)
    return S_a, S_b, S_between, S_a - S_between, S_b - S_between, (S_a + S_b) / 2 - S_between


def prototype(vecs, mechs):
    us = [normalize(vecs[m]) for m in mechs]
    return normalize(torch.stack(us, 0).mean(0))


def prototype_affinity(vecs, target_mech, CO, MG):
    if target_mech in CO:
        own_mechs = [m for m in CO if m != target_mech]
        other_mechs = MG
    else:
        own_mechs = [m for m in MG if m != target_mech]
        other_mechs = CO
    p_own = prototype(vecs, own_mechs)
    p_other = prototype(vecs, other_mechs)
    return cos(vecs[target_mech], p_own) - cos(vecs[target_mech], p_other)


def summarize(values, percentile_keys=(0.025, 0.975)):
    t = torch.tensor(values)
    return {
        'mean': t.mean().item(), 'median': t.median().item(),
        'ci_lo': t.quantile(percentile_keys[0]).item(), 'ci_hi': t.quantile(percentile_keys[1]).item(),
        'n': len(values),
    }


# ── H1: reliability at the fixed layer ──────────────────────────────────

def h1_reliability(diffs_data, id_index, common_ids, CO):
    rng = random.Random(H1_SPLIT_HALF_SEED)
    results = {}
    for m in CO:
        results[m] = {}
        for raw_or_pc in ['raw', 'placebo_calibrated']:
            per_estimator = {}
            for method in ['mean', 'median', 'trimmed_mean']:
                idxs = [id_index[m][pid] for pid in common_ids]
                vecs = diffs_data['diffs'][m][idxs, FIXED_LAYER, :]
                if raw_or_pc == 'placebo_calibrated':
                    idxs_p = [id_index['placebo'][pid] for pid in common_ids]
                    vecs = vecs - diffs_data['diffs']['placebo'][idxs_p, FIXED_LAYER, :]
                direction = aggregate(vecs, method)
                direction_norm = direction.norm().item()
                cos_to_mean = F.cosine_similarity(vecs, direction.unsqueeze(0).expand_as(vecs), dim=-1)
                proj = (vecs @ direction) / (direction.norm() + 1e-12)
                proj_vecs = proj.unsqueeze(-1) * (direction / (direction.norm() + 1e-12)).unsqueeze(0)
                total_ss = (vecs ** 2).sum().item()
                mdef = (proj_vecs ** 2).sum().item() / total_ss if total_ss > 0 else None
                per_estimator[method] = {
                    'direction_norm': direction_norm,
                    'mean_cosine_to_mean_direction': cos_to_mean.mean().item(),
                    'min_cosine_to_mean_direction': cos_to_mean.min().item(),
                    'mean_direction_energy_fraction': mdef,
                }
            results[m][raw_or_pc] = {'per_estimator': per_estimator}

        # split-half (primary combo only)
        n = len(common_ids)
        half = n // 2
        sh_cos = torch.zeros(H1_SPLIT_HALF_REPS)
        for r in range(H1_SPLIT_HALF_REPS):
            shuffled = common_ids.copy()
            rng.shuffle(shuffled)
            idxsA = [id_index[m][pid] for pid in shuffled[:half]]
            idxsB = [id_index[m][pid] for pid in shuffled[half:2 * half]]
            vecsA = diffs_data['diffs'][m][idxsA, FIXED_LAYER, :]
            vecsB = diffs_data['diffs'][m][idxsB, FIXED_LAYER, :]
            idxsA_p = [id_index['placebo'][pid] for pid in shuffled[:half]]
            idxsB_p = [id_index['placebo'][pid] for pid in shuffled[half:2 * half]]
            vecsA = vecsA - diffs_data['diffs']['placebo'][idxsA_p, FIXED_LAYER, :]
            vecsB = vecsB - diffs_data['diffs']['placebo'][idxsB_p, FIXED_LAYER, :]
            dirA, dirB = vecsA.mean(0), vecsB.mean(0)
            sh_cos[r] = F.cosine_similarity(dirA.unsqueeze(0), dirB.unsqueeze(0), dim=-1)
        results[m]['split_half_cosine_mean'] = sh_cos.mean().item()
        results[m]['split_half_cosine_median'] = sh_cos.median().item()
        results[m]['split_half_cosine_ci95'] = [sh_cos.quantile(0.025).item(), sh_cos.quantile(0.975).item()]
        results[m]['split_half_n_reps'] = H1_SPLIT_HALF_REPS
        results[m]['split_half_seed'] = H1_SPLIT_HALF_SEED
    return results


# ── H2: layerwise + D_angle ──────────────────────────────────────────────

def h2_layerwise(diffs_data, id_index, common_ids, CO, n_layers):
    prefix, refusal, persona = 'prefix_injection', 'refusal_suppression', 'persona_roleplay'
    layer_rows = []
    vecs_all = {m: mech_vecs_all_layers(diffs_data, id_index, m, common_ids,
                                         PRIMARY_ESTIMATOR, PRIMARY_RAW_PC) for m in CO}
    norms = {m: vecs_all[m].norm(dim=-1) for m in CO}  # [n_layers]
    cos_pf = F.cosine_similarity(vecs_all[prefix], vecs_all[refusal], dim=-1)
    cos_pr = F.cosine_similarity(vecs_all[persona], vecs_all[refusal], dim=-1)
    cos_pp = F.cosine_similarity(vecs_all[prefix], vecs_all[persona], dim=-1)
    for l in range(n_layers):
        layer_rows.append({
            'layer_index': l, 'layer_ordinal': l + 1,
            'cos_prefix_refusal': cos_pf[l].item(), 'cos_persona_refusal': cos_pr[l].item(),
            'cos_prefix_persona': cos_pp[l].item(),
            'norm_prefix_injection': norms[prefix][l].item(),
            'norm_refusal_suppression': norms[refusal][l].item(),
            'norm_persona_roleplay': norms[persona][l].item(),
        })
    return layer_rows


def early_late_layers(n_layers):
    k = n_layers // 3
    early = list(range(0, k))
    late = list(range(n_layers - k, n_layers))
    return early, late


def dangle_point_estimate(layer_rows, early, late):
    refusal_pairs = [(r['cos_prefix_refusal'] + r['cos_persona_refusal']) / 2 for r in layer_rows]
    prefix_persona = [r['cos_prefix_persona'] for r in layer_rows]
    early_refusal = sum(refusal_pairs[l] for l in early) / len(early)
    late_refusal = sum(refusal_pairs[l] for l in late) / len(late)
    early_pp = sum(prefix_persona[l] for l in early) / len(early)
    late_pp = sum(prefix_persona[l] for l in late) / len(late)
    D_angle = (late_refusal - early_refusal) - (late_pp - early_pp)
    return D_angle, {'early_refusal_pairs': early_refusal, 'late_refusal_pairs': late_refusal,
                      'early_prefix_persona': early_pp, 'late_prefix_persona': late_pp}


def dangle_bootstrap(diffs_data, id_index, common_ids, CO, n_layers, early, late):
    prefix, refusal, persona = 'prefix_injection', 'refusal_suppression', 'persona_roleplay'
    rng = random.Random(DANGLE_SEED)
    vals = []
    for _ in range(DANGLE_N_BOOTSTRAP):
        resampled = [common_ids[rng.randrange(len(common_ids))] for _ in range(len(common_ids))]
        vecs_all = {m: mech_vecs_all_layers(diffs_data, id_index, m, resampled,
                                             PRIMARY_ESTIMATOR, PRIMARY_RAW_PC) for m in CO}
        cos_pf = F.cosine_similarity(vecs_all[prefix], vecs_all[refusal], dim=-1)
        cos_pr = F.cosine_similarity(vecs_all[persona], vecs_all[refusal], dim=-1)
        cos_pp = F.cosine_similarity(vecs_all[prefix], vecs_all[persona], dim=-1)
        refusal_pairs = ((cos_pf + cos_pr) / 2)
        early_refusal = refusal_pairs[early].mean().item()
        late_refusal = refusal_pairs[late].mean().item()
        early_pp = cos_pp[early].mean().item()
        late_pp = cos_pp[late].mean().item()
        vals.append((late_refusal - early_refusal) - (late_pp - early_pp))
    t = torch.tensor(vals)
    return {
        'point_estimate_from_full_data': None,  # filled by caller
        'source_level_bootstrap_mean': t.mean().item(),
        'source_level_bootstrap_ci95': [t.quantile(0.025).item(), t.quantile(0.975).item()],
        'P_lt_0': (t < 0).float().mean().item(),
        'n_bootstrap': DANGLE_N_BOOTSTRAP, 'seed': DANGLE_SEED,
    }


def norm_relative_change(layer_rows, early, late, mech_key):
    early_norm = sum(layer_rows[l][mech_key] for l in early) / len(early)
    late_norm = sum(layer_rows[l][mech_key] for l in late) / len(late)
    return {
        'early_mean_norm': early_norm, 'late_mean_norm': late_norm,
        'relative_change': (late_norm - early_norm) / early_norm if early_norm != 0 else None,
    }


# ── H3: cross-fitted common-component removal ───────────────────────────

def h3_common_component(diffs_data, id_index, common_ids, active_mechanisms, CO, MG):
    rng = random.Random(H3_SEED)
    ids = common_ids.copy()
    results = {}
    for gdef in ['unit_normalized', 'raw_magnitude']:
        symmetrized = {'Delta_CO_perp': [], 'Delta_MG_perp': [], 'V_bilateral': [], 'A_refusal_perp': []}
        for _ in range(H3_N_SPLITS):
            shuffled = ids.copy()
            rng.shuffle(shuffled)
            half = len(shuffled) // 2
            ids_A, ids_B = shuffled[:half], shuffled[half:2 * half]

            vecs_A = {m: mech_vec_at_layer(diffs_data, id_index, m, ids_A, FIXED_LAYER,
                                            PRIMARY_ESTIMATOR, PRIMARY_RAW_PC) for m in active_mechanisms}
            vecs_B = {m: mech_vec_at_layer(diffs_data, id_index, m, ids_B, FIXED_LAYER,
                                            PRIMARY_ESTIMATOR, PRIMARY_RAW_PC) for m in active_mechanisms}

            def common_dir(vecs):
                if gdef == 'unit_normalized':
                    us = [normalize(vecs[m]) for m in active_mechanisms]
                else:
                    us = [vecs[m] for m in active_mechanisms]
                return normalize(torch.stack(us, 0).mean(0))

            g_A = common_dir(vecs_A)
            g_B = common_dir(vecs_B)

            # A_to_B: g from A, remove from B, test on B's residuals
            resid_B = {m: vecs_B[m] - (vecs_B[m] @ g_A) * g_A for m in active_mechanisms}
            _, _, _, dco_AtoB, dmg_AtoB, _ = compute_partition_stats(resid_B, CO, MG)
            aref_AtoB = prototype_affinity(resid_B, TARGET_MECH, CO, MG)

            # B_to_A: g from B, remove from A, test on A's residuals
            resid_A = {m: vecs_A[m] - (vecs_A[m] @ g_B) * g_B for m in active_mechanisms}
            _, _, _, dco_BtoA, dmg_BtoA, _ = compute_partition_stats(resid_A, CO, MG)
            aref_BtoA = prototype_affinity(resid_A, TARGET_MECH, CO, MG)

            dco = (dco_AtoB + dco_BtoA) / 2
            dmg = (dmg_AtoB + dmg_BtoA) / 2
            v_bi = min(dco, dmg)
            aref = (aref_AtoB + aref_BtoA) / 2

            symmetrized['Delta_CO_perp'].append(dco)
            symmetrized['Delta_MG_perp'].append(dmg)
            symmetrized['V_bilateral'].append(v_bi)
            symmetrized['A_refusal_perp'].append(aref)

        results[gdef] = {
            stat: {
                **summarize(vals),
                'interval_name': 'repeated_split_stability_interval',  # NOT a bootstrap CI
                'n_splits': H3_N_SPLITS, 'seed': H3_SEED,
                'sign_proportion_positive': sum(1 for v in vals if v > 0) / len(vals),
            }
            for stat, vals in symmetrized.items()
        }
    return results


# ── H4: leave-one-template-out prototype bootstrap ───────────────────────

def h4_prototype_test(diffs_data, id_index, common_ids, active_mechanisms, CO, MG):
    vecs_full = {m: mech_vec_at_layer(diffs_data, id_index, m, common_ids, FIXED_LAYER,
                                       PRIMARY_ESTIMATOR, PRIMARY_RAW_PC) for m in active_mechanisms}
    point_estimates = {m: prototype_affinity(vecs_full, m, CO, MG) for m in active_mechanisms}

    rng = random.Random(H4_SEED)
    boot_vals = {m: [] for m in active_mechanisms}
    for _ in range(H4_N_BOOTSTRAP):
        resampled = [common_ids[rng.randrange(len(common_ids))] for _ in range(len(common_ids))]
        vecs = {m: mech_vec_at_layer(diffs_data, id_index, m, resampled, FIXED_LAYER,
                                      PRIMARY_ESTIMATOR, PRIMARY_RAW_PC) for m in active_mechanisms}
        for m in active_mechanisms:
            boot_vals[m].append(prototype_affinity(vecs, m, CO, MG))

    results = {}
    for m in active_mechanisms:
        t = torch.tensor(boot_vals[m])
        results[m] = {
            'point_estimate': point_estimates[m],
            'source_level_bootstrap_ci95': [t.quantile(0.025).item(), t.quantile(0.975).item()],
            'P_gt_0': (t > 0).float().mean().item(),
            'n_bootstrap': H4_N_BOOTSTRAP, 'seed': H4_SEED,
        }
    return results


# ── interpretation matrix ────────────────────────────────────────────────

def build_interpretation_matrix(h1, h2_dangle, h3, h4, dangle_point, norm_changes):
    matrix = []

    # Rule 1: measurement instability
    refusal_sh = h1[TARGET_MECH]['split_half_cosine_mean']
    prefix_sh = h1['prefix_injection']['split_half_cosine_mean']
    persona_sh = h1['persona_roleplay']['split_half_cosine_mean']
    refusal_norm = h1[TARGET_MECH][PRIMARY_RAW_PC]['per_estimator'][PRIMARY_ESTIMATOR]['direction_norm']
    prefix_norm = h1['prefix_injection'][PRIMARY_RAW_PC]['per_estimator'][PRIMARY_ESTIMATOR]['direction_norm']
    persona_norm = h1['persona_roleplay'][PRIMARY_RAW_PC]['per_estimator'][PRIMARY_ESTIMATOR]['direction_norm']
    geo_mean_norm = (prefix_norm * persona_norm) ** 0.5
    norm_ratio = refusal_norm / geo_mean_norm if geo_mean_norm > 0 else None

    threshold_sensitivity = {}
    for thr in RELIABILITY_LOW_THRESHOLDS:
        cond_a = (refusal_sh < thr) and (refusal_sh < prefix_sh - 0.10) and (refusal_sh < persona_sh - 0.10)
        cond_b = (norm_ratio is not None) and (norm_ratio < 0.5)
        threshold_sensitivity[str(thr)] = {'condition_a_split_half': cond_a, 'condition_b_norm_ratio': cond_b,
                                            'triggers_rule_1': cond_a or cond_b}
    rule1_triggered = threshold_sensitivity['0.7']['triggers_rule_1']
    matrix.append({
        'rule_id': 1, 'description': 'measurement-instability candidate',
        'evidence': {'refusal_split_half_mean': refusal_sh, 'prefix_split_half_mean': prefix_sh,
                     'persona_split_half_mean': persona_sh, 'refusal_norm': refusal_norm,
                     'geometric_mean_prefix_persona_norm': geo_mean_norm, 'norm_ratio': norm_ratio,
                     'threshold_sensitivity_0.6_0.7_0.8': threshold_sensitivity},
        'verdict': 'supports measurement-instability candidate' if rule1_triggered
                   else 'does not support measurement-instability candidate',
    })

    # Rule 2 & 3: shared-component masking vs stable template-specific divergence
    delta_co_perp_unit = h3['unit_normalized']['Delta_CO_perp']
    aref_bootstrap = h4[TARGET_MECH]
    aref_ci_lo, aref_ci_hi = aref_bootstrap['source_level_bootstrap_ci95']
    dco_perp_ci_lo, dco_perp_ci_hi = delta_co_perp_unit['ci_lo'], delta_co_perp_unit['ci_hi']

    reliability_ok = not rule1_triggered
    aref_stably_negative = aref_ci_hi < 0
    dco_perp_stably_positive = dco_perp_ci_lo > 0
    dco_perp_stably_negative = dco_perp_ci_hi < 0

    matrix.append({
        'rule_id': 2, 'description': 'shared-component masking',
        'evidence': {'reliability_ok': reliability_ok, 'original_Delta_CO_known_negative': True,
                     'Delta_CO_perp_unit_normalized_stability_interval': [dco_perp_ci_lo, dco_perp_ci_hi]},
        'verdict': 'supports shared-component masking' if (reliability_ok and dco_perp_stably_positive)
                   else 'does not support shared-component masking',
    })
    matrix.append({
        'rule_id': 3, 'description': 'stable template-specific geometric divergence',
        'evidence': {'reliability_ok': reliability_ok,
                     'A_refusal_source_level_bootstrap_ci95': [aref_ci_lo, aref_ci_hi],
                     'Delta_CO_perp_unit_normalized_stability_interval': [dco_perp_ci_lo, dco_perp_ci_hi]},
        'verdict': 'supports stable template-specific geometric divergence'
                   if (reliability_ok and aref_stably_negative and not dco_perp_stably_positive)
                   else 'does not support stable template-specific geometric divergence',
    })

    # Rule 4: angular divergence vs magnitude collapse (diff-in-diff)
    dangle_ci_lo, dangle_ci_hi = h2_dangle['source_level_bootstrap_ci95']
    refusal_rel_change = norm_changes[TARGET_MECH]['relative_change']
    prefix_rel_change = norm_changes['prefix_injection']['relative_change']
    persona_rel_change = norm_changes['persona_roleplay']['relative_change']
    refusal_most_extreme_decline = (
        refusal_rel_change is not None and prefix_rel_change is not None and persona_rel_change is not None
        and refusal_rel_change < min(prefix_rel_change, persona_rel_change)
    )
    dangle_stably_negative = dangle_ci_hi < 0
    matrix.append({
        'rule_id': 4, 'description': 'selective angular divergence rather than magnitude collapse',
        'evidence': {'D_angle_point_estimate': dangle_point,
                     'D_angle_source_level_bootstrap_ci95': [dangle_ci_lo, dangle_ci_hi],
                     'norm_relative_change': {'refusal_suppression': refusal_rel_change,
                                               'prefix_injection': prefix_rel_change,
                                               'persona_roleplay': persona_rel_change},
                     'refusal_has_most_extreme_relative_decline': refusal_most_extreme_decline},
        'verdict': 'supports selective angular divergence rather than magnitude collapse'
                   if (dangle_stably_negative and not refusal_most_extreme_decline)
                   else 'does not support selective angular divergence rather than magnitude collapse',
    })

    # Rule 5: estimator sensitivity of the common-component conclusion
    delta_co_perp_raw = h3['raw_magnitude']['Delta_CO_perp']
    unit_positive = delta_co_perp_unit['ci_lo'] > 0
    unit_negative = delta_co_perp_unit['ci_hi'] < 0
    raw_positive = delta_co_perp_raw['ci_lo'] > 0
    raw_negative = delta_co_perp_raw['ci_hi'] < 0
    opposite_conclusions = (unit_positive and raw_negative) or (unit_negative and raw_positive)
    matrix.append({
        'rule_id': 5, 'description': 'common-component conclusion is estimator-sensitive/inconclusive',
        'evidence': {'unit_normalized_stability_interval': [delta_co_perp_unit['ci_lo'], delta_co_perp_unit['ci_hi']],
                     'raw_magnitude_stability_interval': [delta_co_perp_raw['ci_lo'], delta_co_perp_raw['ci_hi']]},
        'verdict': 'common-component conclusion is estimator-sensitive/inconclusive' if opposite_conclusions
                   else 'common-component conclusion is consistent across both common-direction definitions',
    })

    return matrix


# ── figures ──────────────────────────────────────────────────────────────

def plot_layerwise(layer_rows, early, late, out_path_base):
    layers = [r['layer_index'] for r in layer_rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax1.plot(layers, [r['cos_prefix_refusal'] for r in layer_rows], label='cos(prefix, refusal)')
    ax1.plot(layers, [r['cos_persona_refusal'] for r in layer_rows], label='cos(persona, refusal)')
    ax1.plot(layers, [r['cos_prefix_persona'] for r in layer_rows], label='cos(prefix, persona)', color='black')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax1.axvline(FIXED_LAYER, color='red', linestyle=':', label=f'fixed layer {FIXED_LAYER}')
    ax1.axvspan(min(early), max(early), alpha=0.1, color='blue', label='early third')
    ax1.axvspan(min(late), max(late), alpha=0.1, color='orange', label='late third')
    ax1.set_ylabel('cosine similarity')
    ax1.set_title('Meta-Llama-3.1-8B-Instruct: CO pairwise cosine by layer')
    ax1.legend(fontsize=8)

    ax2.plot(layers, [r['norm_prefix_injection'] for r in layer_rows], label='prefix_injection norm')
    ax2.plot(layers, [r['norm_refusal_suppression'] for r in layer_rows], label='refusal_suppression norm')
    ax2.plot(layers, [r['norm_persona_roleplay'] for r in layer_rows], label='persona_roleplay norm')
    ax2.axvline(FIXED_LAYER, color='red', linestyle=':')
    ax2.axvspan(min(early), max(early), alpha=0.1, color='blue')
    ax2.axvspan(min(late), max(late), alpha=0.1, color='orange')
    ax2.set_xlabel('layer index (0-based)')
    ax2.set_ylabel('direction norm')
    ax2.legend(fontsize=8)

    fig.savefig(out_path_base + '.png', dpi=150, bbox_inches='tight')
    fig.savefig(out_path_base + '.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_prototype_affinity(h4, h3, active_mechanisms, out_path_base):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    means = [h4[m]['point_estimate'] for m in active_mechanisms]
    los = [h4[m]['source_level_bootstrap_ci95'][0] for m in active_mechanisms]
    his = [h4[m]['source_level_bootstrap_ci95'][1] for m in active_mechanisms]
    errs = [[m - l for m, l in zip(means, los)], [h - m for m, h in zip(means, his)]]
    colors = ['crimson' if m == TARGET_MECH else 'steelblue' for m in active_mechanisms]
    ax1.bar(range(len(active_mechanisms)), means, yerr=errs, color=colors, capsize=4)
    ax1.axhline(0, color='gray', linestyle='--')
    ax1.set_xticks(range(len(active_mechanisms)))
    ax1.set_xticklabels(active_mechanisms, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('A_m (own-category minus other-category prototype affinity)')
    ax1.set_title('H4: leave-one-out prototype affinity\n(source-level bootstrap CI)')

    stats = ['Delta_CO_perp', 'Delta_MG_perp', 'V_bilateral', 'A_refusal_perp']
    x = range(len(stats))
    width = 0.35
    for i, gdef in enumerate(['unit_normalized', 'raw_magnitude']):
        means2 = [h3[gdef][s]['mean'] for s in stats]
        los2 = [h3[gdef][s]['ci_lo'] for s in stats]
        his2 = [h3[gdef][s]['ci_hi'] for s in stats]
        errs2 = [[m - l for m, l in zip(means2, los2)], [h - m for m, h in zip(means2, his2)]]
        offset = (i - 0.5) * width
        ax2.bar([xi + offset for xi in x], means2, width=width, yerr=errs2, capsize=4, label=gdef)
    ax2.axhline(0, color='gray', linestyle='--')
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(stats, rotation=20, ha='right', fontsize=8)
    ax2.set_ylabel('value (residual space)')
    ax2.set_title('H3: common-component-removed statistics\n(repeated-split stability interval)')
    ax2.legend(fontsize=8)

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
        direction_ids = set(json.load(f)['direction_ids'])

    path = os.path.join(args.output_dir, MODEL_ALIAS, f'paired_diffs_{args.lang}{args.suffix}.pt')
    diffs_data = torch.load(path, map_location='cpu')
    diffs_data['diffs'] = {m: v.float() for m, v in diffs_data['diffs'].items()}
    n_layers = diffs_data['n_layers']
    id_index = {m: {pid: i for i, pid in enumerate(diffs_data['instruction_ids'][m])}
                for m in active_mechanisms + ['placebo']}
    id_sets = [set(diffs_data['instruction_ids'][m]) for m in active_mechanisms + ['placebo']]
    common_ids = sorted(set.intersection(*id_sets))
    print(f"Loaded {path}  n_layers={n_layers}  fixed_layer={FIXED_LAYER}  common_ids={len(common_ids)}")
    if set(common_ids) != direction_ids:
        print(f"  WARNING: common_ids != direction_ids")

    try:
        git_commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=SCRIPT_DIR,
                                     capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        git_commit = 'unknown'

    print("Running H1 (reliability)...")
    h1 = h1_reliability(diffs_data, id_index, common_ids, CO)

    print("Running H2 (layerwise + D_angle)...")
    layer_rows = h2_layerwise(diffs_data, id_index, common_ids, CO, n_layers)
    early, late = early_late_layers(n_layers)
    dangle_point, dangle_components = dangle_point_estimate(layer_rows, early, late)
    dangle_boot = dangle_bootstrap(diffs_data, id_index, common_ids, CO, n_layers, early, late)
    dangle_boot['point_estimate_from_full_data'] = dangle_point
    norm_changes = {m: norm_relative_change(layer_rows, early, late,
                                             f'norm_{m}') for m in CO}

    print("Running H3 (cross-fitted common-component removal, 500 splits x 2 definitions)...")
    h3 = h3_common_component(diffs_data, id_index, common_ids, active_mechanisms, CO, MG)

    print("Running H4 (leave-one-out prototype, 2000-rep bootstrap)...")
    h4 = h4_prototype_test(diffs_data, id_index, common_ids, active_mechanisms, CO, MG)

    print("Building interpretation matrix...")
    matrix = build_interpretation_matrix(h1, dangle_boot, h3, h4, dangle_point, norm_changes)
    for row in matrix:
        print(f"  Rule {row['rule_id']} ({row['description']}): {row['verdict']}")

    # ── write outputs ────────────────────────────────────────────────────
    out_dir = os.path.join(args.output_dir, 'canonical_v2')
    fig_dir = os.path.join(out_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    metadata = {
        'model': MODEL_ALIAS, 'taxonomy_version': taxonomy['taxonomy_version'],
        'config_path': taxonomy['config_path'], 'CO_mechs': CO, 'MG_mechs': MG,
        'active_mechanisms': active_mechanisms, 'target_mechanism': TARGET_MECH,
        'input_file': path, 'git_commit': git_commit, 'n_layers': n_layers,
        'fixed_layer_0based': FIXED_LAYER, 'primary_estimator': PRIMARY_ESTIMATOR,
        'primary_raw_or_pc': PRIMARY_RAW_PC, 'trim_frac': TRIM_FRAC,
        'early_layers': early, 'late_layers': late,
        'h1_split_half_reps': H1_SPLIT_HALF_REPS, 'h1_split_half_seed': H1_SPLIT_HALF_SEED,
        'h3_n_splits': H3_N_SPLITS, 'h3_seed': H3_SEED,
        'h3_interval_name': 'repeated_split_stability_interval',
        'h4_n_bootstrap': H4_N_BOOTSTRAP, 'h4_seed': H4_SEED,
        'h4_interval_name': 'source_level_bootstrap_ci',
        'dangle_n_bootstrap': DANGLE_N_BOOTSTRAP, 'dangle_seed': DANGLE_SEED,
        'reliability_low_thresholds_swept': RELIABILITY_LOW_THRESHOLDS,
        'torch_version': torch.__version__,
        'note': "verdicts use 'supports X' / 'does not support X' language only -- "
                "not 'confirmed', 'proved', or claims about a causal mechanism.",
    }

    results_json = {
        'metadata': metadata,
        'h1_reliability': h1,
        'h2_dangle': dangle_boot,
        'h2_dangle_components': dangle_components,
        'h2_norm_relative_change': norm_changes,
        'h3_common_component': h3,
        'h4_prototype_affinity': h4,
        'interpretation_matrix': matrix,
    }
    with open(os.path.join(out_dir, 'experiment2_llama_co_divergence.json'), 'w') as f:
        json.dump(results_json, f, indent=2)

    with open(os.path.join(out_dir, 'experiment2_llama_co_reliability.csv'), 'w', newline='') as f:
        rows = []
        for m in CO:
            for raw_or_pc in ['raw', 'placebo_calibrated']:
                for method in ['mean', 'median', 'trimmed_mean']:
                    e = h1[m][raw_or_pc]['per_estimator'][method]
                    rows.append({'mechanism': m, 'raw_or_pc': raw_or_pc, 'estimator': method, **e})
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    with open(os.path.join(out_dir, 'experiment2_llama_co_layerwise.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(layer_rows[0].keys()))
        writer.writeheader(); writer.writerows(layer_rows)

    with open(os.path.join(out_dir, 'experiment2_llama_prototype_affinity.csv'), 'w', newline='') as f:
        rows = [{'mechanism': m, **h4[m]} for m in active_mechanisms]
        for r in rows:
            r['source_level_bootstrap_ci_lo'] = r['source_level_bootstrap_ci95'][0]
            r['source_level_bootstrap_ci_hi'] = r['source_level_bootstrap_ci95'][1]
            del r['source_level_bootstrap_ci95']
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    with open(os.path.join(out_dir, 'experiment2_llama_common_component.csv'), 'w', newline='') as f:
        rows = []
        for gdef in ['unit_normalized', 'raw_magnitude']:
            for stat, vals in h3[gdef].items():
                rows.append({'g_definition': gdef, 'statistic': stat, **vals})
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    plot_layerwise(layer_rows, early, late, os.path.join(fig_dir, 'experiment2_llama_co_layerwise'))
    plot_prototype_affinity(h4, h3, active_mechanisms, os.path.join(fig_dir, 'experiment2_llama_prototype_affinity'))

    print(f"\nSaved outputs to {out_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    parser.add_argument('--lang',       type=str, default='en')
    parser.add_argument('--suffix',     type=str, default='_full572_corrected')
    args = parser.parse_args()
    main(args)
