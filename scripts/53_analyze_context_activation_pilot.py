"""
Context Activation Pilot -- analysis of the real paired-difference tensors
produced by scripts/52_extract_context_activations_pilot.py. Implements
EXPERIMENT2_CONTEXT_ACTIVATION_PILOT_PROTOCOL.md Sec 5-8.

EVERY output carries result_status = "PILOT_NON_RESULT". This script does
NOT decide whether a C direction exists -- it reports descriptive
diagnostics only, per the protocol's explicit "no single metric decides"
rule and "pilot SVD is descriptive only, n_family=4" caveat.

Reads (never writes to) the real .pt files -- no model, no tokenizer, no
GPU needed for this script; pure tensor math on the already-extracted
activations. Verifies both payload SHA-256 values against what
context_activation_pilot_metadata.json recorded before trusting them.

Usage:
  python scripts/53_analyze_context_activation_pilot.py \
      --pilot_dir output/context_activation_pilot/Meta-Llama-3.1-8B-Instruct \
      --write_report output/context_activation_pilot/Meta-Llama-3.1-8B-Instruct/context_activation_pilot_analysis.json
"""
import argparse
import datetime
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = SCRIPT_DIR
sys.path.insert(0, SCRIPT_DIR)

FAMILIES = ['ctx_persona', 'ctx_authority', 'ctx_fictional', 'ctx_continuation']
VARIANTS = ['v1', 'v2', 'v3']
POS_NAMES = ['t_inst', 't_post']  # index 0 / 1, per scripts/52's per_layer stacking order
PRIMARY_LAYER = 19
BOOTSTRAP_RESAMPLES = 1000
RANDOM_SEED = 0

TOKEN_AUDIT_PATH = os.path.join(REPO_ROOT, '..', 'output', 'audits', 'context',
                                 'context_templates_token_length_audit.json')
TOKEN_AUDIT_PATH = os.path.normpath(TOKEN_AUDIT_PATH)


def cosine(a, b):
    denom = (a.norm() * b.norm()).item()
    if denom == 0.0:
        return float('nan')
    return (a @ b).item() / denom


def safe_cosine(a, b):
    """v2's cosine: never returns a bare NaN. Zero-norm inputs (observed at
    layer 0 in the v1 all-layer sweep) return an explicit
    {'value': None, 'status': 'UNDEFINED_ZERO_NORM'} instead."""
    denom = (a.norm() * b.norm()).item()
    if denom == 0.0:
        return {'value': None, 'status': 'UNDEFINED_ZERO_NORM'}
    return {'value': (a @ b).item() / denom, 'status': 'OK'}


def pearson_r(xs, ys):
    n = len(xs)
    mean_x, mean_y = sum(xs) / n, sum(ys) / n
    cov = sum((a - mean_x) * (b - mean_y) for a, b in zip(xs, ys))
    var_x = sum((a - mean_x) ** 2 for a in xs)
    var_y = sum((b - mean_y) ** 2 for b in ys)
    if var_x == 0 or var_y == 0:
        return float('nan')
    return cov / ((var_x * var_y) ** 0.5)


def spearman_r(xs, ys):
    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0] * len(vals)
        for r, i in enumerate(order):
            ranks[i] = r
        return ranks
    return pearson_r(rank(xs), rank(ys))


def load_and_verify(pilot_dir, metadata):
    import torch
    from utils.direction_metadata import sha256_of_nested_tensors

    context_path = os.path.join(REPO_ROOT, '..', metadata['context_paired_diffs_path'])
    canonical_path = os.path.join(REPO_ROOT, '..', metadata['canonical_paired_diffs_path'])
    context_path = os.path.normpath(context_path)
    canonical_path = os.path.normpath(canonical_path)

    context_payload = torch.load(context_path, map_location='cpu')
    canonical_payload = torch.load(canonical_path, map_location='cpu')

    actual_context_hash = sha256_of_nested_tensors(context_payload)
    if actual_context_hash != metadata['context_paired_diffs_sha256']:
        raise ValueError(f"context_paired_diffs hash mismatch: file has {actual_context_hash}, "
                          f"metadata recorded {metadata['context_paired_diffs_sha256']}")
    actual_canonical_hash = sha256_of_nested_tensors(canonical_payload)
    if actual_canonical_hash != metadata['canonical_paired_diffs_sha256']:
        raise ValueError(f"canonical_paired_diffs hash mismatch: file has {actual_canonical_hash}, "
                          f"metadata recorded {metadata['canonical_paired_diffs_sha256']}")

    return context_payload, canonical_payload


def bootstrap_ci(deltas, n_resamples=BOOTSTRAP_RESAMPLES, seed=RANDOM_SEED):
    """deltas: [n_instructions, hidden] per-instruction paired differences
    for one (family, variant, layer, position). Resamples WHOLE
    instructions with replacement (never splits an instruction's own
    delta across resamples, since each row here already IS one
    instruction's own delta -- resampling rows directly satisfies the
    "resample by instruction" requirement). Returns mean paired projection
    (each instruction's own delta projected onto the full-sample
    direction) and a 95% CI on that projection's bootstrap distribution."""
    import torch
    n = deltas.shape[0]
    base_dir = deltas.mean(dim=0)
    base_norm = base_dir.norm()
    if base_norm.item() == 0.0:
        return {'mean_paired_projection': 0.0, 'ci_2_5': 0.0, 'ci_97_5': 0.0}
    unit_dir = base_dir / base_norm
    per_instruction_projection = (deltas @ unit_dir)
    mean_paired_projection = per_instruction_projection.mean().item()

    g = torch.Generator().manual_seed(seed)
    resample_means = torch.empty(n_resamples)
    for i in range(n_resamples):
        idx = torch.randint(0, n, (n,), generator=g)
        resample_means[i] = per_instruction_projection[idx].mean()
    sorted_means, _ = torch.sort(resample_means)
    ci_low = sorted_means[int(0.025 * n_resamples)].item()
    ci_high = sorted_means[int(0.975 * n_resamples) - 1].item()
    return {'mean_paired_projection': mean_paired_projection, 'ci_2_5': ci_low, 'ci_97_5': ci_high}


def split_half_cosine(deltas, seed=RANDOM_SEED):
    import torch
    n = deltas.shape[0]
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    half = n // 2
    d1 = deltas[perm[:half]].mean(dim=0)
    d2 = deltas[perm[half:]].mean(dim=0)
    return cosine(d1, d2)


def main(args):
    import torch

    with open(args.metadata_path, encoding='utf-8') as f:
        metadata = json.load(f)
    if metadata.get('result_status') != 'PILOT_NON_RESULT':
        raise ValueError(f"metadata result_status={metadata.get('result_status')!r}, "
                          f"expected PILOT_NON_RESULT -- refusing to analyze")
    if metadata.get('dry_run') is not False:
        raise ValueError("metadata indicates this was a dry_run -- refusing to analyze dry-run "
                          "activations as if they were real (they are zero-filled synthetic tensors)")

    context_payload, canonical_payload = load_and_verify(args.pilot_dir, metadata)
    n_layers_total = context_payload['delta'].shape[2]
    if PRIMARY_LAYER >= n_layers_total:
        raise ValueError(f"primary_layer={PRIMARY_LAYER} out of range for n_layers_total={n_layers_total}")

    label_to_idx = {label: i for i, label in enumerate(context_payload['condition_labels_all_16'])}
    canon_to_idx = {name: i for i, name in enumerate(canonical_payload['condition_names'])}
    for name in ('plain', 'placebo', 'persona_roleplay', 'prefix_injection'):
        if name not in canon_to_idx:
            raise ValueError(f"canonical condition {name!r} missing from canonical_payload")

    def context_deltas(fam, v, layer, pos):
        idx = label_to_idx[f'{fam}_{v}']
        return context_payload['delta'][:, idx, layer, pos, :].float()

    def canonical_deltas_vs_plain(mech, layer, pos):
        m_idx = canon_to_idx[mech]
        p_idx = canon_to_idx['plain']
        return (canonical_payload['activations'][:, m_idx, layer, pos, :]
                - canonical_payload['activations'][:, p_idx, layer, pos, :]).float()

    # ---- per (family, variant, position) at the primary layer ----
    per_variant = {}
    directions_by_fam_var_pos = {}  # (fam,v,pos) -> [hidden] tensor, primary layer only
    for fam in FAMILIES:
        per_variant[fam] = {}
        for v in VARIANTS:
            per_variant[fam][v] = {}
            for pos_idx, pos_name in enumerate(POS_NAMES):
                deltas = context_deltas(fam, v, PRIMARY_LAYER, pos_idx)
                direction = deltas.mean(dim=0)
                directions_by_fam_var_pos[(fam, v, pos_idx)] = direction
                stats = bootstrap_ci(deltas)
                per_variant[fam][v][pos_name] = {
                    'direction_norm': direction.norm().item(),
                    'mean_paired_projection': stats['mean_paired_projection'],
                    'bootstrap_ci_2_5': stats['ci_2_5'],
                    'bootstrap_ci_97_5': stats['ci_97_5'],
                    'split_half_cosine': split_half_cosine(deltas),
                }
            per_variant[fam][v]['t_inst_vs_t_post_cosine'] = cosine(
                directions_by_fam_var_pos[(fam, v, 0)], directions_by_fam_var_pos[(fam, v, 1)])

    # ---- family structure at primary layer, t_post (primary position) ----
    TPOST = 1
    family_structure = {}
    centroids = {}
    for fam in FAMILIES:
        v_dirs = {v: directions_by_fam_var_pos[(fam, v, TPOST)] for v in VARIANTS}
        centroid = sum(v_dirs[v] for v in VARIANTS) / 3.0
        centroids[fam] = centroid
        pairwise = {}
        for i in range(len(VARIANTS)):
            for j in range(i + 1, len(VARIANTS)):
                pairwise[f'{VARIANTS[i]}_{VARIANTS[j]}'] = cosine(v_dirs[VARIANTS[i]], v_dirs[VARIANTS[j]])
        centroid_to_variant = {v: cosine(centroid, v_dirs[v]) for v in VARIANTS}
        loo = {}
        for held_out in VARIANTS:
            others = [v for v in VARIANTS if v != held_out]
            predicted = (v_dirs[others[0]] + v_dirs[others[1]]) / 2.0
            loo[held_out] = cosine(predicted, v_dirs[held_out])
        family_structure[fam] = {
            'centroid_norm': centroid.norm().item(),
            'within_family_pairwise_cosine': pairwise,
            'centroid_to_variant_cosine': centroid_to_variant,
            'leave_one_variant_out_cosine': loo,
        }

    # ---- cross-family structure + descriptive SVD ----
    cross_family_pairwise = {}
    for i in range(len(FAMILIES)):
        for j in range(i + 1, len(FAMILIES)):
            cross_family_pairwise[f'{FAMILIES[i]}_x_{FAMILIES[j]}'] = cosine(
                centroids[FAMILIES[i]], centroids[FAMILIES[j]])
    centroid_matrix = torch.stack([centroids[f] for f in FAMILIES])  # [4, hidden]
    centroid_matrix_centered = centroid_matrix - centroid_matrix.mean(dim=0, keepdim=True)
    try:
        _, s, _ = torch.linalg.svd(centroid_matrix_centered, full_matrices=False)
        singular_values = s.tolist()
        total_sq = sum(x * x for x in singular_values)
        variance_explained = [x * x / total_sq if total_sq > 0 else 0.0 for x in singular_values]
    except Exception as e:
        singular_values, variance_explained = [], []
        print(f"WARNING: SVD failed: {e}")

    # ---- discriminant comparisons vs canonical mechanisms (placebo-calibrated) ----
    d_placebo = canonical_deltas_vs_plain('placebo', PRIMARY_LAYER, TPOST).mean(dim=0)
    d_persona_roleplay = canonical_deltas_vs_plain('persona_roleplay', PRIMARY_LAYER, TPOST).mean(dim=0)
    d_prefix_injection = canonical_deltas_vs_plain('prefix_injection', PRIMARY_LAYER, TPOST).mean(dim=0)
    tilde_persona_roleplay = d_persona_roleplay - d_placebo
    tilde_prefix_injection = d_prefix_injection - d_placebo
    discriminant = {
        'cos_ctx_persona_vs_tilde_persona_roleplay': cosine(centroids['ctx_persona'], tilde_persona_roleplay),
        'cos_ctx_continuation_vs_tilde_prefix_injection': cosine(
            centroids['ctx_continuation'], tilde_prefix_injection),
        'ctx_continuation_vs_payload_splitting': 'NOT_EXTRACTED_THIS_ROUND -- formal-experiment follow-up item',
        'construction_asymmetry_caveat': (
            'Context directions are positive-minus-shared-neutral contrasts; canonical directions are '
            'template-minus-plain, placebo-calibrated contrasts. These cosines are discriminant '
            'diagnostics (does the context direction point somewhere recognizably similar to a '
            'canonical mechanism direction), not a symmetric mechanism-identity test.'
        ),
    }

    # ---- length sensitivity (descriptive only, n=12) ----
    length_sensitivity = {'note': 'descriptive sensitivity diagnostic only, not a statistical conclusion (n=12)'}
    if os.path.exists(TOKEN_AUDIT_PATH):
        with open(TOKEN_AUDIT_PATH, encoding='utf-8') as f:
            token_audit = json.load(f)
        llama_stats = token_audit['per_model_family_stats'].get('Meta-Llama-3.1-8B-Instruct')
        if llama_stats is not None:
            norms, diffs = [], []
            for fam in FAMILIES:
                for v in VARIANTS:
                    norms.append(per_variant[fam][v]['t_post']['direction_norm'])
                    diffs.append(llama_stats[fam]['diff_vs_neutral'][v])
            n = len(norms)
            mean_n, mean_d = sum(norms) / n, sum(diffs) / n
            cov = sum((a - mean_n) * (b - mean_d) for a, b in zip(norms, diffs))
            var_n = sum((a - mean_n) ** 2 for a in norms)
            var_d = sum((b - mean_d) ** 2 for b in diffs)
            pearson_r = cov / ((var_n * var_d) ** 0.5) if var_n > 0 and var_d > 0 else float('nan')
            length_sensitivity.update({
                'pearson_r_direction_norm_vs_token_length_diff': pearson_r, 'n_points': n,
            })
        else:
            length_sensitivity['note'] += ' -- Llama stats not found in token audit, skipped'
    else:
        length_sensitivity['note'] += f' -- {TOKEN_AUDIT_PATH} not found, skipped'

    # ---- all-layer robustness sweep (lightweight: t_post only, avg norm +
    # avg cross-family cosine + the 2 discriminant cosines, per layer) ----
    layers_swept = list(range(n_layers_total))
    avg_norms, avg_cross_cos, disc1_by_layer, disc2_by_layer = [], [], [], []
    for layer in layers_swept:
        layer_dirs = {}
        for fam in FAMILIES:
            v_dirs = [context_deltas(fam, v, layer, TPOST).mean(dim=0) for v in VARIANTS]
            layer_dirs[fam] = sum(v_dirs) / 3.0
        norms_this_layer = [context_deltas(fam, v, layer, TPOST).mean(dim=0).norm().item()
                             for fam in FAMILIES for v in VARIANTS]
        avg_norms.append(sum(norms_this_layer) / len(norms_this_layer))
        pair_cos = [cosine(layer_dirs[FAMILIES[i]], layer_dirs[FAMILIES[j]])
                    for i in range(4) for j in range(i + 1, 4)]
        avg_cross_cos.append(sum(pair_cos) / len(pair_cos))
        d_pl = canonical_deltas_vs_plain('placebo', layer, TPOST).mean(dim=0)
        d_pr = canonical_deltas_vs_plain('persona_roleplay', layer, TPOST).mean(dim=0)
        d_pi = canonical_deltas_vs_plain('prefix_injection', layer, TPOST).mean(dim=0)
        disc1_by_layer.append(cosine(layer_dirs['ctx_persona'], d_pr - d_pl))
        disc2_by_layer.append(cosine(layer_dirs['ctx_continuation'], d_pi - d_pl))

    report = {
        'result_status': 'PILOT_NON_RESULT',
        'note': ('Descriptive diagnostics only. Per '
                 'EXPERIMENT2_CONTEXT_ACTIVATION_PILOT_PROTOCOL.md Sec 5-8: no single metric decides '
                 'whether a C direction exists; the SVD below is descriptive only (n_family=4); '
                 'directional alignment or separation is not by itself sufficient to claim C exists.'),
        'primary_layer': PRIMARY_LAYER,
        'per_variant_layer19': per_variant,
        'family_structure_layer19_tpost': family_structure,
        'cross_family_layer19_tpost': {
            'pairwise_cosine': cross_family_pairwise,
            'svd_descriptive_only_n_family_4': {
                'singular_values': singular_values, 'variance_explained': variance_explained,
                'k2_preregistered_primary': True,
                'note': 'n_family=4 -- far too few points for formal statistical evidence; k is not '
                        're-selected after seeing these values.',
            },
        },
        'discriminant_layer19_tpost': discriminant,
        'length_sensitivity': length_sensitivity,
        'all_layer_robustness_sweep_tpost': {
            'layers': layers_swept,
            'avg_variant_direction_norm': avg_norms,
            'avg_cross_family_pairwise_cosine': avg_cross_cos,
            'cos_ctx_persona_vs_tilde_persona_roleplay_by_layer': disc1_by_layer,
            'cos_ctx_continuation_vs_tilde_prefix_injection_by_layer': disc2_by_layer,
            'note': 'diagnostic/robustness description only, not a substitute for the layer-19 primary report',
        },
        'provenance': {
            'pilot_metadata_path': args.metadata_path,
            'context_paired_diffs_sha256': metadata['context_paired_diffs_sha256'],
            'canonical_paired_diffs_sha256': metadata['canonical_paired_diffs_sha256'],
            'pilot_git_commit': metadata['git_commit'],
            'instruction_ids': metadata['instruction_ids'],
        },
        'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    print(json.dumps({k: v for k, v in report.items()
                       if k not in ('all_layer_robustness_sweep_tpost', 'per_variant_layer19')}, indent=2))

    if os.path.exists(args.write_report):
        print(f"\nRefusing to write: {args.write_report} already exists.")
        sys.exit(1)
    os.makedirs(os.path.dirname(os.path.abspath(args.write_report)), exist_ok=True)
    tmp_path = args.write_report + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(report, f, indent=2)
    os.replace(tmp_path, args.write_report)
    print(f"\nWrote analysis (new file, non-overwriting): {args.write_report}")


SPLIT_HALF_SEEDS = list(range(1000, 1100))  # 100 predefined seeds, per user's item B
BOOTSTRAP_SEED_BASE = 0  # bootstrap_direction_cosine / signed-projection CI use this as their single seed


def repeated_split_half_all_variants(context_payload, label_to_idx, layer, pos_idx, seeds):
    """One instruction partition per seed, shared across all 12 (family,
    variant) combos for that seed -- satisfies "the same instruction's
    conditions must stay in the same half" by construction, since the
    partition is computed once per seed and reused, not re-drawn per
    variant. Returns {fam: {v: {median, q2_5, q97_5, n_seeds}}}."""
    import torch
    n = context_payload['delta'].shape[0]
    half = n // 2
    per_seed_cosines = {fam: {v: [] for v in VARIANTS} for fam in FAMILIES}
    for seed in seeds:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=g)
        idx1, idx2 = perm[:half], perm[half:]
        for fam in FAMILIES:
            for v in VARIANTS:
                idx = label_to_idx[f'{fam}_{v}']
                deltas = context_payload['delta'][:, idx, layer, pos_idx, :].float()
                d1, d2 = deltas[idx1].mean(dim=0), deltas[idx2].mean(dim=0)
                c = safe_cosine(d1, d2)
                per_seed_cosines[fam][v].append(c['value'] if c['status'] == 'OK' else float('nan'))
    summary = {}
    for fam in FAMILIES:
        summary[fam] = {}
        for v in VARIANTS:
            vals, _ = torch.sort(torch.tensor(per_seed_cosines[fam][v]))
            n_s = len(vals)
            summary[fam][v] = {
                'median': vals[n_s // 2].item(),
                'q2_5': vals[int(0.025 * n_s)].item(),
                'q97_5': vals[min(int(0.975 * n_s), n_s - 1)].item(),
                'n_seeds': n_s,
            }
    return summary


def bootstrap_direction_cosine(deltas, n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED_BASE):
    """cosine(bootstrap-resampled direction, full-sample direction) across
    n_resamples instruction-level resamples -- reports median + 95%
    interval. This is a DIRECTION-stability check (does the direction
    itself stay pointed the same way under resampling), distinct from
    bootstrap_ci()'s projection-magnitude CI."""
    import torch
    n = deltas.shape[0]
    full_dir = deltas.mean(dim=0)
    g = torch.Generator().manual_seed(seed)
    cosines = []
    for _ in range(n_resamples):
        idx = torch.randint(0, n, (n,), generator=g)
        resampled_dir = deltas[idx].mean(dim=0)
        c = safe_cosine(resampled_dir, full_dir)
        cosines.append(c['value'] if c['status'] == 'OK' else float('nan'))
    sorted_c, _ = torch.sort(torch.tensor(cosines))
    n_s = len(sorted_c)
    return {
        'median': sorted_c[n_s // 2].item(),
        'ci_2_5': sorted_c[int(0.025 * n_s)].item(),
        'ci_97_5': sorted_c[min(int(0.975 * n_s), n_s - 1)].item(),
    }


def main_v2(args):
    import torch

    with open(args.metadata_path, encoding='utf-8') as f:
        metadata = json.load(f)
    if metadata.get('result_status') != 'PILOT_NON_RESULT':
        raise ValueError(f"metadata result_status={metadata.get('result_status')!r}, "
                          f"expected PILOT_NON_RESULT -- refusing to analyze")
    if metadata.get('dry_run') is not False:
        raise ValueError("metadata indicates this was a dry_run -- refusing to analyze dry-run "
                          "activations as if they were real")

    context_payload, canonical_payload = load_and_verify(args.pilot_dir, metadata)
    n_layers_total = context_payload['delta'].shape[2]
    if PRIMARY_LAYER >= n_layers_total:
        raise ValueError(f"primary_layer={PRIMARY_LAYER} out of range for n_layers_total={n_layers_total}")

    label_to_idx = {label: i for i, label in enumerate(context_payload['condition_labels_all_16'])}
    TINST, TPOST = 0, 1

    def context_deltas(fam, v, layer, pos):
        idx = label_to_idx[f'{fam}_{v}']
        return context_payload['delta'][:, idx, layer, pos, :].float()

    # ---- A. full t_inst battery at layer 19 ----
    per_variant_tinst = {}
    directions_tinst = {}
    for fam in FAMILIES:
        per_variant_tinst[fam] = {}
        for v in VARIANTS:
            deltas = context_deltas(fam, v, PRIMARY_LAYER, TINST)
            direction = deltas.mean(dim=0)
            directions_tinst[(fam, v)] = direction
            proj_stats = bootstrap_ci(deltas)  # signed paired projection + magnitude CI
            dir_cos_stats = bootstrap_direction_cosine(deltas)
            per_variant_tinst[fam][v] = {
                'direction_norm': direction.norm().item(),
                'signed_paired_projection': {
                    'mean': proj_stats['mean_paired_projection'],
                    'bootstrap_ci_2_5': proj_stats['ci_2_5'],
                    'bootstrap_ci_97_5': proj_stats['ci_97_5'],
                    'note': 'magnitude/projection description only -- NOT a claim that the CI failing '
                            'to cross zero proves the direction itself is stable; see '
                            'bootstrap_direction_cosine for that.',
                },
                'bootstrap_direction_cosine': dir_cos_stats,
            }

    repeated_split_half = repeated_split_half_all_variants(
        context_payload, label_to_idx, PRIMARY_LAYER, TINST, SPLIT_HALF_SEEDS)
    for fam in FAMILIES:
        for v in VARIANTS:
            per_variant_tinst[fam][v]['repeated_split_half_cosine'] = repeated_split_half[fam][v]

    # within-family pairwise + leave-one-variant-out at t_inst
    family_structure_tinst = {}
    centroids_tinst = {}
    for fam in FAMILIES:
        v_dirs = {v: directions_tinst[(fam, v)] for v in VARIANTS}
        centroid = sum(v_dirs[v] for v in VARIANTS) / 3.0
        centroids_tinst[fam] = centroid
        pairwise = {}
        for i in range(len(VARIANTS)):
            for j in range(i + 1, len(VARIANTS)):
                pairwise[f'{VARIANTS[i]}_{VARIANTS[j]}'] = safe_cosine(v_dirs[VARIANTS[i]], v_dirs[VARIANTS[j]])
        loo = {}
        for held_out in VARIANTS:
            others = [v for v in VARIANTS if v != held_out]
            predicted = (v_dirs[others[0]] + v_dirs[others[1]]) / 2.0
            loo[held_out] = safe_cosine(predicted, v_dirs[held_out])
        family_structure_tinst[fam] = {
            'centroid_norm': centroid.norm().item(),
            'within_family_pairwise_cosine': pairwise,
            'leave_one_variant_out_cosine': loo,
        }

    # t_inst norm vs token-length-diff correlation (Pearson + Spearman)
    tinst_length_corr = {'note': 'descriptive sensitivity diagnostic only, not a statistical conclusion (n=12)'}
    if os.path.exists(TOKEN_AUDIT_PATH):
        with open(TOKEN_AUDIT_PATH, encoding='utf-8') as f:
            token_audit = json.load(f)
        llama_stats = token_audit['per_model_family_stats'].get('Meta-Llama-3.1-8B-Instruct')
        if llama_stats is not None:
            norms, diffs = [], []
            for fam in FAMILIES:
                for v in VARIANTS:
                    norms.append(per_variant_tinst[fam][v]['direction_norm'])
                    diffs.append(llama_stats[fam]['diff_vs_neutral'][v])
            tinst_length_corr.update({
                'pearson_r': pearson_r(norms, diffs), 'spearman_r': spearman_r(norms, diffs), 'n_points': len(norms),
            })
        else:
            tinst_length_corr['note'] += ' -- Llama stats not found in token audit, skipped'
    else:
        tinst_length_corr['note'] += f' -- {TOKEN_AUDIT_PATH} not found, skipped'

    # ---- D. candidate context structure at t_inst/layer19 (DESCRIPTIVE_PILOT) ----
    full_matrix = {fam_i: {fam_j: safe_cosine(centroids_tinst[fam_i], centroids_tinst[fam_j])
                            for fam_j in FAMILIES} for fam_i in FAMILIES}
    off_diag = [full_matrix[FAMILIES[i]][FAMILIES[j]]['value']
                for i in range(4) for j in range(4) if i != j
                and full_matrix[FAMILIES[i]][FAMILIES[j]]['status'] == 'OK']
    avg_cross_family_cosine = sum(off_diag) / len(off_diag) if off_diag else None

    centroid_matrix = torch.stack([centroids_tinst[f] for f in FAMILIES])
    centroid_matrix_centered = centroid_matrix - centroid_matrix.mean(dim=0, keepdim=True)
    try:
        _, s, _ = torch.linalg.svd(centroid_matrix_centered, full_matrices=False)
        singular_values = s.tolist()
        total_sq = sum(x * x for x in singular_values)
        variance_explained = [x * x / total_sq if total_sq > 0 else 0.0 for x in singular_values]
    except Exception as e:
        singular_values, variance_explained = [], []
        print(f"WARNING: SVD failed: {e}")
    k1_variance = variance_explained[0] if variance_explained else None
    k2_variance = sum(variance_explained[:2]) if len(variance_explained) >= 2 else None

    lofo = {}
    for held_out in FAMILIES:
        others = [f for f in FAMILIES if f != held_out]
        predicted = sum(centroids_tinst[f] for f in others) / len(others)
        lofo[held_out] = safe_cosine(predicted, centroids_tinst[held_out])

    candidate_structure_tinst = {
        'result_status': 'DESCRIPTIVE_PILOT',
        'full_4x4_pairwise_cosine_matrix': full_matrix,
        'average_cross_family_cosine': avg_cross_family_cosine,
        'svd_singular_values': singular_values,
        'svd_variance_explained': variance_explained,
        'k1_variance_explained': k1_variance,
        'k2_variance_explained': k2_variance,
        'leave_one_family_out_cosine': lofo,
        'note': 'n_family=4 -- descriptive only, NOT sufficient on its own to select a formal '
                'dimensionality. Neither k=1 nor k=2 is adopted by this report.',
    }

    # ---- E. ctx_continuation format-position diagnostic ----
    continuation_format_diag = {}
    for v in VARIANTS:
        d_tinst = directions_tinst[('ctx_continuation', v)]
        d_tpost = context_deltas('ctx_continuation', v, PRIMARY_LAYER, TPOST).mean(dim=0)
        norm_tinst, norm_tpost = d_tinst.norm().item(), d_tpost.norm().item()
        diff_vec = d_tpost - d_tinst
        continuation_format_diag[v] = {
            't_inst_norm': norm_tinst,
            't_post_norm': norm_tpost,
            'cosine_t_inst_t_post': safe_cosine(d_tinst, d_tpost),
            'norm_ratio_tpost_over_tinst': (norm_tpost / norm_tinst) if norm_tinst != 0 else None,
            'difference_vector_norm': diff_vec.norm().item(),
            'difference_vector_label': 'format-position-associated residual (d_tpost - d_tinst) -- '
                                        'NOT a causal format direction; no causal claim is made about '
                                        'what produces this residual.',
        }

    report = {
        'result_status': 'PILOT_NON_RESULT',
        'checklist_version': 'v2',
        'primary_position_for_formal_followup': 't_inst',
        'secondary_position': 't_post',
        'design_revision_reason': (
            "Real pilot data (t_post primary run) found ctx_continuation's t_post direction norm "
            "~5-8x larger than t_inst, with cos(t_inst,t_post)=0.10-0.15, while the other 3 families "
            "showed t_inst==t_post exactly (their templates place {instruction} at the very end). "
            "t_post for ctx_continuation sits immediately after a Response:/Speaker B:/New response: "
            "token carrying its own identity/format signal -- t_post is therefore confounded with "
            "trailing-format-token identity for that family and is demoted to a sensitivity analysis; "
            "t_inst does not have this specific confound. See "
            "EXPERIMENT2_CONTEXT_ACTIVATION_PILOT_PROTOCOL.md Sec 13 for the full record (Sec 3's "
            "original t_post-primary design is NOT deleted, only revised going forward)."
        ),
        'note': ('Descriptive pilot diagnostics only. No single metric decides whether a C direction '
                 'exists. Section D (candidate structure) is DESCRIPTIVE_PILOT and not sufficient, on '
                 'its own, to select a formal dimensionality (n_family=4). Neither a single shared '
                 'direction nor k=2 is adopted here -- the 4 family-specific directions are the '
                 'conservative default representation until a formal decision is made.'),
        'primary_layer': PRIMARY_LAYER,
        'per_variant_tinst_layer19': per_variant_tinst,
        'family_structure_tinst_layer19': family_structure_tinst,
        'tinst_length_sensitivity': tinst_length_corr,
        'candidate_context_structure_tinst_layer19': candidate_structure_tinst,
        'ctx_continuation_format_position_diagnostic_layer19': continuation_format_diag,
        'provenance': {
            'pilot_metadata_path': args.metadata_path,
            'context_paired_diffs_sha256': metadata['context_paired_diffs_sha256'],
            'canonical_paired_diffs_sha256': metadata['canonical_paired_diffs_sha256'],
            'pilot_extraction_git_commit': metadata['git_commit'],
            'instruction_ids': metadata['instruction_ids'],
            'analysis_git_commit': None,  # filled in by caller if available; see --write_report notes
        },
        'bootstrap_seeds': {'n_resamples': BOOTSTRAP_RESAMPLES, 'seed': BOOTSTRAP_SEED_BASE},
        'split_half_seeds': SPLIT_HALF_SEEDS,
        'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    try:
        from utils.direction_metadata import current_git_commit
        report['provenance']['analysis_git_commit'] = current_git_commit(os.path.join(REPO_ROOT, '..'))
    except Exception as e:
        print(f"WARNING: could not resolve analysis git commit: {e}")

    print(json.dumps({k: v for k, v in report.items()
                       if k not in ('per_variant_tinst_layer19',)}, indent=2))

    if os.path.exists(args.write_report):
        print(f"\nRefusing to write: {args.write_report} already exists.")
        sys.exit(1)
    os.makedirs(os.path.dirname(os.path.abspath(args.write_report)), exist_ok=True)
    tmp_path = args.write_report + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(report, f, indent=2)
    os.replace(tmp_path, args.write_report)
    print(f"\nWrote v2 analysis (new file, non-overwriting): {args.write_report}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot_dir', type=str, required=True)
    parser.add_argument('--metadata_path', type=str, default=None)
    parser.add_argument('--write_report', type=str, required=True)
    parser.add_argument('--v2', action='store_true',
                         help='Run the t_inst-primary extended analysis (Sec 14 of the protocol) instead '
                              'of the original t_post-primary analysis.')
    args = parser.parse_args()
    if args.metadata_path is None:
        args.metadata_path = os.path.join(args.pilot_dir, 'context_activation_pilot_metadata.json')
    if args.v2:
        main_v2(args)
    else:
        main(args)
