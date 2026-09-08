"""
Formal C-Direction Estimation -- analysis of the real per-model paired-
difference/activation tensors produced by
scripts/54_extract_formal_context_activations.py. Implements
EXPERIMENT2_FORMAL_C_DIRECTION_PROTOCOL.md Sec 7-9.

EVERY output carries result_status = "FORMAL_DIRECTION_ESTIMATION". This
script does NOT decide whether/how a C construct generalizes -- it
reports per-model and cross-model descriptive diagnostics only, per the
frozen family-specific-representation decision
(EXPERIMENT2_CONTEXT_ACTIVATION_PILOT_PROTOCOL.md Sec 15): the primary
output is the 4-tuple of family-specific directions per model; k=2 (or
any k) subspace compression is reported as an EXPLORATORY lens only,
never adopted here.

Reads (never writes to) the real .pt files for all 3 models -- no model,
no tokenizer, no GPU needed for this script.

Cross-model comparison (Sec 9) NEVER compares raw direction coordinates
across models (different models have unrelated hidden spaces) -- only
RSA-style comparisons (Gram-matrix correlation, rank consistency).

Usage:
  python scripts/55_analyze_formal_context_activations.py \
      --formal_dir output/context_activations_formal \
      --model_aliases Qwen2.5-7B-Instruct,Meta-Llama-3.1-8B-Instruct,gemma-2-9b-it \
      --write_report output/context_activations_formal/formal_c_direction_analysis.json
"""
import argparse
import datetime
import itertools
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

FAMILIES = ['ctx_persona', 'ctx_authority', 'ctx_fictional', 'ctx_continuation']
VARIANTS = ['v1', 'v2', 'v3']
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0
SPLIT_HALF_SEEDS = list(range(1000, 1100))


def safe_cosine(a, b):
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


def load_and_verify_model(formal_dir, model_alias):
    import torch
    from utils.direction_metadata import sha256_of_nested_tensors

    model_dir = os.path.join(formal_dir, model_alias)
    metadata_path = os.path.join(model_dir, 'context_activation_formal_metadata.json')
    with open(metadata_path, encoding='utf-8') as f:
        metadata = json.load(f)
    if metadata.get('result_status') != 'FORMAL_DIRECTION_ESTIMATION':
        raise ValueError(f"{model_alias}: metadata result_status={metadata.get('result_status')!r}, "
                          f"expected FORMAL_DIRECTION_ESTIMATION")
    if metadata.get('dry_run') is not False:
        raise ValueError(f"{model_alias}: metadata indicates dry_run -- refusing to analyze as real")

    context_path = os.path.join(REPO_ROOT, metadata['context_paired_diffs_path'])
    canonical_path = os.path.join(REPO_ROOT, metadata['canonical_activations_path'])
    context_payload = torch.load(context_path, map_location='cpu')
    canonical_payload = torch.load(canonical_path, map_location='cpu')

    if sha256_of_nested_tensors(context_payload) != metadata['context_paired_diffs_sha256']:
        raise ValueError(f"{model_alias}: context_paired_diffs hash mismatch")
    if sha256_of_nested_tensors(canonical_payload) != metadata['canonical_activations_sha256']:
        raise ValueError(f"{model_alias}: canonical_activations hash mismatch")

    return metadata, context_payload, canonical_payload


def bootstrap_direction_cosine(deltas, n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
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
    return {'median': sorted_c[n_s // 2].item(), 'ci_2_5': sorted_c[int(0.025 * n_s)].item(),
            'ci_97_5': sorted_c[min(int(0.975 * n_s), n_s - 1)].item()}


def repeated_split_half_all_variants(context_payload, label_to_idx, layer, pos_idx, seeds):
    import torch
    n = context_payload['delta'].shape[0]
    half = n // 2
    per_seed = {fam: {v: [] for v in VARIANTS} for fam in FAMILIES}
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
                per_seed[fam][v].append(c['value'] if c['status'] == 'OK' else float('nan'))
    summary = {}
    for fam in FAMILIES:
        summary[fam] = {}
        for v in VARIANTS:
            vals, _ = torch.sort(torch.tensor(per_seed[fam][v]))
            n_s = len(vals)
            summary[fam][v] = {'median': vals[n_s // 2].item(), 'q2_5': vals[int(0.025 * n_s)].item(),
                                'q97_5': vals[min(int(0.975 * n_s), n_s - 1)].item(), 'n_seeds': n_s}
    return summary


def analyze_one_model(model_alias, metadata, context_payload, canonical_payload):
    import torch

    layer = metadata['primary_layer']
    TINST, TPOST = 0, 1
    label_to_idx = {label: i for i, label in enumerate(context_payload['condition_labels_all_16'])}
    canon_to_idx = {name: i for i, name in enumerate(canonical_payload['condition_names'])}

    def context_deltas(fam, v, ly, pos):
        return context_payload['delta'][:, label_to_idx[f'{fam}_{v}'], ly, pos, :].float()

    def canonical_deltas_vs_plain(mech, ly, pos):
        m_idx, p_idx = canon_to_idx[mech], canon_to_idx['plain']
        return (canonical_payload['activations'][:, m_idx, ly, pos, :]
                - canonical_payload['activations'][:, p_idx, ly, pos, :]).float()

    # ---- A/B: per-variant + per-family at t_inst, primary layer ----
    directions = {}
    per_variant = {}
    for fam in FAMILIES:
        per_variant[fam] = {}
        for v in VARIANTS:
            d_tinst = context_deltas(fam, v, layer, TINST).mean(dim=0)
            d_tpost = context_deltas(fam, v, layer, TPOST).mean(dim=0)
            directions[(fam, v)] = d_tinst
            deltas_tinst = context_deltas(fam, v, layer, TINST)
            per_variant[fam][v] = {
                'direction_norm': d_tinst.norm().item(),
                'bootstrap_direction_cosine': bootstrap_direction_cosine(deltas_tinst),
                'signed_paired_projection_mean': deltas_tinst.mean(dim=0).norm().item(),
                't_inst_vs_t_post_cosine': safe_cosine(d_tinst, d_tpost),
            }
    repeated_split_half = repeated_split_half_all_variants(context_payload, label_to_idx, layer, TINST,
                                                             SPLIT_HALF_SEEDS)
    for fam in FAMILIES:
        for v in VARIANTS:
            per_variant[fam][v]['repeated_split_half_cosine'] = repeated_split_half[fam][v]

    family_structure = {}
    centroids = {}
    for fam in FAMILIES:
        v_dirs = {v: directions[(fam, v)] for v in VARIANTS}
        centroid = sum(v_dirs[v] for v in VARIANTS) / 3.0
        centroids[fam] = centroid
        pairwise = {f'{VARIANTS[i]}_{VARIANTS[j]}': safe_cosine(v_dirs[VARIANTS[i]], v_dirs[VARIANTS[j]])
                    for i in range(3) for j in range(i + 1, 3)}
        loo = {}
        for held_out in VARIANTS:
            others = [v for v in VARIANTS if v != held_out]
            predicted = (v_dirs[others[0]] + v_dirs[others[1]]) / 2.0
            loo[held_out] = safe_cosine(predicted, v_dirs[held_out])
        # bootstrap CI on the family centroid's own norm (magnitude description)
        all_deltas = torch.cat([context_deltas(fam, v, layer, TINST) for v in VARIANTS], dim=0)
        family_structure[fam] = {
            'centroid_norm': centroid.norm().item(),
            'within_family_pairwise_cosine': pairwise,
            'leave_one_variant_out_cosine': loo,
            'centroid_norm_bootstrap_ci': bootstrap_direction_cosine(all_deltas),
        }

    # ---- C: cross-family + k=1/2/3 exploratory ----
    full_matrix = {fi: {fj: safe_cosine(centroids[fi], centroids[fj]) for fj in FAMILIES} for fi in FAMILIES}
    off_diag = [full_matrix[FAMILIES[i]][FAMILIES[j]]['value'] for i in range(4) for j in range(4) if i != j
                and full_matrix[FAMILIES[i]][FAMILIES[j]]['status'] == 'OK']
    avg_cross_family_cosine = sum(off_diag) / len(off_diag) if off_diag else None

    centroid_matrix = torch.stack([centroids[f] for f in FAMILIES])
    centered = centroid_matrix - centroid_matrix.mean(dim=0, keepdim=True)
    U, s, Vt = torch.linalg.svd(centered, full_matrices=False)
    singular_values = s.tolist()
    total_sq = sum(x * x for x in singular_values)
    variance_explained = [x * x / total_sq if total_sq > 0 else 0.0 for x in singular_values]
    k_variance = {f'k{k}': sum(variance_explained[:k]) for k in (1, 2, 3) if k <= len(variance_explained)}

    lofo = {}
    for held_out in FAMILIES:
        others = [f for f in FAMILIES if f != held_out]
        predicted = sum(centroids[f] for f in others) / len(others)
        lofo[held_out] = safe_cosine(predicted, centroids[held_out])

    # k=2 subspace stability: bootstrap + split-half principal angles
    k2_basis_full = Vt[:2] if Vt.shape[0] >= 2 else None  # [2, hidden], rows orthonormal (right singular vectors)

    def k2_subspace_from_centroids(cents):
        m = torch.stack(cents)
        m = m - m.mean(dim=0, keepdim=True)
        _, _, vt = torch.linalg.svd(m, full_matrices=False)
        return vt[:2] if vt.shape[0] >= 2 else vt

    def principal_angles(basis1, basis2):
        # basis1/basis2: [k, hidden], rows orthonormal
        M = basis1 @ basis2.T
        sv = torch.linalg.svdvals(M)
        sv = torch.clamp(sv, -1.0, 1.0)
        return torch.acos(sv).tolist()

    n_instr = context_payload['delta'].shape[0]
    g = torch.Generator().manual_seed(BOOTSTRAP_SEED)
    boot_angles = []
    for _ in range(min(BOOTSTRAP_RESAMPLES, 500)):  # capped: 4-point SVD is cheap but keep this section bounded
        idx = torch.randint(0, n_instr, (n_instr,), generator=g)
        resampled_centroids = []
        for fam in FAMILIES:
            v_dirs_r = [context_deltas(fam, v, layer, TINST)[idx].mean(dim=0) for v in VARIANTS]
            resampled_centroids.append(sum(v_dirs_r) / 3.0)
        basis_r = k2_subspace_from_centroids(resampled_centroids)
        if k2_basis_full is not None and basis_r.shape[0] == 2:
            angles = principal_angles(k2_basis_full, basis_r)
            boot_angles.append(max(angles))  # worst-case principal angle, radians
    boot_angles_t = torch.tensor(boot_angles) if boot_angles else torch.tensor([float('nan')])
    sorted_ba, _ = torch.sort(boot_angles_t)
    n_ba = len(sorted_ba)
    k2_bootstrap_stability = {
        'n_resamples': len(boot_angles), 'median_max_principal_angle_rad': sorted_ba[n_ba // 2].item(),
        'ci_2_5_rad': sorted_ba[int(0.025 * n_ba)].item(),
        'ci_97_5_rad': sorted_ba[min(int(0.975 * n_ba), n_ba - 1)].item(),
    }

    split_angles = []
    for seed in SPLIT_HALF_SEEDS:
        gg = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n_instr, generator=gg)
        half = n_instr // 2
        idx1, idx2 = perm[:half], perm[half:]
        c1, c2 = [], []
        for fam in FAMILIES:
            v1 = [context_deltas(fam, v, layer, TINST)[idx1].mean(dim=0) for v in VARIANTS]
            v2 = [context_deltas(fam, v, layer, TINST)[idx2].mean(dim=0) for v in VARIANTS]
            c1.append(sum(v1) / 3.0)
            c2.append(sum(v2) / 3.0)
        b1, b2 = k2_subspace_from_centroids(c1), k2_subspace_from_centroids(c2)
        if b1.shape[0] == 2 and b2.shape[0] == 2:
            split_angles.append(max(principal_angles(b1, b2)))
    split_angles_t = torch.tensor(split_angles) if split_angles else torch.tensor([float('nan')])
    sorted_sa, _ = torch.sort(split_angles_t)
    n_sa = len(sorted_sa)
    k2_split_half_stability = {
        'n_seeds': len(split_angles), 'median_max_principal_angle_rad': sorted_sa[n_sa // 2].item(),
        'q2_5_rad': sorted_sa[int(0.025 * n_sa)].item(), 'q97_5_rad': sorted_sa[min(int(0.975 * n_sa), n_sa - 1)].item(),
    }

    k2_lofo_reconstruction = {}
    for held_out in FAMILIES:
        others = [f for f in FAMILIES if f != held_out]
        basis = k2_subspace_from_centroids([centroids[f] for f in others])
        held_vec = centroids[held_out]
        proj_coeffs = basis @ held_vec
        reconstruction = proj_coeffs @ basis
        recon_cos = safe_cosine(reconstruction, held_vec)
        residual_norm = (held_vec - reconstruction).norm().item()
        k2_lofo_reconstruction[held_out] = {
            'reconstruction_cosine': recon_cos,
            'residual_norm': residual_norm,
            'residual_fraction_of_held_out_norm': residual_norm / held_vec.norm().item() if held_vec.norm().item() > 0 else None,
        }

    candidate_structure = {
        'result_status': 'DESCRIPTIVE_EXPLORATORY',
        'full_4x4_pairwise_cosine_matrix': full_matrix,
        'average_cross_family_cosine': avg_cross_family_cosine,
        'svd_singular_values': singular_values,
        'svd_variance_explained': variance_explained,
        'k_variance_explained': k_variance,
        'leave_one_family_out_cosine': lofo,
        'k2_bootstrap_subspace_stability': k2_bootstrap_stability,
        'k2_split_half_subspace_stability': k2_split_half_stability,
        'k2_leave_one_family_out_reconstruction': k2_lofo_reconstruction,
        'note': ('n_family=4 -- exploratory only, per the frozen family-specific-representation '
                 'decision (pilot protocol Sec 15). k is NOT adopted here regardless of these numbers.'),
    }

    # ---- D: canonical attack-profile projection ----
    d_placebo = canonical_deltas_vs_plain('placebo', layer, TPOST).mean(dim=0)
    canonical_names = [n for n in canonical_payload['condition_names'] if n not in ('plain', 'placebo')]
    attack_profiles = {}
    for mech in canonical_names:
        tilde_d = canonical_deltas_vs_plain(mech, layer, TPOST).mean(dim=0) - d_placebo
        profile = {}
        for fam in FAMILIES:
            c = safe_cosine(tilde_d, centroids[fam])
            raw_projection = (tilde_d @ centroids[fam]).item()
            normalized_projection = (raw_projection / centroids[fam].norm().item()
                                      if centroids[fam].norm().item() > 0 else None)
            profile[fam] = {'cosine': c, 'raw_signed_projection': raw_projection,
                             'normalized_signed_projection': normalized_projection}
        attack_profiles[mech] = profile

    return {
        'model_alias': model_alias,
        'primary_layer': layer,
        'per_variant_tinst': per_variant,
        'family_structure_tinst': family_structure,
        'candidate_context_structure_tinst': candidate_structure,
        'canonical_attack_profiles_tpost': attack_profiles,
        'centroids_tinst': {fam: centroids[fam] for fam in FAMILIES},  # kept in-memory for RSA; not JSON-serialized directly
    }


def rsa_cross_model(per_model_results):
    """Sec 9: RSA-style cross-model comparison. NEVER compares raw
    direction coordinates -- only (a) correlation between models' 4x4
    family Gram matrices (flattened off-diagonal), (b) rank consistency
    of canonical-mechanism context-profiles."""
    import torch
    aliases = list(per_model_results.keys())
    gram_flat = {}
    for alias in aliases:
        cs = per_model_results[alias]['candidate_context_structure_tinst']['full_4x4_pairwise_cosine_matrix']
        flat = [cs[FAMILIES[i]][FAMILIES[j]]['value'] for i in range(4) for j in range(4) if i != j]
        gram_flat[alias] = flat

    gram_rsa = {}
    for a1, a2 in itertools.combinations(aliases, 2):
        r = pearson_r(gram_flat[a1], gram_flat[a2])
        gram_rsa[f'{a1}_x_{a2}'] = {'pearson_r_of_gram_offdiag': r}

    mech_rank_consistency = {}
    all_mechs = set()
    for alias in aliases:
        all_mechs |= set(per_model_results[alias]['canonical_attack_profiles_tpost'].keys())
    for mech in sorted(all_mechs):
        per_model_ranks = {}
        for alias in aliases:
            profile = per_model_results[alias]['canonical_attack_profiles_tpost'].get(mech)
            if profile is None:
                continue
            cos_vals = [profile[fam]['cosine']['value'] if profile[fam]['cosine']['status'] == 'OK' else None
                        for fam in FAMILIES]
            if any(v is None for v in cos_vals):
                continue
            order = sorted(range(4), key=lambda i: cos_vals[i])
            ranks = [0] * 4
            for r, i in enumerate(order):
                ranks[i] = r
            per_model_ranks[alias] = dict(zip(FAMILIES, ranks))
        pairwise_rank_corr = {}
        for a1, a2 in itertools.combinations(aliases, 2):
            if a1 in per_model_ranks and a2 in per_model_ranks:
                r1 = [per_model_ranks[a1][f] for f in FAMILIES]
                r2 = [per_model_ranks[a2][f] for f in FAMILIES]
                pairwise_rank_corr[f'{a1}_x_{a2}'] = spearman_r(r1, r2)
        mech_rank_consistency[mech] = {'per_model_ranks': per_model_ranks,
                                        'pairwise_spearman_of_ranks': pairwise_rank_corr}

    return {
        'result_status': 'DESCRIPTIVE_EXPLORATORY',
        'note': 'RSA-only: raw direction coordinates are never compared across models (unrelated hidden '
                'spaces). Gram-matrix correlation asks whether families relate to each other in the same '
                'PATTERN across models; rank consistency asks whether relative family ordering per '
                'canonical mechanism is stable across models. Any pattern holding for only 1-2 of 3 '
                'models is model-specific, not generalized.',
        'family_gram_matrix_rsa': gram_rsa,
        'canonical_mechanism_rank_consistency': mech_rank_consistency,
    }


def main(args):
    model_aliases = args.model_aliases.split(',')
    per_model_results = {}
    for alias in model_aliases:
        metadata, context_payload, canonical_payload = load_and_verify_model(args.formal_dir, alias)
        print(f"{alias}: loaded and hash-verified. primary_layer={metadata['primary_layer']}")
        per_model_results[alias] = analyze_one_model(alias, metadata, context_payload, canonical_payload)
        print(f"{alias}: analysis complete.")

    rsa = rsa_cross_model(per_model_results)

    # strip the in-memory-only centroid tensors before JSON serialization
    serializable_per_model = {}
    for alias, r in per_model_results.items():
        r2 = dict(r)
        r2.pop('centroids_tinst', None)
        serializable_per_model[alias] = r2

    report = {
        'result_status': 'FORMAL_DIRECTION_ESTIMATION',
        'note': ('Per-model family-specific C directions (primary representation, per pilot protocol '
                 'Sec 15) plus k=2/k=3 exploratory subspace diagnostics (never adopted here) and '
                 'canonical attack-profile projections. Cross-model comparison is RSA-only -- no raw '
                 'coordinate comparison across models. No claim is made about whether/how the C '
                 'construct generalizes across models.'),
        'primary_representation': 'family_specific_4_tuple',
        'primary_token_position': 't_inst',
        'per_model': serializable_per_model,
        'cross_model_rsa': rsa,
        'bootstrap_seeds': {'n_resamples': BOOTSTRAP_RESAMPLES, 'seed': BOOTSTRAP_SEED},
        'split_half_seeds': SPLIT_HALF_SEEDS,
        'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }

    if os.path.exists(args.write_report):
        print(f"\nRefusing to write: {args.write_report} already exists.")
        sys.exit(1)
    os.makedirs(os.path.dirname(os.path.abspath(args.write_report)), exist_ok=True)
    tmp_path = args.write_report + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(report, f, indent=2)
    os.replace(tmp_path, args.write_report)
    print(f"\nWrote formal analysis (new file, non-overwriting): {args.write_report}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--formal_dir', type=str, required=True)
    parser.add_argument('--model_aliases', type=str,
                         default='Qwen2.5-7B-Instruct,Meta-Llama-3.1-8B-Instruct,gemma-2-9b-it')
    parser.add_argument('--write_report', type=str, required=True)
    args = parser.parse_args()
    main(args)
