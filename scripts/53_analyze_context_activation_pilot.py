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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot_dir', type=str, required=True)
    parser.add_argument('--metadata_path', type=str, default=None)
    parser.add_argument('--write_report', type=str, required=True)
    args = parser.parse_args()
    if args.metadata_path is None:
        args.metadata_path = os.path.join(args.pilot_dir, 'context_activation_pilot_metadata.json')
    main(args)
