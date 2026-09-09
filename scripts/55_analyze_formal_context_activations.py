"""
Formal C-Direction Estimation -- analysis of the real per-model paired-
difference/activation tensors produced by
scripts/54_extract_formal_context_activations.py. Implements
EXPERIMENT2_CONTEXT_REPRESENTATION_PROTOCOL.md Sec 4-9 (which supersedes
EXPERIMENT2_FORMAL_C_DIRECTION_PROTOCOL.md Sec 7-9 for the statistics
below; the RSA/k2-exploratory sections inherited from that earlier
document are unchanged in intent, only adapted to cluster-equal-weight
resampling -- see Sec 3 of this docstring).

EVERY output carries result_status = "FORMAL_DIRECTION_ESTIMATION". This
script does NOT decide whether/how a C construct generalizes -- it
reports per-model and cross-model descriptive diagnostics only, per the
frozen family-specific-representation decision
(EXPERIMENT2_CONTEXT_ACTIVATION_PILOT_PROTOCOL.md Sec 15): the primary
output is the 4-tuple of family-specific directions per model; k=2 (or
any k) subspace compression is reported as an EXPLORATORY lens only,
never adopted here.

Reads (never writes to) the real .pt files for all 3 models, PLUS the
existing refusal_dir_v3_en.pt / harmfulness_dir_v2_en.pt direction files
(via utils.direction_metadata.verify_direction_file, which raises on any
hash/shape/dtype mismatch) for the R/H explanatory-power test (Sec 5). No
model, no tokenizer, no GPU needed for this script.

--- 2026-09-09, second revision (this round) -----------------------------

1. PRIMARY vs SENSITIVITY token position, made explicit in the schema:
   canonical-mechanism comparisons (attack profiles, CO/MG prototypes) are
   now PRIMARY at t_inst -- the SAME position as the context directions
   they're compared against. t_post is retained ONLY as an explicitly
   named secondary sensitivity lens
   (`canonical_attack_profiles_tpost_sensitivity`), never averaged into or
   substituted for the primary result. The previous revision's t_inst/
   t_post inconsistency (CO/MG at t_inst, attack profile at t_post) is
   resolved, not just flagged.

2. CO/MG prototype completeness (protocol Sec 4, tightened): a prototype
   requires ALL 3 of its constituent mechanisms to be well-defined
   (non-near-zero tilde_d_m). If even ONE of the 3 is undefined, the
   WHOLE prototype is `UNDEFINED_INCOMPLETE_CATEGORY` -- it is never
   built from the remaining 2. (The previous revision incorrectly
   tolerated 1 undefined mechanism out of 3; that tolerance is removed.)

3. Cluster EQUAL-WEIGHT aggregation, not just "resampled together": every
   place a direction/centroid/prototype is estimated from the 300-
   instruction axis now computes the WITHIN-cluster mean FIRST (collapsing
   each of the 298 normalized-text clusters to one vector), and only THEN
   averages across the 298 cluster-means with equal weight
   (`compute_cluster_means` / `cluster_equal_weight_mean` below). The
   previous revision's fix (duplicate id-pairs always resampled together)
   was necessary but not sufficient: a flat `.mean(dim=0)` over the
   resampled, flattened id list still let a 2-member cluster outweigh a
   singleton cluster 2:1 in that resample's point estimate. This revision
   removes that residual bias from the point estimate itself, not just
   from which ids move together.

4. R/H dual-position design is now asserted, not assumed: R (refusal_dir_v3)
   is verified to have `semantic_position == 't_post'`, H
   (harmfulness_dir_v2) `semantic_position == 't_inst'`
   (`assert_dual_position_design`), and the analysis output explicitly
   records that R, H, and the context-direction primary representation
   are NOT all measured at the same token position.

Cross-model comparison NEVER compares raw direction coordinates across
models (different models have unrelated hidden spaces) -- only RSA-style
comparisons (Gram-matrix correlation, rank consistency).

Usage:
  python scripts/55_analyze_formal_context_activations.py \
      --formal_dir output/context_activations_formal \
      --model_aliases Qwen2.5-7B-Instruct,Meta-Llama-3.1-8B-Instruct,gemma-2-9b-it \
      --write_report output/context_activations_formal/formal_c_direction_analysis.json
"""
import argparse
import datetime
import hashlib
import itertools
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.insert(0, SCRIPT_DIR)

FAMILIES = ['ctx_persona', 'ctx_authority', 'ctx_fictional', 'ctx_continuation']
VARIANTS = ['v1', 'v2', 'v3']
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0
SPLIT_HALF_SEEDS = list(range(1000, 1100))

CO_MECHS = ['prefix_injection', 'refusal_suppression', 'persona_roleplay']
MG_MECHS = ['encoding_obfuscation', 'payload_splitting', 'distractors_negated']

N_RANDOM_SUBSPACES = 1000
RANDOM_NULL_GLOBAL_SEED = 20260908
EXPECTED_N_INSTRUCTION_CLUSTERS = 298  # per the 2026-09-09 572-pool duplicate-text audit


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


# ---------------------------------------------------------------------------
# Instruction clusters (protocol Sec 6-7) + EQUAL-WEIGHT aggregation.
#
# The correct discipline (this round's fix): for ANY estimate computed from
# the 300-instruction axis, ALWAYS collapse each cluster to its own
# within-cluster mean FIRST (compute_cluster_means), and only average
# ACROSS the resulting `n_clusters` rows with equal weight thereafter.
# Resampling (bootstrap / split-half) then operates on those already-
# collapsed cluster-mean rows -- a 2-member duplicate-text cluster and a
# 1-member singleton cluster both contribute exactly one row, so both get
# exactly 1/n_clusters weight, in the point estimate AND in every resample.
# Merely keeping a duplicate pair's ids together in a resample (the
# previous revision's fix) is necessary but NOT sufficient: a flat
# `.mean(dim=0)` over that resample's flattened, unequal-size id groups
# still overweights larger clusters relative to singletons.
# ---------------------------------------------------------------------------

def load_instruction_clusters(instruction_ids):
    from utils.axis_manifest import normalize_text
    sampled_prompts_path = os.path.join(REPO_ROOT, 'data', 'sampled_prompts.json')
    with open(sampled_prompts_path, encoding='utf-8') as f:
        pool = json.load(f)
    by_id = {item['id']: item for item in pool}
    missing = [i for i in instruction_ids if i not in by_id]
    if missing:
        raise ValueError(f"instruction id(s) not found in sampled_prompts.json: {missing}")
    id_to_idx = {iid: i for i, iid in enumerate(instruction_ids)}
    text_to_ids = {}
    for iid in instruction_ids:
        norm = normalize_text(by_id[iid]['instruction_en'])
        text_to_ids.setdefault(norm, []).append(iid)
    clusters_ids = list(text_to_ids.values())
    clusters_idx = [[id_to_idx[i] for i in c] for c in clusters_ids]
    n_clusters = len(clusters_idx)
    if n_clusters != EXPECTED_N_INSTRUCTION_CLUSTERS:
        raise ValueError(
            f"expected exactly {EXPECTED_N_INSTRUCTION_CLUSTERS} instruction clusters from "
            f"{len(instruction_ids)} direction_ids (per the 2026-09-09 572-pool duplicate-text "
            f"audit), got {n_clusters} -- data/sampled_prompts.json or data/splits.json may have "
            f"changed since that audit; refusing to silently proceed with a different cluster count."
        )
    duplicate_groups = [g for g in clusters_ids if len(g) > 1]
    report = {
        'n_raw_ids': len(instruction_ids),
        'n_unique_text_clusters': n_clusters,
        'duplicate_groups': duplicate_groups,
        'aggregation': 'within_cluster_mean_first_then_equal_weight_across_clusters',
    }
    return clusters_idx, report


def compute_cluster_means(deltas, clusters_idx):
    """deltas: [n_instr, hidden]. Returns [n_clusters, hidden]: row c is the
    WITHIN-cluster mean over deltas[clusters_idx[c]], computed BEFORE any
    across-cluster aggregation. Every downstream statistic in this script
    (point estimate, bootstrap, split-half) consumes this array, never the
    raw per-instruction `deltas` directly -- this is what makes duplicate-
    text clusters and singleton clusters carry equal weight throughout."""
    import torch
    return torch.stack([deltas[torch.tensor(idx, dtype=torch.long)].mean(dim=0) for idx in clusters_idx])


def cluster_equal_weight_mean(deltas, clusters_idx):
    return compute_cluster_means(deltas, clusters_idx).mean(dim=0)


def tile_clusters_across_repeats(clusters_idx, n_instr, n_repeats):
    """For a tensor formed by concatenating n_repeats copies of the
    n_instr-instruction axis (e.g. 3 variants stacked), row i of repeat r
    lives at index i + r*n_instr. Since all n_repeats copies share the same
    underlying instruction identity/text, the pooled cluster for
    text-cluster c is the union of its members across ALL repeats -- e.g. a
    duplicate-text pair, pooled across 3 variants, becomes one 6-member
    pooled cluster, not 3 separate 2-member ones (all 6 rows still get
    exactly 1/n_clusters total weight)."""
    return [[idx + r * n_instr for r in range(n_repeats) for idx in c] for c in clusters_idx]


def cluster_bootstrap_from_cluster_means(cluster_means, n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """cluster_means: [n_clusters, hidden], already within-cluster-averaged.
    A standard (non-clustered) bootstrap over these rows now correctly
    implements equal per-cluster weighting throughout, since each row IS a
    cluster (not an instruction)."""
    import torch
    full_dir = cluster_means.mean(dim=0)
    n_clusters = cluster_means.shape[0]
    g = torch.Generator().manual_seed(seed)
    cosines = []
    for _ in range(n_resamples):
        draws = torch.randint(0, n_clusters, (n_clusters,), generator=g)
        resampled_dir = cluster_means[draws].mean(dim=0)
        c = safe_cosine(resampled_dir, full_dir)
        cosines.append(c['value'] if c['status'] == 'OK' else float('nan'))
    sorted_c, _ = torch.sort(torch.tensor(cosines))
    n_s = len(sorted_c)
    return {'median': sorted_c[n_s // 2].item(), 'ci_2_5': sorted_c[int(0.025 * n_s)].item(),
            'ci_97_5': sorted_c[min(int(0.975 * n_s), n_s - 1)].item(),
            'resample_unit': 'instruction_normalized_text_cluster_equal_weight', 'n_clusters': n_clusters}


def bootstrap_direction_cosine(deltas, clusters_idx, n_resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    cm = compute_cluster_means(deltas, clusters_idx)
    return cluster_bootstrap_from_cluster_means(cm, n_resamples, seed)


def repeated_split_half_all_variants(context_payload, label_to_idx, layer, pos_idx, seeds, clusters_idx):
    import torch
    cluster_means_cache = {}
    for fam in FAMILIES:
        for v in VARIANTS:
            idx = label_to_idx[f'{fam}_{v}']
            deltas = context_payload['delta'][:, idx, layer, pos_idx, :].float()
            cluster_means_cache[(fam, v)] = compute_cluster_means(deltas, clusters_idx)
    n_clusters = len(clusters_idx)
    per_seed = {fam: {v: [] for v in VARIANTS} for fam in FAMILIES}
    for seed in seeds:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n_clusters, generator=g)
        half = n_clusters // 2
        idx1, idx2 = perm[:half], perm[half:]
        for fam in FAMILIES:
            for v in VARIANTS:
                cm = cluster_means_cache[(fam, v)]
                d1, d2 = cm[idx1].mean(dim=0), cm[idx2].mean(dim=0)
                c = safe_cosine(d1, d2)
                per_seed[fam][v].append(c['value'] if c['status'] == 'OK' else float('nan'))
    summary = {}
    for fam in FAMILIES:
        summary[fam] = {}
        for v in VARIANTS:
            vals, _ = torch.sort(torch.tensor(per_seed[fam][v]))
            n_s = len(vals)
            summary[fam][v] = {'median': vals[n_s // 2].item(), 'q2_5': vals[int(0.025 * n_s)].item(),
                                'q97_5': vals[min(int(0.975 * n_s), n_s - 1)].item(), 'n_seeds': n_s,
                                'resample_unit': 'instruction_normalized_text_cluster_equal_weight'}
    return summary


# ---------------------------------------------------------------------------
# CO/MG prototype construction (protocol Sec 4, tightened this round) --
# ALL 3 constituent mechanisms must be well-defined, or the whole
# prototype is UNDEFINED_INCOMPLETE_CATEGORY. Never built from 2-of-3.
# ---------------------------------------------------------------------------

def normalize_or_undefined(vec):
    import torch
    v = vec.float()
    eps = torch.finfo(v.dtype).eps
    scale = v.abs().max().item()
    threshold = eps * max(scale, 1.0) * (v.numel() ** 0.5)
    norm = v.norm().item()
    if norm <= threshold:
        return None, {'status': 'UNDEFINED_NEAR_ZERO_NORM', 'norm': norm, 'threshold': threshold}
    return v / norm, {'status': 'OK', 'norm': norm, 'threshold': threshold}


def build_co_mg_prototype(mech_list, canonical_deltas_vs_plain, d_placebo, layer, pos, clusters_idx):
    per_mech = {}
    units = []
    all_defined = True
    for mech in mech_list:
        d_m = cluster_equal_weight_mean(canonical_deltas_vs_plain(mech, layer, pos), clusters_idx)
        tilde = d_m.float() - d_placebo.float()
        u, status = normalize_or_undefined(tilde)
        per_mech[mech] = status
        if u is not None:
            units.append(u)
        else:
            all_defined = False
    if not all_defined:
        # ANY undefined mechanism invalidates the whole prototype -- never
        # fall back to averaging the remaining (up to 2) defined ones.
        return None, per_mech, 'UNDEFINED_INCOMPLETE_CATEGORY'
    mean_vec = sum(units) / len(units)
    proto, proto_status = normalize_or_undefined(mean_vec)
    if proto is None:
        return None, per_mech, 'UNDEFINED_PROTOTYPE_ZERO_NORM_AFTER_AVERAGING'
    return proto, per_mech, 'OK'


# ---------------------------------------------------------------------------
# R/H explanatory power: E_RH(d), q(d), and a random-2D-subspace null with
# Holm correction (protocol Sec 5). R/H DUAL-POSITION DESIGN (this round,
# made explicit and asserted, not assumed): R (refusal_dir_v3) is built at
# t_post; H (harmfulness_dir_v2) is built at t_inst; the context direction
# d compared against span(R,H) is measured at t_inst (primary). These
# three are NOT all extracted at the same token position -- callers must
# not describe them as such.
# ---------------------------------------------------------------------------

def assert_dual_position_design(r_meta, h_meta, model_alias):
    r_pos = r_meta.get('semantic_position')
    h_pos = h_meta.get('semantic_position')
    if r_pos != 't_post':
        raise ValueError(
            f"{model_alias}: refusal_dir_v3 semantic_position={r_pos!r}, expected 't_post' -- the "
            f"dual-position design (R built at t_post, H built at t_inst) assumption no longer "
            f"holds for this file; refusing to proceed without review.")
    if h_pos != 't_inst':
        raise ValueError(
            f"{model_alias}: harmfulness_dir_v2 semantic_position={h_pos!r}, expected 't_inst' -- "
            f"the dual-position design (R built at t_post, H built at t_inst) assumption no longer "
            f"holds for this file; refusing to proceed without review.")


def load_r_h_vectors(model_alias, primary_layer):
    from utils.direction_metadata import verify_direction_file
    r_path = os.path.join(REPO_ROOT, 'output', 'output_v3_behavioral_refusal',
                           model_alias, 'refusal_dir_v3_en.pt')
    h_path = os.path.join(REPO_ROOT, 'output', 'output_v2_dual_position',
                           model_alias, 'harmfulness_dir_v2_en.pt')
    r_tensor, r_meta = verify_direction_file(r_path)
    h_tensor, h_meta = verify_direction_file(h_path)
    assert_dual_position_design(r_meta, h_meta, model_alias)
    if primary_layer >= r_tensor.shape[0] or primary_layer >= h_tensor.shape[0]:
        raise ValueError(f"{model_alias}: primary_layer={primary_layer} out of range for R shape "
                          f"{list(r_tensor.shape)} / H shape {list(h_tensor.shape)}")
    # R/H tensor index i is taken to correspond 1:1 to hidden_states index i
    # (0 = raw embeddings, i = output of transformer block i) -- evidenced
    # by refusal_dir_v3's val_cohens_d_per_layer[0] == 0.0 (no signal at the
    # untransformed embedding layer) and tensor_shape[0] == n_layers (not
    # n_layers+1), i.e. the tensor simply omits the final block's entry
    # rather than being offset by one. primary_layer < n_layers for all 3
    # models (16<28, 19<32, 25<42), so this indexing needs no adjustment.
    return r_tensor[primary_layer].float(), h_tensor[primary_layer].float(), r_meta, h_meta


def deterministic_subseed(model_alias, layer_index, global_seed):
    s = f"{model_alias}|{layer_index}|{global_seed}"
    h = hashlib.sha256(s.encode('utf-8')).hexdigest()
    return int(h[:15], 16) % (2 ** 31 - 1)


def generate_random_subspace_bases(hidden_dim, model_alias, layer_index,
                                    n_random=N_RANDOM_SUBSPACES, global_seed=RANDOM_NULL_GLOBAL_SEED):
    import torch
    seed = deterministic_subseed(model_alias, layer_index, global_seed)
    g = torch.Generator().manual_seed(seed)
    bases = []
    for _ in range(n_random):
        v = torch.randn(hidden_dim, 2, generator=g)
        q, _ = torch.linalg.qr(v)
        bases.append(q)
    return bases, seed


def null_E_distribution(d, bases):
    d = d.float()
    dn2 = (d @ d).item()
    if dn2 == 0.0:
        return None
    return [(q.T @ d).pow(2).sum().item() / dn2 for q in bases]


def compute_E_RH_and_q(d, r_vec, h_vec):
    import torch
    d = d.float()
    dn2 = (d @ d).item()
    if dn2 == 0.0:
        return None
    basis_raw = torch.stack([r_vec.float(), h_vec.float()], dim=1)
    Q, _ = torch.linalg.qr(basis_raw)
    proj = Q.T @ d
    e_rh = (proj @ proj).item() / dn2
    q_val = max(0.0, 1.0 - e_rh) ** 0.5
    return e_rh, q_val


def rh_significance_test(d, r_vec, h_vec, bases, hidden_dim):
    obs = compute_E_RH_and_q(d, r_vec, h_vec)
    if obs is None:
        return {'status': 'UNDEFINED_ZERO_NORM_D'}
    e_rh, q_val = obs
    null_vals = null_E_distribution(d, bases)
    if null_vals is None:
        return {'status': 'UNDEFINED_ZERO_NORM_D'}
    sorted_null = sorted(null_vals)
    n = len(sorted_null)
    null_median = sorted_null[n // 2]
    null_95 = sorted_null[min(int(0.95 * n), n - 1)]
    observed_percentile = sum(1 for v in sorted_null if v <= e_rh) / n
    unadjusted_p = sum(1 for v in sorted_null if v >= e_rh) / n
    return {
        'status': 'OK',
        'E_RH': e_rh, 'q': q_val,
        'null_median': null_median, 'null_95th_percentile': null_95,
        'observed_percentile_in_null': observed_percentile,
        'unadjusted_p_one_sided': unadjusted_p,
        'significant_at_0_05_one_sided': e_rh > null_95,
        'theoretical_E_random_expectation': 2.0 / hidden_dim,
        'empirical_null_mean': sum(sorted_null) / n,
        'n_random_subspaces': n,
    }


def holm_correction(named_pvalues):
    m = len(named_pvalues)
    if m == 0:
        return {}
    order = sorted(range(m), key=lambda i: named_pvalues[i][1])
    adjusted = [None] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        p = named_pvalues[idx][1]
        adj = min((m - rank) * p, 1.0)
        running_max = max(running_max, adj)
        adjusted[idx] = running_max
    return {named_pvalues[i][0]: adjusted[i] for i in range(m)}


# ---------------------------------------------------------------------------
# Canonical attack-profile projection -- PRIMARY at t_inst (this round),
# t_post kept ONLY as an explicitly named secondary sensitivity lens.
# ---------------------------------------------------------------------------

def compute_attack_profiles(canonical_deltas_vs_plain, canonical_names, centroids, layer, pos, clusters_idx):
    d_placebo = cluster_equal_weight_mean(canonical_deltas_vs_plain('placebo', layer, pos), clusters_idx)
    profiles = {}
    for mech in canonical_names:
        d_m = cluster_equal_weight_mean(canonical_deltas_vs_plain(mech, layer, pos), clusters_idx)
        tilde_d = d_m - d_placebo
        profile = {}
        for fam in FAMILIES:
            c = safe_cosine(tilde_d, centroids[fam])
            raw_projection = (tilde_d @ centroids[fam]).item()
            normalized_projection = (raw_projection / centroids[fam].norm().item()
                                      if centroids[fam].norm().item() > 0 else None)
            profile[fam] = {'cosine': c, 'raw_signed_projection': raw_projection,
                             'normalized_signed_projection': normalized_projection}
        profiles[mech] = profile
    return profiles


def analyze_one_model(model_alias, metadata, context_payload, canonical_payload):
    import torch

    layer = metadata['primary_layer']
    TINST, TPOST = 0, 1
    label_to_idx = {label: i for i, label in enumerate(context_payload['condition_labels_all_16'])}
    canon_to_idx = {name: i for i, name in enumerate(canonical_payload['condition_names'])}

    clusters_idx, cluster_report = load_instruction_clusters(metadata['instruction_ids'])
    n_instr = context_payload['delta'].shape[0]

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
            deltas_tinst = context_deltas(fam, v, layer, TINST)
            deltas_tpost = context_deltas(fam, v, layer, TPOST)
            d_tinst = cluster_equal_weight_mean(deltas_tinst, clusters_idx)
            d_tpost = cluster_equal_weight_mean(deltas_tpost, clusters_idx)
            directions[(fam, v)] = d_tinst
            per_variant[fam][v] = {
                'direction_norm': d_tinst.norm().item(),
                'bootstrap_direction_cosine': bootstrap_direction_cosine(deltas_tinst, clusters_idx),
                'signed_paired_projection_mean': d_tinst.norm().item(),
                't_inst_vs_t_post_cosine': safe_cosine(d_tinst, d_tpost),
            }
    repeated_split_half = repeated_split_half_all_variants(
        context_payload, label_to_idx, layer, TINST, SPLIT_HALF_SEEDS, clusters_idx)
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
        all_deltas = torch.cat([context_deltas(fam, v, layer, TINST) for v in VARIANTS], dim=0)
        pooled_clusters_idx = tile_clusters_across_repeats(clusters_idx, n_instr, len(VARIANTS))
        family_structure[fam] = {
            'centroid_norm': centroid.norm().item(),
            'within_family_pairwise_cosine': pairwise,
            'leave_one_variant_out_cosine': loo,
            'centroid_norm_bootstrap_ci': bootstrap_direction_cosine(all_deltas, pooled_clusters_idx),
        }

    # ---- C: cross-family + k=1/2/3 exploratory (structure unchanged;
    # resampling now cluster-equal-weight) ----
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

    k2_basis_full = Vt[:2] if Vt.shape[0] >= 2 else None

    def k2_subspace_from_centroids(cents):
        m = torch.stack(cents)
        m = m - m.mean(dim=0, keepdim=True)
        _, _, vt = torch.linalg.svd(m, full_matrices=False)
        return vt[:2] if vt.shape[0] >= 2 else vt

    def principal_angles(basis1, basis2):
        M = basis1 @ basis2.T
        sv = torch.linalg.svdvals(M)
        sv = torch.clamp(sv, -1.0, 1.0)
        return torch.acos(sv).tolist()

    k2_cluster_means = {}
    for fam in FAMILIES:
        for v in VARIANTS:
            k2_cluster_means[(fam, v)] = compute_cluster_means(context_deltas(fam, v, layer, TINST), clusters_idx)
    n_clusters = len(clusters_idx)

    g = torch.Generator().manual_seed(BOOTSTRAP_SEED)
    boot_angles = []
    for _ in range(min(BOOTSTRAP_RESAMPLES, 500)):
        draws = torch.randint(0, n_clusters, (n_clusters,), generator=g)
        resampled_centroids = []
        for fam in FAMILIES:
            v_dirs_r = [k2_cluster_means[(fam, v)][draws].mean(dim=0) for v in VARIANTS]
            resampled_centroids.append(sum(v_dirs_r) / 3.0)
        basis_r = k2_subspace_from_centroids(resampled_centroids)
        if k2_basis_full is not None and basis_r.shape[0] == 2:
            angles = principal_angles(k2_basis_full, basis_r)
            boot_angles.append(max(angles))
    boot_angles_t = torch.tensor(boot_angles) if boot_angles else torch.tensor([float('nan')])
    sorted_ba, _ = torch.sort(boot_angles_t)
    n_ba = len(sorted_ba)
    k2_bootstrap_stability = {
        'n_resamples': len(boot_angles), 'median_max_principal_angle_rad': sorted_ba[n_ba // 2].item(),
        'ci_2_5_rad': sorted_ba[int(0.025 * n_ba)].item(),
        'ci_97_5_rad': sorted_ba[min(int(0.975 * n_ba), n_ba - 1)].item(),
        'resample_unit': 'instruction_normalized_text_cluster_equal_weight',
    }

    split_angles = []
    for seed in SPLIT_HALF_SEEDS:
        gg = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n_clusters, generator=gg)
        half = n_clusters // 2
        idx1, idx2 = perm[:half], perm[half:]
        c1, c2 = [], []
        for fam in FAMILIES:
            v1 = [k2_cluster_means[(fam, v)][idx1].mean(dim=0) for v in VARIANTS]
            v2 = [k2_cluster_means[(fam, v)][idx2].mean(dim=0) for v in VARIANTS]
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
        'resample_unit': 'instruction_normalized_text_cluster_equal_weight',
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

    # ---- D: canonical attack-profile projection -- PRIMARY at t_inst,
    # t_post kept only as an explicitly named secondary sensitivity lens.
    canonical_names = [n for n in canonical_payload['condition_names'] if n not in ('plain', 'placebo')]
    attack_profiles_tinst_primary = compute_attack_profiles(
        canonical_deltas_vs_plain, canonical_names, centroids, layer, TINST, clusters_idx)
    attack_profiles_tpost_sensitivity = compute_attack_profiles(
        canonical_deltas_vs_plain, canonical_names, centroids, layer, TPOST, clusters_idx)

    # ---- E: CO/MG prototypes (Sec 4) -- t_inst, matching the context
    # directions' and the attack-profile primary's position ----
    d_placebo_tinst = cluster_equal_weight_mean(canonical_deltas_vs_plain('placebo', layer, TINST), clusters_idx)
    p_CO, co_per_mech, co_status = build_co_mg_prototype(
        CO_MECHS, canonical_deltas_vs_plain, d_placebo_tinst, layer, TINST, clusters_idx)
    p_MG, mg_per_mech, mg_status = build_co_mg_prototype(
        MG_MECHS, canonical_deltas_vs_plain, d_placebo_tinst, layer, TINST, clusters_idx)
    co_mg_prototypes = {
        'p_CO': {'status': co_status, 'per_mechanism': co_per_mech},
        'p_MG': {'status': mg_status, 'per_mechanism': mg_per_mech},
        'note': ('Requires ALL 3 constituent mechanisms well-defined; any single undefined mechanism '
                 'yields UNDEFINED_INCOMPLETE_CATEGORY for the whole prototype (never built from the '
                 'remaining 2). Individual mechanism comparisons (canonical_attack_profiles_tinst_primary '
                 'above) are always reported alongside this prototype and never replaced by it. '
                 'Computed at t_inst (primary), consistent with the context directions.'),
    }

    # ---- F: R/H explanatory power (Sec 5) ----
    hidden_dim = context_payload['delta'].shape[-1]
    try:
        r_vec, h_vec, r_meta, h_meta = load_r_h_vectors(model_alias, layer)
        rh_bases, rh_seed = generate_random_subspace_bases(hidden_dim, model_alias, layer)
        rh_status, rh_error = 'OK', None
    except Exception as e:
        r_vec = h_vec = rh_bases = rh_seed = None
        rh_status, rh_error = 'RH_VERIFICATION_OR_LOAD_FAILED', f"{type(e).__name__}: {e}"

    rh_per_variant = {}
    pvalue_list = []
    if rh_status == 'OK':
        for fam in FAMILIES:
            rh_per_variant[fam] = {}
            for v in VARIANTS:
                test = rh_significance_test(directions[(fam, v)], r_vec, h_vec, rh_bases, hidden_dim)
                rh_per_variant[fam][v] = test
                if test.get('status') == 'OK':
                    pvalue_list.append((f'{fam}|{v}', test['unadjusted_p_one_sided']))
        holm_adjusted = holm_correction(pvalue_list)
        for key, p_adj in holm_adjusted.items():
            fam, v = key.split('|')
            rh_per_variant[fam][v]['holm_adjusted_p'] = p_adj
            rh_per_variant[fam][v]['significant_after_holm_0_05'] = p_adj < 0.05

    rh_explanatory_power = {
        'status': rh_status, 'error': rh_error,
        'position_design': {
            'R_semantic_position': 't_post', 'H_semantic_position': 't_inst',
            'context_C_primary_position': 't_inst',
            'note': ('R, H, and the context direction C are NOT extracted at the same token position. '
                     'R (refusal_dir_v3) is measured at t_post; H (harmfulness_dir_v2) at t_inst; C\'s '
                     'primary representation at t_inst. This is the pre-existing dual-position R/H '
                     'design (asserted via assert_dual_position_design against each file\'s own '
                     'semantic_position metadata field, not assumed) -- E_RH/q below describe the '
                     'layer-l residual-stream subspace spanned by these two position-heterogeneous '
                     'vectors, not a claim that all three share one measurement position.'),
        },
        'random_null_seed_derivation': 'sha256(model_alias|layer_index|20260908) per (model,layer)',
        'random_null_seed_used': rh_seed,
        'n_random_subspaces': N_RANDOM_SUBSPACES,
        'per_variant_primary_layer': rh_per_variant,
        'holm_correction_scope': 'the 12 (family,variant) tests at the primary layer for this model',
        'note': ('E_RH/q computed for context directions at t_inst (primary position). All-layer '
                 'sweeps are NOT computed here (exploratory-only if ever added; out of scope for this '
                 'confirmatory primary-layer test).'),
    }

    # ---- G: non-reducibility (Sec 8-C): family centroid vs p_CO, p_MG,
    # and span(R,H) ----
    non_reducibility = {}
    for fam in FAMILIES:
        centroid = centroids[fam]
        entry = {
            'vs_p_CO': safe_cosine(centroid, p_CO) if p_CO is not None else {'status': co_status},
            'vs_p_MG': safe_cosine(centroid, p_MG) if p_MG is not None else {'status': mg_status},
        }
        if rh_status == 'OK':
            entry['vs_span_R_H'] = rh_significance_test(centroid, r_vec, h_vec, rh_bases, hidden_dim)
        else:
            entry['vs_span_R_H'] = {'status': rh_status}
        non_reducibility[fam] = entry

    return {
        'model_alias': model_alias,
        'primary_layer': layer,
        'instruction_cluster_summary': cluster_report,
        'per_variant_tinst': per_variant,
        'family_structure_tinst': family_structure,
        'candidate_context_structure_tinst': candidate_structure,
        'canonical_attack_profiles_tinst_primary': attack_profiles_tinst_primary,
        'canonical_attack_profiles_tpost_sensitivity': attack_profiles_tpost_sensitivity,
        'co_mg_prototypes_tinst': co_mg_prototypes,
        'rh_explanatory_power_tinst': rh_explanatory_power,
        'non_reducibility_c_layer_tinst': non_reducibility,
        'centroids_tinst': {fam: centroids[fam] for fam in FAMILIES},  # kept in-memory for RSA; not JSON-serialized directly
    }


def rsa_cross_model(per_model_results):
    """RSA-style cross-model comparison. NEVER compares raw direction
    coordinates -- only (a) correlation between models' 4x4 family Gram
    matrices (flattened off-diagonal), (b) rank consistency of
    canonical-mechanism context-profiles (PRIMARY, t_inst)."""
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
        all_mechs |= set(per_model_results[alias]['canonical_attack_profiles_tinst_primary'].keys())
    for mech in sorted(all_mechs):
        per_model_ranks = {}
        for alias in aliases:
            profile = per_model_results[alias]['canonical_attack_profiles_tinst_primary'].get(mech)
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
                'canonical mechanism (t_inst primary) is stable across models. Any pattern holding for '
                'only 1-2 of 3 models is model-specific, not generalized.',
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

    serializable_per_model = {}
    for alias, r in per_model_results.items():
        r2 = dict(r)
        r2.pop('centroids_tinst', None)
        serializable_per_model[alias] = r2

    report = {
        'result_status': 'FORMAL_DIRECTION_ESTIMATION',
        'note': ('Per-model family-specific C directions (primary representation, per pilot protocol '
                 'Sec 15) plus k=2/k=3 exploratory subspace diagnostics (never adopted here), canonical '
                 'attack-profile projections (t_inst primary / t_post sensitivity), CO/MG prototype '
                 'comparisons (t_inst, requires all 3 constituent mechanisms), and R/H explanatory-power '
                 'tests (EXPERIMENT2_CONTEXT_REPRESENTATION_PROTOCOL.md Sec 4-8; R@t_post, H@t_inst, '
                 'C@t_inst -- not a shared position). All instruction-axis aggregation uses within-'
                 'cluster-mean-then-equal-weight (298 normalized-text clusters), not a flat mean over '
                 '300 raw ids. Cross-model comparison is RSA-only. No claim is made about whether/how '
                 'the C construct generalizes across models.'),
        'primary_representation': 'family_specific_4_tuple',
        'primary_token_position': 't_inst',
        'instruction_cluster_aggregation': 'within_cluster_mean_first_then_equal_weight_across_clusters',
        'per_model': serializable_per_model,
        'cross_model_rsa': rsa,
        'bootstrap_seeds': {'n_resamples': BOOTSTRAP_RESAMPLES, 'seed': BOOTSTRAP_SEED},
        'split_half_seeds': SPLIT_HALF_SEEDS,
        'random_subspace_null': {'n_random_subspaces': N_RANDOM_SUBSPACES,
                                  'global_seed': RANDOM_NULL_GLOBAL_SEED},
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
