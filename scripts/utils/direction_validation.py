"""
Shared held-out validation metrics for any two-class direction (refusal_direction
v3's refused-vs-accepted, harmfulness_direction's harmful-vs-harmless, etc.).
Extracted 2026-09-05 from scripts/26_rebuild_refusal_direction_behavioral.py so
scripts/23_extract_reference_directions.py can report the same four lines of
evidence for harmfulness_direction that 26 already reports for refusal_direction_v3
-- per EXPERIMENT2_RH_REBUILD_PROTOCOL.md Sec 2.1/9 point 3: no single metric is a
pass/fail gate, all four are reported together for human review.

Pure torch, no model/GPU dependency -- fully unit-testable with synthetic tensors.
"""
import torch
import torch.nn.functional as F


def cohens_d_per_layer(proj, mask):
    """proj: [n_rows, n_layers]. mask: bool [n_rows], True=positive class.
    DESCRIPTIVE effect size only -- never a standalone pass/fail gate."""
    n_layers = proj.shape[1]
    d = torch.zeros(n_layers)
    for l in range(n_layers):
        p_pos, p_neg = proj[mask, l], proj[~mask, l]
        pooled_std = torch.sqrt((p_pos.var(unbiased=True) + p_neg.var(unbiased=True)) / 2 + 1e-8)
        d[l] = (p_pos.mean() - p_neg.mean()) / pooled_std
    return d


def auc_per_layer(proj, mask):
    """Rank-based (Mann-Whitney U) AUC per layer -- 0.5=no separation, 1.0=
    perfect separation on the validation set. Threshold-free, unlike accuracy
    at a chosen cutoff."""
    n_layers = proj.shape[1]
    n_pos, n_neg = int(mask.sum()), int((~mask).sum())
    auc = torch.full((n_layers,), float('nan'))
    if n_pos == 0 or n_neg == 0:
        return auc
    for l in range(n_layers):
        ranks = proj[:, l].argsort().argsort().float() + 1  # 1-indexed
        rank_sum_pos = ranks[mask].sum()
        u = rank_sum_pos - n_pos * (n_pos + 1) / 2
        auc[l] = u / (n_pos * n_neg)
    return auc


def split_half_reliability(construction_acts, class_mask, seed):
    """Randomly splits the construction rows into two class-stratified
    halves, independently builds a direction (pos_mean - neg_mean) from
    each half, and returns cos(d_half1, d_half2) per layer -- high cosine
    means the direction construction is stable and not an artifact of one
    particular sample. Returns None if either class has <2 examples
    (cannot split -- reported as unavailable, never fabricated)."""
    pos_idx = [i for i, m in enumerate(class_mask.tolist()) if m]
    neg_idx = [i for i, m in enumerate(class_mask.tolist()) if not m]
    if len(pos_idx) < 2 or len(neg_idx) < 2:
        return None
    import random
    rng = random.Random(seed)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    half1 = sorted(pos_idx[:len(pos_idx) // 2] + neg_idx[:len(neg_idx) // 2])
    half2 = sorted(set(range(len(class_mask))) - set(half1))

    def _direction(idx):
        idx_t = torch.tensor(idx)
        sub_mask = class_mask[idx_t]
        sub_acts = construction_acts[idx_t]
        return sub_acts[sub_mask].mean(0) - sub_acts[~sub_mask].mean(0)  # [n_layers, d_model]

    d1, d2 = _direction(half1), _direction(half2)
    return F.cosine_similarity(d1, d2, dim=-1)  # [n_layers]


def bootstrap_cohens_d_ci(proj, mask, n_boot, seed, alpha=0.05):
    """Stratified bootstrap (resample the positive/negative validation rows
    separately, same counts each draw) using the FIXED, already-built
    direction -- percentile CI for Cohen's d. Does not rebuild the direction
    per draw (that's what split_half_reliability is for); this quantifies
    sampling uncertainty in the val-set effect-size estimate itself. Returns
    (None, None) if either class has 0 rows."""
    n_pos, n_neg = int(mask.sum()), int((~mask).sum())
    if n_pos == 0 or n_neg == 0:
        return None, None
    pos_idx = mask.nonzero(as_tuple=True)[0]
    neg_idx = (~mask).nonzero(as_tuple=True)[0]
    g = torch.Generator().manual_seed(seed)
    n_layers = proj.shape[1]
    boots = torch.zeros(n_boot, n_layers)
    for b in range(n_boot):
        p = pos_idx[torch.randint(0, n_pos, (n_pos,), generator=g)]
        n = neg_idx[torch.randint(0, n_neg, (n_neg,), generator=g)]
        boot_mask = torch.cat([torch.ones(n_pos, dtype=torch.bool), torch.zeros(n_neg, dtype=torch.bool)])
        boot_proj = torch.cat([proj[p], proj[n]], dim=0)
        boots[b] = cohens_d_per_layer(boot_proj, boot_mask)
    lo = boots.quantile(alpha / 2, dim=0)
    hi = boots.quantile(1 - alpha / 2, dim=0)
    return lo, hi
