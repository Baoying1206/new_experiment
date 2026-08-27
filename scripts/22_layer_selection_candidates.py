"""
Decision 2 (layer-selection leakage fix): replaces 14_find_safety_layer.py's
un-partitioned, pre-splits.json layer choice (see
output/audits/layer_selection_leakage.md) with a validation-only procedure.

Per the user's explicit instruction, this does NOT pick a single final layer.
It computes several candidate selection rules on validation_ids (72 English
instructions, never direction_ids/test_ids) and reports what each rule would
pick, plus the effect at that layer, side by side -- the choice of which
rule to lock in for Exp1-3's primary-analysis layer is left to the user.

Requires paired_diffs_{lang}{suffix}_validation_ids.pt from
18_extract_paired_diffs.py --ids_key validation_ids (CPU-only, no
model/GPU needed here).

Candidate rules implemented (Decision 2's prioritized criteria, computed
independently -- not combined into one score):

  1. split_half_reliability(L): for each of the 6 mechanisms, repeatedly
     split the 72 validation instructions in half, compute cosine between
     the two half-sample directions at layer L, average over mechanisms and
     reps. High = the mechanism directions themselves are reliably
     estimated at this layer (necessary for anything downstream to be
     trustworthy, not sufficient on its own to justify a "safety-relevant"
     claim).
  2. template_placebo_separation(L): for each mechanism, 1 - cosine(mean
     mechanism direction at L, mean placebo direction at L), averaged over
     mechanisms. High = at this layer, templates produce a direction
     distinguishable from generic template-wrapping noise (placebo).
  3. combined_rank(L): mechanisms where both (1) and (2) rank highly --
     reports the layer minimizing the sum of (1)'s and (2)'s ranks, as one
     natural way to combine them, but reported as one candidate among
     several, not the final answer.
  4. relative_layer_0_6 = floor(0.6 * n_layers): fixed, pre-registered,
     data-independent sensitivity check (mirrors Arditi et al.'s style of
     relative-depth heuristic), reported regardless of what (1)-(3) say.

Rules 1-3's argmax search is restricted to layers [floor(0.2*n_layers),
floor(0.8*n_layers)) -- the earliest layers trivially win both criteria
(near-ceiling split-half reliability and template-placebo separation),
because templated vs plain/placebo text differs lexically at the raw
token-embedding level before any deep computation happens; that's a
meaningless artifact, not a safety-relevant signal. This mirrors
14_find_safety_layer.py's existing exclusion of the last ~20% of layers
(residual-stream norm growth), applied symmetrically to the first ~20% too.
Full per-layer values (including the excluded range) are still saved to
the output JSON for auditability.

NOT computed here: "reference-direction reliability" (split-half
reliability of refusal_direction/harmfulness_direction themselves) -- this
needs the dual-position (t_inst/t_post) direction rebuild from Decision 1,
which is blocked on the token-position audit
(scripts/audits/audit_token_positions.py) and hasn't run yet. Add this
criterion once those directions exist.

Usage:
  python scripts/22_layer_selection_candidates.py \
      --output_dir output --model_alias Qwen2.5-7B-Instruct \
      --lang en --suffix _full572 --n_splithalf 200
"""
import argparse
import json
import os
import random

import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(__file__)
REAL_MECHS = ['prefix_injection', 'refusal_suppression', 'instruction_hierarchy',
              'persona_roleplay', 'fictional_framing', 'encoding_obfuscation']


def build_id_index(diffs_data, mechs):
    return {m: {pid: i for i, pid in enumerate(diffs_data['instruction_ids'][m])} for m in mechs}


def common_ids_for(diffs_data, mechs):
    id_lists = [set(diffs_data['instruction_ids'][m]) for m in mechs]
    return sorted(set.intersection(*id_lists))


def mean_vec(diffs_data, id_index, mech, ids):
    idxs = [id_index[mech][pid] for pid in ids]
    return diffs_data['diffs'][mech][idxs].mean(0)  # [n_layers, d]


def split_half_reliability(diffs_data, id_index, mechs, ids, n_layers, n_reps, rng):
    """Returns tensor[n_layers] -- mean over mechs and reps of split-half cosine."""
    acc = torch.zeros(n_layers)
    for m in mechs:
        cos_reps = torch.zeros(n_reps, n_layers)
        for r in range(n_reps):
            shuffled = ids.copy()
            rng.shuffle(shuffled)
            half = len(shuffled) // 2
            idxA = [id_index[m][pid] for pid in shuffled[:half]]
            idxB = [id_index[m][pid] for pid in shuffled[half:]]
            vecA = diffs_data['diffs'][m][idxA].mean(0)
            vecB = diffs_data['diffs'][m][idxB].mean(0)
            cos_reps[r] = F.cosine_similarity(vecA, vecB, dim=-1)
        acc += cos_reps.mean(0)
    return acc / len(mechs)


def template_placebo_separation(diffs_data, id_index, mechs, ids, n_layers):
    """Returns tensor[n_layers] -- mean over mechs of 1 - cos(mech_dir, placebo_dir)."""
    placebo_vec = mean_vec(diffs_data, id_index, 'placebo', ids)  # [n_layers, d]
    acc = torch.zeros(n_layers)
    for m in mechs:
        mech_vec = mean_vec(diffs_data, id_index, m, ids)
        acc += 1 - F.cosine_similarity(mech_vec, placebo_vec, dim=-1)
    return acc / len(mechs)


def main(args):
    out_dir = os.path.join(args.output_dir, args.model_alias)
    diffs_path = os.path.join(out_dir, f'paired_diffs_{args.lang}{args.suffix}_validation_ids.pt')
    diffs_data = torch.load(diffs_path, map_location='cpu')
    diffs_data['diffs'] = {m: v.float() for m, v in diffs_data['diffs'].items()}
    n_layers = diffs_data['n_layers']
    ids_key = diffs_data.get('ids_key')
    assert ids_key == 'validation_ids', (
        f"expected a validation_ids-built paired_diffs file, got ids_key={ids_key!r} -- "
        f"refusing to select a layer using anything other than the validation split."
    )
    print(f"Loaded {diffs_path}  n_layers={n_layers}  (validation_ids, n={len(diffs_data['instruction_ids']['placebo'])})")

    id_index = build_id_index(diffs_data, REAL_MECHS + ['placebo'])
    common_ids = common_ids_for(diffs_data, REAL_MECHS + ['placebo'])
    print(f"common_ids = {len(common_ids)}\n")

    rng = random.Random(0)
    sh_rel = split_half_reliability(diffs_data, id_index, REAL_MECHS, common_ids,
                                     n_layers, args.n_splithalf, rng)
    tp_sep = template_placebo_separation(diffs_data, id_index, REAL_MECHS, common_ids, n_layers)

    # Restrict argmax search to a pre-registered middle-layer window, excluding
    # both ends -- the earliest layers trivially win both criteria (split-half
    # "reliability" and template-vs-placebo "separation" are both near-ceiling
    # there simply because templated vs plain/placebo text differs lexically at
    # the token-embedding level, before any deep computation; this is a
    # meaningless artifact, not a safety-relevant signal), mirroring
    # 14_find_safety_layer.py's existing exclusion of the last ~20% of layers
    # (residual-stream norm growth) but applied symmetrically to the first
    # ~20% too. Full per-layer arrays are still saved below so this window is
    # auditable, not hidden.
    lo = int(0.2 * n_layers)
    hi = int(0.8 * n_layers)
    print(f"Restricting candidate search to layers [{lo}, {hi}) -- excludes the "
          f"trivially-high-reliability earliest layers and the norm-growth-dominated "
          f"latest layers; full per-layer values are still saved to the output JSON.\n")

    sh_rank = sh_rel.argsort(descending=True).argsort()  # rank 0 = best (full range, for reference)
    tp_rank = tp_sep.argsort(descending=True).argsort()
    usable_rank_sum = sh_rank[lo:hi] + tp_rank[lo:hi]
    combined_best_layer = lo + int(usable_rank_sum.argmin().item())

    sh_best_layer = lo + int(sh_rel[lo:hi].argmax().item())
    tp_best_layer = lo + int(tp_sep[lo:hi].argmax().item())
    relative_0_6_layer = int(0.6 * n_layers)

    print("=== Candidate layer-selection rules (validation_ids only) ===")
    print(f"  1. split_half_reliability argmax:      layer {sh_best_layer}  "
          f"(value={sh_rel[sh_best_layer]:.4f}, range=[{sh_rel.min():.4f},{sh_rel.max():.4f}])")
    print(f"  2. template_placebo_separation argmax:  layer {tp_best_layer}  "
          f"(value={tp_sep[tp_best_layer]:.4f}, range=[{tp_sep.min():.4f},{tp_sep.max():.4f}])")
    print(f"  3. combined (min rank-sum of 1+2):      layer {combined_best_layer}  "
          f"(rank_sum={int(usable_rank_sum[combined_best_layer - lo])})")
    print(f"  4. relative_layer floor(0.6*n_layers):  layer {relative_0_6_layer}  "
          f"(fixed, pre-registered, data-independent -- always reported regardless of 1-3)")
    print(f"\n  NOT computed: reference-direction reliability (needs Decision 1's dual-position "
          f"refusal_direction/harmfulness_direction rebuild, not done yet)")
    print(f"\n  This script does NOT pick a final layer -- report these 4 candidates and their "
          f"agreement/disagreement, then decide.")

    results = {
        'model': args.model_alias, 'lang': args.lang, 'suffix': args.suffix,
        'n_layers': n_layers, 'n_validation_ids': len(common_ids),
        'search_window': {'lo': lo, 'hi': hi, 'note': 'argmax restricted to [lo,hi); '
                           'full per-layer arrays below include excluded layers too'},
        'split_half_reliability_per_layer': sh_rel.tolist(),
        'template_placebo_separation_per_layer': tp_sep.tolist(),
        'candidates': {
            'split_half_reliability_argmax': sh_best_layer,
            'template_placebo_separation_argmax': tp_best_layer,
            'combined_rank_sum_argmin': combined_best_layer,
            'relative_layer_0_6': relative_0_6_layer,
        },
        'candidates_agree': len({sh_best_layer, tp_best_layer, combined_best_layer, relative_0_6_layer}) == 1,
        'note': 'reference_direction_reliability not computed -- pending Decision 1 dual-position rebuild',
    }
    out_path = os.path.join(out_dir, f'layer_selection_candidates_{args.lang}{args.suffix}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',   type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    parser.add_argument('--model_alias',  type=str, required=True)
    parser.add_argument('--lang',         type=str, default='en')
    parser.add_argument('--suffix',       type=str, default='_full572')
    parser.add_argument('--n_splithalf',  type=int, default=200)
    args = parser.parse_args()
    main(args)
