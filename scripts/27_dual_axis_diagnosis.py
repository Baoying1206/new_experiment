"""
Experiment 2's actual diagnosis: given Exp1's finding that Wei et al.'s
binary taxonomy does not correspond to separable geometric clusters, tests
whether the 6 templates instead show continuous, structured variation along
the delta_R (refusal-axis)/delta_H (harmfulness-axis) plane -- and whether a
finer-grained descriptor (template_identity, or model identity/interaction)
explains more of that variation than Wei's coarse binary label does.

CPU-only -- reads delta_r_h_{lang}{suffix}_{ids_key}.pt (from
25_extract_delta_r_h.py) for each model, no GPU needed.

Three things reported, per Decision requirements:

  1. Descriptive: per-mechanism mean/std of delta_R and delta_H (raw and
     placebo-calibrated), per model -- literally where each template lands
     on the delta_R-delta_H plane. This is the "continuous, mixed variation"
     check itself.

  2. Variance decomposition (one-way R^2 = SS_between/SS_total, exact, no
     model-fitting/convergence issues): for delta_R and delta_H separately,
     at EVERY layer, compares R^2(Wei's 2-group CO/MG label) against
     R^2(6-group template_identity) -- since Wei's label is a strict
     coarsening of template_identity, R^2(mechanism) >= R^2(Wei) always by
     construction; the GAP between them is what matters -- a large gap means
     the 6 templates genuinely differ from each other beyond what the binary
     label captures, a near-zero gap means the binary label was already
     capturing most of the structure (which would undercut Exp1's finding,
     so this is also a consistency check against Exp1, not just new
     information).

  3. Cross-model comparison at a fixed, pre-registered relative-depth layer
     (floor(0.6*n_layers) per model, NOT the data-dependent candidates from
     22_layer_selection_candidates.py, to avoid selecting a layer that
     happens to make model/mechanism effects look large): pools all 3
     models' delta_R/delta_H at that layer and decomposes R^2(model),
     R^2(mechanism, pooled across models), R^2(mechanism x model jointly) --
     the approximate interaction term is R^2(joint) - R^2(model) -
     R^2(mechanism), valid under this design's balanced structure (300
     instructions x 6 mechanisms x 3 models). Answers "does template_identity
     or model interaction explain more variance than the taxonomy label."

Also reports cos(refusal_direction, harmfulness_direction) per model (read
from the already-computed output_v2_dual_position/*/reference_direction_diagnostics_en.json,
for refusal_direction v3 vs harmfulness_direction -- recomputed here directly
from the .pt files since that diagnostics file was built against the old v2
refusal_direction), per Decision 2/3's requirement that this always
accompanies any delta_R/delta_H result.

Usage:
  python scripts/27_dual_axis_diagnosis.py \
      --output_dir output --lang en --suffix _full572 --ids_key direction_ids \
      --models Qwen2.5-7B-Instruct,Meta-Llama-3.1-8B-Instruct,gemma-2-9b-it
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)
from utils.direction_metadata import verify_delta_file, verify_direction_file, atomic_json_save, current_git_commit
from _taxonomy_v2_loader import load_taxonomy_v2

# FIXED 2026-09-04 (artifact-lineage audit): same stale-hardcoded-list bug as
# 25_extract_delta_r_h.py had -- 'instruction_hierarchy'/'fictional_framing'
# have not existed in templates_en.json since the wei_canonical_v2
# correction. Read live from the same source of truth every current-taxonomy
# script uses, never hand-copied.
_taxonomy = load_taxonomy_v2()
REAL_MECHS = _taxonomy['active_mechanisms']
CO = _taxonomy['CO_mechs']
MG = _taxonomy['MG_mechs']
WEI_LABEL = {m: 'CO' for m in CO}
WEI_LABEL.update({m: 'MG' for m in MG})


def one_way_r2(values, group_labels):
    """values: 1D tensor. group_labels: list of hashable labels, same length.
    Exact R^2 = SS_between/SS_total for a one-way grouping -- no model
    fitting, so no convergence/regularization concerns."""
    grand_mean = values.mean()
    ss_total = ((values - grand_mean) ** 2).sum()
    if ss_total <= 0:
        return 0.0
    groups = {}
    for v, g in zip(values.tolist(), group_labels):
        groups.setdefault(g, []).append(v)
    ss_between = 0.0
    for g, vs in groups.items():
        vs_t = torch.tensor(vs)
        ss_between += len(vs) * (vs_t.mean() - grand_mean) ** 2
    return (ss_between / ss_total).item()


def load_delta_r_h(output_dir, model_alias, lang, suffix, ids_key):
    """Fail-fast, hash-verified, mechanism-set-checked load -- NEVER a bare
    torch.load(). Raises immediately (FileNotFoundError/ValueError) rather
    than letting a missing/stale/corrupted input surface later as a
    confusing KeyError deep inside the R^2 computation."""
    path = os.path.join(output_dir, model_alias, f'delta_r_h_{lang}{suffix}_{ids_key}.pt')
    payload, meta = verify_delta_file(path, expected_active_mechanisms=REAL_MECHS)
    return payload


def descriptive_summary(data, mech_list=REAL_MECHS):
    n_layers = data['n_layers']
    has_perp = 'delta_H_perp' in data  # older payloads (pre R/H-orthogonality addition) won't have this
    summary = {}
    for mech in mech_list:
        summary[mech] = {
            'delta_R_mean_per_layer': data['delta_R'][mech].mean(0).tolist(),
            'delta_H_mean_per_layer': data['delta_H'][mech].mean(0).tolist(),
            'delta_R_pc_mean_per_layer': data['delta_R_placebo_calibrated'][mech].mean(0).tolist(),
            'delta_H_pc_mean_per_layer': data['delta_H_placebo_calibrated'][mech].mean(0).tolist(),
        }
        if has_perp:
            summary[mech]['delta_H_perp_mean_per_layer'] = data['delta_H_perp'][mech].mean(0).tolist()
            summary[mech]['delta_H_perp_pc_mean_per_layer'] = data['delta_H_perp_placebo_calibrated'][mech].mean(0).tolist()
    return summary


def shape_and_finite_check(data, mech_list=REAL_MECHS):
    """Per-mechanism tensor shape + torch.isfinite() check -- part of the
    single-model pilot report (item 6): confirms the payload's tensors are
    well-formed BEFORE any statistic is computed on them, independent of the
    hash check verify_delta_file already did on load."""
    n_layers = data['n_layers']
    n_instr = data['delta_R'][mech_list[0]].shape[0]
    out = {}
    for mech in mech_list:
        for key in ('delta_R', 'delta_H'):
            t = data[key][mech]
            out[f'{key}[{mech}]'] = {
                'shape': list(t.shape),
                'shape_ok': list(t.shape) == [n_instr, n_layers],
                'all_finite': bool(torch.isfinite(t).all()),
            }
    return out


def direction_norms_and_angle(output_dir, lang, model_alias):
    """Loads (hash-verified) refusal_direction_v3 + harmfulness_direction_v2
    for one model, returns per-layer ||r||, ||h||, cos(r,h) -- the R/H
    independence check (protocol Sec 10), usable standalone in pilot mode
    without needing a delta_r_h payload at all."""
    v2_dir = os.path.join(output_dir, 'output_v2_dual_position', model_alias)
    v3_dir = os.path.join(output_dir, 'output_v3_behavioral_refusal', model_alias)
    refusal_dir, _ = verify_direction_file(os.path.join(v3_dir, f'refusal_dir_v3_{lang}.pt'))
    harmfulness_dir, _ = verify_direction_file(os.path.join(v2_dir, f'harmfulness_dir_v2_{lang}.pt'))
    refusal_dir, harmfulness_dir = refusal_dir.float(), harmfulness_dir.float()
    cos = F.cosine_similarity(refusal_dir, harmfulness_dir, dim=-1)
    return {
        'refusal_direction_norm_per_layer': refusal_dir.norm(dim=-1).tolist(),
        'harmfulness_direction_norm_per_layer': harmfulness_dir.norm(dim=-1).tolist(),
        'cos_r_h_per_layer': cos.tolist(),
    }


def variance_decomposition_per_model(data, key='delta_R_placebo_calibrated'):
    """Returns {layer: {'r2_wei': ..., 'r2_mechanism': ..., 'gap': ...}}."""
    n_layers = data['n_layers']
    n_instr = data[key][REAL_MECHS[0]].shape[0]
    out = {'r2_wei_per_layer': [], 'r2_mechanism_per_layer': [], 'gap_per_layer': []}
    for l in range(n_layers):
        values, mech_labels, wei_labels = [], [], []
        for mech in REAL_MECHS:
            v = data[key][mech][:, l]
            values.extend(v.tolist())
            mech_labels.extend([mech] * n_instr)
            wei_labels.extend([WEI_LABEL[mech]] * n_instr)
        values_t = torch.tensor(values)
        r2_wei = one_way_r2(values_t, wei_labels)
        r2_mech = one_way_r2(values_t, mech_labels)
        out['r2_wei_per_layer'].append(r2_wei)
        out['r2_mechanism_per_layer'].append(r2_mech)
        out['gap_per_layer'].append(r2_mech - r2_wei)
    return out


def cross_model_decomposition(all_data, key='delta_R_placebo_calibrated'):
    """At each model's floor(0.6*n_layers) layer, pools across models and
    decomposes R^2(model), R^2(mechanism, pooled), R^2(mechanism x model)."""
    values, mech_labels, model_labels, joint_labels = [], [], [], []
    layer_used = {}
    for model_alias, data in all_data.items():
        n_layers = data['n_layers']
        l = int(0.6 * n_layers)
        layer_used[model_alias] = l
        n_instr = data[key][REAL_MECHS[0]].shape[0]
        for mech in REAL_MECHS:
            v = data[key][mech][:, l]
            values.extend(v.tolist())
            mech_labels.extend([mech] * n_instr)
            model_labels.extend([model_alias] * n_instr)
            joint_labels.extend([(mech, model_alias)] * n_instr)
    values_t = torch.tensor(values)
    r2_model = one_way_r2(values_t, model_labels)
    r2_mechanism = one_way_r2(values_t, mech_labels)
    r2_joint = one_way_r2(values_t, joint_labels)
    interaction_approx = r2_joint - r2_model - r2_mechanism
    return {
        'layer_used_per_model': layer_used,
        'r2_model': r2_model, 'r2_mechanism_pooled': r2_mechanism,
        'r2_joint_mechanism_x_model': r2_joint,
        'interaction_r2_approx': interaction_approx,
    }


CANONICAL_3_MODELS = {'Qwen2.5-7B-Instruct', 'Meta-Llama-3.1-8B-Instruct', 'gemma-2-9b-it'}


def main(args):
    models = args.models.split(',')
    print(f"Loading delta_R/delta_H for {len(models)} model(s), verifying each before use...")
    all_data = {}
    for m in models:
        all_data[m] = load_delta_r_h(args.output_dir, m, args.lang, args.suffix, args.ids_key)
        print(f"  [{m}] OK -- {all_data[m]['n_layers']} layers, "
              f"{all_data[m]['delta_R'][REAL_MECHS[0]].shape[0]} instructions, hash-verified.")
    print(f"All {len(models)} model(s) loaded and verified.\n")

    # Formal cross-model R^2 analysis (variance decomposition + pooled
    # cross-model decomposition) is only meaningful, and only permitted, once
    # all 3 canonical models' formal delta_r_h artifacts are present and
    # hash-verified (the load loop above already did the verification --
    # this gate only decides whether it's the FULL canonical set, not a
    # subset). Anything else (1 or 2 models, or a non-canonical alias) is
    # treated as a PILOT run: only descriptive/shape/finite/norm/angle
    # output, explicitly never labeled or saved as a formal R^2 result --
    # per item 6 of the 2026-09-04 correction, added specifically because
    # this function used to run variance decomposition and the pooled
    # "cross-model" section unconditionally, even for a single model, which
    # silently produced a degenerate/misleading pooled result (R^2(model)
    # trivially computed on one group).
    is_formal = (len(models) == 3 and set(models) == CANONICAL_3_MODELS)

    results = {'lang': args.lang, 'suffix': args.suffix, 'ids_key': args.ids_key,
               'taxonomy_version': _taxonomy['taxonomy_version'], 'active_mechanisms': REAL_MECHS,
               'git_commit': current_git_commit(SCRIPT_DIR),
               'result_status': 'FORMAL' if is_formal else 'PILOT_NON_RESULT',
               'models_requested': models, 'per_model': {}}

    if not is_formal:
        print(f"=== PILOT MODE ({len(models)} model(s): {models}) ===")
        print("Not the full canonical 3-model set -- variance decomposition (R^2 Wei-vs-mechanism) "
              "and the cross-model section are SKIPPED. Only descriptive projections, tensor "
              "shape/finite checks, direction norms, and R/H angle are reported below. This output "
              "is tagged result_status=PILOT_NON_RESULT and must not be cited as an Experiment 2 result.\n")
        for model_alias, data in all_data.items():
            print(f"[{model_alias}]")
            desc = descriptive_summary(data)
            shape_finite = shape_and_finite_check(data)
            n_bad_shape = sum(1 for v in shape_finite.values() if not v['shape_ok'])
            n_nonfinite = sum(1 for v in shape_finite.values() if not v['all_finite'])
            print(f"  shape/finite: {len(shape_finite)} tensors checked, "
                  f"{n_bad_shape} shape mismatches, {n_nonfinite} with non-finite values")
            entry = {'descriptive': desc, 'shape_and_finite_check': shape_finite}
            try:
                norms_angle = direction_norms_and_angle(args.output_dir, args.lang, model_alias)
                entry['direction_norms_and_r_h_angle'] = norms_angle
                mid = int(0.6 * data['n_layers'])
                print(f"  ||r||[layer {mid}]={norms_angle['refusal_direction_norm_per_layer'][mid]:.3f}  "
                      f"||h||[layer {mid}]={norms_angle['harmfulness_direction_norm_per_layer'][mid]:.3f}  "
                      f"cos(r,h)[layer {mid}]={norms_angle['cos_r_h_per_layer'][mid]:+.4f}")
            except FileNotFoundError as e:
                entry['direction_norms_and_r_h_angle'] = None
                print(f"  direction files not available for norm/angle check: {e}")
            results['per_model'][model_alias] = entry
        out_path = os.path.join(args.output_dir, f'dual_axis_diagnosis_PILOT_{args.lang}{args.suffix}_{args.ids_key}.json')
        atomic_json_save(results, out_path)
        print(f"\nSaved (PILOT_NON_RESULT): {out_path} (atomic)")
        return

    print("=== 1. Descriptive: per-mechanism delta_R / delta_H (placebo-calibrated, mean per layer) ===")
    for model_alias, data in all_data.items():
        print(f"\n[{model_alias}]")
        desc = descriptive_summary(data)
        results['per_model'].setdefault(model_alias, {})['descriptive'] = desc
        n_layers = data['n_layers']
        mid = int(0.6 * n_layers)
        for mech in REAL_MECHS:
            dr = desc[mech]['delta_R_pc_mean_per_layer'][mid]
            dh = desc[mech]['delta_H_pc_mean_per_layer'][mid]
            dh_perp_str = ""
            if 'delta_H_perp_pc_mean_per_layer' in desc[mech]:
                dh_perp_str = f"  delta_H_perp={desc[mech]['delta_H_perp_pc_mean_per_layer'][mid]:+.4f}"
            print(f"  [{WEI_LABEL[mech]}] {mech:24s}  delta_R={dr:+.4f}  delta_H={dh:+.4f}{dh_perp_str}  (layer {mid})")

    print("\n=== 2. Variance decomposition: Wei's binary label vs template_identity ===")
    for model_alias, data in all_data.items():
        print(f"\n[{model_alias}]")
        for key, name in [('delta_R_placebo_calibrated', 'delta_R'), ('delta_H_placebo_calibrated', 'delta_H')]:
            vd = variance_decomposition_per_model(data, key=key)
            results['per_model'][model_alias][f'variance_decomposition_{name}'] = vd
            n_layers = data['n_layers']
            mean_gap = sum(vd['gap_per_layer']) / n_layers
            max_gap_layer = max(range(n_layers), key=lambda l: vd['gap_per_layer'][l])
            print(f"  {name}: mean R2(mechanism)-R2(Wei) across layers = {mean_gap:.4f}  "
                  f"(max gap at layer {max_gap_layer}: "
                  f"R2(Wei)={vd['r2_wei_per_layer'][max_gap_layer]:.4f}, "
                  f"R2(mechanism)={vd['r2_mechanism_per_layer'][max_gap_layer]:.4f})")

    print("\n=== 3. Cross-model: does model identity/interaction explain variance beyond mechanism? ===")
    print("(at each model's floor(0.6*n_layers) layer, a fixed non-data-dependent reference point)")
    for key, name in [('delta_R_placebo_calibrated', 'delta_R'), ('delta_H_placebo_calibrated', 'delta_H')]:
        cmd = cross_model_decomposition(all_data, key=key)
        results[f'cross_model_{name}'] = cmd
        print(f"\n  {name}:  layers used = {cmd['layer_used_per_model']}")
        print(f"    R2(model)                = {cmd['r2_model']:.4f}")
        print(f"    R2(mechanism, pooled)    = {cmd['r2_mechanism_pooled']:.4f}")
        print(f"    R2(mechanism x model)    = {cmd['r2_joint_mechanism_x_model']:.4f}")
        print(f"    interaction R2 (approx)  = {cmd['interaction_r2_approx']:.4f}")

    print("\n=== R/H independence check: cos(r,h) and ||h_perp||/||h|| per model, per layer "
          "(protocol Sec 10 -- no threshold imposed, reported for review) ===")
    for model_alias in models:
        v2_dir = os.path.join(args.output_dir, 'output_v2_dual_position', model_alias)
        v3_dir = os.path.join(args.output_dir, 'output_v3_behavioral_refusal', model_alias)
        refusal_dir, _ = verify_direction_file(os.path.join(v3_dir, f'refusal_dir_v3_{args.lang}.pt'))
        harmfulness_dir, _ = verify_direction_file(os.path.join(v2_dir, f'harmfulness_dir_v2_{args.lang}.pt'))
        refusal_dir, harmfulness_dir = refusal_dir.float(), harmfulness_dir.float()
        cos = F.cosine_similarity(refusal_dir, harmfulness_dir, dim=-1)
        r_dot_r = (refusal_dir * refusal_dir).sum(-1, keepdim=True).clamp_min(1e-12)
        h_dot_r = (harmfulness_dir * refusal_dir).sum(-1, keepdim=True)
        h_perp = harmfulness_dir - (h_dot_r / r_dot_r) * refusal_dir
        h_perp_ratio = h_perp.norm(dim=-1) / harmfulness_dir.norm(dim=-1).clamp_min(1e-12)
        results['per_model'][model_alias]['cos_refusal_v3_harmfulness_v2_per_layer'] = cos.tolist()
        results['per_model'][model_alias]['h_perp_norm_ratio_per_layer'] = h_perp_ratio.tolist()
        print(f"  [{model_alias}] mean cos = {cos.mean():.4f}  range=[{cos.min():.4f},{cos.max():.4f}]  "
              f"mean ||h_perp||/||h|| = {h_perp_ratio.mean():.4f}")

    out_path = os.path.join(args.output_dir, f'dual_axis_diagnosis_{args.lang}{args.suffix}_{args.ids_key}.json')
    atomic_json_save(results, out_path)
    print(f"\nSaved: {out_path} (atomic)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    parser.add_argument('--lang',       type=str, default='en')
    parser.add_argument('--suffix',     type=str, default='_full572')
    parser.add_argument('--ids_key',    type=str, default='direction_ids')
    parser.add_argument('--models',     type=str,
                         default='Qwen2.5-7B-Instruct,Meta-Llama-3.1-8B-Instruct,gemma-2-9b-it')
    args = parser.parse_args()
    main(args)
