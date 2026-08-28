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

import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(__file__)
REAL_MECHS = ['prefix_injection', 'refusal_suppression', 'instruction_hierarchy',
              'persona_roleplay', 'fictional_framing', 'encoding_obfuscation']
CO = ['prefix_injection', 'refusal_suppression', 'instruction_hierarchy']
MG = ['persona_roleplay', 'fictional_framing', 'encoding_obfuscation']
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
    path = os.path.join(output_dir, model_alias, f'delta_r_h_{lang}{suffix}_{ids_key}.pt')
    return torch.load(path, map_location='cpu')


def descriptive_summary(data, mech_list=REAL_MECHS):
    n_layers = data['n_layers']
    summary = {}
    for mech in mech_list:
        summary[mech] = {
            'delta_R_mean_per_layer': data['delta_R'][mech].mean(0).tolist(),
            'delta_H_mean_per_layer': data['delta_H'][mech].mean(0).tolist(),
            'delta_R_pc_mean_per_layer': data['delta_R_placebo_calibrated'][mech].mean(0).tolist(),
            'delta_H_pc_mean_per_layer': data['delta_H_placebo_calibrated'][mech].mean(0).tolist(),
        }
    return summary


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


def main(args):
    models = args.models.split(',')
    all_data = {m: load_delta_r_h(args.output_dir, m, args.lang, args.suffix, args.ids_key) for m in models}

    results = {'lang': args.lang, 'suffix': args.suffix, 'ids_key': args.ids_key, 'per_model': {}}

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
            print(f"  [{WEI_LABEL[mech]}] {mech:24s}  delta_R={dr:+.4f}  delta_H={dh:+.4f}  (layer {mid})")

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

    print("\n=== cos(refusal_direction_v3, harmfulness_direction_v2) per model, per layer ===")
    for model_alias in models:
        v2_dir = os.path.join(args.output_dir, 'output_v2_dual_position', model_alias)
        v3_dir = os.path.join(args.output_dir, 'output_v3_behavioral_refusal', model_alias)
        refusal_dir = torch.load(os.path.join(v3_dir, f'refusal_dir_v3_{args.lang}.pt'), map_location='cpu').float()
        harmfulness_dir = torch.load(os.path.join(v2_dir, f'harmfulness_dir_v2_{args.lang}.pt'), map_location='cpu').float()
        cos = F.cosine_similarity(refusal_dir, harmfulness_dir, dim=-1)
        results['per_model'][model_alias]['cos_refusal_v3_harmfulness_v2_per_layer'] = cos.tolist()
        print(f"  [{model_alias}] mean cos = {cos.mean():.4f}  range=[{cos.min():.4f},{cos.max():.4f}]")

    out_path = os.path.join(args.output_dir, f'dual_axis_diagnosis_{args.lang}{args.suffix}_{args.ids_key}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


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
