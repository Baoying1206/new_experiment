"""
Completeness/quality audit for delta_r_h_{lang}{suffix}_test_ids.pt (the
Exp3 test-set extraction, 25_extract_delta_r_h.py --ids_key test_ids) --
reads the actual tensors, never parses slurm log text. CPU-only.

Checks per model, exactly the list requested for Exp3 test-data sign-off:
  1. instruction_ids count == len(splits.json['test_ids'])
  2. instruction_ids set == splits.json['test_ids'] set (exact, not just count)
  3. instruction_ids set disjoint from splits.json['direction_ids']
  4. delta_R/delta_H tensor shape per mechanism == [n_test_ids, n_layers]
  5. all 6 real mechanisms + placebo present in delta_R and delta_H
  6. torch.isfinite() all True (no NaN/Inf) for delta_R, delta_H, and both
     placebo-calibrated versions
  7. valid_mask true-count and failures count/rate
  8. delta_R_placebo_calibrated / delta_H_placebo_calibrated keys present
  9. full distribution stats (min/max/mean/median/5%-trimmed-mean/quantiles
     0,1,5,50,95,99,100) per mechanism, at the fixed floor(0.6*n_layers)
     layer -- for every model, not just Gemma, so any model's extreme
     values get the same scrutiny.
  10. Gemma-specific: full per-instruction delta_R distribution for
      fictional_framing at layer 25, to check whether the -241.68 mean
      figure quoted earlier is a genuine central tendency or driven by a
      handful of extreme instructions (reports the individual extreme
      values with their instruction ids, not just aggregate stats).

Usage:
  python scripts/audits/audit_delta_r_h_test_ids.py --output_dir output --lang en --suffix _full572
"""
import argparse
import json
import os

import torch

SCRIPT_DIR = os.path.dirname(__file__)
SPLITS_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'data', 'splits.json')
REAL_MECHS = ['prefix_injection', 'refusal_suppression', 'instruction_hierarchy',
              'persona_roleplay', 'fictional_framing', 'encoding_obfuscation']
ALL_CONDS = REAL_MECHS + ['placebo']
FIXED_LAYER_FRACTION = 0.6

MODELS = ['Qwen2.5-7B-Instruct', 'Meta-Llama-3.1-8B-Instruct', 'gemma-2-9b-it']


def trimmed_mean(x, frac=0.05):
    n = x.numel()
    k = int(n * frac)
    if k == 0:
        return x.mean().item()
    sorted_x, _ = torch.sort(x)
    return sorted_x[k:n - k].mean().item()


def quantiles(x):
    qs = [0.0, 0.01, 0.05, 0.50, 0.95, 0.99, 1.0]
    return {f'q{int(q*100)}': torch.quantile(x, q).item() for q in qs}


def distribution_stats(x):
    return {
        'n': x.numel(), 'min': x.min().item(), 'max': x.max().item(),
        'mean': x.mean().item(), 'median': x.median().item(),
        'trimmed_mean_5pct': trimmed_mean(x, 0.05),
        'quantiles': quantiles(x),
        'n_nonfinite': int((~torch.isfinite(x)).sum().item()),
    }


def audit_model(output_dir, model_alias, lang, suffix, splits, warnings):
    path = os.path.join(output_dir, model_alias, f'delta_r_h_{lang}{suffix}_test_ids.pt')
    result = {'path': path, 'exists': os.path.exists(path)}
    if not result['exists']:
        warnings.append(f"[{model_alias}] MISSING: {path} -- test_ids extraction not yet run for this model.")
        return result

    data = torch.load(path, map_location='cpu')
    test_ids_expected = set(splits['test_ids'])
    direction_ids = set(splits['direction_ids'])
    ids_actual = set(data['instruction_ids'])

    result['ids_key_in_file'] = data.get('ids_key')
    result['n_instruction_ids'] = len(data['instruction_ids'])
    result['n_expected_test_ids'] = len(test_ids_expected)
    result['count_matches'] = len(data['instruction_ids']) == len(test_ids_expected)
    result['id_set_exact_match'] = ids_actual == test_ids_expected
    result['ids_missing_from_file'] = sorted(test_ids_expected - ids_actual)
    result['ids_extra_in_file'] = sorted(ids_actual - test_ids_expected)
    result['overlap_with_direction_ids'] = sorted(ids_actual & direction_ids)
    result['disjoint_from_direction_ids'] = len(ids_actual & direction_ids) == 0

    if not result['count_matches']:
        warnings.append(f"[{model_alias}] instruction_ids count {result['n_instruction_ids']} "
                         f"!= expected {result['n_expected_test_ids']}")
    if not result['id_set_exact_match']:
        warnings.append(f"[{model_alias}] instruction_ids set does NOT exactly match splits.json['test_ids'] -- "
                         f"missing={len(result['ids_missing_from_file'])} extra={len(result['ids_extra_in_file'])}")
    if not result['disjoint_from_direction_ids']:
        warnings.append(f"[{model_alias}] CRITICAL: {len(result['overlap_with_direction_ids'])} ids overlap "
                         f"with direction_ids -- this would leak profile-construction data into test evaluation.")

    n_layers = data['n_layers']
    n = len(data['instruction_ids'])
    result['n_layers'] = n_layers

    result['mechanisms_present'] = {
        'delta_R': sorted(data['delta_R'].keys()), 'delta_H': sorted(data['delta_H'].keys()),
    }
    missing_mechs_R = set(ALL_CONDS) - set(data['delta_R'].keys())
    missing_mechs_H = set(ALL_CONDS) - set(data['delta_H'].keys())
    if missing_mechs_R or missing_mechs_H:
        warnings.append(f"[{model_alias}] missing mechanisms -- delta_R missing {missing_mechs_R}, "
                         f"delta_H missing {missing_mechs_H}")

    result['shapes'] = {}
    for key in ('delta_R', 'delta_H'):
        for mech in ALL_CONDS:
            if mech in data[key]:
                shape = tuple(data[key][mech].shape)
                result['shapes'][f'{key}.{mech}'] = shape
                if shape != (n, n_layers):
                    warnings.append(f"[{model_alias}] {key}.{mech} shape {shape} != expected ({n}, {n_layers})")

    result['has_placebo_calibrated'] = (
        'delta_R_placebo_calibrated' in data and 'delta_H_placebo_calibrated' in data
    )
    if not result['has_placebo_calibrated']:
        warnings.append(f"[{model_alias}] missing delta_R_placebo_calibrated/delta_H_placebo_calibrated keys")

    result['nonfinite_counts'] = {}
    for key in ('delta_R', 'delta_H', 'delta_R_placebo_calibrated', 'delta_H_placebo_calibrated'):
        if key not in data:
            continue
        for mech, tensor in data[key].items():
            n_bad = int((~torch.isfinite(tensor)).sum().item())
            if n_bad > 0:
                result['nonfinite_counts'][f'{key}.{mech}'] = n_bad
                warnings.append(f"[{model_alias}] {key}.{mech} has {n_bad} non-finite values")

    valid_mask = data.get('valid_mask', {})
    result['valid_mask_true_counts'] = {mech: int(m.sum().item()) for mech, m in valid_mask.items()}
    failures = data.get('failures', [])
    result['n_failures'] = len(failures)
    result['failure_rate'] = len(failures) / (n * len(ALL_CONDS)) if n > 0 else None
    result['sample_failures'] = failures[:10]

    fixed_layer = int(FIXED_LAYER_FRACTION * n_layers)
    result['fixed_layer'] = fixed_layer
    result['distribution_stats_at_fixed_layer'] = {}
    for mech in REAL_MECHS:
        if mech in data.get('delta_R_placebo_calibrated', {}):
            result['distribution_stats_at_fixed_layer'][f'delta_R_pc.{mech}'] = distribution_stats(
                data['delta_R_placebo_calibrated'][mech][:, fixed_layer])
        if mech in data.get('delta_H_placebo_calibrated', {}):
            result['distribution_stats_at_fixed_layer'][f'delta_H_pc.{mech}'] = distribution_stats(
                data['delta_H_placebo_calibrated'][mech][:, fixed_layer])

    if model_alias == 'gemma-2-9b-it' and 'fictional_framing' in data.get('delta_R_placebo_calibrated', {}):
        layer_25 = 25
        if layer_25 < n_layers:
            per_instr = data['delta_R_placebo_calibrated']['fictional_framing'][:, layer_25]
            ids_list = data['instruction_ids']
            paired = sorted(zip(ids_list, per_instr.tolist()), key=lambda x: x[1])
            result['gemma_fictional_framing_layer25_diagnosis'] = {
                'n': len(paired), 'mean': per_instr.mean().item(), 'median': per_instr.median().item(),
                'std': per_instr.std().item(),
                'most_negative_10': paired[:10],
                'most_positive_10': paired[-10:],
                'quantiles': quantiles(per_instr),
            }

    return result


def main(args):
    with open(SPLITS_PATH) as f:
        splits = json.load(f)

    warnings = []
    results = {'lang': args.lang, 'suffix': args.suffix, 'per_model': {}}
    for model_alias in args.models.split(','):
        print(f"\n=== {model_alias} ===")
        r = audit_model(args.output_dir, model_alias, args.lang, args.suffix, splits, warnings)
        results['per_model'][model_alias] = r
        if not r.get('exists'):
            print("  MISSING -- not extracted yet.")
            continue
        print(f"  n_instruction_ids={r['n_instruction_ids']}  expected={r['n_expected_test_ids']}  "
              f"count_matches={r['count_matches']}  id_set_exact_match={r['id_set_exact_match']}")
        print(f"  disjoint_from_direction_ids={r['disjoint_from_direction_ids']}")
        print(f"  n_layers={r['n_layers']}  fixed_layer(floor(0.6*n_layers))={r['fixed_layer']}")
        print(f"  has_placebo_calibrated={r['has_placebo_calibrated']}")
        print(f"  n_failures={r['n_failures']}  failure_rate={r['failure_rate']}")
        print(f"  nonfinite_counts={r['nonfinite_counts'] or 'none'}")
        if 'gemma_fictional_framing_layer25_diagnosis' in r:
            d = r['gemma_fictional_framing_layer25_diagnosis']
            print(f"  [Gemma fictional_framing layer25] mean={d['mean']:.2f} median={d['median']:.2f} "
                  f"std={d['std']:.2f}")
            print(f"    most negative 10 (id, value): {d['most_negative_10']}")

    results['warnings'] = warnings
    print(f"\n=== {len(warnings)} warnings ===")
    for w in warnings:
        print(f"  {w}")

    out_path = os.path.join(args.output_dir, 'delta_r_h_test_ids_audit.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=os.path.join(SCRIPT_DIR, '..', '..', 'output'))
    parser.add_argument('--lang',       type=str, default='en')
    parser.add_argument('--suffix',     type=str, default='_full572')
    parser.add_argument('--models',     type=str, default=','.join(MODELS))
    args = parser.parse_args()
    main(args)
