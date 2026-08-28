"""
Read-only audit of paired_diffs_{lang}_full572_corrected.pt for the 3
models -- CPU-only, needs only torch (no `pipeline` import), reads the .pt
files directly, never touches completions (completions can't be used to
verify NaN/Inf in the extracted activations).

Per explicit instruction: does NOT assume the schema is
{'instruction_ids': {mech: [...]}, 'diffs': {mech: tensor}, ...} just
because that's what 18_extract_paired_diffs.py is known to currently save --
records the actual top-level keys first, then adapts: if 'instruction_ids'
is a dict, treats it as per-mechanism; if it's a list, treats it as a single
global id list shared across mechanisms; anything else is reported as an
unrecognized shape rather than silently coerced.

Usage:
  python scripts/audits/audit_paired_diffs_corrected.py \
      --output_dir output --lang en --suffix _full572_corrected
"""
import argparse
import json
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
from _taxonomy_v2_loader import load_taxonomy_v2

SPLITS_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'data', 'splits.json')
MODELS = ['Qwen2.5-7B-Instruct', 'Meta-Llama-3.1-8B-Instruct', 'gemma-2-9b-it']


def audit_one(path, direction_ids, active_mechanisms):
    result = {'path': path, 'exists': os.path.exists(path)}
    if not result['exists']:
        return result

    data = torch.load(path, map_location='cpu')
    top_level_keys = sorted(data.keys())
    result['top_level_keys'] = top_level_keys

    # --- instruction_ids: discover actual shape, don't assume ---
    ids_field = data.get('instruction_ids')
    if isinstance(ids_field, dict):
        result['instruction_ids_shape'] = 'per_mechanism_dict'
        per_mech_ids = {mech: list(ids) for mech, ids in ids_field.items()}
    elif isinstance(ids_field, list):
        result['instruction_ids_shape'] = 'single_global_list'
        per_mech_ids = None
        result['global_instruction_ids_count'] = len(ids_field)
        result['global_instruction_ids_unique_count'] = len(set(ids_field))
    elif ids_field is None:
        result['instruction_ids_shape'] = 'MISSING -- no instruction_ids field at all'
        per_mech_ids = None
    else:
        result['instruction_ids_shape'] = f'UNRECOGNIZED type: {type(ids_field)}'
        per_mech_ids = None

    direction_ids_set = set(direction_ids)
    if per_mech_ids is not None:
        result['per_mechanism'] = {}
        first_order = None
        orders_match_across_mechanisms = True
        for mech, ids in per_mech_ids.items():
            ids_set = set(ids)
            entry = {
                'count': len(ids),
                'unique_count': len(ids_set),
                'has_duplicates': len(ids) != len(ids_set),
                'set_equals_direction_ids': ids_set == direction_ids_set,
                'missing_from_this_mech': sorted(direction_ids_set - ids_set),
                'extra_in_this_mech': sorted(ids_set - direction_ids_set),
            }
            result['per_mechanism'][mech] = entry
            if first_order is None:
                first_order = ids
            elif ids != first_order:
                orders_match_across_mechanisms = False
        result['instruction_id_order_identical_across_mechanisms'] = orders_match_across_mechanisms
        if not orders_match_across_mechanisms:
            result['order_note'] = (
                "Instruction ID order DIFFERS across mechanisms -- any downstream code must "
                "index via an id->position lookup per mechanism, never assume row i in one "
                "mechanism's tensor corresponds to row i in another's."
            )
    elif ids_field is not None and isinstance(ids_field, list):
        ids_set = set(ids_field)
        result['global_ids'] = {
            'set_equals_direction_ids': ids_set == direction_ids_set,
            'missing_from_global': sorted(direction_ids_set - ids_set),
            'extra_in_global': sorted(ids_set - direction_ids_set),
        }

    result['n_layers_field'] = data.get('n_layers')
    result['d_model_field'] = data.get('d_model')
    result['ids_key_field'] = data.get('ids_key')

    diffs = data.get('diffs')
    result['diffs_present'] = diffs is not None
    if diffs is not None:
        mechs_present = sorted(diffs.keys())
        result['mechanisms_in_diffs'] = mechs_present
        expected = set(active_mechanisms) | {'placebo'}
        result['expected_mechanisms'] = sorted(expected)
        result['missing_mechanisms'] = sorted(expected - set(mechs_present))
        result['unexpected_extra_mechanisms'] = sorted(set(mechs_present) - expected)

        result['tensor_shapes_dtypes'] = {}
        result['nonfinite_counts'] = {}
        for mech, tensor in diffs.items():
            result['tensor_shapes_dtypes'][mech] = {
                'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
            }
            n_bad = int((~torch.isfinite(tensor.float())).sum().item())
            result['nonfinite_counts'][mech] = n_bad

    # --- fields the user asked about, reported honestly whether present or not ---
    result['has_valid_mask_field'] = 'valid_mask' in data
    result['has_failures_field'] = 'failures' in data
    if result['has_valid_mask_field']:
        vm = data['valid_mask']
        result['valid_mask_true_counts'] = {
            mech: int(m.sum().item()) for mech, m in vm.items()
        } if isinstance(vm, dict) else 'unrecognized valid_mask shape'
    if result['has_failures_field']:
        result['n_failures'] = len(data['failures'])

    # --- placebo-calibration status: confirmed by inspecting 18_extract_paired_diffs.py's
    # own save code (not a guess) -- it saves ONLY 'diffs' (raw, uncalibrated per-mechanism
    # diffs including placebo's own raw diff), no placebo-calibrated fields at all. Verify
    # that expectation against what's actually in this file rather than asserting it blind.
    result['has_placebo_calibrated_fields_in_file'] = any(
        'calibrat' in k.lower() for k in top_level_keys
    )
    result['placebo_calibration_note'] = (
        "18_extract_paired_diffs.py's save code only writes raw 'diffs' (uncalibrated, "
        "including placebo's own raw diff) -- placebo calibration must be computed downstream "
        "by subtracting diffs['placebo'] per-instruction, never assume it's pre-done."
    )

    return result


def main(args):
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    direction_ids = splits['direction_ids']

    taxonomy = load_taxonomy_v2()
    active_mechanisms = taxonomy['active_mechanisms']
    print(f"Loaded taxonomy v2 config: active_mechanisms={active_mechanisms}\n")

    results = {'lang': args.lang, 'suffix': args.suffix,
               'n_direction_ids': len(direction_ids), 'per_model': {}}

    for model_alias in MODELS:
        print(f"=== {model_alias} ===")
        path = os.path.join(args.output_dir, model_alias, f'paired_diffs_{args.lang}{args.suffix}.pt')
        r = audit_one(path, direction_ids, active_mechanisms)
        results['per_model'][model_alias] = r
        if not r['exists']:
            print(f"  MISSING: {path}\n")
            continue
        print(f"  top_level_keys: {r['top_level_keys']}")
        print(f"  instruction_ids_shape: {r['instruction_ids_shape']}")
        if 'instruction_id_order_identical_across_mechanisms' in r:
            print(f"  order identical across mechanisms: {r['instruction_id_order_identical_across_mechanisms']}")
        print(f"  n_layers_field: {r['n_layers_field']}  d_model_field: {r['d_model_field']}")
        print(f"  mechanisms_in_diffs: {r.get('mechanisms_in_diffs')}")
        print(f"  missing_mechanisms: {r.get('missing_mechanisms')}")
        print(f"  unexpected_extra_mechanisms: {r.get('unexpected_extra_mechanisms')}")
        print(f"  nonfinite_counts: {r.get('nonfinite_counts')}")
        print(f"  has_valid_mask_field: {r['has_valid_mask_field']}  has_failures_field: {r['has_failures_field']}")
        print(f"  has_placebo_calibrated_fields_in_file: {r['has_placebo_calibrated_fields_in_file']}")
        if r.get('per_mechanism'):
            all_match = all(v['set_equals_direction_ids'] for v in r['per_mechanism'].values())
            print(f"  ALL mechanisms' id set == direction_ids exactly: {all_match}")
        print()

    out_dir = os.path.join(args.output_dir, 'canonical_v2')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'paired_diffs_audit.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=os.path.join(SCRIPT_DIR, '..', '..', 'output'))
    parser.add_argument('--lang',       type=str, default='en')
    parser.add_argument('--suffix',     type=str, default='_full572_corrected')
    args = parser.parse_args()
    main(args)
