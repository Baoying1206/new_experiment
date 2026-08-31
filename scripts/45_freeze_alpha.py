"""
Freezes one alpha per (model, method) for method in {fixed_wei, adaptive}
(the only two conditions eligible for alpha selection under
protocol_version=exp3_reduced_v1 -- Global/Placebo never participate, see
EXPERIMENT3_PROTOCOL.md section 4). Uses select_alpha (imported from
_defence_metrics.py, never reimplemented): minimize macro-ASR subject to
benign macro-FRR not exceeding the No-defence benign macro-FRR by more than
BENIGN_FRR_MAX_INCREASE_PP (5) percentage points; ties -> smallest alpha;
no eligible non-zero alpha -> freeze 0.0.

Reads, for one model:
  experiment3_validation_join_audit_{model}_{method}.json  (fixed_wei, adaptive)
    -- overall_pass is checked and REQUIRED true; refuses otherwise, never
       freezes an alpha from data that failed its own integrity checks.
  experiment3_validation_summary_{model}_{method}.json     (fixed_wei, adaptive)
    -- per_alpha_macro's macro_asr/macro_benign_frr, by alpha.
  experiment3_no_defence_baseline_{model}.json
    -- overall_pass REQUIRED true; benign_macro_frr used as the constraint baseline.

Writes experiment3_alpha_freeze_{model}.json (frozen alpha + reason +
max_allowed_frr per method, plus the exact inputs used, git commit,
timestamp) -- pure CPU/Python, no GPU, no new generation/judging.

Usage:
  python scripts/45_freeze_alpha.py --model_idx 1 --output_path output
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)
import _defence_metrics as dm  # torch-free

MODELS = dm.MODEL_PATHS
ALPHA_ELIGIBLE_METHODS = ('fixed_wei', 'adaptive')


class UnverifiedInputError(RuntimeError):
    pass


def load_method_summary(output_path, model_alias, method):
    out_dir = os.path.join(output_path, 'canonical_v2')
    audit_path = os.path.join(out_dir, f'experiment3_validation_join_audit_{model_alias}_{method}.json')
    summary_path = os.path.join(out_dir, f'experiment3_validation_summary_{model_alias}_{method}.json')
    with open(audit_path) as f:
        audit = json.load(f)
    if not audit.get('overall_pass'):
        raise UnverifiedInputError(
            f"{audit_path} has overall_pass={audit.get('overall_pass')!r} -- refusing to use "
            f"{model_alias} x {method} for alpha freezing until its integrity checks pass."
        )
    with open(summary_path) as f:
        summary = json.load(f)
    macro_asr_by_alpha = {float(a): v['macro_asr'] for a, v in summary['per_alpha_macro'].items()}
    macro_frr_by_alpha = {float(a): v['macro_benign_frr'] for a, v in summary['per_alpha_macro'].items()}
    return macro_asr_by_alpha, macro_frr_by_alpha, audit_path, summary_path


def load_no_defence_baseline(output_path, model_alias):
    out_dir = os.path.join(output_path, 'canonical_v2')
    baseline_path = os.path.join(out_dir, f'experiment3_no_defence_baseline_{model_alias}.json')
    with open(baseline_path) as f:
        baseline = json.load(f)
    if not baseline.get('overall_pass'):
        raise UnverifiedInputError(
            f"{baseline_path} has overall_pass={baseline.get('overall_pass')!r} -- refusing to use "
            f"the {model_alias} No-defence baseline for alpha freezing until its checks pass."
        )
    return baseline, baseline_path


def main(args):
    model_alias, _ = MODELS[args.model_idx]
    print(f"=== Alpha freezing: {model_alias} ===")

    baseline, baseline_path = load_no_defence_baseline(args.output_path, model_alias)
    no_defence_macro_frr = baseline['benign_macro_frr']
    print(f"No-defence benign_macro_frr = {no_defence_macro_frr} "
          f"(No-defence harmful_macro_asr = {baseline['harmful_macro_asr']}, for reference only, "
          f"not part of the alpha-selection constraint)")

    results = {}
    for method in ALPHA_ELIGIBLE_METHODS:
        macro_asr_by_alpha, macro_frr_by_alpha, audit_path, summary_path = load_method_summary(
            args.output_path, model_alias, method)
        chosen_alpha, reason, max_allowed_frr = dm.select_alpha(
            macro_asr_by_alpha, macro_frr_by_alpha, no_defence_macro_frr)
        print(f"\n{method}: macro_asr_by_alpha={macro_asr_by_alpha}")
        print(f"{method}: macro_frr_by_alpha={macro_frr_by_alpha}")
        print(f"{method}: max_allowed_frr={max_allowed_frr} (no_defence {no_defence_macro_frr} "
              f"+ {dm.BENIGN_FRR_MAX_INCREASE_PP}pp)")
        print(f"{method}: FROZEN alpha={chosen_alpha}  reason={reason}")
        results[method] = {
            'frozen_alpha': chosen_alpha, 'reason': reason, 'max_allowed_frr': max_allowed_frr,
            'macro_asr_by_alpha': macro_asr_by_alpha, 'macro_frr_by_alpha': macro_frr_by_alpha,
            'macro_asr_at_frozen_alpha': macro_asr_by_alpha.get(chosen_alpha),
            'macro_frr_at_frozen_alpha': macro_frr_by_alpha.get(chosen_alpha),
            'join_audit_input_path': audit_path, 'summary_input_path': summary_path,
        }

    out_dir = os.path.join(args.output_path, 'canonical_v2')
    os.makedirs(out_dir, exist_ok=True)
    result = {
        'model': model_alias, 'protocol_version': dm.PROTOCOL_VERSION,
        'alpha_eligible_methods': list(ALPHA_ELIGIBLE_METHODS),
        'candidates': dm.VALIDATION_ALPHAS, 'benign_frr_max_increase_pp': dm.BENIGN_FRR_MAX_INCREASE_PP,
        'no_defence_benign_macro_frr': no_defence_macro_frr,
        'no_defence_harmful_macro_asr': baseline['harmful_macro_asr'],
        'no_defence_baseline_input_path': baseline_path,
        'results': results,
        'git_commit': dm.git_commit_hash(), 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    }
    result_path = os.path.join(out_dir, f'experiment3_alpha_freeze_{model_alias}.json')
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {result_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_idx', type=int, required=True)
    parser.add_argument('--output_path', type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    args = parser.parse_args()
    main(args)
