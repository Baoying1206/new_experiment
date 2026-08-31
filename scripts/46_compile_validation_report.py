"""
Pure aggregation, no new computation: reads the already-produced
experiment3_validation_join_audit_{model}_{method}.json,
experiment3_validation_summary_{model}_{method}.json (fixed_wei, adaptive),
experiment3_no_defence_baseline_{model}.json, and
experiment3_alpha_freeze_{model}.json for all 3 models, and packages them
into one consolidated report (JSON, machine-readable) plus a printed
markdown-style summary table. Refuses (raises) if any required input file
is missing or if a required overall_pass is not True -- never silently
reports from unverified data.

Also re-runs the static test-data-not-read check (grep-equivalent) and
embeds its result in the report.

Usage:
  python scripts/46_compile_validation_report.py --output_path output
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)
import _defence_metrics as dm  # torch-free

MODELS = dm.MODEL_PATHS
ALPHA_ELIGIBLE_METHODS = ('fixed_wei', 'adaptive')


class MissingInputError(RuntimeError):
    pass


def load_json(path):
    if not os.path.exists(path):
        raise MissingInputError(f"required input file missing: {path}")
    with open(path) as f:
        return json.load(f)


def check_test_data_not_read():
    """Static source-code check: no script in the validation pipeline may
    open() or json.load() test_ids/benign_test_100.json. BENIGN_TEST_PATH's
    own definition line is the only permitted occurrence anywhere."""
    targets = [
        '40_defence_generation_driver.py', '41_join_and_summarize_defence_validation.py',
        '44_summarize_no_defence_baseline.py', '45_freeze_alpha.py', '_defence_metrics.py',
    ]
    findings = {}
    for fname in targets:
        path = os.path.join(SCRIPT_DIR, fname)
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
        benign_test_path_lines = [i + 1 for i, l in enumerate(lines) if 'BENIGN_TEST_PATH' in l]
        bare_test_ids_lines = [i + 1 for i, l in enumerate(lines)
                                if re.search(r'\btest_ids\b', l) and 'validation_ids' not in l]
        findings[fname] = {
            'BENIGN_TEST_PATH_occurrences': benign_test_path_lines,
            'bare_test_ids_occurrences': bare_test_ids_lines,
        }
    ok = all(len(v['BENIGN_TEST_PATH_occurrences']) <= 1 and len(v['bare_test_ids_occurrences']) == 0
              for v in findings.values())
    return {'pass': ok, 'per_file': findings,
            'note': 'BENIGN_TEST_PATH may appear at most once (its own definition, never opened); '
                    'bare test_ids (not validation_ids) must never appear.'}


def collect_model(output_path, model_idx):
    model_alias, _ = MODELS[model_idx]
    out_dir = os.path.join(output_path, 'canonical_v2')

    baseline = load_json(os.path.join(out_dir, f'experiment3_no_defence_baseline_{model_alias}.json'))
    if not baseline.get('overall_pass'):
        raise MissingInputError(f"{model_alias} No-defence baseline has overall_pass="
                                 f"{baseline.get('overall_pass')!r}")

    alpha_freeze = load_json(os.path.join(out_dir, f'experiment3_alpha_freeze_{model_alias}.json'))

    methods = {}
    for method in ALPHA_ELIGIBLE_METHODS:
        audit = load_json(os.path.join(out_dir, f'experiment3_validation_join_audit_{model_alias}_{method}.json'))
        if not audit.get('overall_pass'):
            raise MissingInputError(f"{model_alias} x {method} join_audit has overall_pass="
                                     f"{audit.get('overall_pass')!r}")
        summary = load_json(os.path.join(out_dir, f'experiment3_validation_summary_{model_alias}_{method}.json'))
        methods[method] = {
            'join_audit_overall_pass': audit['overall_pass'],
            'generation_rows': audit['generation_rows'],
            'unique_judgement_rows': audit['unique_judgement_rows'],
            'cache_hit_rate': audit['cache_hit_rate'],
            'parse_failures': audit['parse_failures'],
            'direction_config_hash': audit['direction_config_hash'],
            'generation_config_hash': audit['generation_config_hash'],
            'per_alpha_macro': summary['per_alpha_macro'],
            'per_alpha_template': summary['per_alpha_template'],
            'frozen_alpha': alpha_freeze['results'][method]['frozen_alpha'],
            'frozen_alpha_reason': alpha_freeze['results'][method]['reason'],
            'frozen_alpha_max_allowed_frr': alpha_freeze['results'][method]['max_allowed_frr'],
            'macro_asr_at_frozen_alpha': alpha_freeze['results'][method]['macro_asr_at_frozen_alpha'],
            'macro_frr_at_frozen_alpha': alpha_freeze['results'][method]['macro_frr_at_frozen_alpha'],
        }

    return {
        'model': model_alias,
        'no_defence_benign_macro_frr': baseline['benign_macro_frr'],
        'no_defence_harmful_macro_asr': baseline['harmful_macro_asr'],
        'no_defence_benign_per_template': baseline['benign_per_template'],
        'no_defence_harmful_per_template': baseline['harmful_per_template'],
        'methods': methods,
    }


def main(args):
    per_model = {}
    total_new_generations = 0
    for model_idx in sorted(MODELS.keys()):
        model_alias, _ = MODELS[model_idx]
        per_model[model_alias] = collect_model(args.output_path, model_idx)
        for method in ALPHA_ELIGIBLE_METHODS:
            total_new_generations += per_model[model_alias]['methods'][method]['generation_rows']
        total_new_generations += 480  # no_defence_benign per model (not stored in join_audit schema)

    test_data_check = check_test_data_not_read()

    report = {
        'protocol_version': dm.PROTOCOL_VERSION,
        'primary_conditions': dm.PRIMARY_CONDITIONS,
        'supplementary_conditions': dm.SUPPLEMENTARY_CONDITIONS,
        'per_model': per_model,
        'total_new_target_generations_fixed_wei_adaptive_plus_no_defence_benign': total_new_generations,
        'expected_per_protocol_doc': 23328,
        'test_data_not_read_check': test_data_check,
        'git_commit': dm.git_commit_hash(), 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    }

    out_dir = os.path.join(args.output_path, 'canonical_v2')
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, 'experiment3_validation_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"protocol_version = {report['protocol_version']}")
    print(f"test_data_not_read_check pass = {test_data_check['pass']}")
    print(f"total_new_target_generations (fixed_wei+adaptive+no_defence_benign) = "
          f"{total_new_generations} (expected {report['expected_per_protocol_doc']})")
    print()
    for model_alias, m in per_model.items():
        print(f"=== {model_alias} ===")
        print(f"  No-defence: benign_macro_frr={m['no_defence_benign_macro_frr']:.4f}  "
              f"harmful_macro_asr={m['no_defence_harmful_macro_asr']:.4f}")
        for method, d in m['methods'].items():
            print(f"  {method}: frozen_alpha={d['frozen_alpha']}  reason={d['frozen_alpha_reason']}  "
                  f"macro_asr@alpha={d['macro_asr_at_frozen_alpha']:.4f}  "
                  f"macro_frr@alpha={d['macro_frr_at_frozen_alpha']:.4f}  "
                  f"max_allowed_frr={d['frozen_alpha_max_allowed_frr']:.4f}  "
                  f"parse_failures={d['parse_failures']}  cache_hit_rate={d['cache_hit_rate']:.4f}")
        print()

    print(f"Saved: {report_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    args = parser.parse_args()
    main(args)
