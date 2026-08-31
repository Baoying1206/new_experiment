"""
Builds experiment3_defence_frozen_config.json from the already-produced
validation-phase artifacts (experiment3_alpha_freeze_{model}.json,
experiment3_validation_join_audit_{model}_{method}.json) -- pure
aggregation, computes nothing new. This is the SINGLE file the test-phase
driver is allowed to read alpha/hash/grouping from; it must never
reselect alpha, direction, layer, or grouping itself.

Hard-asserts (refuses to write, raises immediately) if:
  - any frozen alpha does not match the exact expected values given by the
    user's test-phase authorization,
  - direction_config_hash/generation_config_hash differ between fixed_wei
    and adaptive for the same model (they must be identical -- same model,
    same layer, same direction-construction data),
  - the embedded Adaptive grouping (a static copy, cross-checked against
    scripts/37_defence_directions_and_hooks.py's FROZEN_ADAPTIVE_GROUPING
    via git log showing zero modifying commits since introduction, commit
    6d03822) does not match what's on disk right now.

Usage:
  python scripts/47_freeze_test_config.py --output_path output
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

EXPECTED_ALPHA = {
    'Qwen2.5-7B-Instruct': {'fixed_wei': 1.5, 'adaptive': 1.5},
    'Meta-Llama-3.1-8B-Instruct': {'fixed_wei': 0.5, 'adaptive': 1.5},
    'gemma-2-9b-it': {'fixed_wei': 0.5, 'adaptive': 0.25},
}

FIXED_LAYERS = {'Qwen2.5-7B-Instruct': 16, 'Meta-Llama-3.1-8B-Instruct': 19, 'gemma-2-9b-it': 25}

# Static copy of scripts/37_defence_directions_and_hooks.py's FROZEN_ADAPTIVE_GROUPING.
# Not imported directly (that module pulls torch, unwanted in this CPU-only script).
# Integrity is instead verified by `git log --oneline -- scripts/37_defence_directions_and_hooks.py`
# showing exactly one commit (6d03822, the commit that introduced it) -- i.e. it has never
# been modified since being frozen. See EXPERIMENT3_PROTOCOL.md section 2.
EXPECTED_ADAPTIVE_GROUPING = {
    'Qwen2.5-7B-Instruct': {
        'template_specific': ['prefix_injection'],
        'subgroups': {
            'CO_reduced': ['refusal_suppression', 'persona_roleplay'],
            'MG_full': ['encoding_obfuscation', 'payload_splitting', 'distractors_negated'],
        },
    },
    'Meta-Llama-3.1-8B-Instruct': {
        'template_specific': ['refusal_suppression', 'distractors_negated'],
        'subgroups': {
            'CO_reduced': ['prefix_injection', 'persona_roleplay'],
            'MG_reduced': ['encoding_obfuscation', 'payload_splitting'],
        },
    },
    'gemma-2-9b-it': {
        'template_specific': ['prefix_injection'],
        'subgroups': {
            'CO_reduced': ['refusal_suppression', 'persona_roleplay'],
            'MG_full': ['encoding_obfuscation', 'payload_splitting', 'distractors_negated'],
        },
    },
}


class FrozenConfigMismatchError(RuntimeError):
    pass


def verify_adaptive_grouping_unmodified():
    """Confirms scripts/37_defence_directions_and_hooks.py has exactly one
    commit in its git history (the commit that introduced FROZEN_ADAPTIVE_GROUPING) --
    i.e. it was never touched after being frozen. Also textually extracts the
    live dict from the file and compares it against EXPECTED_ADAPTIVE_GROUPING
    above (belt-and-suspenders: catches an uncommitted, unstaged edit that a
    pure git-log check would miss)."""
    import subprocess
    path = os.path.join(SCRIPT_DIR, '37_defence_directions_and_hooks.py')
    log = subprocess.run(['git', 'log', '--oneline', '--', path], cwd=SCRIPT_DIR,
                          capture_output=True, text=True, check=True).stdout.strip().splitlines()
    if len(log) != 1:
        raise FrozenConfigMismatchError(
            f"scripts/37_defence_directions_and_hooks.py has {len(log)} commits in its git "
            f"history (expected exactly 1, the commit that froze FROZEN_ADAPTIVE_GROUPING) -- "
            f"it may have been modified since freezing. Log: {log}"
        )

    with open(path, encoding='utf-8') as f:
        source = f.read()
    match = re.search(r'FROZEN_ADAPTIVE_GROUPING = (\{.*?\n\})\n', source, re.DOTALL)
    if not match:
        raise FrozenConfigMismatchError("could not locate FROZEN_ADAPTIVE_GROUPING in the live source file")
    live_dict = eval(match.group(1), {'__builtins__': {}})  # static Python-literal dict, no arbitrary code
    for model, expected in EXPECTED_ADAPTIVE_GROUPING.items():
        if live_dict.get(model) != expected:
            raise FrozenConfigMismatchError(
                f"live FROZEN_ADAPTIVE_GROUPING[{model!r}] does not match the expected frozen "
                f"value.\n  live:     {live_dict.get(model)}\n  expected: {expected}"
            )
    return log[0]


def main(args):
    out_dir = os.path.join(args.output_path, 'canonical_v2')
    per_model = {}
    for model_idx in sorted(MODELS.keys()):
        model_alias, _ = MODELS[model_idx]
        alpha_freeze = json.load(open(os.path.join(out_dir, f'experiment3_alpha_freeze_{model_alias}.json')))

        hashes = {}
        for method in ALPHA_ELIGIBLE_METHODS:
            audit = json.load(open(os.path.join(
                out_dir, f'experiment3_validation_join_audit_{model_alias}_{method}.json')))
            frozen_alpha = alpha_freeze['results'][method]['frozen_alpha']
            expected_alpha = EXPECTED_ALPHA[model_alias][method]
            if frozen_alpha != expected_alpha:
                raise FrozenConfigMismatchError(
                    f"{model_alias} x {method}: frozen alpha={frozen_alpha} does not match the "
                    f"authorized expected value {expected_alpha}. Refusing to write frozen config."
                )
            hashes[method] = (audit['direction_config_hash'], audit['generation_config_hash'])

        if hashes['fixed_wei'] != hashes['adaptive']:
            raise FrozenConfigMismatchError(
                f"{model_alias}: direction/generation_config_hash differ between fixed_wei and "
                f"adaptive ({hashes['fixed_wei']} vs {hashes['adaptive']}) -- these must be "
                f"identical (same model, same fixed layer, same direction data)."
            )
        direction_config_hash, generation_config_hash = hashes['fixed_wei']

        per_model[model_alias] = {
            'fixed_layer_0based': FIXED_LAYERS[model_alias],
            'direction_config_hash': direction_config_hash,
            'generation_config_hash': generation_config_hash,
            'alpha': {method: alpha_freeze['results'][method]['frozen_alpha'] for method in ALPHA_ELIGIBLE_METHODS},
            'alpha_freeze_reason': {method: alpha_freeze['results'][method]['reason']
                                     for method in ALPHA_ELIGIBLE_METHODS},
            'adaptive_grouping': EXPECTED_ADAPTIVE_GROUPING.get(model_alias),  # Gemma: no template_specific override needed at this dict (see note)
        }

    adaptive_grouping_commit = verify_adaptive_grouping_unmodified()

    config = {
        'protocol_version': dm.PROTOCOL_VERSION,
        'primary_conditions': ['no_defence', 'fixed_wei', 'adaptive'],
        'supplementary_conditions': dm.SUPPLEMENTARY_CONDITIONS,
        'per_model': per_model,
        'adaptive_grouping_source_commit': adaptive_grouping_commit,
        'test_split': {'harmful_test_ids_count': 200, 'benign_test_100_count': 100, 'n_templates': 6},
        'git_commit_at_freeze': dm.git_commit_hash(),
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    }

    os.makedirs(out_dir, exist_ok=True)
    config_path = os.path.join(out_dir, 'experiment3_defence_frozen_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print("All alpha values match the authorized expected values.")
    print("direction_config_hash/generation_config_hash consistent between fixed_wei and adaptive, per model.")
    print(f"Adaptive grouping verified unmodified since commit {adaptive_grouping_commit}.")
    for model_alias, m in per_model.items():
        print(f"  {model_alias}: layer={m['fixed_layer_0based']}  alpha={m['alpha']}")
    print(f"\nSaved: {config_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    args = parser.parse_args()
    main(args)
