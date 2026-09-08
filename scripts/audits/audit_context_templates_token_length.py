"""
Lexical/token-length static audit of templates/templates_context_v1.json's
16 template texts, using each of the 3 project models' own tokenizer.

TOKENIZER ONLY -- never loads model weights, never runs a forward pass.
Uses AutoTokenizer.from_pretrained(path, local_files_only=True), which
raises immediately (no network fallback, no silent download) if the
tokenizer is not already present locally at the given path. If that
happens, this script stops with a clear error -- it never attempts to
download anything itself.

Measures the WRAPPING PART only: each template with the literal
'{instruction}' placeholder removed (replaced with ''), tokenized with
add_special_tokens=False so the count reflects pure wrapper-text content,
not each model's constant chat-template special-token overhead (which
would just shift every number by the same constant and add noise to the
cross-family/cross-variant comparisons this script cares about).

Descriptive statistics only -- per EXPERIMENT2_CONTEXT_RECONFIGURATION_PROTOCOL.md,
no template is rewritten here, and no hard pass/fail threshold is applied.
The 1.5x-IQR "notably different" flag is the standard, pre-existing
statistical convention (not chosen after seeing this data to reach a
particular conclusion) and is a descriptive label only, not a filter --
nothing is excluded or modified based on it.

Never reads test_ids. Never writes over an existing file -- refuses if
--write_report's target already exists, and only accepts paths under
output/audits/context/.

Usage:
  python scripts/audits/audit_context_templates_token_length.py \
      --model_paths /home/h24/baga0553/models/Qwen2.5-7B-Instruct,/home/h24/baga0553/models/Llama-3.1-8B-Instruct,/home/h24/baga0553/models/gemma-2-9b-it \
      --model_aliases Qwen2.5-7B-Instruct,Meta-Llama-3.1-8B-Instruct,gemma-2-9b-it \
      --write_report output/audits/context/context_templates_token_length_audit.json
"""
import argparse
import datetime
import hashlib
import json
import os
import platform
import statistics
import sys

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
from audit_context_templates_dry_run import load_context_templates, all_template_strings, TEMPLATES_PATH

DEFAULT_REPORT_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'output', 'audits', 'context')


def wrapper_text(template):
    """The template with the {instruction} placeholder removed -- isolates
    the wrapping context's own token contribution from the (variable-length,
    per-instruction) content that would otherwise dominate the count."""
    return template.replace('{instruction}', '')


def count_tokens(tokenizer, text):
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def iqr_flag(value, group_values):
    """Standard 1.5x-IQR convention -- descriptive flag only, not a filter,
    nothing is excluded based on it. Uses statistics.quantiles() (stdlib,
    n=4, exclusive method) rather than a hand-rolled index formula, which
    is unreliable at small n. Each FAMILY only has 3 positive variants --
    too few for a meaningful IQR on its own -- so this is applied to the
    POOLED set of all 12 positive variants (across all 4 families) per
    model, not to each family's 3 values in isolation. Returns False
    (unflagged) if group_values has fewer than 4 points."""
    if len(group_values) < 4:
        return False
    q1, _, q3 = statistics.quantiles(group_values, n=4)
    iqr = q3 - q1
    return value < (q1 - 1.5 * iqr) or value > (q3 + 1.5 * iqr)


def main(args):
    model_paths = args.model_paths.split(',')
    model_aliases = args.model_aliases.split(',')
    assert len(model_paths) == len(model_aliases), (
        f"--model_paths ({len(model_paths)}) and --model_aliases ({len(model_aliases)}) "
        f"must have the same count"
    )

    data = load_context_templates()
    all_texts = list(all_template_strings(data))  # 16 (family, vkey, text) tuples

    from transformers import AutoTokenizer

    per_model = {}
    for path, alias in zip(model_paths, model_aliases):
        print(f"Loading tokenizer ONLY (local_files_only=True, no model weights, no forward pass): "
              f"{alias} <- {path}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        except Exception as e:
            print(f"\nSTOPPING: could not load tokenizer for {alias} from {path} without a network "
                  f"fetch (local_files_only=True raised). Not attempting to download. Error: "
                  f"{type(e).__name__}: {e}")
            sys.exit(1)
        print(f"  OK -- tokenizer loaded for {alias}.\n")

        lengths = {}
        for fam_name, vkey, text in all_texts:
            n_tok = count_tokens(tokenizer, wrapper_text(text))
            lengths[(fam_name, vkey)] = n_tok
        per_model[alias] = lengths

        # ---- per-model result-integrity checks (16 total, 12 positive + 4
        # neutral, no duplicate template_id, no NaN/missing value) ----
        template_ids = [
            f"{fam}_{vkey}" if vkey != 'family_specific_neutral_control' else f"{fam}_neutral"
            for fam, vkey in lengths.keys()
        ]
        assert len(lengths) == 16, f"{alias}: expected 16 template lengths, got {len(lengths)}"
        assert len(template_ids) == len(set(template_ids)), f"{alias}: duplicate template_id detected"
        n_pos = sum(1 for fam, vkey in lengths.keys() if vkey != 'family_specific_neutral_control')
        n_neu = sum(1 for fam, vkey in lengths.keys() if vkey == 'family_specific_neutral_control')
        assert n_pos == 12 and n_neu == 4, f"{alias}: expected 12 positive + 4 neutral, got {n_pos}+{n_neu}"
        assert all(isinstance(v, int) and v >= 0 for v in lengths.values()), \
            f"{alias}: found a non-integer or negative token length (NaN/missing surrogate)"
        print(f"  Integrity check OK: 16/16 templates tokenized (12 positive + 4 neutral), "
              f"{len(set(template_ids))} unique template_ids, no NaN/missing values.")

    # ---- per-family, per-model descriptive stats ----
    report = {'result_status': 'STATIC_AUDIT_NON_RESULT',
              'note': 'Descriptive token-length statistics for the 16 template WRAPPERS '
                      '({instruction} removed) only. No hard threshold; 1.5x-IQR flags are '
                      'descriptive labels only, not exclusions. No template was modified.',
              'models': model_aliases, 'per_model_family_stats': {}, 'cross_family_avg': {}}

    for alias in model_aliases:
        lengths = per_model[alias]
        print(f"\n=== {alias} ===")
        fam_avgs = {}
        # Pooled across all 4 families' 3 positive variants (n=12) -- a
        # single family's own 3 values are too few for a meaningful IQR;
        # this asks "is this variant notably different from the other 11
        # positive variants overall", not "from its own 2 family-mates".
        all_pos_lengths_this_model = [
            lengths[(fam, f'v{i}')] for fam in data['families'] for i in (1, 2, 3)
        ]
        for fam_name in sorted(data['families'].keys()):
            pos_lengths = [lengths[(fam_name, f'v{i}')] for i in (1, 2, 3)]
            neutral_len = lengths[(fam_name, 'family_specific_neutral_control')]
            diffs = [p - neutral_len for p in pos_lengths]
            rng = max(pos_lengths) - min(pos_lengths)
            std = statistics.stdev(pos_lengths) if len(pos_lengths) > 1 else 0.0
            pos_mean = statistics.mean(pos_lengths)
            fam_avgs[fam_name] = statistics.mean(pos_lengths + [neutral_len])

            print(f"  {fam_name}:")
            for i, (p, d) in enumerate(zip(pos_lengths, diffs), start=1):
                flag = (' [FLAG: notably different from the pooled 12-variant distribution, '
                         '1.5xIQR convention]') if iqr_flag(p, all_pos_lengths_this_model) else ''
                print(f"    v{i}: {p} tokens  (diff vs neutral: {d:+d}){flag}")
            print(f"    neutral: {neutral_len} tokens")
            print(f"    within-family positive range={rng}  stdev={std:.2f}")

            all_positives_longer_than_neutral = all(d > 0 for d in diffs)

            report['per_model_family_stats'].setdefault(alias, {})[fam_name] = {
                'positive_token_lengths': dict(zip(['v1', 'v2', 'v3'], pos_lengths)),
                'neutral_token_length': neutral_len,
                'diff_vs_neutral': dict(zip(['v1', 'v2', 'v3'], diffs)),
                'positive_only_mean': pos_mean,
                'within_family_range': rng,
                'within_family_stdev': std,
                'all_positives_longer_than_neutral': all_positives_longer_than_neutral,
                'pooled_iqr_flag': {
                    f'v{i}': iqr_flag(p, all_pos_lengths_this_model)
                    for i, p in enumerate(pos_lengths, start=1)
                },
            }

        cross_fam_vals = list(fam_avgs.values())
        cross_fam_range = max(cross_fam_vals) - min(cross_fam_vals)
        print(f"\n  cross-family average wrapper length: "
              f"{ {k: round(v, 1) for k, v in fam_avgs.items()} }")
        print(f"  cross-family range: {cross_fam_range:.1f}")
        report['cross_family_avg'][alias] = {
            'per_family_avg': fam_avgs, 'cross_family_range': cross_fam_range,
        }

    # ---- cross-model outlier agreement: for each of the 12 positive
    # variants, do all 3 models' pooled_iqr_flag agree? ----
    outlier_agreement = {}
    disagreements = []
    for fam_name in sorted(data['families'].keys()):
        for i in (1, 2, 3):
            key = f"{fam_name}_v{i}"
            flags = {
                alias: report['per_model_family_stats'][alias][fam_name]['pooled_iqr_flag'][f'v{i}']
                for alias in model_aliases
            }
            agree = len(set(flags.values())) == 1
            outlier_agreement[key] = {'flags_by_model': flags, 'all_models_agree': agree}
            if not agree:
                disagreements.append(key)
    report['outlier_agreement'] = outlier_agreement
    print(f"\nCross-model outlier agreement: {len(disagreements)}/12 positive variants have disagreeing "
          f"pooled_iqr_flag across the 3 tokenizers"
          + (f" ({disagreements})" if disagreements else " -- all 3 tokenizers agree on every variant."))

    # ---- provenance metadata ----
    with open(TEMPLATES_PATH, 'rb') as f:
        source_template_sha256 = hashlib.sha256(f.read()).hexdigest()
    import transformers as _transformers
    report['source_template_path'] = 'templates/templates_context_v1.json'
    report['source_template_sha256'] = source_template_sha256
    report['tokenizer_paths'] = dict(zip(model_aliases, model_paths))
    report['python_version'] = platform.python_version()
    report['transformers_version'] = _transformers.__version__
    report['generated_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    print(f"\nsource_template_sha256={source_template_sha256}")
    print(f"python_version={report['python_version']}  transformers_version={report['transformers_version']}")

    if args.write_report:
        if os.path.exists(args.write_report):
            print(f"\nRefusing to write: {args.write_report} already exists.")
            sys.exit(1)
        report_dir = os.path.dirname(os.path.abspath(args.write_report))
        expected_dir = os.path.abspath(DEFAULT_REPORT_DIR)
        if report_dir != expected_dir:
            print(f"\nRefusing to write outside {DEFAULT_REPORT_DIR}: got {args.write_report}")
            sys.exit(1)
        os.makedirs(report_dir, exist_ok=True)
        with open(args.write_report, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote report (new file, non-overwriting): {args.write_report}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_paths', type=str, required=True)
    parser.add_argument('--model_aliases', type=str, required=True)
    parser.add_argument('--write_report', type=str, default=None)
    args = parser.parse_args()
    main(args)
