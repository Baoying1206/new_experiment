"""
Synthetic-data tests for scripts/44_summarize_no_defence_baseline.py.
Builds fake no_defence_benign generation+judgement files (arbitrary ids,
since this script never cross-references benign_validation_80.json) and a
fake completions_en_full572_corrected.json + no_defence_harmful judgement
file using REAL validation_ids/active_mechanisms (since
load_harmful_reused_records filters against the real data/splits.json and
asserts an exact 72x6 count). No GPU, no torch.

Usage:
  python scripts/audits/audit_no_defence_baseline_dry_run.py
"""
import json
import os
import shutil
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
import _defence_metrics as dm
from _taxonomy_v2_loader import load_taxonomy_v2
from importlib import import_module
baseline_mod = import_module('44_summarize_no_defence_baseline')

SPLITS_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'data', 'splits.json')


def make_judgement(key, request_harmful=1, refusal=0, response_harmful=1, parse_success=True, **passthrough):
    j = {'judge_cache_key': key, 'raw_judge_output': 'fake', 'request_harmful': request_harmful,
         'refusal': refusal, 'response_harmful': response_harmful, 'parse_success': parse_success,
         'parse_error_reason': None}
    j.update(passthrough)
    return j


def main():
    taxonomy = load_taxonomy_v2()
    active_mechanisms = taxonomy['active_mechanisms']
    assert len(active_mechanisms) == 6

    with open(SPLITS_PATH) as f:
        validation_ids = sorted(json.load(f)['validation_ids'])
    assert len(validation_ids) == 72

    tmpdir = tempfile.mkdtemp()
    try:
        model_alias = 'FakeModel'
        out_dir = os.path.join(tmpdir, 'canonical_v2')
        os.makedirs(out_dir, exist_ok=True)
        model_dir = os.path.join(tmpdir, model_alias)
        os.makedirs(model_dir, exist_ok=True)

        # ---- benign: 80 arbitrary ids x 6 real mechanisms = 480 ----
        benign_ids = [f'b{i}' for i in range(80)]
        benign_gen_rows = []
        for bid in benign_ids:
            for mech in active_mechanisms:
                benign_gen_rows.append({
                    'record_key': f'{bid}|{mech}', 'model': model_alias, 'split': 'validation',
                    'instruction_id': bid, 'benign_or_harmful': 'benign', 'template': mech,
                    'method': 'no_defence', 'alpha': None, 'protocol_version': 'exp3_reduced_v1',
                    'direction_config_hash': 'DCH', 'generation_config_hash': 'GCH',
                    'response': f'benign resp {bid} {mech}', 'generation_tokens': '1 2 3',
                    'generation_length': 3, 'stop_reason': 'eos', 'prompt_token_count': 5,
                    'instruction_en': f'benign instr {bid}',
                })
        benign_gen_path = os.path.join(out_dir, f'experiment3_validation_generations_{model_alias}_no_defence_benign.jsonl')
        with open(benign_gen_path, 'w', encoding='utf-8') as f:
            for r in benign_gen_rows:
                f.write(json.dumps(r) + '\n')

        # Refuse exactly 1/6 of templates deterministically to get a known macro FRR.
        benign_judge_rows = []
        seen_keys = set()
        for r in benign_gen_rows:
            key = dm.judge_cache_key(r['instruction_en'], r['response'])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            refusal = 1 if r['template'] == active_mechanisms[0] else 0
            benign_judge_rows.append(make_judgement(key, request_harmful=0, refusal=refusal, response_harmful=0))
        benign_judge_path = os.path.join(out_dir, f'experiment3_validation_judgements_{model_alias}_no_defence_benign.jsonl')
        with open(benign_judge_path, 'w', encoding='utf-8') as f:
            for j in benign_judge_rows:
                f.write(json.dumps(j) + '\n')

        # ---- harmful: real 72 validation_ids x 6 real mechanisms = 432, via fake completions file ----
        completions = []
        for hid in validation_ids:
            for mech in active_mechanisms:
                completions.append({
                    'id': hid, 'condition': mech, 'instruction_en': f'harmful instr {hid}',
                    'response': f'harmful resp {hid} {mech}',
                })
        comp_path = os.path.join(model_dir, 'completions_en_full572_corrected.json')
        with open(comp_path, 'w', encoding='utf-8') as f:
            json.dump(completions, f)

        # ASR = 1 for exactly one template, 0 for the rest -> known macro ASR = 1/6.
        harmful_judge_rows = []
        seen_keys = set()
        for c in completions:
            key = dm.judge_cache_key(c['instruction_en'], c['response'])
            if key in seen_keys:
                continue
            seen_keys.add(key)
            is_success_template = c['condition'] == active_mechanisms[0]
            harmful_judge_rows.append(make_judgement(
                key, request_harmful=1, refusal=0 if is_success_template else 1,
                response_harmful=1 if is_success_template else 0))
        harmful_judge_path = os.path.join(out_dir, f'experiment3_validation_judgements_{model_alias}_no_defence_harmful.jsonl')
        with open(harmful_judge_path, 'w', encoding='utf-8') as f:
            for j in harmful_judge_rows:
                f.write(json.dumps(j) + '\n')

        # ---- Test 1: benign join/macro FRR ----
        gen_path, judge_path, gen_rows, judge_rows = baseline_mod.load_benign_records(tmpdir, model_alias)
        assert len(gen_rows) == 480 and len(judge_rows) == 480, (len(gen_rows), len(judge_rows))
        benign_macro, benign_checks, benign_joined = baseline_mod.summarize_benign(gen_rows, judge_rows)
        assert len(benign_joined) == 480
        assert abs(benign_macro['macro_frr'] - (1 / 6)) < 1e-9, benign_macro['macro_frr']
        assert all(v.get('pass', True) for v in benign_checks.values())
        print(f"Test 1 PASSED: benign join produces 480 rows, macro_frr=1/6 exactly "
              f"(1 of 6 templates refuses 100%, the rest 0%): {benign_macro['macro_frr']}")

        # ---- Test 2: harmful join/macro ASR ----
        comp_path_out, harmful_records = baseline_mod.load_harmful_reused_records(tmpdir, model_alias, active_mechanisms)
        assert len(harmful_records) == 432
        harmful_judge_rows_loaded = dm.load_jsonl(harmful_judge_path)
        harmful_macro, harmful_checks, harmful_joined = baseline_mod.summarize_harmful(harmful_records, harmful_judge_rows_loaded)
        assert len(harmful_joined) == 432
        assert abs(harmful_macro['macro_asr'] - (1 / 6)) < 1e-9, harmful_macro['macro_asr']
        assert all(v.get('pass', True) for v in harmful_checks.values())
        print(f"Test 2 PASSED: harmful join produces 432 rows, macro_asr=1/6 exactly: {harmful_macro['macro_asr']}")

        # ---- Test 3: full main() end-to-end via subprocess-free direct call, check overall_pass + output file ----
        import argparse
        args = argparse.Namespace(model_idx=None, output_path=tmpdir)
        # main() uses MODELS[args.model_idx] to get model_alias -- patch by monkeypatching MODELS lookup
        # instead of relying on the real MODEL_PATHS dict (which has real aliases, not FakeModel).
        orig_models = dict(baseline_mod.MODELS)
        baseline_mod.MODELS[999] = (model_alias, '/fake/path')
        args.model_idx = 999
        try:
            baseline_mod.main(args)
        finally:
            baseline_mod.MODELS.clear()
            baseline_mod.MODELS.update(orig_models)

        result_path = os.path.join(out_dir, f'experiment3_no_defence_baseline_{model_alias}.json')
        with open(result_path) as f:
            result = json.load(f)
        assert result['overall_pass'] is True, result
        assert abs(result['benign_macro_frr'] - (1 / 6)) < 1e-9
        assert abs(result['harmful_macro_asr'] - (1 / 6)) < 1e-9
        print("Test 3 PASSED: main() runs end-to-end, writes a result file with overall_pass=True "
              "and the expected benign_macro_frr/harmful_macro_asr.")

        # ---- Test 4: wrong harmful record count raises (regression against silently accepting bad data) ----
        bad_completions = completions[:-1]  # drop one record -> 431, not 432
        bad_comp_path = os.path.join(model_dir, 'completions_en_full572_corrected.json')
        with open(bad_comp_path, 'w', encoding='utf-8') as f:
            json.dump(bad_completions, f)
        try:
            baseline_mod.load_harmful_reused_records(tmpdir, model_alias, active_mechanisms)
            raise SystemExit("FAILED: expected AssertionError for a wrong reused-record count")
        except AssertionError as e:
            assert 'expected' in str(e)
            print(f"Test 4 PASSED: a wrong reused-harmful-record count raises immediately: {str(e)[:70]}")
        finally:
            with open(comp_path, 'w', encoding='utf-8') as f:
                json.dump(completions, f)

        print()
        print("ALL NO-DEFENCE BASELINE TESTS PASSED.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
