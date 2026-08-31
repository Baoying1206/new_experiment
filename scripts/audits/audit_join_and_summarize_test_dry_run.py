"""
Synthetic-data tests for scripts/50_join_and_summarize_test.py. Covers all
3 test-phase methods (fixed_wei, adaptive, no_defence). Uses REAL
data/splits.json test_ids for the no_defence harmful reused-completions
path (load_no_defence_harmful_reused cross-references it and asserts an
exact 200x6 count). No GPU, no torch.

Usage:
  python scripts/audits/audit_join_and_summarize_test_dry_run.py
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
test_join_mod = import_module('50_join_and_summarize_test')

SPLITS_PATH = os.path.join(SCRIPT_DIR, '..', '..', 'data', 'splits.json')


def make_gen_row(iid, template, boh, method, alpha, resp_suffix=''):
    return {
        'record_key': f'{iid}|{template}|{method}', 'model': 'FakeModel', 'split': 'test',
        'instruction_id': iid, 'benign_or_harmful': boh, 'template': template,
        'method': method, 'alpha': alpha, 'protocol_version': 'exp3_reduced_v1',
        'direction_config_hash': 'DCH', 'generation_config_hash': 'GCH',
        'response': f'resp {iid} {template}{resp_suffix}', 'generation_tokens': '1 2 3',
        'generation_length': 3, 'stop_reason': 'eos', 'prompt_token_count': 5,
        'instruction_en': f'instr {iid}',
    }


def make_judgement(key, request_harmful, refusal, response_harmful, parse_success=True):
    return {'judge_cache_key': key, 'raw_judge_output': 'fake', 'request_harmful': request_harmful,
            'refusal': refusal, 'response_harmful': response_harmful, 'parse_success': parse_success,
            'parse_error_reason': None}


def main():
    taxonomy = load_taxonomy_v2()
    active_mechanisms = taxonomy['active_mechanisms']
    assert len(active_mechanisms) == 6
    with open(SPLITS_PATH) as f:
        test_ids = sorted(json.load(f)['test_ids'])
    assert len(test_ids) == 200

    tmpdir = tempfile.mkdtemp()
    try:
        model_alias = 'FakeModel'
        out_dir = os.path.join(tmpdir, 'canonical_v2')
        os.makedirs(out_dir, exist_ok=True)
        model_dir = os.path.join(tmpdir, model_alias)
        os.makedirs(model_dir, exist_ok=True)

        # ---- fixed_wei: 1200 harmful (200 arbitrary ids) + 600 benign (100 arbitrary ids) ----
        harmful_ids_synth = [f'h{i}' for i in range(200)]
        benign_ids_synth = [f'b{i}' for i in range(100)]
        gen_rows = []
        for iid in harmful_ids_synth:
            for mech in active_mechanisms:
                gen_rows.append(make_gen_row(iid, mech, 'harmful', 'fixed_wei', 1.5))
        for iid in benign_ids_synth:
            for mech in active_mechanisms:
                gen_rows.append(make_gen_row(iid, mech, 'benign', 'fixed_wei', 1.5))
        gen_path = os.path.join(out_dir, f'experiment3_test_generations_{model_alias}_fixed_wei.jsonl')
        with open(gen_path, 'w') as f:
            for r in gen_rows:
                f.write(json.dumps(r) + '\n')

        judge_rows = []
        seen = set()
        for r in gen_rows:
            key = dm.judge_cache_key(r['instruction_en'], r['response'])
            if key in seen:
                continue
            seen.add(key)
            if r['benign_or_harmful'] == 'harmful':
                judge_rows.append(make_judgement(key, 1, 0, 1))  # ASR=1 everywhere -> macro_asr=1.0
            else:
                judge_rows.append(make_judgement(key, 0, 0, 0))  # never refuses -> macro_frr=0.0
        judge_path = os.path.join(out_dir, f'experiment3_test_judgements_{model_alias}_fixed_wei.jsonl')
        with open(judge_path, 'w') as f:
            for j in judge_rows:
                f.write(json.dumps(j) + '\n')

        # frozen config (needed by main(), not by load_and_join directly)
        with open(os.path.join(out_dir, 'experiment3_defence_frozen_config.json'), 'w') as f:
            json.dump({'per_model': {model_alias: {'alpha': {'fixed_wei': 1.5, 'adaptive': 0.25}}}}, f)

        join_result = test_join_mod.load_and_join(tmpdir, model_alias, 'fixed_wei', active_mechanisms)
        assert len(join_result['gen_rows']) == 1800
        assert len(join_result['joined_rows']) == 1800
        gen_checks = test_join_mod.preflight_generation_checks(join_result['gen_rows'])
        assert all(v.get('pass', True) for v in gen_checks.values()), gen_checks
        per_template = test_join_mod.summarize_per_template(join_result['joined_rows'])
        macro = test_join_mod.summarize_macro(per_template)
        assert abs(macro['macro_asr'] - 1.0) < 1e-9, macro
        assert abs(macro['macro_benign_frr'] - 0.0) < 1e-9, macro
        print("Test 1 PASSED: fixed_wei join produces 1800 rows (1200 harmful + 600 benign), "
              f"macro_asr={macro['macro_asr']}, macro_benign_frr={macro['macro_benign_frr']} exactly as constructed.")

        audit = test_join_mod.build_audit(model_alias, 'fixed_wei', join_result, gen_checks, 1.5)
        assert audit['overall_pass'] is True, audit
        print("Test 2 PASSED: build_audit reports overall_pass=True for a fully clean synthetic dataset.")

        # ---- no_defence: benign fresh gen (100x6) + harmful reused completions (REAL 200 test_ids x6) ----
        nd_benign_gen_rows = []
        for iid in benign_ids_synth:
            for mech in active_mechanisms:
                nd_benign_gen_rows.append(make_gen_row(iid, mech, 'benign', 'no_defence', None))
        nd_benign_gen_path = os.path.join(out_dir, f'experiment3_test_generations_{model_alias}_no_defence_benign.jsonl')
        with open(nd_benign_gen_path, 'w') as f:
            for r in nd_benign_gen_rows:
                f.write(json.dumps(r) + '\n')

        completions = []
        for hid in test_ids:
            for mech in active_mechanisms:
                completions.append({'id': hid, 'condition': mech, 'instruction_en': f'harmful instr {hid}',
                                     'response': f'harmful resp {hid} {mech}'})
        comp_path = os.path.join(model_dir, 'completions_en_full572_corrected.json')
        with open(comp_path, 'w') as f:
            json.dump(completions, f)

        nd_judge_rows = []
        seen2 = set()
        for r in nd_benign_gen_rows:
            key = dm.judge_cache_key(r['instruction_en'], r['response'])
            if key in seen2:
                continue
            seen2.add(key)
            nd_judge_rows.append(make_judgement(key, 0, 1, 0))  # always refuses -> macro_frr=1.0
        nd_benign_judge_path = os.path.join(out_dir, f'experiment3_test_judgements_{model_alias}_no_defence_benign.jsonl')
        with open(nd_benign_judge_path, 'w') as f:
            for j in nd_judge_rows:
                f.write(json.dumps(j) + '\n')

        nd_harmful_judge_rows = []
        seen3 = set()
        for c in completions:
            key = dm.judge_cache_key(c['instruction_en'], c['response'])
            if key in seen3:
                continue
            seen3.add(key)
            nd_harmful_judge_rows.append(make_judgement(key, 1, 1, 0))  # always refuses -> macro_asr=0.0
        nd_harmful_judge_path = os.path.join(out_dir, f'experiment3_test_judgements_{model_alias}_no_defence_harmful.jsonl')
        with open(nd_harmful_judge_path, 'w') as f:
            for j in nd_harmful_judge_rows:
                f.write(json.dumps(j) + '\n')

        nd_join = test_join_mod.load_and_join(tmpdir, model_alias, 'no_defence', active_mechanisms)
        assert len(nd_join['gen_rows']) == 1800, len(nd_join['gen_rows'])
        assert len(nd_join['joined_rows']) == 1800
        nd_gen_checks = test_join_mod.preflight_generation_checks(nd_join['gen_rows'])
        assert all(v.get('pass', True) for v in nd_gen_checks.values()), nd_gen_checks
        nd_per_template = test_join_mod.summarize_per_template(nd_join['joined_rows'])
        nd_macro = test_join_mod.summarize_macro(nd_per_template)
        assert abs(nd_macro['macro_asr'] - 0.0) < 1e-9, nd_macro
        assert abs(nd_macro['macro_benign_frr'] - 1.0) < 1e-9, nd_macro
        print("Test 3 PASSED: no_defence join combines fresh benign gen (600) + reused harmful "
              f"completions (1200, REAL test_ids) = 1800 rows, macro_asr={nd_macro['macro_asr']}, "
              f"macro_benign_frr={nd_macro['macro_benign_frr']} exactly as constructed.")

        nd_audit = test_join_mod.build_audit(model_alias, 'no_defence', nd_join, nd_gen_checks, None)
        assert nd_audit['overall_pass'] is True, nd_audit
        print("Test 4 PASSED: no_defence build_audit reports overall_pass=True.")

        # ---- Test 5: full main() end-to-end for fixed_wei ----
        import argparse
        orig_models = dict(test_join_mod.MODELS)
        test_join_mod.MODELS[999] = (model_alias, '/fake')
        try:
            test_join_mod.main(argparse.Namespace(model_idx=999, method='fixed_wei', output_path=tmpdir))
        finally:
            test_join_mod.MODELS.clear()
            test_join_mod.MODELS.update(orig_models)
        joined_path = os.path.join(out_dir, f'experiment3_test_joined_{model_alias}_fixed_wei.jsonl')
        summary_path = os.path.join(out_dir, f'experiment3_test_summary_{model_alias}_fixed_wei.json')
        audit_path = os.path.join(out_dir, f'experiment3_test_join_audit_{model_alias}_fixed_wei.json')
        assert os.path.exists(joined_path) and os.path.exists(summary_path) and os.path.exists(audit_path)
        with open(audit_path) as f:
            assert json.load(f)['overall_pass'] is True
        print("Test 5 PASSED: main() runs end-to-end for fixed_wei and writes all 3 output files "
              "with overall_pass=True.")

        print()
        print("ALL TEST-SPLIT JOIN/SUMMARIZE TESTS PASSED.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
