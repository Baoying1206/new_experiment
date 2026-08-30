"""
Minimal REAL-GPU pilot for the Exp3 defence protocol -- NOT the full driver,
NOT a source of any result that will be reported. Purpose is narrowly:
  1. confirm the hook (scripts/37) actually fires exactly once per batch on a
     real model, at the real fixed layer, with real left-padded batching;
  2. get a real GPU name/count (sacct/sinfo gave nothing on this cluster --
     see output/canonical_v2/experiment3_throughput_pilot.json's warnings);
  3. get a real, ISOLATED target-generation-only throughput number and a
     separate, isolated WildGuard-only throughput number (the historical
     logs fuse both into one timestamp, per prior audit);
  4. eyeball 3-5 real completions with vs without the Global direction
     applied, as a sanity check -- not evidence of defence efficacy.

Scope, deliberately tiny: ONE model (Meta-Llama-3.1-8B-Instruct), N_PILOT
(default 4) real validation_ids harmful instructions, ONE template
(persona_roleplay), ONE condition (Global), ONE alpha (1.0). No alpha sweep,
no benign data, no other models, no other conditions -- if this breaks, it
should break cheap and be easy to diagnose.

Requires the cluster environment (pipeline/ package, real model weights,
real paired_diffs_en_full572_corrected.pt for Llama) -- NOT locally runnable.

Usage:
  python scripts/39_defence_pilot.py --model_path /path/to/Meta-Llama-3.1-8B-Instruct
"""
import argparse
import json
import os
import subprocess
import sys
import time

import torch

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)
from importlib import import_module
exp3 = import_module('35_common_direction_coverage_audit')
hooks_mod = import_module('37_defence_directions_and_hooks')
from _taxonomy_v2_loader import load_taxonomy_v2

MODEL_ALIAS = 'Meta-Llama-3.1-8B-Instruct'
FIXED_LAYER = exp3.FIXED_LAYERS[MODEL_ALIAS]
N_PILOT = 4
TEMPLATE_MECH = 'persona_roleplay'
ALPHA = 1.0
MAX_NEW_TOKENS = 200

SPLITS_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'splits.json')
POOL_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'sampled_prompts.json')
TEMPLATES_PATH = os.path.join(SCRIPT_DIR, '..', 'templates', 'templates_en.json')


def print_gpu_info():
    print("=== nvidia-smi ===")
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used', '--format=csv'],
            capture_output=True, text=True, timeout=15,
        )
        print(out.stdout.strip() or f"(empty stdout; stderr={out.stderr.strip()})")
    except Exception as e:
        print(f"nvidia-smi failed: {e}")
    print()


def load_direction_data(output_dir):
    path = os.path.join(output_dir, MODEL_ALIAS, 'paired_diffs_en_full572_corrected.pt')
    print(f"Loading {path} ...")
    diffs_data = torch.load(path, map_location='cpu')
    diffs_data['diffs'] = {m: v.float() for m, v in diffs_data['diffs'].items()}
    taxonomy = load_taxonomy_v2()
    active_mechs = taxonomy['active_mechanisms']
    id_index = {m: {pid: i for i, pid in enumerate(diffs_data['instruction_ids'][m])}
                for m in active_mechs + ['placebo']}
    with open(SPLITS_PATH) as f:
        direction_ids = sorted(json.load(f)['direction_ids'])

    dtilde = {}
    for m in active_mechs:
        dtilde[m] = exp3.direction_at_layer(diffs_data, id_index, m, direction_ids, FIXED_LAYER,
                                             'mean', 'placebo_calibrated')
    dtilde['placebo'] = exp3.direction_at_layer(diffs_data, id_index, 'placebo', direction_ids, FIXED_LAYER,
                                                 'mean', 'raw')
    print(f"Built dtilde for {len(dtilde)} mechanisms at layer {FIXED_LAYER}. "
          f"Norms: {{m: round(v.norm().item(),3) for m,v in dtilde.items()}}")
    for m, v in dtilde.items():
        print(f"  {m}: norm={v.norm().item():.4f}")
    return dtilde, taxonomy


def build_pilot_prompts():
    with open(SPLITS_PATH) as f:
        validation_ids = sorted(json.load(f)['validation_ids'])[:N_PILOT]
    with open(POOL_PATH, encoding='utf-8') as f:
        pool = {p['id']: p for p in json.load(f)}
    with open(TEMPLATES_PATH, encoding='utf-8') as f:
        templates = json.load(f)['templates']
    template_text = templates[TEMPLATE_MECH]['text']

    rows = []
    for pid in validation_ids:
        instruction_en = pool[pid]['instruction_en']
        rendered = template_text.format(instruction=instruction_en)
        rows.append({'id': pid, 'instruction': rendered, 'instruction_en': instruction_en,
                      'condition': TEMPLATE_MECH})
    print(f"Built {len(rows)} pilot prompts from validation_ids: {[r['id'] for r in rows]}")
    return rows


def main(args):
    print_gpu_info()

    dtilde, taxonomy = load_direction_data(args.output_dir)
    conds = hooks_mod.build_all_condition_directions(dtilde, MODEL_ALIAS)
    c_G = conds['global']['*']
    print(f"Global c_G norm = {c_G.norm().item():.4f}, alpha={ALPHA}")

    dataset = build_pilot_prompts()

    print("\nLoading model (this is the expensive step)...")
    t0 = time.time()
    from pipeline.model_utils.model_factory import construct_model_base
    model_base = construct_model_base(args.model_path, lang='en')
    t_load = time.time() - t0
    print(f"Model load time: {t_load:.1f}s")

    layer_module = model_base.model_block_modules[FIXED_LAYER]
    print(f"Hooking layer module: {type(layer_module).__name__} at index {FIXED_LAYER}")

    batch_size = len(dataset)
    per_row_c = (ALPHA * c_G).unsqueeze(0).repeat(batch_size, 1)

    # ---- with-hook (Global) generation ----
    audit_log = []
    hook_fn, state = hooks_mod.make_prefill_last_token_hook(per_row_c, audit_log=audit_log)
    print("\nGenerating WITH Global direction hook...")
    t0 = time.time()
    completions_hooked = model_base.generate_completions(
        dataset, fwd_pre_hooks=[], fwd_hooks=[(layer_module, hook_fn)],
        batch_size=batch_size, max_new_tokens=MAX_NEW_TOKENS,
    )
    t_gen_hooked = time.time() - t0
    print(f"With-hook generation time: {t_gen_hooked:.1f}s for {len(completions_hooked)} rows "
          f"({len(completions_hooked)/t_gen_hooked*3600:.0f} rows/hour)")
    print(f"Hook state: {state}")
    print(f"Audit log (should be empty): {audit_log}")
    try:
        hooks_mod.assert_single_intervention(state)
        hook_check_passed = True
        print("assert_single_intervention PASSED: intervention_count == 1 for this batch.")
    except AssertionError as e:
        hook_check_passed = False
        print(f"assert_single_intervention FAILED: {e}")
        print("*** This pilot run's hooked completions should be treated as INVALID. ***")

    # ---- no-defence generation (same prompts, no hook) ----
    print("\nGenerating WITHOUT any hook (No-defence baseline)...")
    t0 = time.time()
    completions_nodefence = model_base.generate_completions(
        dataset, fwd_pre_hooks=[], fwd_hooks=[],
        batch_size=batch_size, max_new_tokens=MAX_NEW_TOKENS,
    )
    t_gen_nodefence = time.time() - t0
    print(f"No-defence generation time: {t_gen_nodefence:.1f}s for {len(completions_nodefence)} rows "
          f"({len(completions_nodefence)/t_gen_nodefence*3600:.0f} rows/hour)")

    print("\nFreeing target model GPU memory...")
    model_base.del_model()
    del model_base
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # ---- WildGuard, isolated timing ----
    # Reuse 03_generate_and_label.py's exact WILDGUARD_PROMPT/_parse_wildguard --
    # do NOT hand-roll a different prompt template, to stay consistent with the
    # unified judge that will be used for the real defence evaluation.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'script03', os.path.join(SCRIPT_DIR, '03_generate_and_label.py'))
    script03 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script03)

    print("\nLoading WildGuard...")
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    guard_tok = AutoTokenizer.from_pretrained('allenai/wildguard')
    guard_tok.padding_side = 'left'
    guard_model = AutoModelForCausalLM.from_pretrained(
        'allenai/wildguard', torch_dtype=torch.bfloat16, device_map='auto',
    ).eval()
    t_wg_load = time.time() - t0
    print(f"WildGuard load time: {t_wg_load:.1f}s")

    def judge(completions, label):
        texts = [script03.WILDGUARD_PROMPT.format(prompt=c['instruction_en'], response=c['response'])
                 for c in completions]
        enc = guard_tok(texts, return_tensors='pt', padding=True, truncation=True,
                        max_length=2048, add_special_tokens=False).to(guard_model.device)
        t0 = time.time()
        with torch.no_grad():
            out = guard_model.generate(**enc, max_new_tokens=32, pad_token_id=guard_tok.eos_token_id)
        t_judge = time.time() - t0
        print(f"[{label}] WildGuard judge time: {t_judge:.2f}s for {len(completions)} rows "
              f"({len(completions)/t_judge*3600:.0f} rows/hour, isolated)")
        for i, c in enumerate(completions):
            new_ids = out[i][enc.input_ids.shape[-1]:]
            raw = guard_tok.decode(new_ids, skip_special_tokens=True)
            parsed = script03._parse_wildguard(raw)
            print(f"  [{label}] id={c['id']}  raw_judge_output={raw!r}  parsed={parsed}")
        return t_judge

    t_wg_hooked = judge(completions_hooked, 'with-hook')
    t_wg_nodefence = judge(completions_nodefence, 'no-defence')

    print("\n=== Sample outputs (manual eyeball only, not a result) ===")
    for c_h, c_n in zip(completions_hooked, completions_nodefence):
        print(f"--- id={c_h['id']} ---")
        print(f"  [with-hook]   {c_h['response'][:200]!r}")
        print(f"  [no-defence]  {c_n['response'][:200]!r}")

    report = {
        'model_alias': MODEL_ALIAS, 'fixed_layer': FIXED_LAYER, 'alpha': ALPHA,
        'n_pilot': N_PILOT, 'template_mech': TEMPLATE_MECH, 'max_new_tokens': MAX_NEW_TOKENS,
        'model_load_time_s': t_load,
        'target_gen_with_hook_time_s': t_gen_hooked, 'target_gen_with_hook_rows_per_hour': len(completions_hooked) / t_gen_hooked * 3600,
        'target_gen_no_defence_time_s': t_gen_nodefence, 'target_gen_no_defence_rows_per_hour': len(completions_nodefence) / t_gen_nodefence * 3600,
        'wildguard_load_time_s': t_wg_load,
        'wildguard_judge_with_hook_time_s': t_wg_hooked, 'wildguard_judge_no_defence_time_s': t_wg_nodefence,
        'hook_state': state, 'hook_audit_log': audit_log, 'hook_check_passed': hook_check_passed,
    }
    out_dir = os.path.join(args.output_dir, 'canonical_v2')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'experiment3_throughput_pilot.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    args = parser.parse_args()
    main(args)
