"""
Exp3 TEST phase (one-shot, held-out). Reuses 40_defence_generation_driver.py's
already-proven infrastructure (direction construction from direction_ids,
prefill-last-token hook + chunking, run_judge, record_key/record_is_valid,
_build_generation_record) via direct import -- NOT reimplemented. Adds only
what's test-specific:

  - test-split prompt builders (test_ids / benign_test_100.json instead of
    validation_ids / benign_validation_80.json)
  - the single FROZEN alpha per (model, method), read ONLY from
    experiment3_defence_frozen_config.json -- there is no --alpha CLI flag
    here, by design, so an alpha can never be passed in by accident.
  - a hard stop (FrozenConfigViolationError, no generation/output written)
    if the runtime-computed direction_config_hash/generation_config_hash,
    the fixed layer, or the live Adaptive grouping disagree with what was
    frozen in experiment3_defence_frozen_config.json BEFORE any test data
    was read.

--method is restricted to {no_defence, fixed_wei, adaptive} -- global/
placebo are not test-phase methods (not in the choices list at all, so an
attempt to pass them fails at argparse before any code here runs).

Usage:
  python scripts/49_defence_test_driver.py --method fixed_wei --model_idx 1 \
      --stage generate --output_path output
  python scripts/49_defence_test_driver.py --method fixed_wei --model_idx 1 \
      --stage judge --output_path output
  python scripts/49_defence_test_driver.py --method no_defence --no_defence_target benign \
      --model_idx 1 --stage generate --output_path output
  python scripts/49_defence_test_driver.py --method no_defence --no_defence_target benign \
      --model_idx 1 --stage judge --output_path output
  python scripts/49_defence_test_driver.py --method no_defence --no_defence_target harmful_rejudge \
      --model_idx 1 --output_path output
"""
import argparse
import gc
import json
import os
import sys
from importlib import import_module

import torch

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)
drv = import_module('40_defence_generation_driver')
from _taxonomy_v2_loader import load_taxonomy_v2

MODEL_PATHS = drv.MODEL_PATHS
TEST_METHODS = ('fixed_wei', 'adaptive')  # no_defence handled by its own dedicated functions, as in validation


class FrozenConfigViolationError(RuntimeError):
    pass


def load_frozen_config(output_path):
    path = os.path.join(output_path, 'canonical_v2', 'experiment3_defence_frozen_config.json')
    with open(path) as f:
        return json.load(f)


def verify_against_frozen_config(frozen_config, model_alias, direction_config_hash, generation_config_hash):
    """Hard stop (check 8): the test driver may never proceed if anything
    that should have been frozen at the validation/test boundary has
    drifted. Checks hashes, fixed layer, AND the live Adaptive grouping --
    not just alpha (alpha is looked up from this same file, never
    independently selected, so there is no separate "alpha check" here)."""
    if model_alias not in frozen_config['per_model']:
        raise FrozenConfigViolationError(f"{model_alias} not present in frozen config -- refusing to proceed.")
    frozen = frozen_config['per_model'][model_alias]

    if direction_config_hash != frozen['direction_config_hash']:
        raise FrozenConfigViolationError(
            f"{model_alias}: runtime direction_config_hash={direction_config_hash!r} != "
            f"frozen {frozen['direction_config_hash']!r}. STOPPING -- no test output written."
        )
    if generation_config_hash != frozen['generation_config_hash']:
        raise FrozenConfigViolationError(
            f"{model_alias}: runtime generation_config_hash={generation_config_hash!r} != "
            f"frozen {frozen['generation_config_hash']!r}. STOPPING -- no test output written."
        )
    live_layer = drv.exp3_coverage.FIXED_LAYERS[model_alias]
    if live_layer != frozen['fixed_layer_0based']:
        raise FrozenConfigViolationError(
            f"{model_alias}: live fixed_layer={live_layer} != frozen {frozen['fixed_layer_0based']}. "
            f"STOPPING -- no test output written."
        )
    live_grouping = drv.hooks_mod.FROZEN_ADAPTIVE_GROUPING.get(model_alias)
    if live_grouping != frozen.get('adaptive_grouping'):
        raise FrozenConfigViolationError(
            f"{model_alias}: live Adaptive grouping != frozen grouping. STOPPING -- no test output written.\n"
            f"  live:   {live_grouping}\n  frozen: {frozen.get('adaptive_grouping')}"
        )


def frozen_alpha_for(frozen_config, model_alias, method):
    return frozen_config['per_model'][model_alias]['alpha'][method]


# ── test-split prompt builders (mirrors drv.build_validation_*_prompts) ──

def build_test_harmful_prompts(active_mechanisms):
    script02 = drv._import_by_path('script02', os.path.join(SCRIPT_DIR, '02_build_templated_data.py'))
    with open(drv.SPLITS_PATH) as f:
        test_ids = sorted(json.load(f)['test_ids'])
    with open(drv.POOL_PATH, encoding='utf-8') as f:
        pool = {p['id']: p for p in json.load(f)}
    with open(drv.TEMPLATES_PATH, encoding='utf-8') as f:
        templates = json.load(f)['templates']
    prompts = []
    for hid in test_ids:
        instruction_en = pool[hid]['instruction_en']
        for mech in active_mechanisms:
            rendered = script02.build_condition(mech, templates[mech]['text'], mech, instruction_en)
            prompts.append({'instruction_id': hid, 'benign_or_harmful': 'harmful', 'template': mech,
                             'instruction': rendered, 'instruction_en': instruction_en})
    assert len(prompts) == len(test_ids) * len(active_mechanisms) == 200 * 6, \
        f"expected 1200 harmful test prompts, got {len(prompts)}"
    return prompts, test_ids


def build_test_benign_prompts(active_mechanisms):
    script02 = drv._import_by_path('script02', os.path.join(SCRIPT_DIR, '02_build_templated_data.py'))
    with open(drv.BENIGN_TEST_PATH, encoding='utf-8') as f:
        benign_pool = json.load(f)
    with open(drv.TEMPLATES_PATH, encoding='utf-8') as f:
        templates = json.load(f)['templates']
    prompts = []
    for b in benign_pool:
        instruction_en = b['instruction']
        for mech in active_mechanisms:
            rendered = script02.build_condition(mech, templates[mech]['text'], mech, instruction_en)
            prompts.append({'instruction_id': b['benign_id'], 'benign_or_harmful': 'benign', 'template': mech,
                             'instruction': rendered, 'instruction_en': instruction_en})
    assert len(prompts) == len(benign_pool) * len(active_mechanisms) == 100 * 6, \
        f"expected 600 benign test prompts, got {len(prompts)}"
    return prompts, [b['benign_id'] for b in benign_pool]


# ── generation ─────────────────────────────────────────────────────────

def run_test_intervention_method(args):
    """One (model, method) job, method in {fixed_wei, adaptive}, SINGLE
    frozen alpha (no sweep) -- mirrors
    drv.run_validation_intervention_method's chunking/hook/resume pattern."""
    method = args.method
    assert method in TEST_METHODS, f"{method!r} must be one of {TEST_METHODS}"
    model_alias, model_path = MODEL_PATHS[args.model_idx]
    taxonomy = load_taxonomy_v2()
    active_mechanisms = taxonomy['active_mechanisms']
    fixed_layer = drv.exp3_coverage.FIXED_LAYERS[model_alias]

    frozen_config = load_frozen_config(args.output_path)
    alpha = frozen_alpha_for(frozen_config, model_alias, method)

    direction_config_hash, _ = drv.compute_direction_config_hash(model_alias)
    generation_config_hash, _ = drv.compute_generation_config_hash(model_path)
    verify_against_frozen_config(frozen_config, model_alias, direction_config_hash, generation_config_hash)
    print(f"=== Test: {model_alias} x {method} (FROZEN alpha={alpha}) ===")
    print(f"direction_config_hash={direction_config_hash[:16]}...  generation_config_hash="
          f"{generation_config_hash[:16]}...  -- VERIFIED against frozen config.")

    harmful_prompts, harmful_ids = build_test_harmful_prompts(active_mechanisms)
    benign_prompts, benign_ids = build_test_benign_prompts(active_mechanisms)
    all_prompts = harmful_prompts + benign_prompts
    print(f"{len(harmful_prompts)} harmful + {len(benign_prompts)} benign = {len(all_prompts)} target "
          f"records for this (model, method) at the single frozen alpha.")

    out_dir = os.path.join(args.output_path, 'canonical_v2')
    os.makedirs(out_dir, exist_ok=True)
    gen_path = os.path.join(out_dir, f'experiment3_test_generations_{model_alias}_{method}.jsonl')
    valid_keys, n_invalid, _ = drv.load_valid_existing_keys(gen_path)
    print(f"Resume: {len(valid_keys)} valid existing records, {n_invalid} invalid/truncated (will be regenerated).")

    todo = []
    for p in all_prompts:
        key = drv.record_key(model_alias, 'test', p['instruction_id'], p['benign_or_harmful'],
                              p['template'], method, alpha, drv.PROTOCOL_VERSION,
                              direction_config_hash, generation_config_hash)
        if key not in valid_keys:
            todo.append(p)
    print(f"{len(todo)} records remaining to generate.")
    if not todo:
        print("Nothing to do -- all records already valid.")
        return

    path = os.path.join(args.output_path, model_alias, 'paired_diffs_en_full572_corrected.pt')
    diffs_data = torch.load(path, map_location='cpu')
    diffs_data['diffs'] = {m: v.float() for m, v in diffs_data['diffs'].items()}
    id_index = {m: {pid: i for i, pid in enumerate(diffs_data['instruction_ids'][m])}
                for m in active_mechanisms + ['placebo']}
    with open(drv.SPLITS_PATH) as f:
        direction_ids = sorted(json.load(f)['direction_ids'])
    dtilde = {m: drv.exp3_coverage.direction_at_layer(diffs_data, id_index, m, direction_ids, fixed_layer,
                                                        'mean', 'placebo_calibrated') for m in active_mechanisms}
    dtilde['placebo'] = drv.exp3_coverage.direction_at_layer(diffs_data, id_index, 'placebo', direction_ids,
                                                               fixed_layer, 'mean', 'raw')
    conds = drv.hooks_mod.build_all_condition_directions(dtilde, model_alias)

    print("Loading model...")
    from pipeline.model_utils.model_factory import construct_model_base
    model_base = construct_model_base(model_path, lang='en')
    layer_module = model_base.model_block_modules[fixed_layer]
    eos_id = model_base.tokenizer.eos_token_id
    print(f"Model loaded. Hooking layer {fixed_layer} ({type(layer_module).__name__}).")

    batch_size = drv.CONDITION_BATCH_SIZE_OVERRIDE.get(model_alias, 60)
    n_written = 0
    for start in range(0, len(todo), batch_size):
        chunk_prompts = todo[start:start + batch_size]
        tok = model_base.tokenize_instructions_fn(instructions=[p['instruction'] for p in chunk_prompts], system=None)
        prompt_tok_counts = tok.attention_mask.sum(dim=1).tolist()

        per_row_c = torch.stack([
            alpha * drv.direction_vector_for(conds, method, p['template']) for p in chunk_prompts
        ], 0)
        audit_log = []
        hook_fn, state = drv.hooks_mod.make_prefill_last_token_hook(per_row_c, audit_log=audit_log)
        completions = model_base.generate_completions(
            chunk_prompts, fwd_pre_hooks=[], fwd_hooks=[(layer_module, hook_fn)],
            batch_size=len(chunk_prompts), max_new_tokens=drv.MAX_NEW_TOKENS,
        )
        drv.hooks_mod.assert_single_intervention(state)
        if audit_log:
            print(f"  WARNING at batch starting {start}: {audit_log}")

        batch_records = []
        for c, item, ptc in zip(completions, chunk_prompts, prompt_tok_counts):
            for k in ('instruction_id', 'benign_or_harmful', 'template'):
                c[k] = item[k]
            batch_records.append(drv._build_generation_record(
                c, item, eos_id, model_alias, 'test', method, alpha,
                direction_config_hash, generation_config_hash, ptc, 'primary'))
        drv.append_jsonl(gen_path, batch_records)
        n_written += len(batch_records)
        print(f"  batch {start}-{start + len(chunk_prompts)}: wrote {len(batch_records)} records "
              f"({n_written}/{len(todo)} this run)")

    print("\nFreeing target model GPU memory...")
    model_base.del_model()
    del model_base
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Done. Generations at: {gen_path}")


def run_test_no_defence_benign(args):
    """Generates the 100x6=600 benign test No-defence responses (no hook)."""
    model_alias, model_path = MODEL_PATHS[args.model_idx]
    taxonomy = load_taxonomy_v2()
    active_mechanisms = taxonomy['active_mechanisms']
    direction_config_hash, _ = drv.compute_direction_config_hash(model_alias)  # recorded for traceability, unused
    generation_config_hash, _ = drv.compute_generation_config_hash(model_path)

    benign_prompts, benign_ids = build_test_benign_prompts(active_mechanisms)
    print(f"=== Test No-defence benign: {model_alias} === ({len(benign_prompts)} prompts)")

    out_dir = os.path.join(args.output_path, 'canonical_v2')
    os.makedirs(out_dir, exist_ok=True)
    gen_path = os.path.join(out_dir, f'experiment3_test_generations_{model_alias}_no_defence_benign.jsonl')
    valid_keys, n_invalid, _ = drv.load_valid_existing_keys(gen_path)
    print(f"Resume: {len(valid_keys)} valid, {n_invalid} invalid.")

    todo = [p for p in benign_prompts if drv.record_key(
        model_alias, 'test', p['instruction_id'], p['benign_or_harmful'], p['template'],
        'no_defence', None, drv.PROTOCOL_VERSION, direction_config_hash, generation_config_hash) not in valid_keys]
    print(f"{len(todo)} remaining.")
    if not todo:
        print("Nothing to do.")
        return

    from pipeline.model_utils.model_factory import construct_model_base
    model_base = construct_model_base(model_path, lang='en')
    eos_id = model_base.tokenizer.eos_token_id
    batch_size = drv.CONDITION_BATCH_SIZE_OVERRIDE.get(model_alias, 60)

    for start in range(0, len(todo), batch_size):
        chunk = todo[start:start + batch_size]
        tok = model_base.tokenize_instructions_fn(instructions=[p['instruction'] for p in chunk], system=None)
        prompt_tok_counts = tok.attention_mask.sum(dim=1).tolist()
        completions = model_base.generate_completions(
            chunk, fwd_pre_hooks=[], fwd_hooks=[], batch_size=len(chunk), max_new_tokens=drv.MAX_NEW_TOKENS)
        batch_records = []
        for c, item, ptc in zip(completions, chunk, prompt_tok_counts):
            for k in ('instruction_id', 'benign_or_harmful', 'template'):
                c[k] = item[k]
            batch_records.append(drv._build_generation_record(
                c, item, eos_id, model_alias, 'test', 'no_defence', None,
                direction_config_hash, generation_config_hash, ptc, 'primary'))
        drv.append_jsonl(gen_path, batch_records)
        print(f"  wrote {len(batch_records)} records")

    model_base.del_model()
    del model_base
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Done. Generations at: {gen_path}")


def run_test_no_defence_harmful_rejudge(args):
    """No target-model generation -- reuses completions_en_full572_corrected.json's
    already-generated test_ids x 6-mechanism responses (generated without any
    steering) and runs ONLY the strict WildGuard judge on them, filtered to
    test_ids (not validation_ids)."""
    model_alias, _ = MODEL_PATHS[args.model_idx]
    taxonomy = load_taxonomy_v2()
    active_mechanisms = taxonomy['active_mechanisms']
    print(f"=== Test No-defence harmful rejudge: {model_alias} ===")

    comp_path = os.path.join(args.output_path, model_alias, 'completions_en_full572_corrected.json')
    with open(comp_path, encoding='utf-8') as f:
        data = json.load(f)
    completions = data['completions'] if isinstance(data, dict) and 'completions' in data else data

    with open(drv.SPLITS_PATH) as f:
        test_ids = set(json.load(f)['test_ids'])
    records = [c for c in completions if c['id'] in test_ids and c['condition'] in active_mechanisms]
    assert len(records) == len(test_ids) * len(active_mechanisms) == 200 * 6, \
        f"expected 1200 reused no-defence harmful test records, got {len(records)}"
    print(f"Found {len(records)} reused No-defence harmful responses to rejudge.")

    for r in records:
        r['instruction_id'], r['benign_or_harmful'], r['template'] = r['id'], 'harmful', r['condition']
        r['condition'] = r['method'] = 'no_defence'
        r['alpha'] = None
        r['protocol_version'] = drv.PROTOCOL_VERSION

    out_dir = os.path.join(args.output_path, 'canonical_v2')
    os.makedirs(out_dir, exist_ok=True)
    judge_path = os.path.join(out_dir, f'experiment3_test_judgements_{model_alias}_no_defence_harmful.jsonl')
    existing = drv.load_jsonl(judge_path)
    existing_cache_keys = {j['judge_cache_key'] for j in existing}

    script03 = drv._import_by_path('script03', os.path.join(SCRIPT_DIR, '03_generate_and_label.py'))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Loading WildGuard...")
    guard_tok = AutoTokenizer.from_pretrained('allenai/wildguard')
    guard_tok.padding_side = 'left'
    guard_model = AutoModelForCausalLM.from_pretrained('allenai/wildguard', torch_dtype=torch.bfloat16,
                                                        device_map='auto').eval()

    cache = {j['judge_cache_key']: j for j in existing}
    written = {'n': 0}

    def _persist(batch):
        new = [j for j in batch if j['judge_cache_key'] not in existing_cache_keys]
        if new:
            drv.append_jsonl(judge_path, new)
            existing_cache_keys.update(j['judge_cache_key'] for j in new)
            written['n'] += len(new)
            print(f"  wrote {len(new)} judgements ({written['n']} so far)")

    judgements, jmetrics, cache = drv.run_judge(records, guard_model, guard_tok, script03,
                                                 cache=cache, on_new_batch=_persist)
    print(f"Judge metrics: {jmetrics}. Wrote {written['n']} new judgements to {judge_path} (incremental).")


def _test_gen_and_judge_paths(args, model_alias):
    out_dir = os.path.join(args.output_path, 'canonical_v2')
    if args.method == 'no_defence':
        assert args.no_defence_target == 'benign', (
            "no_defence harmful judging is produced by run_test_no_defence_harmful_rejudge itself"
        )
        stem = f'{model_alias}_no_defence_benign'
    else:
        stem = f'{model_alias}_{args.method}'
    gen_path = os.path.join(out_dir, f'experiment3_test_generations_{stem}.jsonl')
    judge_path = os.path.join(out_dir, f'experiment3_test_judgements_{stem}.jsonl')
    return gen_path, judge_path


def run_test_judge(args):
    model_alias, _ = MODEL_PATHS[args.model_idx]
    gen_path, judge_path = _test_gen_and_judge_paths(args, model_alias)
    print(f"=== Test Judge: {model_alias} x {args.method}"
          f"{' (' + args.no_defence_target + ')' if args.method == 'no_defence' else ''} ===")

    all_rows = drv.load_jsonl(gen_path)
    records = [r for r in all_rows if drv.record_is_valid(r)]
    n_invalid = len(all_rows) - len(records)
    print(f"Loaded {len(records)} valid generation records from {gen_path} ({n_invalid} invalid/truncated skipped).")
    if not records:
        print("Nothing to judge.")
        return

    existing = drv.load_jsonl(judge_path)
    existing_cache_keys = {j['judge_cache_key'] for j in existing}
    cache = {j['judge_cache_key']: j for j in existing}

    script03 = drv._import_by_path('script03', os.path.join(SCRIPT_DIR, '03_generate_and_label.py'))
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("Loading WildGuard...")
    guard_tok = AutoTokenizer.from_pretrained('allenai/wildguard')
    guard_tok.padding_side = 'left'
    guard_model = AutoModelForCausalLM.from_pretrained('allenai/wildguard', torch_dtype=torch.bfloat16,
                                                        device_map='auto').eval()

    written = {'n': 0}

    def _persist(batch):
        new = [j for j in batch if j['judge_cache_key'] not in existing_cache_keys]
        if new:
            drv.append_jsonl(judge_path, new)
            existing_cache_keys.update(j['judge_cache_key'] for j in new)
            written['n'] += len(new)
            print(f"  wrote {len(new)} judgements ({written['n']} so far)")

    judgements, jmetrics, cache = drv.run_judge(records, guard_model, guard_tok, script03,
                                                 cache=cache, on_new_batch=_persist)
    print(f"Judge metrics: {jmetrics}. Wrote {written['n']} new judgements to {judge_path} (incremental).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_idx', type=int, required=True)
    parser.add_argument('--method', choices=['no_defence', 'fixed_wei', 'adaptive'], required=True)
    parser.add_argument('--output_path', type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    parser.add_argument('--no_defence_target', choices=['benign', 'harmful_rejudge'], default=None)
    parser.add_argument('--stage', choices=['generate', 'judge'], default='generate')
    args = parser.parse_args()

    if args.method == 'no_defence':
        assert args.no_defence_target in ('benign', 'harmful_rejudge'), \
            "--method no_defence requires --no_defence_target benign|harmful_rejudge"
        if args.no_defence_target == 'harmful_rejudge':
            run_test_no_defence_harmful_rejudge(args)
        elif args.stage == 'generate':
            run_test_no_defence_benign(args)
        else:
            run_test_judge(args)
    elif args.stage == 'generate':
        run_test_intervention_method(args)
    else:
        run_test_judge(args)


if __name__ == '__main__':
    main()
