"""
Core analysis script for the jailbreak-template pilot.

For each language, extracts per-sample per-layer prompt activations (last token,
no generation -- consistent with jailbreak_vector/refusal_direction elsewhere in
this project) for all 8 conditions (plain + 6 templates + placebo), then:

1. Computes a paired template_direction per (language, mechanism):
       mean_over_matched_prompt_ids( act(templated) - act(plain) )
   Paired-by-prompt-id, so no category-balancing is needed here (unlike
   jailbreak_vector) -- every sampled prompt contributes to every mechanism's
   estimate under the exact same content.

2. Same-mechanism, cross-language cosine similarity:
       cos( template_direction[mech][langA], template_direction[mech][langB] )
   -- tests whether a given jailbreak mechanism's activation-space signature is
   language-invariant.

3. Same-language, cross-mechanism cosine similarity:
       cos( template_direction[mechA][lang], template_direction[mechB][lang] )
   -- replicates/extends Kirch et al.'s "transfer is attack-family-specific"
   finding to this project's data.

4. Bootstrap (resample the 75 prompt ids with replacement, 1000 iters) gives a
   percentile CI on every cosine similarity above, since a bare point estimate
   from ~75 paired prompts isn't enough to tell a real difference from noise.

5. Bypass-rate uplift per condition vs plain, per language -- in particular
   placebo vs the 6 real templates, to distinguish "this language shares the
   jailbreak mechanism, just weaker" from "this language's safety behavior is
   fragile to any perturbation, real or not."

Requires pipeline/ + GPU, same as the rest of experiment_thesis/scripts/. NOT
runnable on a laptop without the model. Run 03_generate_and_label.py for all
three languages first.

Usage:
  python scripts/04_extract_directions_and_analyze.py \
      --model_path /path/to/Qwen2.5-7B-Instruct \
      --model_alias Qwen2.5-7B-Instruct \
      --output_dir output \
      --batch_size 8 \
      --n_bootstrap 1000
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import add_hooks  # verify this import path on the cluster

SCRIPT_DIR = os.path.dirname(__file__)
PILOT_LANGS = ['en', 'zh', 'de', 'ko', 'ar', 'th', 'yo', 'sw', 'am']
MECHANISMS = ['prefix_injection', 'refusal_suppression', 'instruction_hierarchy',
              'persona_roleplay', 'fictional_framing', 'encoding_obfuscation', 'placebo']


# ── Activation extraction (per-sample, not averaged; same pattern as train_jailbreak_probes.py) ──

def get_activations_pre_hook(layer, cache, position, offset):
    def hook_fn(module, input):
        activation = input[0]
        cache[offset:offset + activation.shape[0], layer, :] = (
            activation[:, position, :].detach().to(cache.dtype).cpu()
        )
    return hook_fn


def get_all_activations(model_base, instructions, batch_size, position=-1):
    model = model_base.model
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    n_samples = len(instructions)
    all_acts = torch.zeros((n_samples, n_layers, d_model), dtype=torch.float32)

    for i in tqdm(range(0, n_samples, batch_size), desc="extracting activations"):
        batch = instructions[i:i + batch_size]
        fwd_pre_hooks = [
            (model_base.model_block_modules[layer],
             get_activations_pre_hook(layer=layer, cache=all_acts, position=position, offset=i))
            for layer in range(n_layers)
        ]
        inputs = model_base.tokenize_instructions_fn(instructions=batch)
        with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
            with torch.no_grad():
                model(input_ids=inputs.input_ids.to(model.device),
                      attention_mask=inputs.attention_mask.to(model.device))
    return all_acts  # [n_samples, n_layers, d_model]


# ── Per-language extraction ────────────────────────────────────────────────────

def extract_lang_data(model_base, completions_path, batch_size):
    """Returns acts_by_id_cond: {id: {condition: act_tensor[n_layers, d_model]}},
    plus bypass_by_id_cond: {id: {condition: 0/1}}."""
    with open(completions_path, encoding='utf-8') as f:
        completions = json.load(f)['completions']

    instructions = [c['instruction'] for c in completions]
    all_acts = get_all_activations(model_base, instructions, batch_size)  # [n, n_layers, d_model]

    acts_by_id_cond = {}
    bypass_by_id_cond = {}
    for i, c in enumerate(completions):
        acts_by_id_cond.setdefault(c['id'], {})[c['condition']] = all_acts[i]
        bypass_by_id_cond.setdefault(c['id'], {})[c['condition']] = (
            1 if c.get('wildguard', {}).get('refusal', 1) == 0 else 0
        )
    return acts_by_id_cond, bypass_by_id_cond


# ── Paired template_direction with bootstrap ───────────────────────────────────

def paired_diffs(acts_by_id_cond, mechanism):
    """[n_prompts, n_layers, d_model] of act(templated) - act(plain), only ids with both."""
    diffs = []
    for pid, by_cond in acts_by_id_cond.items():
        if mechanism in by_cond and 'plain' in by_cond:
            diffs.append(by_cond[mechanism] - by_cond['plain'])
    if not diffs:
        return None
    return torch.stack(diffs, dim=0)  # [n_prompts, n_layers, d_model]


def bootstrap_cos_ci(diffs_a, diffs_b, n_bootstrap=1000, seed=0):
    """
    diffs_a, diffs_b: [n_a, n_layers, d_model] and [n_b, n_layers, d_model] paired-diff
    samples (not necessarily same n or same prompt ids -- e.g. cross-language).
    Returns per-layer {point, ci_low, ci_high} from resampling each side independently.
    """
    rng = random.Random(seed)
    n_a, n_layers, _ = diffs_a.shape
    n_b = diffs_b.shape[0]

    point = F.cosine_similarity(diffs_a.mean(0).float(), diffs_b.mean(0).float(), dim=-1)  # [n_layers]

    boot_sims = np.zeros((n_bootstrap, n_layers))
    for b in range(n_bootstrap):
        idx_a = [rng.randrange(n_a) for _ in range(n_a)]
        idx_b = [rng.randrange(n_b) for _ in range(n_b)]
        mean_a = diffs_a[idx_a].mean(0).float()
        mean_b = diffs_b[idx_b].mean(0).float()
        boot_sims[b] = F.cosine_similarity(mean_a, mean_b, dim=-1).numpy()

    ci_low = np.percentile(boot_sims, 2.5, axis=0)
    ci_high = np.percentile(boot_sims, 97.5, axis=0)
    return {
        'point': point.tolist(),
        'ci_low': ci_low.tolist(),
        'ci_high': ci_high.tolist(),
        'avg_point': float(point.mean()),
        'avg_ci_low': float(ci_low.mean()),
        'avg_ci_high': float(ci_high.mean()),
    }


# ── Main ────────────────────────────────────────────────────────────────────────

def main(args):
    model_alias = args.model_alias or os.path.basename(args.model_path)
    out_dir = os.path.join(args.output_dir, model_alias)
    os.makedirs(out_dir, exist_ok=True)

    print("Loading model...")
    model_base = construct_model_base(args.model_path, lang='en')
    print(f"  Loaded: {model_alias}\n")

    lang_acts = {}
    lang_bypass = {}
    for lang in PILOT_LANGS:
        completions_path = os.path.join(out_dir, f'completions_{lang}.json')
        if not os.path.exists(completions_path):
            raise FileNotFoundError(f"Missing {completions_path}; run 03_generate_and_label.py for {lang} first.")
        print(f"[{lang}] Extracting activations...")
        acts_by_id_cond, bypass_by_id_cond = extract_lang_data(model_base, completions_path, args.batch_size)
        lang_acts[lang] = acts_by_id_cond
        lang_bypass[lang] = bypass_by_id_cond
        torch.cuda.empty_cache()

    # Paired diffs per (lang, mechanism)
    diffs = {}  # diffs[lang][mechanism] = [n_prompts, n_layers, d_model]
    for lang in PILOT_LANGS:
        diffs[lang] = {}
        for mech in MECHANISMS:
            d = paired_diffs(lang_acts[lang], mech)
            if d is not None:
                diffs[lang][mech] = d
                print(f"[{lang}][{mech}] n_paired_prompts={d.shape[0]}")

    results = {'model': model_alias, 'langs': PILOT_LANGS, 'mechanisms': MECHANISMS}

    # 1. Same-mechanism, cross-language
    print("\nSame-mechanism cross-language cosine similarity...")
    cross_lang = {}
    for mech in MECHANISMS:
        for i, l1 in enumerate(PILOT_LANGS):
            for l2 in PILOT_LANGS[i + 1:]:
                if mech not in diffs.get(l1, {}) or mech not in diffs.get(l2, {}):
                    continue
                key = f'{mech}__{l1}_vs_{l2}'
                res = bootstrap_cos_ci(diffs[l1][mech], diffs[l2][mech], args.n_bootstrap)
                cross_lang[key] = res
                print(f"  [{mech}] {l1} vs {l2}: avg_cos={res['avg_point']:.3f} "
                      f"[{res['avg_ci_low']:.3f}, {res['avg_ci_high']:.3f}]")
    results['same_mechanism_cross_language'] = cross_lang

    # 2. Same-language, cross-mechanism
    print("\nSame-language cross-mechanism cosine similarity...")
    cross_mech = {}
    real_mechs = [m for m in MECHANISMS if m != 'placebo']
    for lang in PILOT_LANGS:
        for i, m1 in enumerate(real_mechs):
            for m2 in real_mechs[i + 1:]:
                if m1 not in diffs.get(lang, {}) or m2 not in diffs.get(lang, {}):
                    continue
                key = f'{lang}__{m1}_vs_{m2}'
                res = bootstrap_cos_ci(diffs[lang][m1], diffs[lang][m2], args.n_bootstrap)
                cross_mech[key] = res
                print(f"  [{lang}] {m1} vs {m2}: avg_cos={res['avg_point']:.3f} "
                      f"[{res['avg_ci_low']:.3f}, {res['avg_ci_high']:.3f}]")
    results['same_language_cross_mechanism'] = cross_mech

    # 3. Bypass-rate uplift per condition vs plain, per language
    print("\nBypass-rate uplift vs plain baseline...")
    bypass_rates = {}
    for lang in PILOT_LANGS:
        by_cond = {}
        for pid, conds in lang_bypass[lang].items():
            for cond, val in conds.items():
                by_cond.setdefault(cond, []).append(val)
        rates = {cond: float(np.mean(vals)) for cond, vals in by_cond.items()}
        bypass_rates[lang] = rates
        plain_rate = rates.get('plain', float('nan'))
        print(f"  [{lang}] plain={plain_rate:.2f}", end="  ")
        for mech in MECHANISMS:
            if mech in rates:
                print(f"{mech}={rates[mech]:.2f}(+{rates[mech]-plain_rate:+.2f})", end="  ")
        print()
    results['bypass_rates'] = bypass_rates

    out_path = os.path.join(out_dir, 'pilot_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',   type=str, required=True)
    parser.add_argument('--model_alias',  type=str, default=None)
    parser.add_argument('--output_dir',   type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    parser.add_argument('--batch_size',   type=int, default=8)
    parser.add_argument('--n_bootstrap',  type=int, default=1000)
    args = parser.parse_args()
    if args.model_alias is None:
        args.model_alias = os.path.basename(args.model_path)
    main(args)
