"""
Does the geometric shift magnitude (||template_direction||) predict the actual
behavioral jailbreak effect (ΔASR = bypass_rate[mechanism] - bypass_rate[plain])?

This is the step that turns "we found representation shifts that correlate
across languages" into "these shifts are tied to actual safety failure, not
just template-recognition geometry" -- the validity gap flagged a while back:
a template can produce a very stable, cross-lingually-shared direction while
having zero effect on whether the model actually complies.

Method: for each (language, mechanism), compute ||template_direction|| (L2
norm, averaged across layers) from a fresh activation extraction (04 doesn't
persist the raw per-prompt diffs, only the final cosine-similarity stats).
Pair each with the already-computed ΔASR from pilot_results.json's
bypass_rates section. Report Pearson and Spearman correlation across all
(language, mechanism) pairs, plus the same broken down by mechanism and by
language.

Requires the same pipeline/ + GPU environment as 03/04/06/07.

Usage:
  python scripts/08_magnitude_vs_behavior.py \
      --model_path /path/to/Qwen2.5-7B-Instruct \
      --model_alias Qwen2.5-7B-Instruct \
      --output_dir output \
      --batch_size 8
"""
import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import add_hooks  # verify this import path on the cluster

SCRIPT_DIR = os.path.dirname(__file__)
PILOT_LANGS = ['en', 'zh', 'de', 'ko', 'ar', 'th', 'yo', 'sw', 'am']
REAL_MECHS = ['prefix_injection', 'refusal_suppression', 'instruction_hierarchy',
              'persona_roleplay', 'fictional_framing', 'encoding_obfuscation']


# ── Activation extraction (same pattern as 04/06/07) ───────────────────────────

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
    return all_acts


def extract_lang_data(model_base, completions_path, batch_size):
    with open(completions_path, encoding='utf-8') as f:
        completions = json.load(f)['completions']
    instructions = [c['instruction'] for c in completions]
    all_acts = get_all_activations(model_base, instructions, batch_size)
    acts_by_id_cond = {}
    for i, c in enumerate(completions):
        acts_by_id_cond.setdefault(c['id'], {})[c['condition']] = all_acts[i]
    return acts_by_id_cond


def template_direction_norm(acts_by_id_cond, mechanism):
    diffs = []
    for pid, by_cond in acts_by_id_cond.items():
        if mechanism in by_cond and 'plain' in by_cond:
            diffs.append(by_cond[mechanism] - by_cond['plain'])
    if not diffs:
        return None
    direction = torch.stack(diffs, dim=0).mean(0)  # [n_layers, d_model]
    return direction.float().norm(dim=-1).mean().item()  # avg L2 norm across layers


# ── Correlation (no scipy dependency assumed -- implement Pearson/Spearman manually) ──

def pearson(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = sum((x - mx) ** 2 for x in xs) ** 0.5
    sy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (sx * sy) if sx > 0 and sy > 0 else float('nan')


def spearman(xs, ys):
    def rank(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0] * len(vals)
        for r, i in enumerate(order):
            ranks[i] = r
        return ranks
    return pearson(rank(xs), rank(ys))


# ── Main ────────────────────────────────────────────────────────────────────────

def main(args):
    model_alias = args.model_alias or os.path.basename(args.model_path)
    out_dir = os.path.join(args.output_dir, model_alias)

    with open(os.path.join(out_dir, 'pilot_results.json')) as f:
        pilot = json.load(f)
    bypass_rates = pilot['bypass_rates']  # {lang: {condition: rate}}

    print("Loading model...")
    model_base = construct_model_base(args.model_path, lang='en')
    print(f"  Loaded: {model_alias}\n")

    records = []  # each: {lang, mechanism, magnitude, delta_asr}
    for lang in PILOT_LANGS:
        completions_path = os.path.join(out_dir, f'completions_{lang}.json')
        if not os.path.exists(completions_path):
            continue
        print(f"[{lang}] Extracting activations...")
        acts = extract_lang_data(model_base, completions_path, args.batch_size)

        plain_rate = bypass_rates.get(lang, {}).get('plain')
        if plain_rate is None:
            continue
        for mech in REAL_MECHS:
            mag = template_direction_norm(acts, mech)
            mech_rate = bypass_rates.get(lang, {}).get(mech)
            if mag is None or mech_rate is None:
                continue
            delta_asr = mech_rate - plain_rate
            records.append({'lang': lang, 'mechanism': mech, 'magnitude': mag, 'delta_asr': delta_asr})
            print(f"  [{lang}][{mech}] ||direction||={mag:.2f}  ΔASR={delta_asr:+.3f}")
        torch.cuda.empty_cache()

    xs = [r['magnitude'] for r in records]
    ys = [r['delta_asr'] for r in records]
    overall_pearson = pearson(xs, ys)
    overall_spearman = spearman(xs, ys)
    print(f"\n=== Overall (n={len(records)}) ===")
    print(f"  Pearson r = {overall_pearson:.3f}")
    print(f"  Spearman rho = {overall_spearman:.3f}")

    by_mech = {}
    for mech in REAL_MECHS:
        sub = [r for r in records if r['mechanism'] == mech]
        if len(sub) >= 3:
            r_p = pearson([s['magnitude'] for s in sub], [s['delta_asr'] for s in sub])
            by_mech[mech] = r_p
            print(f"  [{mech}] Pearson r (across {len(sub)} languages) = {r_p:.3f}")

    by_lang = {}
    for lang in PILOT_LANGS:
        sub = [r for r in records if r['lang'] == lang]
        if len(sub) >= 3:
            r_p = pearson([s['magnitude'] for s in sub], [s['delta_asr'] for s in sub])
            by_lang[lang] = r_p
            print(f"  [{lang}] Pearson r (across {len(sub)} mechanisms) = {r_p:.3f}")

    out_path = os.path.join(out_dir, 'magnitude_vs_behavior.json')
    with open(out_path, 'w') as f:
        json.dump({
            'model': model_alias,
            'records': records,
            'overall_pearson': overall_pearson,
            'overall_spearman': overall_spearman,
            'by_mechanism_pearson': by_mech,
            'by_language_pearson': by_lang,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',  type=str, required=True)
    parser.add_argument('--model_alias', type=str, default=None)
    parser.add_argument('--output_dir',  type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    parser.add_argument('--batch_size',  type=int, default=8)
    args = parser.parse_args()
    if args.model_alias is None:
        args.model_alias = os.path.basename(args.model_path)
    main(args)
