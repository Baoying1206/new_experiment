"""
Leave-one-category-out robustness check for the two core findings:
  1. Resource-tier continuum: cos(HL) < cos(LM)
  2. Real templates > placebo (sign test across the 6 tier-pair comparisons)

For each of the 8 coarse harm categories, excludes all prompts in that
category and recomputes template_direction + the two core comparisons using
only the remaining ~65-68 prompts. If a finding survives every single
category exclusion, it isn't being driven by one dominant harm topic.

Activation extraction happens once (GPU-heavy part); the 8 leave-one-out
recomputations are pure CPU-side averaging over the already-extracted
per-prompt activations, so this doesn't cost 8x the GPU time.

Requires the same pipeline/ + GPU environment as 03/04/06.

Usage:
  python scripts/07_leave_one_category_out.py \
      --model_path /path/to/Qwen2.5-7B-Instruct \
      --model_alias Qwen2.5-7B-Instruct \
      --output_dir output \
      --batch_size 8
"""
import argparse
import json
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import add_hooks  # verify this import path on the cluster

SCRIPT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
PILOT_LANGS = ['en', 'zh', 'de', 'ko', 'ar', 'th', 'yo', 'sw', 'am']
REAL_MECHS = ['prefix_injection', 'refusal_suppression', 'instruction_hierarchy',
              'persona_roleplay', 'fictional_framing', 'encoding_obfuscation']
TIERS = {'en': 'H', 'zh': 'H', 'de': 'H', 'ko': 'M', 'ar': 'M', 'th': 'M',
         'yo': 'L', 'sw': 'L', 'am': 'L'}


# ── Activation extraction (same pattern as 04/06) ──────────────────────────────

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
    """Returns acts_by_id_cond: {id: {condition: act_tensor}}."""
    with open(completions_path, encoding='utf-8') as f:
        completions = json.load(f)['completions']
    instructions = [c['instruction'] for c in completions]
    all_acts = get_all_activations(model_base, instructions, batch_size)
    acts_by_id_cond = {}
    id_category = {}
    for i, c in enumerate(completions):
        acts_by_id_cond.setdefault(c['id'], {})[c['condition']] = all_acts[i]
        id_category[c['id']] = c['category']
    return acts_by_id_cond, id_category


def template_direction(acts_by_id_cond, mechanism, exclude_ids):
    diffs = []
    for pid, by_cond in acts_by_id_cond.items():
        if pid in exclude_ids:
            continue
        if mechanism in by_cond and 'plain' in by_cond:
            diffs.append(by_cond[mechanism] - by_cond['plain'])
    if not diffs:
        return None
    return torch.stack(diffs, dim=0).mean(0)  # [n_layers, d_model]


def cos_sim(a, b):
    return F.cosine_similarity(a.float(), b.float(), dim=-1).mean().item()


# ── Main ────────────────────────────────────────────────────────────────────────

def main(args):
    model_alias = args.model_alias or os.path.basename(args.model_path)
    out_dir = os.path.join(args.output_dir, model_alias)

    print("Loading model...")
    model_base = construct_model_base(args.model_path, lang='en')
    print(f"  Loaded: {model_alias}\n")

    lang_acts = {}
    lang_id_category = {}
    for lang in PILOT_LANGS:
        completions_path = os.path.join(out_dir, f'completions_{lang}.json')
        if not os.path.exists(completions_path):
            print(f"[{lang}] Missing, skipping.")
            continue
        print(f"[{lang}] Extracting activations...")
        acts, id_cat = extract_lang_data(model_base, completions_path, args.batch_size)
        lang_acts[lang] = acts
        lang_id_category[lang] = id_cat
        torch.cuda.empty_cache()

    all_categories = sorted(set(next(iter(lang_id_category.values())).values()))
    print(f"\nCategories found: {all_categories}\n")

    def compute_core_findings(exclude_category):
        """Returns (hl_mean, lm_mean, real_vs_placebo_wins_out_of_6)."""
        # template_direction per (lang, mechanism), excluding prompts in exclude_category
        directions = {}
        for lang in lang_acts:
            exclude_ids = {pid for pid, cat in lang_id_category[lang].items() if cat == exclude_category}
            directions[lang] = {}
            for mech in REAL_MECHS + ['placebo']:
                d = template_direction(lang_acts[lang], mech, exclude_ids)
                if d is not None:
                    directions[lang][mech] = d

        tier_pair_vals = {mech: {} for mech in REAL_MECHS + ['placebo']}
        langs = list(directions.keys())
        for i, l1 in enumerate(langs):
            for l2 in langs[i + 1:]:
                tp = ''.join(sorted([TIERS[l1], TIERS[l2]]))
                for mech in REAL_MECHS + ['placebo']:
                    if mech in directions[l1] and mech in directions[l2]:
                        sim = cos_sim(directions[l1][mech], directions[l2][mech])
                        tier_pair_vals[mech].setdefault(tp, []).append(sim)

        def pooled(mech_list, tp):
            vals = []
            for mech in mech_list:
                vals.extend(tier_pair_vals[mech].get(tp, []))
            return sum(vals) / len(vals) if vals else float('nan')

        hl = pooled(REAL_MECHS, 'HL')
        lm = pooled(REAL_MECHS, 'LM')

        wins = 0
        for tp in ['HH', 'MM', 'LL', 'HM', 'HL', 'LM']:
            real = pooled(REAL_MECHS, tp)
            placebo = pooled(['placebo'], tp)
            if real > placebo:
                wins += 1

        return hl, lm, wins

    print("=== Baseline (no exclusion) ===")
    hl0, lm0, wins0 = compute_core_findings(exclude_category=None)
    print(f"  HL={hl0:.3f}  LM={lm0:.3f}  continuum_holds={hl0 < lm0}  real>placebo_wins={wins0}/6\n")

    results = {'model': model_alias, 'baseline': {'HL': hl0, 'LM': lm0, 'continuum_holds': hl0 < lm0, 'wins': wins0},
               'leave_one_out': {}}

    print("=== Leave-one-category-out ===")
    for cat in all_categories:
        hl, lm, wins = compute_core_findings(exclude_category=cat)
        continuum_holds = hl < lm
        print(f"  [exclude {cat}] HL={hl:.3f}  LM={lm:.3f}  continuum_holds={continuum_holds}  real>placebo_wins={wins}/6")
        results['leave_one_out'][cat] = {'HL': hl, 'LM': lm, 'continuum_holds': continuum_holds, 'wins': wins}

    out_path = os.path.join(out_dir, 'leave_one_category_out.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
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
