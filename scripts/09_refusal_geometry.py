"""
Geometric relationship between template_direction and the original experiment's
refusal_direction / harmfulness_direction.

Motivation: Wei et al. 2023 split jailbreak failure modes into two families --
"competing objectives" (the model recognizes the request is harmful but is
pushed to comply anyway, e.g. prefix_injection, refusal_suppression,
instruction_hierarchy) vs "mismatched generalization" (the model's harm
detection itself is bypassed, e.g. persona_roleplay, fictional_framing,
encoding_obfuscation). That taxonomy makes a testable geometric prediction:

  - competing_objectives templates should push activations substantially
    along refusal_direction (they work BY suppressing the refusal signal),
    so cos(template_direction, refusal_direction) should be relatively large
    in magnitude.
  - mismatched_generalization templates should act more on harm *recognition*
    than on the refusal response itself, so they should align more with
    harmfulness_direction than with refusal_direction, and leave more of
    template_direction orthogonal to refusal_direction.

refusal_direction / harmfulness_direction come from experiment_thesis's
extract_jailbreak_vectors.py (mean(harmful) - mean(harmless) resp.
mean(bypassed) - mean(harmless), last-token position, same model_block_modules
hook convention as this project's extraction code) -- so cosine comparison
against this project's template_direction is apples-to-apples as long as both
were computed on the same model checkpoint.

NOTE: as of the last sync, experiment_thesis's saved refusal_dir_{lang}.pt /
harmfulness_dir_{lang}.pt exist for en/zh/de/ko/ar/th/yo but NOT sw/am (those
two just weren't included in the run that produced the currently-saved
files, even though LANGS in that script already covers all 16). If sw/am
.pt files are missing, this script skips them with a warning rather than
failing -- rerun experiment_thesis/scripts/extract_jailbreak_vectors.py for
sw/am to fill the gap before treating the low-resource tier as complete here.

Usage:
  python scripts/09_refusal_geometry.py \
      --model_path /path/to/Qwen2.5-7B-Instruct \
      --model_alias Qwen2.5-7B-Instruct \
      --output_dir output \
      --refusal_dir_root /home/h24/baga0553/experiment_thesis/output/jailbreak_analysis \
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
PILOT_LANGS = ['en', 'zh', 'de', 'ko', 'ar', 'th', 'yo', 'sw', 'am']
REAL_MECHS = ['prefix_injection', 'refusal_suppression', 'instruction_hierarchy',
              'persona_roleplay', 'fictional_framing', 'encoding_obfuscation']
TIERS = {'en': 'H', 'zh': 'H', 'de': 'H', 'ko': 'M', 'ar': 'M', 'th': 'M',
         'yo': 'L', 'sw': 'L', 'am': 'L'}


# ── Activation extraction (same pattern as 04/06/07/08) ────────────────────────

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


def template_direction(acts_by_id_cond, mechanism):
    diffs = []
    for pid, by_cond in acts_by_id_cond.items():
        if mechanism in by_cond and 'plain' in by_cond:
            diffs.append(by_cond[mechanism] - by_cond['plain'])
    if not diffs:
        return None
    return torch.stack(diffs, dim=0).mean(0)  # [n_layers, d_model]


# ── Geometry vs a reference direction (refusal_direction / harmfulness_direction) ──

def geometry_vs_reference(direction, reference):
    """
    direction, reference: [n_layers, d_model].
    Returns per-layer cosine sim (averaged across layers) plus the fraction of
    `direction`'s norm that lies along `reference` (projection decomposition),
    averaged across layers.
    """
    cos = F.cosine_similarity(direction.float(), reference.float(), dim=-1)  # [n_layers]

    ref_unit = F.normalize(reference.float(), dim=-1)
    parallel_mag = (direction.float() * ref_unit).sum(dim=-1).abs()  # [n_layers]
    total_mag = direction.float().norm(dim=-1)  # [n_layers]
    frac_parallel = (parallel_mag / total_mag.clamp_min(1e-8))  # [n_layers]

    return {
        'cos_mean': float(cos.mean()),
        'cos_by_layer': cos.tolist(),
        'frac_parallel_mean': float(frac_parallel.mean()),
    }


# ── Main ────────────────────────────────────────────────────────────────────────

def main(args):
    model_alias = args.model_alias or os.path.basename(args.model_path)
    out_dir = os.path.join(args.output_dir, model_alias)

    mech_cat_path = os.path.join(SCRIPT_DIR, '..', 'templates', 'templates_en.json')
    with open(mech_cat_path, encoding='utf-8') as f:
        mech_categories = json.load(f)['mechanism_categories']
    mech_to_cat = {m: cat for cat, mechs in mech_categories.items() for m in mechs}

    print("Loading model...")
    model_base = construct_model_base(args.model_path, lang='en')
    print(f"  Loaded: {model_alias}\n")

    records = []  # {lang, mechanism, category, tier, cos_refusal, frac_refusal, cos_harm, frac_harm}
    skipped_langs = []

    for lang in PILOT_LANGS:
        completions_path = os.path.join(out_dir, f'completions_{lang}.json')
        refusal_path = os.path.join(args.refusal_dir_root, model_alias, f'refusal_dir_{lang}.pt')
        harm_path = os.path.join(args.refusal_dir_root, model_alias, f'harmfulness_dir_{lang}.pt')

        if not os.path.exists(completions_path):
            print(f"[{lang}] Missing completions, skipping.")
            skipped_langs.append((lang, 'no completions'))
            continue
        has_refusal = os.path.exists(refusal_path)
        has_harm = os.path.exists(harm_path)
        if not has_refusal and not has_harm:
            print(f"[{lang}] Missing both refusal_dir and harmfulness_dir at {args.refusal_dir_root}, skipping. "
                  f"Rerun experiment_thesis/scripts/extract_jailbreak_vectors.py for this language.")
            skipped_langs.append((lang, 'no refusal/harmfulness direction'))
            continue
        if not has_harm:
            print(f"[{lang}] harmfulness_dir missing (jailbreak_vector likely had 0 usable categories for "
                  f"this language) -- proceeding with refusal_direction only.")

        print(f"[{lang}] Extracting activations...")
        acts = extract_lang_data(model_base, completions_path, args.batch_size)
        refusal_dir = torch.load(refusal_path, map_location='cpu') if has_refusal else None
        harm_dir = torch.load(harm_path, map_location='cpu') if has_harm else None

        for mech in REAL_MECHS:
            d = template_direction(acts, mech)
            if d is None:
                continue
            g_ref = geometry_vs_reference(d, refusal_dir) if refusal_dir is not None else None
            g_harm = geometry_vs_reference(d, harm_dir) if harm_dir is not None else None
            records.append({
                'lang': lang, 'mechanism': mech,
                'category': mech_to_cat.get(mech, 'unknown'), 'tier': TIERS[lang],
                'cos_refusal': g_ref['cos_mean'] if g_ref else None,
                'frac_refusal': g_ref['frac_parallel_mean'] if g_ref else None,
                'cos_harmfulness': g_harm['cos_mean'] if g_harm else None,
                'frac_harmfulness': g_harm['frac_parallel_mean'] if g_harm else None,
            })
            ref_str = f"{g_ref['cos_mean']:+.3f}" if g_ref else "n/a"
            harm_str = f"{g_harm['cos_mean']:+.3f}" if g_harm else "n/a"
            print(f"  [{lang}][{mech}] cos(refusal)={ref_str}  cos(harmfulness)={harm_str}")
        torch.cuda.empty_cache()

    def group_mean(key, filter_fn):
        vals = [r[key] for r in records if filter_fn(r) and r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    by_category = {}
    for cat in ['competing_objectives', 'mismatched_generalization']:
        by_category[cat] = {
            'cos_refusal_mean': group_mean('cos_refusal', lambda r, c=cat: r['category'] == c),
            'cos_harmfulness_mean': group_mean('cos_harmfulness', lambda r, c=cat: r['category'] == c),
            'frac_refusal_mean': group_mean('frac_refusal', lambda r, c=cat: r['category'] == c),
        }
    def fmt(v):
        return f"{v:.3f}" if v is not None else "n/a"

    print("\n=== By mechanism category ===")
    for cat, stats in by_category.items():
        print(f"  {cat}: cos(refusal)={fmt(stats['cos_refusal_mean'])}  "
              f"cos(harmfulness)={fmt(stats['cos_harmfulness_mean'])}  "
              f"frac_along_refusal={fmt(stats['frac_refusal_mean'])}")

    by_tier = {}
    for tier in ['H', 'M', 'L']:
        by_tier[tier] = {
            'cos_refusal_mean': group_mean('cos_refusal', lambda r, t=tier: r['tier'] == t),
            'cos_harmfulness_mean': group_mean('cos_harmfulness', lambda r, t=tier: r['tier'] == t),
        }
    print("\n=== By resource tier ===")
    for tier, stats in by_tier.items():
        print(f"  {tier}: cos(refusal)={fmt(stats['cos_refusal_mean'])}  "
              f"cos(harmfulness)={fmt(stats['cos_harmfulness_mean'])}")

    out_path = os.path.join(out_dir, 'refusal_geometry.json')
    with open(out_path, 'w') as f:
        json.dump({
            'model': model_alias,
            'records': records,
            'by_category': by_category,
            'by_tier': by_tier,
            'skipped_langs': skipped_langs,
        }, f, indent=2)
    print(f"\nSaved: {out_path}")
    if skipped_langs:
        print(f"Skipped languages (incomplete data): {skipped_langs}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',       type=str, required=True)
    parser.add_argument('--model_alias',      type=str, default=None)
    parser.add_argument('--output_dir',       type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    parser.add_argument('--refusal_dir_root', type=str, required=True,
                         help='experiment_thesis output dir containing {model_alias}/refusal_dir_{lang}.pt etc.')
    parser.add_argument('--batch_size',       type=int, default=8)
    args = parser.parse_args()
    if args.model_alias is None:
        args.model_alias = os.path.basename(args.model_path)
    main(args)
