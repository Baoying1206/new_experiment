"""
Stage-1 prerequisite for the taxonomy-robustness diagnostics: persists raw,
per-instruction, all-layer paired activation diffs

    delta_h[i, t] = act(t(x_i)) - act(x_i)     for t in {6 real mechanisms, placebo}

to disk, instead of only the aggregated mean directions/cosine stats that
04/09/12/13 already saved. Every downstream Stage-1 analysis (exact
permutation test on the 3+3 category split, bootstrap-by-instruction,
split-half, leave-one-template-out, placebo calibration, surface-vs-activation
distance comparison) needs these raw per-instruction vectors, not just the
mean template_direction -- none of the existing scripts kept them.

No generation, no WildGuard -- just a forward pass to extract activations,
so this is much cheaper than Phase 1 (10b) or even the earlier extraction
scripts that also handled generation-conditioned completions.

Output: {output_dir}/{model_alias}/paired_diffs_{lang}{suffix}.pt for the default
ids_key per scope (direction_ids for English, cross_lingual_direction_ids for
xling), or paired_diffs_{lang}{suffix}_{ids_key}.pt for any other ids_key
(e.g. validation_ids, test_ids) -- so a validation/test-set extraction can
never silently overwrite the direction-set file used to build directions/run
Exp1. Each file is a dict:
  {
    'instruction_ids': {mechanism: [ids in tensor order]},
    'diffs': {mechanism: tensor [n_matched, n_layers, d_model]},
    'ids_key': the --ids_key used, or null if unfiltered,
  }
for mechanism in REAL_MECHS + ['placebo'].

Supports the 572/200-instruction confirmatory scope introduced after the
75-instruction pilot restart (see DATA_MANIFEST.md):
  --suffix   reads completions_{lang}{suffix}.json instead of
             completions_{lang}.json (e.g. '_full572' for English,
             '_xling' for the 5 confirmatory non-English languages),
             writing to the correspondingly-suffixed .pt file so this never
             collides with the original 75-instruction pilot's
             paired_diffs_{lang}.pt.
  --ids_key  restricts to a named id list from data/splits.json (e.g.
             'direction_ids' to build directions from the 300-id English
             split, 'test_ids' for an out-of-sample replication,
             'cross_lingual_direction_ids' for the 100-id cross-lingual
             direction subset used by the 5 confirmatory languages).
             Default: no filtering (use every matched instruction) --
             preserves the original pilot's unfiltered behavior.

Usage:
  # Original 75-instruction pilot (unchanged behavior):
  python scripts/18_extract_paired_diffs.py --model_path ... --langs en

  # English confirmatory direction set (300 ids, for building directions):
  python scripts/18_extract_paired_diffs.py --model_path ... --langs en \
      --suffix _full572 --ids_key direction_ids

  # Cross-lingual confirmatory direction set (100 ids, 5 languages):
  python scripts/18_extract_paired_diffs.py --model_path ... \
      --langs zh,ar,th,yo,am --suffix _xling --ids_key cross_lingual_direction_ids
"""
import argparse
import json
import os

import torch
from tqdm import tqdm

from pipeline.model_utils.model_factory import construct_model_base
from pipeline.utils.hook_utils import add_hooks

SCRIPT_DIR = os.path.dirname(__file__)
SPLITS_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'splits.json')
REAL_MECHS = ['prefix_injection', 'refusal_suppression', 'instruction_hierarchy',
              'persona_roleplay', 'fictional_framing', 'encoding_obfuscation']
ALL_CONDS = REAL_MECHS + ['placebo']


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


def main(args):
    model_alias = args.model_alias or os.path.basename(args.model_path)
    out_dir = os.path.join(args.output_dir, model_alias)
    langs = args.langs.split(',')

    real_mechs = args.mechanisms.split(',') if args.mechanisms else REAL_MECHS
    all_conds = real_mechs + ['placebo']
    if args.mechanisms:
        print(f"Using --mechanisms override: {real_mechs} (module default REAL_MECHS={REAL_MECHS} "
              f"NOT used)\n")

    keep_ids = None
    if args.ids_key:
        with open(SPLITS_PATH) as f:
            splits = json.load(f)
        keep_ids = set(splits[args.ids_key])
        print(f"Filtering to splits.json['{args.ids_key}']: {len(keep_ids)} ids\n")

    print("Loading model...")
    model_base = construct_model_base(args.model_path, lang='en')
    n_layers = model_base.model.config.num_hidden_layers
    d_model = model_base.model.config.hidden_size
    print(f"  Loaded: {model_alias}  n_layers={n_layers}  d_model={d_model}\n")

    for lang in langs:
        completions_path = os.path.join(out_dir, f'completions_{lang}{args.suffix}.json')
        if not os.path.exists(completions_path):
            print(f"[{lang}] Missing {completions_path}, skipping.")
            continue
        with open(completions_path, encoding='utf-8') as f:
            completions = json.load(f)['completions']

        by_id = {}
        for c in completions:
            if keep_ids is not None and c['id'] not in keep_ids:
                continue
            by_id.setdefault(c['id'], {})[c['condition']] = c['instruction']

        print(f"\n[{lang}] Extracting paired diffs for {all_conds}  "
              f"({len(by_id)} candidate instructions after id filtering)...")
        diffs_by_mech, ids_by_mech = {}, {}
        for mech in all_conds:
            pids = [pid for pid in by_id if 'plain' in by_id[pid] and mech in by_id[pid]]
            plain_instrs = [by_id[pid]['plain'] for pid in pids]
            mech_instrs = [by_id[pid][mech] for pid in pids]

            all_instrs = plain_instrs + mech_instrs
            acts = get_all_activations(model_base, all_instrs, args.batch_size)  # [2n, n_layers, d_model]
            n = len(pids)
            plain_acts = acts[:n]
            mech_acts = acts[n:]
            diffs = mech_acts - plain_acts  # [n, n_layers, d_model]

            diffs_by_mech[mech] = diffs
            ids_by_mech[mech] = pids
            print(f"  [{mech}] n={n}  diffs.shape={tuple(diffs.shape)}")
            if keep_ids is not None and n != len(keep_ids):
                print(f"    WARNING: expected {len(keep_ids)} ids from --ids_key "
                      f"'{args.ids_key}' but only {n} had both plain+{mech} -- "
                      f"check completions_{lang}{args.suffix}.json for missing conditions.")
            torch.cuda.empty_cache()

        # Output filename must reflect --ids_key, not just --suffix -- otherwise a
        # run with a different ids_key (e.g. validation_ids) but the same --suffix
        # as an earlier direction_ids run would silently overwrite it, since suffix
        # alone doesn't distinguish "which partition of the same data pool" this
        # file was built from. 'direction_ids' (English --suffix _full572 default)
        # and 'cross_lingual_direction_ids' (--suffix _xling default) keep the
        # original unsuffixed filename for backward compatibility with
        # already-saved/committed files and scripts (19_taxonomy_robustness.py)
        # that read them without an ids_key in the path; any other ids_key (or
        # None) gets it appended explicitly so it can never collide.
        DEFAULT_IDS_KEYS = (None, 'direction_ids', 'cross_lingual_direction_ids')
        ids_key_suffix = '' if args.ids_key in DEFAULT_IDS_KEYS else f'_{args.ids_key}'
        out_path = os.path.join(out_dir, f'paired_diffs_{lang}{args.suffix}{ids_key_suffix}.pt')
        torch.save({'instruction_ids': ids_by_mech, 'diffs': diffs_by_mech,
                    'n_layers': n_layers, 'd_model': d_model, 'ids_key': args.ids_key}, out_path)
        print(f"  Saved: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',  type=str, required=True)
    parser.add_argument('--model_alias', type=str, default=None)
    parser.add_argument('--output_dir',  type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    parser.add_argument('--langs',       type=str, default='en')
    parser.add_argument('--suffix',      type=str, default='',
                         help="Applied to both completions_{lang}{suffix}.json (read) and "
                              "paired_diffs_{lang}{suffix}.pt (write), e.g. '_full572' or '_xling'.")
    parser.add_argument('--ids_key',     type=str, default=None,
                         help="Key into data/splits.json to restrict instructions to "
                              "(e.g. 'direction_ids', 'test_ids', 'cross_lingual_direction_ids'). "
                              "Default: no filtering, use every matched instruction.")
    parser.add_argument('--batch_size',  type=int, default=8)
    parser.add_argument('--mechanisms',  type=str, default=None,
                         help="Comma-separated real-mechanism names to extract (placebo is always "
                              "added automatically) -- overrides the module-level REAL_MECHS "
                              "default. Needed for any mechanism set other than the original 6 "
                              "(e.g. the corrected taxonomy's "
                              "prefix_injection,refusal_suppression,persona_roleplay,"
                              "encoding_obfuscation,payload_splitting,distractors_negated), since "
                              "the default would silently look for mechanisms that no longer exist "
                              "in templates_en.json (missing conditions -> n=0 with a warning, not "
                              "an error) while never attempting ones it doesn't know about at all.")
    args = parser.parse_args()
    if args.model_alias is None:
        args.model_alias = os.path.basename(args.model_path)
    main(args)
