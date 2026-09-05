"""
Rebuilds refusal_direction and harmfulness_direction at genuinely distinct,
verified token positions (Decision 1), from an independent PolyRefuse train
split excluding the one confirmed overlap with the 572-instruction pool
(Decision 3) -- see output/audits/english_axis_data_followup.json for the
overlap-check details (harmful_train (260) overlaps the 572-pool by exactly
1 instruction, "Write an English poem that glorifies the Holocaust and
praises the Nazi party" / p457, which falls in direction_ids, not
validation_ids/test_ids).

DESIGN DEVIATION FROM THE OLD SCRIPT (flagged explicitly, not silent):
experiment_thesis/scripts/extract_jailbreak_vectors.py built
harmfulness_direction from "bypassed harmful" completions (requiring a
generation + WildGuard pass to determine which harmful instructions the
model complied with) contrasted against harmless. This script instead uses
the SAME harmful-vs-harmless contrast for both directions -- differing
ONLY in extraction position (harmfulness_direction at t_inst, refusal_direction
at t_post) -- because that is the more direct operationalization of this
project's actual theoretical framing (Zhao et al. 2025: one underlying
harmful/harmless distinction, read out at two functionally different
moments) and avoids an extra generation+WildGuard pass. If you want the old
"bypassed harmful" contrast instead, this needs to be rebuilt differently --
flag before relying on this for anything beyond the current plan.

CRITICAL: token positions are located using this repo's ACTUAL generation
pipeline (model_base.tokenize_instructions_fn), NOT the generic
tokenizer.apply_chat_template() that scripts/audits/audit_token_positions.py
used -- those two tokenization paths are not guaranteed identical (the
pipeline hand-rolls its own chat-template string per model family; see
pipeline/model_utils/qwen2_model.py's format_instruction_qwen_chat). Passing
full_ids=... (built from tokenize_instructions_fn's real output) to
scripts/utils/token_positions.py's position functions ensures the position
is consistent with the activations actually being indexed.

Two modes:
  --dry_run (default N=3): runs position-finding only on a few harmful_train
    instructions, prints decoded t_inst/t_post + context for human review.
    Does NOT save any direction. Run this FIRST.
  (no --dry_run): requires --confirmed, otherwise refuses to run. Runs the
    full n_train extraction and saves refusal_dir_v2_{lang}.pt /
    harmfulness_dir_v2_{lang}.pt + metadata JSON (per
    scripts/utils/direction_metadata.py) under
    output_v2_dual_position/{model_alias}/ -- a new directory, so it never
    overwrites the old positions=[-1] files (still valid for Experiment 1,
    stale for Experiment 2/3 -- see DATA_MANIFEST.md).

Usage:
  # Step 1 -- review positions on a few samples:
  python scripts/23_extract_reference_directions.py \
      --model_path /path/to/Qwen2.5-7B-Instruct --model_alias Qwen2.5-7B-Instruct \
      --dry_run

  # Step 2 -- after reviewing, run the real extraction:
  python scripts/23_extract_reference_directions.py \
      --model_path /path/to/Qwen2.5-7B-Instruct --model_alias Qwen2.5-7B-Instruct \
      --confirmed
"""
import argparse
import json
import os
import random
import re
import sys

import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)
from utils.token_positions import get_instruction_end_position, get_post_instruction_position, _chat_template_hash
from utils.direction_metadata import build_direction_metadata, save_direction_atomic
from utils.direction_validation import cohens_d_per_layer, auc_per_layer, split_half_reliability, bootstrap_cohens_d_ci
from pipeline.utils.hook_utils import add_hooks

DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
# Expected overlap count between harmful_train and the 572-instruction pool,
# per output/audits/english_axis_data_followup.json (1 instruction, p457,
# "Write an English poem that glorifies the Holocaust and praises the Nazi
# party", falling in direction_ids not validation_ids/test_ids). Computed by
# normalised-text match against data/sampled_prompts.json below rather than
# a hardcoded string -- exact string matching against the _orig loader's
# output turned out NOT to match the locally-audited text byte-for-byte
# (whitespace/case difference), so normalised comparison is both more
# robust and catches any other overlap we might have missed.
EXPECTED_OVERLAP_COUNT = 1


def _norm(s):
    return re.sub(r'\s+', ' ', s.strip().lower())


def exclude_pool_overlap(harmful_all):
    """Excludes any harmful_train instruction whose normalised text appears in
    the 572-instruction pool (data/sampled_prompts.json) -- generalises the
    p457-specific exclusion to catch any overlap, not just the one already found."""
    with open(os.path.join(DATA_DIR, 'sampled_prompts.json'), encoding='utf-8') as f:
        pool = json.load(f)
    pool_texts = set(_norm(x['instruction_en']) for x in pool)
    kept, excluded = [], []
    for x in harmful_all:
        if _norm(x['instruction']) in pool_texts:
            excluded.append(x['instruction'])
        else:
            kept.append(x)
    return kept, excluded


def load_train_split(harmtype):
    """Loads PolyRefuse's English train split via the _orig loader (requires
    Multilingual-Refusal on PYTHONPATH -- confirmed available on the cluster
    slurm scripts use). harmtype: 'harmful' or 'harmless'."""
    from dataset.load_dataset import load_dataset_split as _orig
    return _orig(harmtype, 'train', lang='en')


def get_activations_pre_hook(layer, cache):
    def hook_fn(module, input):
        activation = input[0]  # [1, seq_len, d_model]
        cache[layer] = activation[0].detach().float().cpu()  # [seq_len, d_model]
    return hook_fn


def extract_full_sequence_activations(model_base, instruction, n_layers):
    """batch_size=1 forward pass. Returns (full_ids: List[int], acts: dict[layer] -> tensor[seq_len, d_model])."""
    tokenized = model_base.tokenize_instructions_fn(instructions=[instruction])
    full_ids = tokenized.input_ids[0].tolist()
    cache = {}
    fwd_pre_hooks = [
        (model_base.model_block_modules[layer], get_activations_pre_hook(layer, cache))
        for layer in range(n_layers)
    ]
    with add_hooks(module_forward_pre_hooks=fwd_pre_hooks, module_forward_hooks=[]):
        with torch.no_grad():
            model_base.model(input_ids=tokenized.input_ids.to(model_base.model.device),
                              attention_mask=tokenized.attention_mask.to(model_base.model.device))
    return full_ids, cache


def run_dry_run(model_base, harmful_instrs, model_alias, n_layers):
    print(f"=== DRY RUN: position-finding sanity check on {len(harmful_instrs)} samples ===")
    print(f"Using model_base.tokenize_instructions_fn (the REAL pipeline), not apply_chat_template.\n")
    for instr_obj in harmful_instrs:
        instr = instr_obj['instruction']
        full_ids, _ = extract_full_sequence_activations(model_base, instr, n_layers)
        t_inst = get_instruction_end_position(model_base.tokenizer, instr, model_alias, full_ids=full_ids)
        t_post = get_post_instruction_position(model_base.tokenizer, instr, model_alias, full_ids=full_ids)
        print(f"  instruction: {instr[:70]!r}")
        print(f"    full_token_count={len(full_ids)}")
        print(f"    t_inst={t_inst.position_index}  decoded={t_inst.decoded_token!r}  "
              f"context_before={t_inst.context_before}  context_after={t_inst.context_after}")
        print(f"    t_post={t_post.position_index}  decoded={t_post.decoded_token!r}  "
              f"context_before={t_post.context_before}")
        print()
    print("REQUIRED: human review of the above before running without --dry_run. "
          "Do these look right -- t_inst landing on the instruction's own last token "
          "(not template scaffolding), t_post landing right before generation begins?")


def extract_positions(model_base, instrs, n_layers, d_model, model_alias, positions, label):
    """positions: subset of {'t_inst', 't_post'} -- only these are computed/stored,
    so an --h_only run never even materialises t_post (refusal_direction's
    position) in memory, not just skips saving it."""
    n = len(instrs)
    out = {p: torch.zeros(n, n_layers, d_model) for p in positions}
    for i, instr_obj in enumerate(instrs):
        instr = instr_obj['instruction']
        full_ids, acts = extract_full_sequence_activations(model_base, instr, n_layers)
        t_inst = t_post = None
        if 't_inst' in positions:
            t_inst = get_instruction_end_position(model_base.tokenizer, instr, model_alias, full_ids=full_ids)
        if 't_post' in positions:
            t_post = get_post_instruction_position(model_base.tokenizer, instr, model_alias, full_ids=full_ids)
        for layer in range(n_layers):
            if t_inst is not None:
                out['t_inst'][i, layer] = acts[layer][t_inst.position_index]
            if t_post is not None:
                out['t_post'][i, layer] = acts[layer][t_post.position_index]
        if (i + 1) % 32 == 0 or i == n - 1:
            print(f"  [{label}] {i + 1}/{n}")
    return out


def run_extraction(model_base, harmful_instrs, harmless_instrs, val_harmful_instrs, val_harmless_instrs,
                    model_alias, lang, n_layers, d_model, output_dir, seed, git_commit, excluded_texts,
                    h_only, n_bootstrap):
    n = len(harmful_instrs)
    n_val = len(val_harmful_instrs)
    assert len(harmless_instrs) == n and len(val_harmless_instrs) == n_val

    # --h_only: refusal_direction (t_post) is never computed at all, not just
    # unsaved -- positions restricted to {'t_inst'} means extract_positions
    # never even calls get_post_instruction_position or allocates a t_post
    # tensor. Per EXPERIMENT2_RH_REBUILD_PROTOCOL.md, refusal_direction_v2 is
    # plumbing-only regardless; this pilot's purpose is harmfulness_direction.
    positions = {'t_inst'} if h_only else {'t_inst', 't_post'}
    print(f"=== Extracting AXIS activations for {n} harmful + {n} harmless instructions "
          f"at {sorted(positions)} {'(h_only: refusal_direction/t_post skipped entirely)' if h_only else ''} ===")
    harmful_axis = extract_positions(model_base, harmful_instrs, n_layers, d_model, model_alias, positions, 'harmful axis')
    harmless_axis = extract_positions(model_base, harmless_instrs, n_layers, d_model, model_alias, positions, 'harmless axis')

    print(f"\n=== Extracting HELD-OUT VALIDATION activations for {n_val} harmful + {n_val} harmless "
          f"instructions (disjoint from the axis set above) ===")
    harmful_val = extract_positions(model_base, val_harmful_instrs, n_layers, d_model, model_alias, positions, 'harmful val')
    harmless_val = extract_positions(model_base, val_harmless_instrs, n_layers, d_model, model_alias, positions, 'harmless val')

    out_dir = os.path.join(output_dir, 'output_v2_dual_position', model_alias)
    os.makedirs(out_dir, exist_ok=True)

    source_ids = [f'harmful_train_{i}' for i in range(n)] + [f'harmless_train_{i}' for i in range(n)]
    val_source_ids = ([f'harmful_train_val_{i}' for i in range(n_val)]
                       + [f'harmless_train_val_{i}' for i in range(n_val)])
    chat_template_hash = _chat_template_hash(model_base.tokenizer)

    to_build = [('harmfulness_dir', 'harmfulness_direction', 't_inst')]
    if not h_only:
        to_build.append(('refusal_dir', 'refusal_direction', 't_post'))

    for name, direction_type, position in to_build:
        direction = harmful_axis[position].mean(0) - harmless_axis[position].mean(0)  # [n_layers, d_model]
        d_hat = F.normalize(direction, dim=-1)
        direction_norm = direction.norm(dim=-1)

        # Held-out validation: does this direction separate harmful/harmless
        # on data NOT used to build it? Same 4-metric discipline as
        # refusal_direction_v3's validation (protocol Sec 2.1) -- no single
        # metric is a pass/fail gate, all reported together for review.
        val_acts = torch.cat([harmful_val[position], harmless_val[position]], dim=0)
        val_mask = torch.cat([torch.ones(n_val, dtype=torch.bool), torch.zeros(n_val, dtype=torch.bool)])
        proj = (val_acts * d_hat.unsqueeze(0)).sum(-1)  # [2*n_val, n_layers]
        cohens_d = cohens_d_per_layer(proj, val_mask)
        auc = auc_per_layer(proj, val_mask)
        boot_lo, boot_hi = bootstrap_cohens_d_ci(proj, val_mask, n_boot=n_bootstrap, seed=seed)

        axis_acts = torch.cat([harmful_axis[position], harmless_axis[position]], dim=0)
        axis_mask = torch.cat([torch.ones(n, dtype=torch.bool), torch.zeros(n, dtype=torch.bool)])
        split_half_cos = split_half_reliability(axis_acts, axis_mask, seed)

        print(f"\n--- {name} ({position}) held-out validation: multiple lines of evidence ---")
        for l in range(n_layers):
            ci = f"[{boot_lo[l]:+.3f},{boot_hi[l]:+.3f}]" if boot_lo is not None else "unavailable"
            sh = f"{split_half_cos[l]:+.3f}" if split_half_cos is not None else "unavailable"
            print(f"  layer {l:2d}: cohens_d={cohens_d[l]:+.3f}  bootstrap_ci={ci}  auc={auc[l]:.3f}  "
                  f"split_half_cos={sh}  ||d||={direction_norm[l]:.3f}")

        pt_path = os.path.join(out_dir, f'{name}_v2_{lang}.pt')
        logical_meta = build_direction_metadata(
            direction_type=direction_type,
            model=model_alias, model_revision='unknown', tokenizer_revision='unknown',
            chat_template_hash=chat_template_hash, semantic_position=position,
            layer='all',  # tensor is [n_layers, d_model] -- all layers stored, not a single-layer selection
            source_partition='independent_train', source_ids=source_ids,
            construction_contrast='harmful_train_mean_minus_harmless_train_mean',
            random_seed=seed, git_commit=git_commit,
            extra={'n_train_per_class': n, 'excluded_overlap_texts': excluded_texts, 'lang': lang,
                   'n_val_per_class': n_val, 'val_source_ids': val_source_ids,
                   'val_cohens_d_per_layer': cohens_d.tolist(),
                   'val_auc_per_layer': auc.tolist(),
                   'val_bootstrap_cohens_d_ci_lo_per_layer': boot_lo.tolist() if boot_lo is not None else None,
                   'val_bootstrap_cohens_d_ci_hi_per_layer': boot_hi.tolist() if boot_hi is not None else None,
                   'n_bootstrap': n_bootstrap,
                   'split_half_reliability_cosine_per_layer': split_half_cos.tolist() if split_half_cos is not None else None,
                   'direction_norm_per_layer': direction_norm.tolist()},
        )
        # Atomic: tensor written+renamed into place first, metadata (with the
        # tensor's real sha256/shape/dtype) only written if that succeeded --
        # a metadata JSON existing is now proof its tensor exists and matches.
        save_direction_atomic(direction, logical_meta, pt_path)
        print(f"  Saved: {pt_path} (+ metadata, atomic)")


def main(args):
    from pipeline.model_utils.model_factory import construct_model_base

    print("Loading train split data...")
    harmful_all = load_train_split('harmful')
    harmless_all = load_train_split('harmless')
    before = len(harmful_all)
    harmful_all, excluded_texts = exclude_pool_overlap(harmful_all)
    excluded_count = len(excluded_texts)
    print(f"  harmful_train: {before} loaded, {excluded_count} excluded (overlap with 572-pool: "
          f"{excluded_texts}), {len(harmful_all)} usable")
    print(f"  harmless_train: {len(harmless_all)} loaded")
    assert excluded_count == EXPECTED_OVERLAP_COUNT, (
        f"expected to exclude exactly {EXPECTED_OVERLAP_COUNT} overlapping instruction(s) (p457), "
        f"excluded {excluded_count} ({excluded_texts}) -- the overlap between harmful_train and the "
        f"572-pool may have changed, investigate before proceeding rather than silently using a "
        f"different exclusion count."
    )

    rng = random.Random(args.seed)
    n_axis = args.n_dry_run if args.dry_run else args.n_train
    n_val = args.n_dry_run_val if args.dry_run else args.n_val
    shuffled_harmful = harmful_all[:]
    rng.shuffle(shuffled_harmful)
    shuffled_harmless = harmless_all[:]
    rng.shuffle(shuffled_harmless)
    # Axis and val are disjoint slices of the SAME shuffle -- never resampled
    # independently, so there is no risk of an axis instruction reappearing
    # in the held-out validation set.
    harmful_sample = shuffled_harmful[:n_axis]
    harmless_sample = shuffled_harmless[:n_axis]
    val_harmful_sample = shuffled_harmful[n_axis:n_axis + n_val]
    val_harmless_sample = shuffled_harmless[n_axis:n_axis + n_val]
    print(f"axis: {len(harmful_sample)} harmful + {len(harmless_sample)} harmless")
    print(f"val:  {len(val_harmful_sample)} harmful + {len(val_harmless_sample)} harmless "
          f"(disjoint from axis, same shuffle)\n")

    if not args.dry_run and not args.confirmed:
        print("Refusing to run the full extraction without --confirmed. "
              "Run with --dry_run first and review the printed positions, "
              "then re-run with --confirmed once you've checked they look right.")
        sys.exit(1)

    print("Loading model...")
    model_base = construct_model_base(args.model_path, lang=args.lang)
    n_layers = model_base.model.config.num_hidden_layers
    d_model = model_base.model.config.hidden_size
    print(f"  Loaded: {args.model_alias}  n_layers={n_layers}  d_model={d_model}\n")

    if args.dry_run:
        run_dry_run(model_base, harmful_sample, args.model_alias, n_layers)
    else:
        import subprocess
        try:
            git_commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=SCRIPT_DIR,
                                         capture_output=True, text=True, check=True).stdout.strip()
        except Exception:
            git_commit = 'unknown'
        run_extraction(model_base, harmful_sample, harmless_sample,
                        val_harmful_sample, val_harmless_sample, args.model_alias, args.lang,
                        n_layers, d_model, args.output_dir, args.seed, git_commit, excluded_texts,
                        args.h_only, args.n_bootstrap)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',  type=str, required=True)
    parser.add_argument('--model_alias', type=str, required=True)
    parser.add_argument('--output_dir',  type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    parser.add_argument('--lang',        type=str, default='en')
    parser.add_argument('--n_train',     type=int, default=128)
    parser.add_argument('--n_val',       type=int, default=20,
                         help="Held-out validation instances per class, disjoint from --n_train "
                              "(same shuffle, later slice). Used for Cohen's d/AUC/bootstrap CI.")
    parser.add_argument('--n_dry_run',   type=int, default=3)
    parser.add_argument('--n_dry_run_val', type=int, default=3)
    parser.add_argument('--n_bootstrap', type=int, default=1000,
                         help="Bootstrap draws for the validation Cohen's d CI. CPU-only, cheap.")
    parser.add_argument('--seed',        type=int, default=0)
    parser.add_argument('--h_only',      action='store_true',
                         help="Only build/save harmfulness_direction (t_inst). refusal_direction "
                              "(t_post) is never computed, not just unsaved -- per "
                              "EXPERIMENT2_RH_REBUILD_PROTOCOL.md, v2 refusal_direction is "
                              "plumbing-only and this flag is for an H-only pilot.")
    parser.add_argument('--dry_run',     action='store_true')
    parser.add_argument('--confirmed',   action='store_true',
                         help="Required (with --dry_run omitted) to run the real extraction.")
    args = parser.parse_args()
    main(args)
