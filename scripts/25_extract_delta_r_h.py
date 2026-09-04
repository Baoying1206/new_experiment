"""
Experiment 2 core data: extracts delta_R (refusal-axis projection) and
delta_H (harmfulness-axis projection) for each of the 6 mechanisms + placebo,
per instruction, per layer -- computed via projection of the template-induced
activation shift onto EACH axis AT ITS OWN position (t_post for
refusal_direction, t_inst for harmfulness_direction), NOT averaged/conflated
onto a single shared position. This is the literal Experiment 2 requirement.

  delta_R_i,mech = dot(act(mech, x_i, t_post) - act(plain, x_i, t_post_of_plain),
                        refusal_direction_hat)   [per layer]
  delta_H_i,mech = dot(act(mech, x_i, t_inst) - act(plain, x_i, t_inst_of_plain),
                        harmfulness_direction_hat)   [per layer]

refusal_direction is the v3 refused-vs-accepted behavioral direction
(scripts/26_rebuild_refusal_direction_behavioral.py, output_v3_behavioral_refusal/),
matching arXiv 2507.11878's actual contrast -- NOT the v2 harmful-vs-harmless
simplification (output_v2_dual_position/), superseded once v3 validated well
(large held-out Cohen's d) and showed lower collinearity with
harmfulness_direction than v2 did. harmfulness_direction is still v2
(harmful-vs-harmless at t_inst already matches the paper).

t_post is the templated/plain prompt's own last token (always well-defined,
same method as before). t_inst for a TEMPLATED condition is located
structurally (scripts/utils/token_positions.py's get_user_turn_end_position:
the last token before the user-turn's closing special token), NOT via
subsequence search against the raw instruction text -- the raw instruction
text is not recoverable as a literal substring for every mechanism
(encoding_obfuscation transforms it), so subsequence search (used for the
plain condition and for building the reference directions from raw
harmful/harmless instructions) is not usable here in general.

Also computes the placebo-calibrated versions (delta_R_PC = delta_R_mech -
delta_R_placebo, matched by instruction id) -- same logic as
19_taxonomy_robustness.py's placebo-calibrated re-test, extended to delta_R/delta_H.

Requires:
  - output_v2_dual_position/{model_alias}/refusal_dir_v2_{lang}.pt,
    harmfulness_dir_v2_{lang}.pt (scripts/23_extract_reference_directions.py --confirmed)
  - completions_{lang}{suffix}.json filtered to direction_ids (the same 300-id
    scope Experiment 1 used)

Usage:
  python scripts/25_extract_delta_r_h.py \
      --model_path /path/to/Qwen2.5-7B-Instruct --model_alias Qwen2.5-7B-Instruct \
      --output_dir output --lang en --suffix _full572 --ids_key direction_ids
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, SCRIPT_DIR)
from utils.token_positions import get_post_instruction_position, get_user_turn_end_position
from utils.direction_metadata import build_delta_metadata, save_delta_atomic, verify_direction_file, current_git_commit
from pipeline.utils.hook_utils import add_hooks
from _taxonomy_v2_loader import load_taxonomy_v2

SPLITS_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'splits.json')
# FIXED 2026-09-04 (artifact-lineage audit): this used to be a hardcoded list
# containing 'instruction_hierarchy'/'fictional_framing' -- pre-correction
# stand-in names that have not existed in templates_en.json since the
# taxonomy was corrected to wei_canonical_v2. A script pinned to those names
# can only ever match zero completions under the current 6-mechanism
# taxonomy (ids ends up empty), silently producing a degenerate/empty
# output. active_mechanisms is now read fresh from templates_en.json via
# _taxonomy_v2_loader, the same source of truth every other current-taxonomy
# script uses -- never hand-copied again.
REAL_MECHS = load_taxonomy_v2()['active_mechanisms']
ALL_CONDS = REAL_MECHS + ['placebo']


def get_activations_pre_hook(layer, cache):
    def hook_fn(module, input):
        cache[layer] = input[0][0].detach().float().cpu()  # [seq_len, d_model]
    return hook_fn


def extract_sequence_and_positions(model_base, text, model_alias, n_layers):
    """batch_size=1 forward pass. Returns (t_inst_idx, t_post_idx, acts: dict[layer] -> tensor[seq_len, d_model])."""
    tokenized = model_base.tokenize_instructions_fn(instructions=[text])
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
    t_inst = get_user_turn_end_position(model_base.tokenizer, full_ids, model_alias)
    t_post = get_post_instruction_position(model_base.tokenizer, text, model_alias, full_ids=full_ids)
    return t_inst.position_index, t_post.position_index, cache


def main(args):
    from pipeline.model_utils.model_factory import construct_model_base

    # refusal_direction version is now selectable (protocol default: v3,
    # refused-vs-accepted, matching arXiv 2507.11878's actual contrast) --
    # --refusal_version v2 exists specifically so a minimal pilot/dry-run can
    # exercise this script's delta-computation machinery WITHOUT requiring a
    # v3 rebuild (which needs generation + WildGuard judging, out of scope
    # for a dry-run). harmfulness_direction is always v2 (harmful-vs-harmless
    # at t_inst) -- unchanged, per EXPERIMENT2_RH_REBUILD_PROTOCOL.md.
    v3_dir = os.path.join(args.output_dir, 'output_v3_behavioral_refusal', args.model_alias)
    v2_dir = os.path.join(args.output_dir, 'output_v2_dual_position', args.model_alias)
    if args.refusal_version == 'v3':
        refusal_pt = os.path.join(v3_dir, f'refusal_dir_v3_{args.lang}.pt')
    else:
        refusal_pt = os.path.join(v2_dir, f'refusal_dir_v2_{args.lang}.pt')
    harmfulness_pt = os.path.join(v2_dir, f'harmfulness_dir_v2_{args.lang}.pt')

    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    keep_ids = set(splits[args.ids_key])
    print(f"Filtering to splits.json['{args.ids_key}']: {len(keep_ids)} ids\n")

    completions_path = os.path.join(args.output_dir, args.model_alias, f'completions_{args.lang}{args.suffix}.json')
    with open(completions_path, encoding='utf-8') as f:
        completions = json.load(f)['completions']
    by_id = {}
    for c in completions:
        if c['id'] not in keep_ids:
            continue
        by_id.setdefault(c['id'], {})[c['condition']] = c['instruction']
    ids = sorted(pid for pid in by_id if 'plain' in by_id[pid] and all(m in by_id[pid] for m in ALL_CONDS))
    if args.limit_ids is not None:
        ids = ids[:args.limit_ids]  # pilot/dry-run scale cap -- deterministic (sorted then truncated), not random
    print(f"{len(ids)} instructions with plain + all 7 conditions present"
          f"{f' (capped to --limit_ids {args.limit_ids})' if args.limit_ids is not None else ''}\n")

    out_dir = os.path.join(args.output_dir, args.model_alias)
    out_path = os.path.join(out_dir, f'delta_r_h_{args.lang}{args.suffix}_{args.ids_key}.pt')

    if args.dry_run:
        print("=== DRY RUN (no model loaded, no GPU work) ===")
        print(f"  model_alias:            {args.model_alias}")
        print(f"  ids_key:                {args.ids_key}")
        print(f"  refusal_version:        {args.refusal_version}")
        print(f"  expected instruction count (splits.json['{args.ids_key}']): {len(keep_ids)}")
        print(f"  actual matched instruction count (plain + all 7 conditions present"
              f"{', capped' if args.limit_ids is not None else ''}): {len(ids)}")
        if args.limit_ids is None and len(ids) != len(keep_ids):
            print(f"  WARNING: matched count != splits.json count -- some ids in "
                  f"splits.json['{args.ids_key}'] are missing from completions_{args.lang}{args.suffix}.json "
                  f"(missing conditions or entirely absent id). {len(keep_ids) - len(ids)} ids would be dropped.")
        print(f"  conditions: {ALL_CONDS} ({len(REAL_MECHS)} real mechanisms + placebo, "
              f"read live from templates_en.json -- never hardcoded)")
        print(f"  axis file (refusal_direction {args.refusal_version}): {refusal_pt}  "
              f"exists={os.path.exists(refusal_pt)}  metadata_exists={os.path.exists(refusal_pt[:-3] + '.json')}")
        print(f"  axis file (harmfulness_direction v2): {harmfulness_pt}  "
              f"exists={os.path.exists(harmfulness_pt)}  metadata_exists={os.path.exists(harmfulness_pt[:-3] + '.json')}")
        if os.path.exists(refusal_pt):
            n_layers_expected = torch.load(refusal_pt, map_location='cpu').shape[0]
            print(f"  expected delta_R/delta_H tensor shape per condition: "
                  f"[{len(ids)}, {n_layers_expected}]  (n_instructions, n_layers)")
        print(f"  output path: {out_path}")
        if os.path.exists(out_path):
            print(f"  WARNING: output path already exists -- running without --dry_run WILL OVERWRITE IT "
                  f"(atomically -- the old file stays intact until the new one is fully written).")
        else:
            print(f"  output path does not exist yet -- safe to write.")
        print("\nDRY RUN complete -- no GPU job, no files written, no direction files were even opened "
              "(existence/metadata only checked via os.path.exists). Re-run without --dry_run to extract.")
        return

    # Fail fast on BOTH direction dependencies before loading the model --
    # verify_direction_file hash-checks each against its own metadata, not
    # just a bare existence/torch.load check.
    print(f"Verifying direction dependencies before loading the model...")
    refusal_dir, _ = verify_direction_file(refusal_pt)
    harmfulness_dir, _ = verify_direction_file(harmfulness_pt)
    refusal_dir, harmfulness_dir = refusal_dir.float(), harmfulness_dir.float()
    refusal_hat = F.normalize(refusal_dir, dim=-1)        # [n_layers, d_model]
    harmfulness_hat = F.normalize(harmfulness_dir, dim=-1)  # [n_layers, d_model]
    print(f"  OK -- both present and hash-verified.\n")

    # R/H independence check (EXPERIMENT2_RH_REBUILD_PROTOCOL.md Sec 10, added
    # 2026-09-04): cos(r,h) per layer, plus the harmfulness direction
    # orthogonalized against refusal -- h_perp = h - (h.r / r.r) r -- using
    # the RAW (non-unit) direction vectors, matching the protocol's formula
    # literally. No cosine threshold is imposed here; both raw and
    # orthogonalized projections are always computed and saved, and the
    # reader judges collinearity from the reported numbers, not a gate.
    cos_r_h = F.cosine_similarity(refusal_dir, harmfulness_dir, dim=-1)  # [n_layers]
    r_dot_r = (refusal_dir * refusal_dir).sum(-1, keepdim=True).clamp_min(1e-12)
    h_dot_r = (harmfulness_dir * refusal_dir).sum(-1, keepdim=True)
    harmfulness_perp = harmfulness_dir - (h_dot_r / r_dot_r) * refusal_dir  # [n_layers, d_model]
    harmfulness_perp_hat = F.normalize(harmfulness_perp, dim=-1)
    h_perp_norm_ratio = harmfulness_perp.norm(dim=-1) / harmfulness_dir.norm(dim=-1).clamp_min(1e-12)  # [n_layers]
    print(f"  cos(refusal_direction, harmfulness_direction) per layer: "
          f"mean={cos_r_h.mean():+.4f}  range=[{cos_r_h.min():+.4f},{cos_r_h.max():+.4f}]")
    print(f"  ||h_perp||/||h|| per layer (1.0=fully orthogonal already, 0.0=h is collinear with r): "
          f"mean={h_perp_norm_ratio.mean():.4f}  range=[{h_perp_norm_ratio.min():.4f},{h_perp_norm_ratio.max():.4f}]\n")

    assert len(ids) > 0, (
        f"0 instructions matched (plain + all {len(ALL_CONDS)} conditions present) for "
        f"ids_key={args.ids_key!r} in {completions_path} -- refusing to proceed and silently write "
        f"an empty/degenerate output. This is exactly the failure mode the pre-rebuild version of "
        f"this script had no protection against (its hardcoded, stale mechanism list could never "
        f"match any real completion, silently producing an empty tensor)."
    )

    print("Loading model...")
    model_base = construct_model_base(args.model_path, lang=args.lang)
    n_layers = model_base.model.config.num_hidden_layers
    print(f"  Loaded: {args.model_alias}  n_layers={n_layers}\n")

    assert refusal_dir.shape[0] == n_layers, (
        f"refusal_direction has {refusal_dir.shape[0]} layers, model has {n_layers} -- "
        f"these v2 directions were not built for this model, refusing to proceed."
    )

    delta_R = {mech: torch.zeros(len(ids), n_layers) for mech in ALL_CONDS}
    delta_H = {mech: torch.zeros(len(ids), n_layers) for mech in ALL_CONDS}
    delta_H_perp = {mech: torch.zeros(len(ids), n_layers) for mech in ALL_CONDS}
    valid_mask = {mech: torch.zeros(len(ids), dtype=torch.bool) for mech in ALL_CONDS}
    failures = []

    for i, pid in enumerate(ids):
        try:
            plain_text = by_id[pid]['plain']
            t_inst_p, t_post_p, acts_p = extract_sequence_and_positions(
                model_base, plain_text, args.model_alias, n_layers)
            plain_at_inst = torch.stack([acts_p[l][t_inst_p] for l in range(n_layers)])  # [n_layers, d]
            plain_at_post = torch.stack([acts_p[l][t_post_p] for l in range(n_layers)])  # [n_layers, d]
        except ValueError as e:
            failures.append({'id': pid, 'condition': 'plain', 'error': str(e)})
            continue  # can't compute any mechanism's diff without plain -- skip this instruction entirely

        for mech in ALL_CONDS:
            try:
                mech_text = by_id[pid][mech]
                t_inst_m, t_post_m, acts_m = extract_sequence_and_positions(
                    model_base, mech_text, args.model_alias, n_layers)
                mech_at_inst = torch.stack([acts_m[l][t_inst_m] for l in range(n_layers)])
                mech_at_post = torch.stack([acts_m[l][t_post_m] for l in range(n_layers)])

                diff_at_inst = mech_at_inst - plain_at_inst  # [n_layers, d]
                diff_at_post = mech_at_post - plain_at_post  # [n_layers, d]

                delta_R[mech][i] = (diff_at_post * refusal_hat).sum(-1)
                delta_H[mech][i] = (diff_at_inst * harmfulness_hat).sum(-1)
                delta_H_perp[mech][i] = (diff_at_inst * harmfulness_perp_hat).sum(-1)
                valid_mask[mech][i] = True
            except ValueError as e:
                failures.append({'id': pid, 'condition': mech, 'error': str(e)})

        if (i + 1) % 32 == 0 or i == len(ids) - 1:
            print(f"  {i + 1}/{len(ids)}  ({len(failures)} failures so far)")

    if failures:
        print(f"\n{len(failures)} position-finding failures out of {len(ids) * len(ALL_CONDS)} "
              f"attempts -- these instructions/conditions are excluded from delta_R/delta_H "
              f"(marked invalid in valid_mask, zeros in the tensor, NOT silently treated as 0-effect):")
        for f in failures[:20]:
            print(f"  [{f['id']}][{f['condition']}] {f['error'][:150]}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more (see saved .pt file's 'failures' field)")
        failure_rate = len(failures) / (len(ids) * len(ALL_CONDS))
        assert failure_rate < 0.05, (
            f"failure rate {failure_rate:.1%} is too high to proceed silently -- "
            f"investigate the position-finding errors above before trusting this data."
        )

    # placebo-calibrated versions (raw H and orthogonalized H alike)
    delta_R_pc = {mech: delta_R[mech] - delta_R['placebo'] for mech in REAL_MECHS}
    delta_H_pc = {mech: delta_H[mech] - delta_H['placebo'] for mech in REAL_MECHS}
    delta_H_perp_pc = {mech: delta_H_perp[mech] - delta_H_perp['placebo'] for mech in REAL_MECHS}

    payload = {
        'instruction_ids': ids, 'n_layers': n_layers, 'ids_key': args.ids_key,
        'delta_R': delta_R, 'delta_H': delta_H, 'delta_H_perp': delta_H_perp,
        'delta_R_placebo_calibrated': delta_R_pc, 'delta_H_placebo_calibrated': delta_H_pc,
        'delta_H_perp_placebo_calibrated': delta_H_perp_pc,
        'valid_mask': valid_mask, 'failures': failures,
    }
    logical_meta = build_delta_metadata(
        model=args.model_alias, lang=args.lang, suffix=args.suffix, ids_key=args.ids_key,
        active_mechanisms=REAL_MECHS, n_instructions=len(ids), n_layers=n_layers,
        refusal_direction_path=refusal_pt, harmfulness_direction_path=harmfulness_pt,
        token_position_R='t_post', token_position_H='t_inst', estimator='mean',
        git_commit=current_git_commit(SCRIPT_DIR),
        extra={'refusal_version': args.refusal_version, 'n_failures': len(failures),
               'limit_ids': args.limit_ids,
               'cos_r_h_per_layer': cos_r_h.tolist(),
               'h_perp_norm_ratio_per_layer': h_perp_norm_ratio.tolist()},
    )
    os.makedirs(out_dir, exist_ok=True)
    # Atomic: payload written+renamed first, metadata (with the payload's own
    # content hash) only written if that succeeded -- same discipline as the
    # direction files. A metadata JSON existing is now proof the .pt exists,
    # matches, and was computed against the CURRENT canonical mechanism set.
    save_delta_atomic(payload, logical_meta, out_path)
    print(f"\nSaved: {out_path} (+ metadata, atomic)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path',  type=str, required=True)
    parser.add_argument('--model_alias', type=str, required=True)
    parser.add_argument('--output_dir',  type=str, default=os.path.join(SCRIPT_DIR, '..', 'output'))
    parser.add_argument('--lang',        type=str, default='en')
    parser.add_argument('--suffix',      type=str, default='_full572')
    parser.add_argument('--ids_key',     type=str, default='direction_ids')
    parser.add_argument('--refusal_version', type=str, default='v3', choices=['v2', 'v3'],
                         help="Which refusal_direction to project delta_R onto. Protocol default is "
                              "v3 (behavioral, refused-vs-accepted). v2 (harmful-vs-harmless) exists "
                              "so a minimal pilot/dry-run can test this script's delta-computation "
                              "machinery without needing a v3 rebuild (which requires generation + "
                              "WildGuard judging).")
    parser.add_argument('--limit_ids',   type=int, default=None,
                         help="Cap the matched instruction list to the first N (after sorting) -- for "
                              "a minimal pilot run. Deterministic, not a random sample. None = no cap "
                              "(the real full-scale run).")
    parser.add_argument('--dry_run',     action='store_true',
                         help="Print planned config (model/ids_key/counts/shapes/paths) and exit "
                              "without loading the model or writing any file.")
    args = parser.parse_args()
    main(args)
