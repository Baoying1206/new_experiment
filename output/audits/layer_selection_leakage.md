# Layer-Selection Leakage Audit

Scope: `scripts/14_find_safety_layer.py` and its one downstream consumer,
`scripts/16_single_layer_geometry.py`. Code-inspection only, no execution
needed — confirmed by direct read + grep of both files plus
`04_extract_directions_and_analyze.py` and `10a_calibrate_injection_alpha.py`.

## What 14_find_safety_layer.py actually does

- Reads `refusal_dir_{lang}.pt` (`[n_layers, d_model]`) from
  `--refusal_dir_root` (an `experiment_thesis/output/jailbreak_analysis`
  path) for all 9 pilot languages × 3 models.
- Criterion: layer with the largest L2 norm of `refusal_direction`, searched
  only over the first 80% of layers (`cutoff = int(0.8 * n_layers)`) to avoid
  the trivial last-layer argmax caused by residual-stream norm growth.
- Picks `mode_layer` (most common per-language peak) and `mean_layer` per
  model. Writes `output/safety_layer_identification.json`.
- **Not** based on classification accuracy, ASR, or any behavioral/test
  outcome — it is a pure function of `refusal_direction`'s geometry. So the
  literal thing the user was worried about ("layer chosen because it gives
  the best test ASR / best binary-classification separation") does **not**
  happen here — confirmed false by reading the code, not assumed.

## Where the actual leak is

`grep -n "splits.json\|validation_ids\|test_ids\|direction_ids\|ids_key\|--split"` across
`04_extract_directions_and_analyze.py`, `10a_calibrate_injection_alpha.py`,
and `14_find_safety_layer.py` returns **zero matches in all three files.**

`refusal_dir_{lang}.pt` (the input to 14) is built by
`experiment_thesis/scripts/extract_jailbreak_vectors.py`, which:
- predates `data/splits.json` entirely (it was run against the original
  9-language × 75-instruction pilot, before the leak-free 572-instruction
  English split existed),
- uses `positions=[-1]` for both refusal_direction and harmfulness_direction
  (see `EXPERIMENT_REDUCTION_PLAN.md` — the position-conflation finding),
- has no concept of `direction_ids`/`validation_ids`/`test_ids` at all.

So the leak is **not** "14 peeks at test behavioral results to pick a layer."
It is: **the layer-selection input itself (`refusal_dir_{lang}.pt`) was built
from data that is not partitioned relative to the current
direction/validation/test split at all** — i.e. the norms 14 is maximizing
over were computed from a pool that, relative to `data/splits.json`,
contains an unknown mix of what are now direction_ids, validation_ids, and
test_ids, undifferentiated. Any layer choice made this way cannot be
described as "selected using the validation set" — it was selected using
an un-partitioned pre-split pool, which is neither correct validation-only
selection nor test-outcome leakage in the classic sense, but is also not
usable as-is for a study that requires validation/test separation.

## Downstream consumer: 16_single_layer_geometry.py

- Takes `--layer` as an explicit CLI argument (not re-derived) — good, no
  double-dipping/cherry-picking within 16 itself.
- But its `--layer` value in practice comes from 14's `mode_layer`/
  `per_lang_peak_layer`, which inherits the same un-partitioned-input problem
  above.
- 16 then recomputes `cos(refusal)`/`cos(harmfulness)`/`frac_along_refusal`
  at that single layer per language/model — i.e. it is a **primary-analysis
  layer selection** for exactly the kind of geometry test Experiment 1/2
  need, so this is squarely in scope for the "must be fixed" requirement.

## Outputs that must be marked stale

| File | Depends on old layer selection? | Status |
|---|---|---|
| `output/safety_layer_identification.json` | Is the old layer selection itself | `stale` — built from un-partitioned `refusal_dir_{lang}.pt`, no split-awareness |
| Any `16_single_layer_geometry.py` run using `mode_layer`/`per_lang_peak_layer` from the file above | Yes, directly | `stale_for_dual_axis_claim` (no such run's output file currently exists in `output/` — 16 has not been executed yet, confirmed by absence of a matching output file in `output/*.json`) |
| `output/layerwise_profile.json`, `output/layerwise_cross_language.json` | No — these come from `12_layerwise_profile.py`/`13_layerwise_cross_language.py`, which report across ALL layers, not a single selected layer; not consumers of 14's output (confirmed: `grep -rln "safety_layer_identification\|mode_layer" scripts/` returns only `14_find_safety_layer.py` and `16_single_layer_geometry.py`) | not stale from this issue, but still built on old single-position `positions=[-1]` directions — separately flagged in `EXPERIMENT_REDUCTION_PLAN.md` |

## Required fix (per Decision 2)

1. Rebuild `refusal_dir_{lang}.pt` / `harmfulness_dir_{lang}.pt` for English
   using the new dual-position extraction (`t_inst`/`t_post`,
   `scripts/utils/token_positions.py`) and an explicit `--split` argument so
   direction-construction only touches `direction_ids` (300, or the
   cross-fitting scheme's 4-fold subsets — see `axis_source_overlap.md`'s
   Decision-3 recommendation).
2. Add `--split validation` to `14_find_safety_layer.py`: it must compute its
   peak-norm criterion (or the newly-preferred split-half
   reliability / template-placebo separation / reference-direction
   reliability criteria per Decision 2) using activations from
   `validation_ids` (72) only — never `test_ids`, never the full
   un-partitioned pool.
3. Add the pre-registered relative-layer sensitivity check:
   `layer = floor(0.6 * n_layers)` computed alongside whatever the
   validation-selected layer is, and report both — this is a fixed rule, not
   dependent on any data, so it can be added without touching leakage
   concerns.
4. Do **not** unilaterally fix a final selection threshold/rule yet — per the
   user's explicit instruction, output candidate rules and their effects on
   `validation_ids` first, then let the user pick, before locking a rule in
   for the primary Experiment 1–3 analysis layer.
