> **SUPERSEDED (2026-09-09).** Replaced by
> `EXPERIMENT2_CONTEXT_REPRESENTATION_PROTOCOL.md`, which incorporates:
> the corrected Experiment 1 conclusion (CO/MG geometric validity is
> model-dependent, not "fails in all 3 models"), the frozen CO/MG
> prototype construction (per-mechanism normalize-then-average), the
> frozen random-subspace null (Holm-corrected), and instruction-cluster
> (not per-instruction) bootstrap resampling to account for duplicate
> normalized-text instructions in `direction_ids`. This document's
> 21,600-forward extraction design (Sec 2-3) and script references
> (`scripts/54`/`55`) remain the basis for the current design -- only the
> analysis-side statistics and the Experiment 1 framing changed. Kept for
> history; do not delete or revert. Do not read the rest of this file as
> current guidance without cross-checking the superseding document.

# Formal C-Direction Estimation Protocol -- Design Stage

Extends `EXPERIMENT2_CONTEXT_ACTIVATION_PILOT_PROTOCOL.md` (the "pilot
protocol" below) from a single-model, 30-instruction pilot to the formal,
3-model, 300-instruction estimation. **This document is design-only** --
no model has been run against it. It authorizes design, code, and
dry-run testing only.

Everything the pilot protocol already established and this document does
not explicitly revise carries over unchanged: the `locate_instruction_end`
longest-common-prefix position-finding method (pilot protocol Sec 3/13),
the paired-difference direction formula (Sec 4), the atomic-write /
non-overwrite / access-log discipline, and the general fail-fast
philosophy. This document does **not** re-derive those; it only states
what's different at formal scale and what the pilot's frozen decision
(Sec 15) means for the formal design.

## 1. Frozen inputs from the pilot (not re-litigated here)

Per pilot protocol Sec 15 (2026-09-08 freeze):
- **Primary token position: `t_inst`.** `t_post` is a secondary
  format-position sensitivity analysis only.
- **Primary C representation: family-specific.**
  `C_m = (C_persona, C_authority, C_fictional, C_continuation)` -- no
  single shared direction, no pre-fixed `k`.
- **`k=2` (or any `k`) SVD compression is secondary/exploratory only**,
  never adopted before the formal experiment's own subspace-stability
  diagnostics are seen (Sec 8 below).
- **All 4 families proceed.** None dropped.
- Formal results must be described as a **"context-framing-induced
  effect"**, never a "pure semantic context direction" -- the
  length/token-count confound is reduced but not eliminated by the
  `t_inst` switch (pilot: r=0.76 Pearson / 0.85 Spearman, n=12, still
  present after the switch, and its *sign* differs from `t_post`'s,
  which is itself informative about position-dependence).

## 2. Scope

- **Models**: `Qwen2.5-7B-Instruct`, `Meta-Llama-3.1-8B-Instruct`,
  `gemma-2-9b-it` -- all 3, each processed independently as its own run
  (never mixed into one tensor; see Sec 9 on cross-model comparison).
- **Instructions**: `data/splits.json`'s `direction_ids` key, **all
  300** (not a 30-instruction subset). `validation_ids`/`test_ids`
  remain forbidden, enforced the same way as the pilot (an `AccessLog`
  that only ever indexes `direction_ids`).
- **Conditions**: the same 16 context conditions
  (`templates/templates_context_v1.json`, 4 families x (3 positive + 1
  neutral)) as the pilot. The pilot's 4 canonical comparison conditions
  (`plain`/`placebo`/`persona_roleplay`/`prefix_injection`) are also
  re-extracted per model, per instruction, for the canonical
  attack-profile projection (Sec 7) -- **6 canonical mechanisms total**
  are needed for Sec 7's full picture (the pilot's 4 plus
  `refusal_suppression`, `encoding_obfuscation`, `payload_splitting`,
  `distractors_negated`; the pilot only extracted 4 of the project's 6
  active canonical mechanisms and explicitly marked the
  `ctx_continuation` vs. `payload_splitting` comparison
  `NOT_EXTRACTED_THIS_ROUND` -- the formal run closes that gap and
  extracts **all 6** active canonical mechanisms, not just 4).
- **Primary layer**: unchanged, `floor(0.6 * n_layers)` per model
  (Qwen=16/28, Llama=19/32, Gemma=25/42 -- pre-registered, from the R/H
  rebuild work, never re-selected from outcome data). All layers are
  still collected for robustness description, primary layer is the
  formal-report basis.
- **Token positions**: both `t_inst` (primary) and `t_post` (secondary)
  collected for every forward pass, same as the pilot.
- **Estimator**: `mean` (paired mean difference), same as the pilot.
- **Bootstrap**: 2000 resamples (pilot used 1000; doubled for the formal
  run per this round's instruction), resampled by instruction.
- **`result_status` for every formal extraction/analysis output**:
  `FORMAL_DIRECTION_ESTIMATION` -- **not** `PILOT_NON_RESULT` (this is
  the graduation point from pilot to formal scale), but also **not**
  `VALIDATED_C_DIRECTION` or any language claiming the C-dimension
  question itself is settled -- `FORMAL_DIRECTION_ESTIMATION` describes
  the *scale and rigor* of the estimation, not a conclusion about
  whether/how C generalizes as a construct.

## 3. Forward-pass count (dynamically asserted, never hardcoded)

Expected: `3 models x 300 instructions x 20 conditions (16 context + 4
canonical-for-discriminant... see Sec 2 above -- actually 6 canonical for
the full attack-profile) = ...`. The exact figure depends on the final
canonical-mechanism count decided in Sec 2 (4 vs 6); **whichever is
used, the extraction script must derive both the condition count and the
total forward-pass count from the actual constructed condition list at
run time** (`assert n_conditions == len(context_conditions) +
len(canonical_conditions)`, `assert n_forward_passes == n_ids *
n_conditions * n_models` or per-model equivalent), exactly as the pilot
script already does -- **the `14,400` figure named in this round's
request (3 x 300 x 16) covers context conditions only; if the 6-canonical
extension in Sec 2 is adopted, the real total is higher and must be
computed, not assumed to be 14,400.** This document does not pre-commit
to a single hardcoded total for this reason.

## 4. Per-instruction paired differences (not aggregated)

Per model, saves the **full per-instruction paired-difference tensor**
for all 300 instructions x 16 context conditions x n_layers x 2
positions -- never only a per-family mean. This mirrors the pilot's
`context_paired_diffs` schema exactly, just at `n_instructions=300`
instead of 30. Canonical conditions are saved as raw per-condition
activations (not pre-differenced), same rationale as the pilot (lets
`d_m`, `d_placebo`, and placebo-calibrated `tilde_d_m` all be recomputed
downstream from one saved tensor).

## 5. Fail-fast gates (extends pilot protocol Sec 10)

All of the pilot's 14 fail-fast rules apply per model, plus:
15. `n_ids` must equal exactly 300 for every model (not silently
    proceeding with fewer if `splits.json`'s `direction_ids` shrinks).
16. Per-model layer count / primary layer must match the pre-registered
    value for that specific model alias (Qwen=16/28, Llama=19/32,
    Gemma=25/42) -- a config mismatch for ANY one model halts only that
    model's run, not the others (each model's extraction is independent).
17. The condition set (context + canonical) must be identical across all
    3 models -- if a per-model template-rendering quirk changes which
    conditions exist for one model, this is a hard stop, not a per-model
    divergence to silently tolerate.

## 6. Output schema (per model)

```
output/context_activations_formal/<model_alias>/
  context_paired_diffs_FORMAL_direction300.pt
  canonical_activations_FORMAL_direction300.pt
  context_token_position_audit_FORMAL.json
  context_activation_formal_summary.json
  context_activation_formal_metadata.json
```

Metadata fields mirror the pilot's (Sec 9 of the pilot protocol) with
`result_status: "FORMAL_DIRECTION_ESTIMATION"`, `n_ids: 300`,
`ids_key: "direction_ids"`, `bootstrap_resamples: 2000`, plus the same
provenance hash set (template content, checklist, token audit,
provenance attestation, splits file, model config).

## 7. Canonical attack-profile projection (per model)

For each of the (up to) 6 active canonical mechanisms `m`, compute the
placebo-calibrated direction `tilde_d_m = d_m - d_placebo` (same formula
as the pilot's discriminant check, Sec 7 of the pilot protocol), then
project it onto each of the 4 family-specific directions:

```
profile(m) = ( cos(tilde_d_m, C_persona), cos(tilde_d_m, C_authority),
               cos(tilde_d_m, C_fictional), cos(tilde_d_m, C_continuation) )
```

**Signed values are kept** (not absolute value) -- both the raw
projection (`tilde_d_m . C_f`, unnormalized, reflecting `tilde_d_m`'s
magnitude along `C_f`) and the cosine (normalized) are reported per
`(m, family)` pair, so a downstream reader can distinguish "small angle,
small magnitude" from "small angle, large magnitude." This produces one
4-dimensional "context-family profile" per canonical mechanism per
model -- a descriptive fingerprint of which context families a given
attack mechanism's direction most resembles, not a claim that the
mechanism *causes* or *is* that context family's effect.

## 8. `k=2` as exploratory-only (extends pilot protocol Sec 15 item 4)

Per model, computed but **not adopted** as the formal representation:
- `k=1`, `k=2`, `k=3` variance-explained from the SVD of the 4
  (centered) family centroids.
- **Bootstrap stability of the `k=2` subspace**: for each of the 2000
  instruction-level resamples, recompute the 4 family centroids and
  their top-2 SVD subspace; report the distribution of principal angles
  (or subspace cosine) between each resampled subspace and the
  full-sample `k=2` subspace.
- **Split-half principal-angle / subspace stability**: same repeated
  (>=100 seed) instruction-partition design as the pilot's split-half,
  but comparing the `k=2` subspace fit on each half rather than a single
  direction's cosine.
- **Leave-one-family-out (LOFO) reconstruction**: project the held-out
  family's centroid onto the `k=2` subspace fit from the other 3
  families; report both the reconstruction cosine and the residual norm
  (fraction of the held-out centroid's norm not captured by the
  subspace).
- Only after all of the above are seen, across all 3 models, does the
  user decide whether `k=2` (or any shared-subspace representation)
  replaces the family-specific default for any downstream use --
  **this decision is explicitly out of scope for this document.**

## 9. Cross-model comparison (RSA, not direct coordinate comparison)

Per this round's explicit instruction, **raw direction coordinates are
never compared across models** (different models have different hidden
spaces with no shared basis -- a coordinate-level comparison would be
meaningless). Instead:
- **Representational similarity analysis (RSA)**: for each model,
  compute the 4x4 family-pairwise-cosine Gram matrix (Sec 8-C style, at
  `t_inst`/primary layer). Compare Gram matrices **across models** via
  correlation of their (flattened, off-diagonal) entries -- this asks
  "do the families relate to each other in the same *pattern* across
  models," not "do they point the same way in some shared space."
- **Canonical mechanism context-profile ranking**: for each model, rank
  the 4 families by their cosine to a given canonical mechanism's
  `tilde_d_m` (Sec 7); compare the **rank order** across models
  (Spearman correlation of ranks), not raw cosine magnitudes.
- **Per-dimension rank consistency**: for each of the 4 families, is its
  relative position (highest/lowest cosine to a given canonical
  mechanism, or highest/lowest within-family stability) consistent
  across the 3 models, or does it vary?
- **Explicit "model-specific structure" reporting**: any pattern that
  holds for only 1-2 of the 3 models is named as such, not
  generalized -- this protocol does not presuppose the 3 models share
  the same C-family structure.

## 10. Not run this round

No model has been loaded. No GPU job has been submitted. No
`validation_ids`/`test_ids` have been read. No `.generate(...)` call, no
WildGuard, no steering exists anywhere in the formal scripts (Sec 12
below). This document, the formal extraction script
(`scripts/54_extract_formal_context_activations.py`), and the formal
analysis script (`scripts/55_analyze_formal_context_activations.py`)
authorize design and dry-run testing only -- the 3-model, 300-instruction
GPU run requires separate, explicit authorization from the user before
any `sbatch` submission.
