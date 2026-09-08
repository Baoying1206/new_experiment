# Context Activation Pilot Protocol -- Design Stage

Records the frozen design for a small, single-model **pilot** activation
extraction over the 4 context-reconfiguration families
(`templates/templates_context_v1.json`, `status:
HUMAN_AND_TOKEN_AUDITED_READY_FOR_ACTIVATION_PILOT`, per
`EXPERIMENT2_CONTEXT_RECONFIGURATION_PROTOCOL.md`).

**This document is design-only.** No model has been run against it. No
extraction script exists yet (see §12 for the planned, not-yet-created,
script names). Everything in this document is either a frozen design
decision, a fail-fast rule, or a dry-run (synthetic, CPU-only) test --
never a claim about real activation data, because none exists.

## 0. Purpose and non-purpose

The pilot exists to verify that the **generation -> position-finding ->
activation-extraction -> direction-computation** pipeline is mechanically
correct for context-template prompts specifically (multi-line templates,
shared neutral controls, response-position cues) *before* committing to a
full formal run across all 3 models and the full `direction_ids` set.

**The pilot is explicitly NOT**:
- A test of H1/H2/H3 (§6) -- `n_family=4` gives an SVD with far too few
  points to be anything but a descriptive diagnostic (§6 says this
  explicitly; do not read pilot SVD results as a hypothesis-test outcome).
- A claim that any C direction exists, is shared, or is independent of
  canonical mechanisms.
- A source of ASR/FRR or any behavioral (compliance/refusal) result --
  no completions are generated, no WildGuard judging occurs (§1).
- Formal evidence for a thesis chapter. Every output file's
  `result_status` must read `PILOT_NON_RESULT`.

## 1. Pilot scope (frozen)

- **Model**: `Meta-Llama-3.1-8B-Instruct` only. Rationale (recorded, not
  re-litigated here): Llama showed near-orthogonal R/H at its primary
  layer in the R-axis work (`cos(R,H)=+0.048` at layer 19), giving the
  cleanest signal for a first pipeline check; Qwen/Gemma are deferred to
  the formal run.
- **Data**: the first 30 IDs of `data/splits.json`'s `direction_ids`, in
  the order they appear in that file:
  `p002, p003, p004, p005, p007, p009, p010, p012, p015, p017, p021,
  p022, p024, p025, p026, p027, p028, p030, p032, p033, p034, p038,
  p039, p043, p044, p047, p048, p049, p050, p052`.
  **`validation_ids` and `test_ids` are forbidden** -- any code path that
  reads either list for use as pilot instructions is a fail-fast
  violation (§10 item 5).
- **Primary layer**: layer 19 (Llama's pre-registered primary layer,
  `floor(0.6*32)`, unchanged from the R/H rebuild). **All-layer results**
  are collected too, but used only for diagnostic/robustness description
  -- never as a substitute primary-layer claim, and never used to
  cherry-pick a better-looking layer post hoc.
- **No generation**: prompts are tokenized and forwarded once
  (teacher-forced on the rendered prompt itself) to collect hidden
  states at the relevant positions -- the model is never asked to
  produce new tokens.
- **No WildGuard**: no compliance/refusal judgment of any kind is
  computed or needed for this pilot.
- **No steering, no alpha selection, no ASR/FRR**: this pilot measures
  representation geometry only, never behavioral intervention.
- **Every output file's `result_status` must be `PILOT_NON_RESULT`.**

## 2. Conditions (computed, not hardcoded)

Two condition families, applied to each of the 30 instructions:

**Context conditions (16)**: all 4 families of
`templates/templates_context_v1.json` -- 3 positive variants + 1
`family_specific_neutral_control` each (12 + 4 = 16).

**Canonical comparison conditions (4)**: `plain` (no template --
`02_build_templated_data.py`'s `condition: 'plain', mechanism: 'none'`
row), `placebo` (the `templates_en.json['placebo']` template), and two
canonical mechanisms: `persona_roleplay`, `prefix_injection` (chosen for
their §7 discriminant comparisons against `ctx_persona` and
`ctx_continuation` respectively -- see §7).

**Total conditions per instruction = 16 + 4 = 20.** Total forward passes
= 30 instructions x 20 conditions = **600**.

This 600 figure MUST be asserted at runtime from the actual condition set
actually constructed (e.g. `assert n_conditions == 20` derived from
`len(all_template_strings(...)) + len(CANONICAL_COMPARISON_CONDITIONS)`,
and `assert n_forward_passes == len(instruction_ids) * n_conditions`) --
never hardcoded as a bare literal with no derivation, per the user's
explicit instruction. See §11's dry-run test for the synthetic version of
this assertion.

## 3. Token positions: t_inst and t_post

Both positions are collected for every one of the 600 forward passes.
This reuses the existing, already-built infrastructure in
`scripts/utils/token_positions.py` rather than a new implementation:

- **`t_post`** (primary measurement): `get_post_instruction_position()`
  -- the last token of the fully-rendered prompt. Rationale: by this
  point the model has processed the complete context manipulation
  (persona assignment, authority claim, fictional frame, or continuation
  cue), so this is where "has the context effect happened" is best read.
- **`t_inst`** (position-sensitivity diagnostic): the last token of the
  **`{instruction}` content itself**, located via
  `get_instruction_end_position()`'s literal-subsequence search (not
  `get_user_turn_end_position()`'s structural end-of-turn search --
  the latter finds the end of the *entire rendered user turn*, which for
  a context template is close to `t_post`, not the instruction span
  specifically; the literal-subsequence method is what actually answers
  "where does the injected instruction end").

**Why both, and why this matters differently per family**: `ctx_persona`,
`ctx_authority`, and `ctx_fictional` place their context-manipulation
text *before* `{instruction}`, so `{instruction}` sits at or near the end
of the template -- `t_inst` and `t_post` are close together (differing
mainly by chat-template closing tokens). `ctx_continuation` places
response-position cues (`Response:` / `Speaker B:` / `New response:`)
*after* `{instruction}`, so `t_inst` necessarily falls **before** those
cues -- `t_inst` for `ctx_continuation` reflects the state *before* the
continuation-specific structural cue is seen, and must not be
interpreted as capturing the full continuation manipulation.

**Pre-registered interpretation rules**:
- `t_post` is the primary measurement for all families.
- `t_inst` is a position-sensitivity diagnostic, not an alternative
  primary measurement.
- For `ctx_continuation` specifically, `t_inst` cannot be read as "the
  full continuation effect" because it necessarily precedes the
  response-position cue -- any `ctx_continuation` conclusion drawn only
  from `t_inst` is invalid on its face.
- If a conclusion holds at `t_post` but not at `t_inst` (or vice versa),
  the format-position-cue confound (already flagged in
  `EXPERIMENT2_CONTEXT_RECONFIGURATION_PROTOCOL.md` Sec 7) must be
  considered before any mechanism-level interpretation.
- **The token position is never chosen after seeing which one gives a
  cleaner result.** Both are always reported.

**Instruction character-span to token-span mapping must be verified per
rendered sample.** A random manual audit (human-reviewed, per the
existing convention in `scripts/audits/audit_token_positions.py`) must
cover, at minimum:
- 4 families x (1 positive + 1 neutral) = 8 context-condition samples,
- plus `plain`, `placebo`, `persona_roleplay`, `prefix_injection`,
- for a total of **at least 12** condition samples reviewed by a human
  before any pilot result is treated as trustworthy. This audit is
  planned as an extension of the existing `audit_token_positions.py`
  pattern (§12) -- not built this round.

## 4. Context-variant direction formulas (frozen)

For family `f`, variant `v`, instruction `i`, layer `l`, position `p`:

```
delta_{i,f,v}^(l,p) = h_{i,f,v,+}^(l,p) - h_{i,f,neutral}^(l,p)
```

(the shared `family_specific_neutral_control` activation for instruction
`i`, family `f` -- the SAME neutral is reused across all 3 of that
family's variants, per
`EXPERIMENT2_CONTEXT_RECONFIGURATION_PROTOCOL.md` Sec 3.)

Variant direction (mean over the 30 instructions):

```
d_{f,v}^(l,p) = mean_i delta_{i,f,v}^(l,p)
```

Family direction (mean over the 3 variants):

```
d_f^(l,p) = (1/3) * sum_v d_{f,v}^(l,p)
```

**Repeated-measures warning (binding on all downstream statistics)**:
because the same neutral is reused across a family's 3 variants, the 3
`delta_{i,f,v}` values for a given instruction `i` are **not**
independent of each other. Any statistical test (bootstrap CI,
significance test) must resample **whole instructions** (all of that
instruction's condition activations together), never treat the
30 instructions x 3 variants = 90 `(instruction, variant)` pairs as 90
independent samples.

## 5. Diagnostics to report

Per variant, per family, per layer, per token position:
- direction norm (`||d_{f,v}^(l,p)||`)
- mean paired projection (mean of `delta_{i,f,v} . d_{f,v} / ||d_{f,v}||`
  over instructions, or equivalent paired-projection statistic)
- bootstrap 95% CI, **1000 resamples, resampled by instruction** (per §4)
- split-half reliability (cosine between directions built from two random
  instruction halves)
- pairwise cosine among the 3 variants within a family
- cosine between the family centroid (`d_f`) and each of its 3 variants
- **leave-one-variant-out**: for each variant, predict it from the mean
  of the other 2 variants (cosine of the held-out variant's own direction
  vs. the 2-variant mean), per family
- cosine between the `t_inst`-based and `t_post`-based direction for the
  same `(f, v, l)`
- positive-vs-neutral token-length difference (already measured in the
  real token-length audit; reported alongside for the sensitivity
  analysis in §8)

**No single metric decides whether "C" exists.** All of the above are
reported together; none is elevated to a pass/fail gate in the pilot.

## 6. Candidate C-structure analysis (descriptive only)

Four non-exclusive, non-presupposed candidate outcomes, unchanged from
`EXPERIMENT2_CONTEXT_RECONFIGURATION_PROTOCOL.md` Sec 5:
- **H0**: no stable context structure.
- **H1**: the 4 families share one approximately common direction.
- **H2**: the 4 families form a low-dimensional shared context subspace.
- **H3**: only family-specific (or model-specific) directions exist, no
  shared structure.

**Pilot SVD is descriptive only**:
- Input: the 4 family centroid directions (`d_f^(l,p)` for the 4
  families), at the primary layer/position.
- Pre-registered candidate `k=2` (unchanged from the frozen design).
- Report singular values and fraction of variance explained.
- **`n_family=4` is far too few points for this to be formal statistical
  evidence of anything** -- it must never be described as such in any
  write-up derived from the pilot.
- **`k` is never re-selected after seeing the pilot's singular values** to
  make a post-hoc-chosen `k` look like it was the pre-registered decision.

## 7. Discriminant checks against canonical mechanisms

Canonical mechanism directions are **re-extracted from the same 30
`direction_ids`** used for the context conditions (not reused from any
existing artifact lacking matching per-instruction/per-position
metadata -- e.g. not reused from `output/output_v3_behavioral_refusal/`
or any full-572-pool artifact, which do not share this pilot's exact
instruction set or position conventions):

```
d_m = mean_i[ h(T_m(x_i)) - h(plain(x_i)) ]
tilde_d_m = d_m - d_placebo
```

(placebo-calibrated, per the existing `templates_en.json['placebo']`
convention.)

**Primary discriminant comparisons**:
- `cos(d_ctx_persona, tilde_d_persona_roleplay)` -- tests the §4
  boundary caveat in the context-reconfiguration protocol (ctx_persona
  vs. persona_roleplay).
- `cos(d_ctx_continuation, tilde_d_prefix_injection)` -- tests the §7
  canonical-overlap judgment recorded for `ctx_continuation`.

**Secondary comparison**:
- `ctx_continuation` vs. `payload_splitting` -- **only if
  `payload_splitting` is extracted this round**. If not extracted (the
  pilot's canonical comparison set is `plain`/`placebo`/
  `persona_roleplay`/`prefix_injection` only -- §2), this comparison is
  marked `NOT_EXTRACTED_THIS_ROUND -- formal-experiment follow-up item`
  in the pilot output. **No comparison result may be fabricated or
  approximated for a mechanism that was not actually extracted.**

**Construction-asymmetry caveat (must appear in any write-up of these
cosines)**: the context direction (`d_{f,v}` in §4) is a
**positive-minus-shared-neutral** contrast; the canonical direction
(`tilde_d_m` above) is a **template-minus-plain, placebo-calibrated**
contrast. These are constructed differently. A cosine between them is a
**discriminant diagnostic** (does the context direction point somewhere
recognizably similar to a canonical mechanism's direction), **not** a
symmetric, apples-to-apples mechanism-identity test.

## 8. Length-sensitivity cross-check

From the real, committed token-length audit
(`output/audits/context/context_templates_token_length_audit.json`,
attested content-identical to the currently pilot-approved template text
via `output/audits/context/context_templates_token_audit_provenance_attestation.json`):
- `ctx_authority`, `ctx_fictional`, `ctx_persona`: all 3 positive variants
  are systematically **longer** than their family's neutral, in all 3
  models.
- `ctx_continuation`: all 3 positive variants are systematically
  **shorter** than their family's neutral (**-4 to -5 tokens**), in all
  3 models -- the opposite direction.
- 0/12 pooled positive variants flagged by the 1.5x-IQR convention, in
  any model.

**Pre-registered interpretation rules** (unchanged from the round that
accepted the token-length audit):
- If `ctx_continuation` -- despite its reversed length-gap direction --
  still aligns with the other 3 families in activation space (§6), a
  shared structure is **not easily explained** by positive-minus-neutral
  length direction alone.
- If `ctx_continuation` separates from the other 3 families, this cannot
  by itself distinguish a genuine mechanism difference from a length
  confound or the format-position-cue confound (§3) -- both must be
  considered.
- The pilot reports the **descriptive correlation** between direction
  norm (or primary paired projection) and positive-vs-neutral
  token-length difference, across the 12 positive variants. With only
  12 points this is a **sensitivity diagnostic only**, never a
  standalone statistical conclusion.

## 9. Pilot output schema

Proposed layout:
```
output/context_activation_pilot/
  Meta-Llama-3.1-8B-Instruct/
    context_activations_PILOT_direction30.pt
    context_directions_PILOT_direction30.pt
    context_activation_pilot_summary.json
    context_activation_pilot_metadata.json
```

`context_activation_pilot_metadata.json` must record, at minimum:
- `result_status = "PILOT_NON_RESULT"`
- `git_commit` (the commit the extraction script itself was run at)
- `model_path`, `model_alias`
- `model_config_hash` (hash of the model's `config.json`, or equivalent
  -- confirms which exact model weights/config were loaded, per this
  project's existing convention of never trusting a bare alias string)
- `source_template_file_sha256` (of `templates/templates_context_v1.json`
  as loaded at run time)
- `template_content_sha256` (per
  `scripts/utils/context_template_provenance.py`'s schema -- confirms the
  16 rendered texts, not just the whole file)
- `human_review_checklist_sha256` (the currently-effective checklist,
  `context_templates_human_review_checklist_v5.json` as of this writing
  -- must be re-resolved to "whichever is latest" at run time, not
  hardcoded to v5, since a later review round could supersede it)
- `token_audit_report_sha256`
  (`context_templates_token_length_audit.json`)
- `provenance_attestation_sha256`
  (`context_templates_token_audit_provenance_attestation.json`)
- `splits_file_sha256` (`data/splits.json`)
- `instruction_ids` (the exact 30 IDs, verbatim)
- `ids_key = "direction_ids"`
- `test_data_read = false` (and `validation_data_read = false`)
- `layers` (full list collected) and `primary_layer = 19`
- `token_positions = ["t_inst", "t_post"]`
- `dtype` (activation storage dtype)
- `estimator` (mean-difference, per §4's formula)
- `bootstrap_resamples = 1000`, `bootstrap_resample_unit = "instruction"`
- `random_seed`
- output tensor shapes and their own SHA-256 (per this project's
  atomic-save convention in `scripts/utils/direction_metadata.py` --
  the real extraction script must use `save_direction_atomic`/
  `verify_direction_file` from that module, not a bare `torch.save`)

## 10. Fail-fast rules (binding on the real extraction script, not yet built)

The real extraction script must refuse to run (raise, not warn) if any of
the following hold:
1. Current `templates_context_v1.json` content hash
   (`template_content_sha256`) does not match the pilot-approved,
   attested content hash.
2. `templates_context_v1.json`'s `status` is not
   `HUMAN_AND_TOKEN_AUDITED_READY_FOR_ACTIVATION_PILOT`.
3. The currently-effective human-review checklist (latest `vN`) or the
   formal token-length audit report is missing.
4. Fewer than 30 `direction_ids` are available (e.g. `splits.json`
   schema changed).
5. Any `validation_ids` or `test_ids` value is read for use as a pilot
   instruction, by any code path.
6. The actual constructed condition count is not exactly 20 (§2).
7. Any of the 30 instructions is missing one or more of the 20
   conditions (incomplete condition set).
8. `{instruction}` character-span-to-token-span localization fails for
   any rendered sample (per `get_instruction_end_position`'s own
   fail-loud behavior -- must not be caught and silently skipped).
9. `t_inst` or `t_post` resolves to an out-of-bounds index for the
   actual tokenized sequence length.
10. The loaded model's alias or layer count does not match the
    pilot's frozen expectation (Llama-3.1-8B-Instruct, 32 layers,
    primary layer 19).
11. Any collected activation tensor contains `NaN` or `Inf`.
12. Any target output path already exists (non-overwrite, per this
    project's standing convention).
13. The running script's own git working tree does not match what its
    own `metadata.json` claims (`git_commit` mismatch against
    `git rev-parse HEAD` at run time).
14. Any code path attempts to call a generation (`.generate(...)`) or
    WildGuard-judging function -- these must not exist in this script at
    all, not merely be unreached.

## 11. Dry-run test suite (this round's deliverable; synthetic, CPU-only)

A new dry-run test module, using synthetic/mock data structures only --
**no torch, no transformers, no real tokenizer or model** -- verifies, at
minimum:
1. Condition count derivation: 16 context + 4 canonical = 20; 30
   instructions x 20 = 600 -- computed from the actual constructed
   condition list, not hardcoded.
2. No code path in the dry-run harness references `validation_ids` or
   `test_ids`.
3. Shared-neutral pairing: each of a family's 3 `delta` values pairs
   against the SAME neutral activation for a given instruction (not 3
   different neutrals).
4. `t_inst`/`t_post` localization logic (using a mock tokenizer) finds
   the correct spans, including for a multiline `ctx_continuation`-style
   template where trailing content follows `{instruction}`.
5. Multiline `ctx_continuation` rendering (with `\n` and trailing
   `Response:`/`Speaker B:`/`New response:` markers) round-trips
   correctly through the position-finding logic.
6. Leave-one-variant-out indexing: for each of the 3 variants, the
   "other two" set excludes exactly the held-out one and includes
   exactly the other two, for all 3 rotations.
7. Bootstrap resampling draws whole instructions (not
   `instruction, variant` pairs) -- a mock resampler is checked to never
   split a single instruction's 3 variants across resample membership
   inconsistently.
8. Non-overwrite: a mock "write" step refuses when the target path
   already exists.
9. Every mock output payload carries `result_status ==
   "PILOT_NON_RESULT"`.
10. The canonical-comparison condition set uses the exact same 30
    instruction IDs as the context conditions (no silent substitution of
    a different ID list).
11. Content-hash verification: a synthetic "current" template dict with
    an altered variant text is correctly detected as
    `template_content_sha256` mismatch against a synthetic "approved"
    hash.
12. NaN/Inf rejection: a synthetic activation array containing `NaN` (or
    `Inf`) is correctly rejected by the fail-fast check.
13. Static source-scan: confirms no `.generate(`, `AutoModelForCausalLM`,
    `wildguard`, or similar model-loading/generation/judging symbol
    appears anywhere in the dry-run module or its imports (guards
    against the fail-fast item 14 rule being violated even in the
    design-stage code itself).

This round implements the dry-run test module and any pure-Python
schema/helper code it needs. **It does not implement the real extraction
script** (see §12).

## 12. Not built this round (explicitly deferred)

- `scripts/24_extract_context_activations_pilot.py` (or similarly named)
  -- the real, torch/transformers-dependent extraction script. Not
  created this round.
- An extension of `scripts/audits/audit_token_positions.py` scoped to
  the pilot's 12-condition manual-audit requirement (§3). Not created
  this round.
- Any code that calls `.generate(...)`, loads WildGuard, or computes
  ASR/FRR. Out of scope for the pilot entirely (§1), not merely deferred.

No model has been run. No GPU job has been submitted. No
`validation_ids`/`test_ids` have been read. This document authorizes
design and dry-run testing only.

## 13. Pilot-driven design revision: primary token position (2026-09-08)

**This section revises, but does not delete, §3's original design.** §3
above is left completely unchanged as the historical record of the
original pre-registration (`t_post` primary, `t_inst` a position-
sensitivity diagnostic). Everything in this section reflects a decision
made *after* seeing real pilot data, which is exactly the kind of
revision §3 itself anticipated needing ("if a conclusion holds at `t_post`
but not at `t_inst`... the format-position-cue confound... must be
considered") -- it is recorded here as a distinct, dated, formal-
experiment-preparation revision, not a silent edit of §3.

**Revision**: for the *formal* (non-pilot) C-direction analysis going
forward, **`t_inst` becomes the primary token position**; `t_post` is
demoted to a **format-position sensitivity analysis**, not a co-equal
primary measurement.

**Basis for this decision** -- the real pilot run (extraction commit
`7626eb9a482ef7f3952726769ed89e1b4b895a4d`; input tensors
`context_paired_diffs_PILOT_direction30.pt`
sha256=`a90ccf5a8b7c78d8cc7833b01d56631be038ba47faee346bded8c99c256d0f98`,
`canonical_paired_diffs_PILOT_direction30.pt`
sha256=`0ce90f91d99bf4924b0354e938c1844026a348df38ba5ed79da21841d85bfb52`;
analysis commit `1a6cc72296abdba533ae37a61762bdcce9a9aa85`) found that at
`t_post`, `ctx_continuation`'s direction norm (5.03-9.03 at layer 19) was
roughly 5-8x larger than at `t_inst` (0.87-1.18), with
`cos(t_inst, t_post) = 0.10-0.15` (i.e. the two token positions point in
almost unrelated directions for this family only -- the other 3 families
showed `t_inst == t_post` to floating-point precision, since their
templates place `{instruction}` at the very end with nothing following).
Combined with the real, previously-documented finding that
`ctx_continuation`'s `t_post` position sits immediately after a
`Response:`/`Speaker B:`/`New response:` token that carries its own
strong identity/format signal (§3's original caveat, §7 of
`EXPERIMENT2_CONTEXT_RECONFIGURATION_PROTOCOL.md`), the most defensible
reading is that `t_post` for `ctx_continuation` is confounded with
which trailing format token appears there, not a clean read of "the
context manipulation's effect at the point generation begins." `t_inst`
-- the instruction's own last token, before any such trailing cue -- does
not have this specific confound, even though (per §3) it carries its own,
different limitation (it precedes the full context manipulation for
families that place structure after the instruction, though in practice
none of the 4 families do that).

**What this revision does NOT do**:
- It does not adopt a single shared C direction.
- It does not pre-adopt `k=2` for any shared subspace -- `k=1`/`k=2`
  variance-explained are both reported as descriptive pilot diagnostics
  only (§14 below), and the formal `k` decision remains open.
- Until that decision is made, the **4 family-specific directions are
  used as the conservative default representation** for any further
  work -- i.e. no shared-direction or shared-subspace assumption is
  baked into anything downstream of this revision.
- It does not retract or invalidate any `t_post`-based number already
  reported (§3, and the `t_post`-based results already reported to the
  user from `context_activation_pilot_analysis.json`) -- those remain
  valid `t_post` (now: format-position sensitivity) results, just no
  longer the primary basis for a formal C-direction conclusion.

## 14. Extended t_inst analysis (v2) -- descriptive pilot diagnostics only

Computed by `scripts/53_analyze_context_activation_pilot.py`'s `v2`
output (`context_activation_pilot_analysis_v2.json`, a new file --
`context_activation_pilot_analysis.json` from the original `t_post`-
primary run is kept, not overwritten). Reads only the two already-
extracted `.pt` tensors above; no model, no GPU, no new activation
extraction.

- Full `t_inst` battery per (family, variant) at layer 19: direction
  norm, signed paired projection with bootstrap CI, repeated split-half
  cosine (>=100 predefined seeds, each seed a fresh random 15/15
  instruction partition, reported as median + 2.5%/97.5% quantiles across
  seeds -- **the 100+ seeds are not treated as independent samples for
  any significance test**, only as a descriptive spread), bootstrap
  direction-cosine stability (1000 instruction-level resamples, cosine of
  each resampled direction against the full-sample direction, reported as
  median + 95% interval -- the earlier "CI doesn't cross zero proves
  stable direction" phrasing is retracted; the norm CI is magnitude
  description only).
- Full 4x4 family-centroid cosine matrix at `t_inst`/layer19 (not just
  the upper triangle), average cross-family cosine, SVD singular values,
  `k=1` and `k=2` variance-explained (both reported, neither adopted),
  and leave-one-family-out cosine (predict a held-out family's centroid
  from the mean of the other 3, cosine to the actual). **All of this is
  labeled `DESCRIPTIVE_PILOT` and is explicitly not sufficient, on its
  own, to select a formal dimensionality** (`n_family=4`, same caveat as
  §6).
- `ctx_continuation` format-position diagnostic: `t_inst` direction,
  `t_post` direction, `cos(t_inst, t_post)`, `norm_ratio =
  norm(t_post)/norm(t_inst)`, and a difference vector `d_tpost -
  d_tinst`, which is labeled a **format-position-associated residual**,
  not a causal "format direction" -- no causal claim is made about what
  produces this residual.
- Any cosine computed between two zero-norm vectors (observed at layer 0
  in the original `t_post` sweep) is reported as
  `{"value": null, "status": "UNDEFINED_ZERO_NORM"}`, never a bare `NaN`.
