# Experiment 2 R/H Rebuild Protocol

Records the frozen definitions, responsibilities, and fail-fast rules for
rebuilding refusal_direction (R), harmfulness_direction (H), and per-template
delta_R/delta_H from scratch, following the 2026-09-04 artifact-lineage audit
that found the previously-reported delta_R/delta_H/R² numbers untraceable to
any real computed tensor (see the audit report in conversation; the
retracted values are enumerated in §7 below and must never be restored, used
as defaults, or used as test-expectation values in any test written against
this protocol).

This document is the single source of truth for Stage 2 of the revised
experimental plan (taxonomy geometry → R/H functional profile → context
reconfiguration → intervention prediction). Stage 1 (taxonomy geometry) is
unchanged and covered by the existing Experiment 1 pipeline. Stages 3-4
(context reconfiguration, intervention prediction) are out of scope for this
document and not authorized to start.

## 1. Root cause of the previous failure (why this rebuild is needed)

Independent of whether any prior run's tensors exist anywhere (unresolved,
per the lineage audit), two scripts in the pre-rebuild pipeline
(`scripts/25_extract_delta_r_h.py`, `scripts/27_dual_axis_diagnosis.py`) had
a real, version-independent bug: `REAL_MECHS` was a hardcoded Python list
containing `instruction_hierarchy` and `fictional_framing` -- mechanism
names from a **pre-correction** taxonomy that have not existed in
`templates/templates_en.json` since the project was corrected to
`taxonomy_version: wei_canonical_v2` (6 mechanisms: `prefix_injection`,
`refusal_suppression`, `persona_roleplay`, `encoding_obfuscation`,
`payload_splitting`, `distractors_negated`). Confirmed via
`git log -p --follow` on both files: this was present since each file's
**first commit** -- never a regression, never previously correct. Any
attempt to run either script against current-taxonomy completions data
would find zero matching instructions (or, for 27, `KeyError` on the first
lookup) -- neither script could ever have produced valid current-taxonomy
delta_R/delta_H/R² data, regardless of where or whether it was executed.

Both scripts are fixed (see the accompanying implementation report) to read
`active_mechanisms` fresh from `scripts/_taxonomy_v2_loader.load_taxonomy_v2()`
every run -- the same source of truth `33_canonical_taxonomy_geometry.py`
(Experiment 1) already uses. This is now enforced by a regression test
(`scripts/audits/audit_rh_rebuild_dry_run.py` Test 10) that greps the actual
script source for the stale names as live string literals.

## 2. Frozen mathematical definitions

### 2.1 refusal_direction (R) -- canonical version: v3 (behavioral, independent common axis)

**REVISED 2026-09-04 (circularity correction) -- supersedes the original
8-condition-pooled design below.** The original plan built the axis-building
sample by rendering harmful instructions under plain + all 6 canonical
mechanisms + placebo -- i.e. the SAME 6 templates Experiment 2 later
analyzes via delta_R. This is circular: template identity can correlate
with the refused/accepted label (some mechanisms are known to bypass
refusal far more often than others), so a direction built this way risks
partly encoding "which template" rather than "refused vs accepted", and
then re-projecting the same templates onto it would partly be measuring
that confound back out again.

**Frozen principle (primary analysis, non-negotiable):**

- Each model has exactly **one common refusal_direction** for the primary
  analysis -- not one per mechanism. `leave-one-mechanism-out` (LOMO)
  directions are a **secondary robustness check only** (§2.1.1) and must
  never be substituted for, or mixed with, the primary axis's delta_R in
  any reported result.
- The axis-building sample **must not include any of the 6 canonical
  mechanism templates** (`prefix_injection`, `refusal_suppression`,
  `persona_roleplay`, `encoding_obfuscation`, `payload_splitting`,
  `distractors_negated`). Independent prompt families only.
- All 6 mechanisms' delta_R (Experiment 2's actual result) are projected
  onto this SAME axis -- so they remain directly comparable on one shared
  coordinate, which is what makes a unified R/H/C functional profile
  possible at all.

```
v_R^(l) = mean(h^(l)(x) | x refused)  -  mean(h^(l)(x) | x accepted)
```

evaluated at token position **t_post** (the rendered prompt's own last
token, before generation begins), separately per layer `l`, over an
axis-building sample of harmful instructions **drawn from an independent
prompt-family source not among the 6 canonical templates** (source TBD --
see the accompanying data-availability report; no source is finalized as of
this writing), labeled refused/accepted by WildGuard. Both classes are
harmful instructions only (never harmless) -- "accepted" here means "the
harmful request was complied with", not "a benign request was correctly
helped". This matches arXiv 2507.11878's actual contrast (Zhao et al.
2025), not the earlier harmful-vs-harmless simplification.

#### 2.1.1 LOMO -- secondary robustness only

Leave-one-mechanism-out directions (`R_{-m}`, built excluding mechanism m's
own axis rows, if/when the axis source ever includes mechanism-like rows at
all) may be computed as a **robustness check**: compare `R_{-m}`'s
delta_R for mechanism m against the primary axis's delta_R for mechanism m.
Large disagreement is a flag for further investigation. **LOMO results are
reported separately, in their own output section, and must never be
blended into, averaged with, or substituted for the primary-axis delta_R**
-- per mechanism, using a different axis per mechanism would put the 6
mechanisms back on 6 different coordinate systems, defeating the point of a
shared R axis for the functional profile.

**Sign convention**: positive projection onto `v_R_hat` = more
refusal-associated. `v_R_hat = v_R / ||v_R||`.

**Validation** (required before trusting a rebuilt v_R for any given
model; corrected 2026-09-04 -- Cohen's d alone, with or without a hard
threshold, is not sufficient and must never be used as a standalone
pass/fail gate). `scripts/26_rebuild_refusal_direction_behavioral.py`
computes and records ALL FOUR of the following on the held-out validation
split, per layer, and no single one of them decides validity on its own --
they are reported together in the direction's metadata `extra` field for
human review:

1. **Cohen's d** between refused/accepted projections -- a descriptive
   effect size only, conventional small/medium/large bands (~0.2/0.5/0.8)
   are informative context, not a threshold.
2. **Bootstrap confidence interval** for Cohen's d (`--n_bootstrap`,
   default 1000; stratified resampling of the validation rows, fixed
   direction) -- quantifies sampling uncertainty in the effect-size
   estimate itself; a wide CI (e.g. spanning 0) is informative even if the
   point estimate looks large.
3. **Split-half reliability**: the axis set is randomly split into two
   class-stratified halves, a direction is built independently from each
   half, and `cos(d_R_half1, d_R_half2)` is reported per layer -- tests
   whether the direction construction is stable across resampling of the
   axis-building data itself, not just whether it happens to separate one
   particular validation split.
4. **Rank-based AUC** (Mann-Whitney U) between refused/accepted projections
   on the validation set -- a threshold-free separation measure,
   complementing Cohen's d (which assumes roughly-Gaussian, similar-variance
   classes) with a nonparametric one.
5. **Direction norm** `||v_R^(l)||` per layer, reported alongside (not a
   validity signal by itself, but needed context: a near-zero-norm direction
   with a large Cohen's d is a red flag for numerical instability, not a
   trustworthy result).

No fixed pass/threshold is pre-registered on any of these. A direction is
judged trustworthy only by a human reviewing all five together (§9 point 3).

**v2 (harmful-vs-harmless) fallback**: retained as `--refusal_version v2`
purely as a **plumbing/pipeline check** (file I/O, projection mechanics,
fail-fast dependency verification) -- **never used for a reported result,
and never treated as validating the formal v3/independent-axis R
definition**, only for exercising `25_extract_delta_r_h.py`'s machinery
mechanically. Any output produced with `--refusal_version v2` must be
labeled as such in its metadata (`refusal_version` field) and is not
eligible to be cited as an Experiment 2 result.

**v3 construction is gated**: `refusal_direction_v3` may only be
(re)constructed once an axis source satisfying ALL of §2.1's independence
requirements AND §3's data-responsibility table has been identified and
confirmed available (existence, format, real refused/accepted label
counts) -- not assumed. As of this writing, no such source has been
confirmed; see the accompanying data-availability report for candidates
and their status. `scripts/26_rebuild_refusal_direction_behavioral.py`'s
current (pre-2026-09-04-correction) axis construction, which pools the 6
canonical templates, must NOT be run for a reported result until it is
revised to draw from a confirmed independent source.

### 2.2 harmfulness_direction (H) -- canonical version: v2 (unchanged)

```
v_H^(l) = mean(h^(l)(x) | x harmful)  -  mean(h^(l)(x) | x harmless)
```

evaluated at token position **t_inst** (the raw instruction's own last
token, located structurally -- see §3), from an independent harmful/harmless
train pool (§3). This was already correctly aligned with the paper and is
NOT being rebuilt in the sense of changing its contrast -- only its
save/load path is being hardened (atomicity, hashing; see §6).

### 2.3 delta_R / delta_H (per template, per instruction, per layer)

For each canonical mechanism `m` and instruction `x_i` (drawn from
`direction_ids`, disjoint from R/H's own source data -- see §3):

```
delta_R_{i,m}^(l) = dot( h^(l)(mech_m(x_i), t_post) - h^(l)(plain(x_i), t_post),  v_R_hat^(l) )
delta_H_{i,m}^(l) = dot( h^(l)(mech_m(x_i), t_inst) - h^(l)(plain(x_i), t_inst),  v_H_hat^(l) )
```

Each axis is projected at **its own** position (R at t_post, H at t_inst) --
never averaged/conflated onto a shared position. `plain(x_i)` is the
unwrapped instruction; `t_inst`/`t_post` for the templated condition are
located via `get_user_turn_end_position`/`get_post_instruction_position`
(structural, not subsequence-search, since `encoding_obfuscation` transforms
the raw instruction text and is not recoverable as a literal substring).

**Placebo calibration**: `delta_R_pc_{i,m} = delta_R_{i,m} - delta_R_{i,placebo}`,
matched by instruction id (and symmetrically for delta_H). Placebo-calibrated
values, not raw, are the primary quantities for any downstream analysis
(variance decomposition, functional profile) -- raw values are retained in
the same output file for reference, never discarded.

**Estimator**: arithmetic mean only, no trimmed-mean/median variant, for
both R/H direction construction and the per-template delta aggregation.

### 2.4 R/H independence check (added 2026-09-04)

`v_R` and `v_H` are constructed from different contrasts (refused-vs-accepted
vs. harmful-vs-harmless) and different token positions, but nothing
guarantees they are geometrically independent -- if they turned out to be
near-collinear at a given layer, delta_R and delta_H there would be largely
redundant, not two distinct axes. This is checked, per layer, per model, and
reported alongside every delta_R/delta_H result -- **no cosine threshold is
imposed**; the numbers are reported for interpretation, not gated on:

```
cos(r^(l), h^(l))                                            -- raw collinearity
h_perp^(l) = h^(l) - ( h^(l)^T r^(l) / r^(l)^T r^(l) ) r^(l)  -- H orthogonalized against R
||h_perp^(l)|| / ||h^(l)||                                    -- fraction of H's magnitude NOT explained by R
```

`25_extract_delta_r_h.py` computes `h_perp_hat` once per run (from the raw,
non-unit-normalized `v_R`/`v_H`, matching the formula above literally) and
additionally projects every instruction/mechanism/layer's t_inst activation
diff onto it, producing `delta_H_perp` alongside the existing `delta_R` /
`delta_H` (and the corresponding placebo-calibrated version,
`delta_H_perp_placebo_calibrated`) in the same output payload. **Every
formal delta_R/delta_H result must report all three of: raw delta_H,
orthogonalized delta_H_perp, and `||h_perp||/||h||`** -- not delta_H alone.
If `cos(r,h)` is large at a layer, delta_H_perp is the more informative
quantity there for judging whether a template's effect on H is truly
distinct from its effect on R.

## 3. Data responsibilities (direction / validation / test)

| Data role | Source | Used for |
|---|---|---|
| R/H reference-direction construction (axis + validation split) | PolyRefuse **train** split (`harmful_train`/`harmless_train`), independent of the 572-instruction pool, with the 1 confirmed overlap (`p457`) excluded | Building `v_R`, `v_H` only |
| Template direction / delta_R / delta_H | `direction_ids` (300, from the 572-instruction pool) | `25_extract_delta_r_h.py`'s per-template projections; Experiment 1's taxonomy-geometry template directions |
| Alpha/layer selection, functional-profile validation (future Stage 2/3 work) | `validation_ids` (72) | Never used to construct R/H or template directions |
| Held out | `test_ids` (200), `benign_test_100` | **Not read by anything in this protocol.** No script covered by this document opens these files. |

R/H reference-direction data (PolyRefuse train) and template-direction data
(`direction_ids`) are deliberately **disjoint sources**, not just disjoint
ID sets within the same pool -- this was already the existing design
(`23_extract_reference_directions.py`/`26_rebuild_refusal_direction_behavioral.py`)
and is unchanged by this rebuild.

## 4. Token position, fixed layer, estimator (frozen, unchanged from Exp1/3)

- **t_inst**: the raw instruction's own last token (structural location via
  `scripts/utils/token_positions.py`, using this repo's actual generation
  pipeline tokenization, not generic `apply_chat_template`).
- **t_post**: the fully-rendered prompt's own last token, before generation.
- **Fixed layer** (primary analysis): `floor(0.6 * n_layers)` per model --
  Qwen2.5-7B-Instruct: layer 16, Meta-Llama-3.1-8B-Instruct: layer 19,
  gemma-2-9b-it: layer 25. This is a **pre-registered, data-independent**
  rule (mirrors Arditi et al.'s relative-depth heuristic) -- it is never
  selected by any behavioral/effect-size outcome. All-layer results are
  reported alongside as a robustness check; the primary layer choice is
  never revised based on what any layer's result looks like.
- R/H direction **tensors themselves store all layers** (`[n_layers, d_model]`)
  -- the fixed-layer rule applies at the point of projection/summary/
  reporting, not at extraction time.

## 5. Output file format (frozen schema)

Every direction file (`refusal_dir_v{2,3}_{lang}.pt`, `harmfulness_dir_v2_{lang}.pt`)
is written **only** via `scripts/utils/direction_metadata.py::save_direction_atomic()`,
producing a `.pt` (tensor) + same-stem `.json` (metadata) pair. Every
delta_R/delta_H file (`delta_r_h_{lang}{suffix}_{ids_key}.pt`) is written
**only** via `save_delta_atomic()`, same pairing convention.

Direction metadata (`REQUIRED_FIELDS` in `direction_metadata.py`):
`direction_type, model, model_revision, tokenizer_revision, chat_template_hash,
semantic_position, layer, source_partition, source_ids_hash, sample_count,
construction_contrast, git_commit, random_seed, tensor_sha256, tensor_shape,
tensor_dtype`.

Delta metadata (`DELTA_REQUIRED_FIELDS`):
`model, lang, suffix, ids_key, active_mechanisms, n_instructions, n_layers,
refusal_direction_path, refusal_direction_sha256, harmfulness_direction_path,
harmfulness_direction_sha256, token_position_R, token_position_H, estimator,
git_commit, payload_sha256`.

No script covered by this protocol may write a `.pt` file directly with
`torch.save()`, or a metadata `.json` with a bare `json.dump()` -- only
through the atomic helpers above.

## 6. Fail-fast rules (all enforced in code, not just documented)

1. **Atomicity**: every tensor write goes to a `.tmp` path first, then
   `os.replace()` into the real path -- a process killed mid-write leaves at
   most a stray `.tmp` file, never a truncated "real" file.
2. **Tensor gates metadata**: the tensor is written+renamed successfully
   BEFORE its metadata sidecar is written. A metadata JSON existing is proof
   its tensor exists -- never a placeholder for a tensor that doesn't.
3. **Hash-verified reads**: any script that loads a direction or delta file
   must use `verify_direction_file()`/`verify_delta_file()` -- never a bare
   `torch.load()`. These recompute the file's content hash and raise
   immediately on any mismatch against the recorded metadata.
4. **Missing input = immediate failure, before expensive work**: every
   script that depends on a direction file checks it (via the verify
   functions above) **before** loading the target model / doing any
   generation, never after.
5. **Mechanism-set check**: `verify_delta_file(..., expected_active_mechanisms=...)`
   rejects any payload whose mechanism keys don't match the CURRENT
   `load_taxonomy_v2()['active_mechanisms']` -- this is the direct fix for
   §1's bug class, enforced at load time, not just at extraction time.
6. **Empty-result guard**: `25_extract_delta_r_h.py` asserts
   `len(ids) > 0` after filtering to the requested split + required
   conditions, before loading the model -- refuses to silently produce and
   save a degenerate all-empty tensor.
7. **R² / variance-decomposition scripts never run on incomplete input**:
   `27_dual_axis_diagnosis.py` loads (and verifies) every requested model's
   delta_r_h file before computing anything; any single model's load
   failure stops the whole run before any R² number is computed for any
   model.
8. **No retracted numbers as defaults, test values, or example results**:
   `-16.15`, `-36.75`, `-241.68`, `0.8137`, `0.7922`, and any claim that
   mechanism-label R² exceeds Wei-label R² from the pre-rebuild pipeline are
   retracted (§7) and must not appear in this codebase as a default
   parameter, a synthetic-test expected value, or a worked example.
9. **Cross-model R² is gated on the exact canonical 3-model set** (added
   2026-09-04): `27_dual_axis_diagnosis.py --models` may be called with any
   subset (e.g. a 1-model pilot), but variance decomposition (R² Wei-vs-
   mechanism) and the pooled cross-model section only run when `--models`
   is exactly `{Qwen2.5-7B-Instruct, Meta-Llama-3.1-8B-Instruct,
   gemma-2-9b-it}` (`CANONICAL_3_MODELS`) -- anything else produces a
   `result_status: PILOT_NON_RESULT` output (descriptive + shape/finite +
   direction-norm/angle only, saved to a `_PILOT_` filename) and never a
   silently-degenerate "cross-model" R² number computed on fewer than 3
   groups.
10. **Audit/verification scripts must not write over their own audited
    files without review** (added 2026-09-04, after an incident -- see
    §12): before running any script under `scripts/audits/` against a
    committed output path, confirm it does not overwrite that path as a
    side effect of "checking" it, or redirect its output elsewhere first.
    A script whose job is to verify a file is not automatically read-only.

## 7. Retracted values (must never be restored, cited, or used as test/example data)

`-16.15`, `-36.75`, `-241.68`, `0.8137`, `0.7922`, and any prior claim that
R²(mechanism) demonstrably exceeded R²(Wei) using real 3-model delta_R/delta_H
data. New results are accepted **only** if every link in the chain
`claim → summary → tensor → input IDs → script/config → commit` is present
and verifiable (§8) -- a new number that happens to be close to a retracted
one is not evidence for or against anything; no attempt should be made to
reproduce or approach these specific values.

## 8. Evidence chain requirement (going forward, all Stage 2+ results)

Every reported number must be traceable through all of:
`claim → summary JSON → tensor (.pt, hash-verified) → input instruction IDs
(source_ids_hash) → script + exact CLI args + config → git commit`.
If any link is missing, the result is `UNVERIFIED` and must not be written
into the thesis. `verify_direction_file()`/`verify_delta_file()` are the
mechanical enforcement of the tensor↔summary↔commit links; `source_ids_hash`
covers the input-IDs link; every script's metadata records its own
`git_commit`.

## 9. Audit requirements for any future extraction run

Before trusting any new R/H/delta artifact:
1. `verify_direction_file()`/`verify_delta_file()` must pass (hash match).
2. The metadata's `active_mechanisms` (delta files) must equal the live
   `load_taxonomy_v2()['active_mechanisms']` at read time, not just at
   write time (guards against `templates_en.json` being edited between the
   write and a later read).
3. For refusal_direction v3 specifically: all five validation metrics from
   §2.1 (Cohen's d, its bootstrap CI, split-half reliability, AUC, direction
   norm -- all stored in metadata `extra`) must be reviewed together before
   use. This protocol does not pre-register a hard pass/fail threshold on
   any of them, so this is a required human review step, not an automated
   gate -- and reviewing Cohen's d in isolation does not satisfy this
   requirement.
4. `git_commit` in the metadata must correspond to a commit that includes
   the current (fixed) version of the extracting script -- a direction file
   whose metadata's `git_commit` predates the taxonomy-loader fix (§1) must
   be treated as suspect and re-derived, not trusted.
5. For any delta_R/delta_H artifact: `cos(r,h)` and `||h_perp||/||h||`
   (§2.4, stored per-layer in the delta file's metadata `extra`) must be
   reported alongside any result that uses raw `delta_H` -- if collinearity
   is high at the layer of interest, `delta_H_perp` should be reported too,
   not delta_H alone.

## 10. R/H independence check -- see §2.4.

## 11. Pilot design (Pilot A / Pilot B, corrected 2026-09-04)

Per the 2026-09-04 correction, "pilot" is split into two explicitly separate
tiers with different scope and different (non-)use of GPU:

**Pilot A -- synthetic data only, no model, no GPU.**
`scripts/audits/audit_rh_rebuild_dry_run.py` (12 tests). Covers: atomic
save/load, hash verification, LOGICAL/TENSOR field schema, taxonomy-loader
source-of-truth (rejects stale mechanism names), placebo calibration
arithmetic, R/H orthogonalization arithmetic (`h_perp`), and the
`27_dual_axis_diagnosis.py` single-model pilot gate (§6 rule 9). Every
tensor in Pilot A is hand-constructed (`torch.randn`/hand-picked values) --
never presented as, or capable of producing, an experimental result. Run
freely at any time; no authorization needed beyond normal code-review.

**Pilot B -- Llama-3.1-8B-Instruct, 20-30 non-test instructions, REAL model
+ REAL activations, validates the FORMAL R/H definitions (H at t_inst, R-v3
at t_post, not the v2 fallback).** This is a real GPU task and is NOT
authorized to run under the current round's constraints. **Pilot B is
additionally BLOCKED** (as of 2026-09-04) on an independent R-axis source
being identified and confirmed -- see the accompanying data-availability
report for candidates and why none is yet confirmed usable. Designing
Pilot B's exact command sequence is deferred until an axis source is
selected; `scripts/26_rebuild_refusal_direction_behavioral.py`'s current
axis construction (pools the 6 canonical templates) must not be the one
Pilot B exercises for anything beyond plumbing.

The previously-described "minimal dry-run" (v2 refusal + `--limit_ids 30`)
tests `25_extract_delta_r_h.py`'s plumbing (file I/O, projection mechanics,
fail-fast checks) but does **not** exercise the formal v3/t_post refusal
definition -- it must never be described as validating the protocol's actual
R definition, only as a machinery/pipeline check equivalent in scope to
Pilot A but using a real (if non-canonical) direction file.

## 12. Incident record

**2026-09-04**: two committed audit-report JSONs
(`output/canonical_v2/paired_diffs_audit.json`,
`output/canonical_v2/experiment3_corrected_completions_audit.json`) were
found locally modified in the working tree, with `paired_diffs_audit.json`
degraded from a full cluster-verified report to a 3-line "file not found"
stub. Root cause: `scripts/audits/audit_paired_diffs_corrected.py` (run
during an earlier investigative session) is not read-only -- it overwrites
its own committed output path, and on this Mac checkout (which never has
`.pt` files synced, per `.gitignore`) it reported "exists: false" and
destroyed the previously-recorded cluster-side verification. Both files
were restored via `git restore` (uncommitted, so no data was permanently
lost) and confirmed byte-identical to `HEAD` afterward. See §6 rule 10 for
the resulting standing rule. A byproduct file,
`output/delta_r_h_test_ids_audit.json` (from the same unauthorized rerun,
confirming only that `delta_r_h_..._test_ids.pt` does not exist for any
model -- no new claim beyond that), was moved to
`output/audits/incidents/delta_r_h_test_ids_audit_NON_RESULT.json` and
tagged `result_status: NON_RESULT`.

## 13. Independent R-axis candidate data (read-only search, 2026-09-04)

No source is finalized. Sorry-Bench is explicitly **not confirmed
available** anywhere in this checkout or its sibling repos -- it is not
listed as a candidate below and must not be assumed available in any future
draft. Candidates found by local filesystem search (see the accompanying
report for the full table and every check run):

- `related_work/Multilingual-Refusal/dataset/splits/harmful_train.json`
  (260, native English, no jailbreak template, no native ID field) --
  genuinely independent of the 572-pool (1/260 overlap, `p457`, already
  excluded by existing code) and already the source `23_extract_reference_directions.py`/
  `26_rebuild_refusal_direction_behavioral.py` load via `dataset.load_dataset_split`.
  Has NO existing refused/accepted labels -- would need real generation +
  WildGuard judging (not run this round) to become axis data.
- `experiment_thesis/output/ja_vector_sweep/*/en/20250519-232436/1/completions/harmful_baseline_evaluations.json`
  (per model, 572 rows, template-free, real WildGuard refused/accepted
  labels already computed) -- **confirmed 100% text-overlap with the
  572-pool** (`data/sampled_prompts.json`), so this specific file cannot be
  used as independent axis data. Its refused/accepted counts are important
  negative evidence regardless: Llama 561/11, Gemma 568/4 -- a plain
  (no-template) presentation of harmful instructions yields an extremely
  small "accepted" class (single digits to low teens out of 572), meaning
  even a genuinely independent equivalent (e.g. the same plain-generation
  methodology re-run on the disjoint 260-item `harmful_train` above) should
  be expected to yield very few accepted examples (~5 at the same rate) --
  likely too few for a stable direction, a real feasibility concern to
  resolve before committing to any particular axis-construction plan.
- `related_work/Multilingual-Refusal/dataset/splits/jailbreakbench_test.json`
  (100 rows) -- its first instruction is textually identical to the
  572-pool's first instruction; likely shares substantial provenance with
  PolyRefuse's own harmful_test source. Not confirmed independent; would
  need a full overlap check before use, not assumed clean.
- `related_work/Multilingual-Refusal/dataset/splits/{or_bench_hard,over_refusal,ok_or,xstest}_*`
  -- over-refusal/borderline-request benchmarks (OR-Bench, XSTest), not
  genuinely-harmful-content datasets; wrong construct (calibration against
  false refusal on borderline-legitimate requests, not refusal-vs-compliance
  on real harmful requests) for this axis. Not recommended as an R-axis
  source.
- `experiment_thesis/output/transfer/*` -- **audited 2026-09-04, REJECTED.**
  Contains `{model}/{target_lang}/transfer_from_{source_lang}.json`, each
  with 572 `completions` rows + a `bypass_rate`. Inspected structure and
  content directly (no script run): these are **NOT** natural refused/
  accepted generations -- `transfer_results.json` records `alpha=20.0,
  k_star=25`, i.e. this is a jailbreak-VECTOR-INJECTION experiment (the
  `jb_vec_{lang}.pt` bypassed-minus-refused vector, itself built from
  `harmful_baseline_evaluations.json`, is added to activations at inference
  to force compliance). Sampled completions show the expected injection
  artifact (degenerate repetitive output at high bypass rates, e.g. 96.9%
  for `sw_from_sw`) and `instruction_en` fields match the 572-pool's texts.
  Doubly unusable as axis data: (1) same 572-pool text overlap as
  `harmful_baseline_evaluations.json`, and (2) the "accepted" label here is
  an artifact of injecting a vector derived from that same overlapping
  data, not a natural refusal/compliance contrast -- using it would be
  circular in a second, independent way.

**No confirmed, ready-to-use independent axis source with existing
refused/accepted labels currently exists**, and none is assumed available.
Sorry-Bench specifically remains an unconfirmed **candidate external
source only** -- not present in this checkout or either sibling repo, not
obtained, not to be described as available in any future draft until it is
actually located and audited. The one realistic near-term path identified
so far is generate+judge on `harmful_train` (260, independent, but likely
severe class imbalance per the negative evidence above) via the existing
plain-condition-only method -- not run this round (requires model + WildGuard).

## 15. Independent R-axis manifest schema (added 2026-09-04)

`scripts/utils/axis_manifest.py` -- pure Python/JSON, no torch/model
dependency. A manifest is a JSON file `{"rows": [...]}`; every row must
have all of:

`dataset_name, dataset_version, source_path, source_file_sha256,
stable_source_id, normalized_text_hash, prompt_family, condition,
model_alias, response_id, refusal_label, label_source, split,
overlaps_572_pool, contains_canonical_template`.

**Stable ID rule**: when the source dataset has no native ID field (true of
every PolyRefuse-derived file found so far -- confirmed via direct
inspection, e.g. `related_work/Multilingual-Refusal/dataset/splits/
harmful_train.json` rows are only `{instruction, category}`), the ID is
`{dataset_name}:{original_row_index}:{normalized_text_hash[:16]}`
(`make_stable_source_id()`) -- **never** a post-shuffle positional index
(the `harmful_train_axis_{i}` scheme this replaces couldn't even
reproducibly name "the same row" across two runs with different shuffles).

**`validate_axis_manifest()` checks, in order, before any model load**:
1. every row has all required fields;
2. `overlaps_572_pool` is `False` for every row, AND independently
   recomputed against the 572-pool's own text hashes (never trusts the
   manifest's self-reported flag alone);
3. `contains_canonical_template` is `False` for every row, AND
   independently rechecked (`prompt_family` must not equal any of the 6
   canonical mechanism names);
4. `stable_source_id` values are unique;
5. `split` values are restricted to `{axis, val}` -- `test_ids`/
   `direction_ids`/`validation_ids` (572-pool concepts) must never appear;
6. the `axis` split has at least one `refusal_label == 1` (refused) AND one
   `== 0` (accepted) row;
7. every unique `source_path`'s file, hashed right now, matches its row's
   recorded `source_file_sha256` (catches source-file drift since the
   manifest was built).

Any failure raises `ValueError`/`FileNotFoundError` with a specific,
actionable message -- no partial acceptance, no silent skip.

## 16. `scripts/26_rebuild_refusal_direction_behavioral.py` gate (added 2026-09-04)

`main()` now requires exactly one of `--axis_manifest PATH` or
`--legacy_pooled_templates CONFIRM_LEGACY_POOLED_TEMPLATES_PROVISIONAL`
(an exact-string sentinel, not a plain boolean flag, specifically so a
typo or copy-pasted `--legacy_pooled_templates true`-style flag cannot
accidentally enable it) -- checked before any `pipeline`/`transformers`
import, any data load, or any model load. Neither given, or both given,
exits(1) immediately.

- `--axis_manifest`: validates via §15's checker; on success, prints the
  per-split refused/accepted counts and stops (activation extraction from
  manifest rows is not yet implemented -- requires a model, out of scope
  this round). On failure, the validator's exception propagates
  immediately, before any model-related import.
- `--legacy_pooled_templates` (exact sentinel required): runs the OLD,
  KNOWN-CIRCULAR construction (pools the 6 canonical templates, §2.1)
  unchanged in its math, but writes to
  `output_v3_behavioral_refusal_LEGACY_PROVISIONAL/{model_alias}/` -- a
  directory name structurally distinct from the real result path, never
  reusable by accident -- with metadata `extra.status =
  'LEGACY_PROVISIONAL_POOLED_TEMPLATES_NOT_FOR_RESULTS'`. As of this round,
  no code path in this script writes to the real (non-suffixed)
  `output_v3_behavioral_refusal/` directory at all -- by design, since
  neither an approved manifest nor a non-circular construction exists yet.
- Both paths still use `save_direction_atomic()` (tensor written+renamed
  before metadata; metadata always includes hash/shape/dtype; §5/§6) -- no
  orphan-tensor risk beyond what every other direction file in this
  pipeline already accepts (a crash strictly between the two atomic writes
  is detectable later via `verify_direction_file()`, per §6 rule 3).

## 14. Partial-commit dependency finding (2026-09-04, RESOLVED)

**Resolved**: `scripts/26_rebuild_refusal_direction_behavioral.py` is now
included in this round's proposed commit (its axis-source gate, §16, was
added this round), so the interface mismatch described below no longer
applies to the actual proposed commit -- kept here as a record of the
finding and the reasoning, not as an open issue.

`scripts/23_extract_reference_directions.py`, `25_extract_delta_r_h.py`,
and `27_dual_axis_diagnosis.py` do not import or call anything from
`scripts/26_rebuild_refusal_direction_behavioral.py` -- committing them
without 26 is safe on its own. However: the currently-committed (HEAD)
version of `scripts/26_rebuild_refusal_direction_behavioral.py` calls
`build_direction_metadata()` then `save_direction_metadata()` directly,
without ever supplying `tensor_sha256`/`tensor_shape`/`tensor_dtype`. The
NEW `direction_metadata.py` (part of this round's proposed commit) makes
`save_direction_metadata()` validate the full `REQUIRED_FIELDS`
(LOGICAL + TENSOR) and raise `ValueError` if the tensor fields are
missing. Consequence: **if the new `direction_metadata.py` is committed
while `scripts/26...` is left at its current HEAD version, running
`scripts/26...` (as committed) will crash at the metadata-save step** --
after the direction tensor itself has already been saved successfully
(`torch.save` runs first), so no data is lost, but it leaves an orphaned
tensor with no metadata sidecar (the exact anti-pattern this rebuild exists
to prevent) and the script exits with an unhandled exception. This is
currently inert (nobody is authorized to run `scripts/26` this round), and
will be resolved automatically once the follow-up commit updates it to use
`save_direction_atomic()`. Flagged here so it is not forgotten before
`scripts/26` is ever run again from a checkout that has the new
`direction_metadata.py` but not yet the follow-up fix.
