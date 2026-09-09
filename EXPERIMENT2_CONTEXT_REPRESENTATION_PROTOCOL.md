# Context Representation Protocol -- Formal Design (2026-09-09)

Supersedes `EXPERIMENT2_FORMAL_C_DIRECTION_PROTOCOL.md` (kept, marked
SUPERSEDED, not deleted). Extends and does not contradict
`EXPERIMENT2_CONTEXT_ACTIVATION_PILOT_PROTOCOL.md` (the pilot protocol),
whose `t_inst`-primary decision (Sec 13/15) this document inherits.
**Design-only until Sec 10's authorization gate is explicitly lifted by
the user.**

## 0. Research questions (current, replaces the earlier "does a unified C
direction exist" framing)

1. Is the CO/MG taxonomy's internal geometric validity for the 6
   canonical mechanisms stable across models?
2. Is CO/MG sufficient to represent contextual-reconfiguration attacks?
3. If context attacks do not stably reduce to CO or MG, is there evidence
   supporting contextual reconfiguration as an additional, internally
   heterogeneous third superordinate category?

Contextual reconfiguration (`ctx_persona`/`ctx_authority`/`ctx_fictional`/
`ctx_continuation`) is a **candidate** third category, not an established
conclusion. This document does not presuppose the 4 families form one
geometric cluster, nor that any single family survives as evidence for a
third category.

## 1. Experiment 1 -- corrected conclusion (frozen, do not re-litigate)

Per the P0 lineage audit (2026-09-08/09): `rank_partitions()` in
`scripts/33_canonical_taxonomy_geometry.py` ranks by `T = (S_a+S_b)/2 -
S_between` descending (rank 1 = highest T = best-separated 2-cluster
fit); tie tolerance `1e-12`, no ties observed. Verified against
`output/canonical_v2/experiment1_taxonomy_geometry.json`
(`git_commit=306d8028...`, `n_common_ids=300` for all 3 models,
`warnings=[]`, single commit `c56fac7`, never overwritten):

| Model | Wei rank /10 | bootstrap P(rank=1), n=2000 | Delta_CO 95% CI | Delta_MG 95% CI |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 1 | 1.0 | [0.0805, 0.0874] | [0.1323, 0.1450] |
| Meta-Llama-3.1-8B-Instruct | 3 | 0.0 | **[-0.0665, -0.0460]** | [0.1496, 0.1686] |
| gemma-2-9b-it | 1 | 1.0 | [0.0458, 0.0579] | [0.2946, 0.3099] |

**Frozen sentence for the paper** (use verbatim, do not paraphrase into
"fails in all 3 models" or "completely overturned"):

> "The internal geometric validity of the CO/MG taxonomy is
> model-dependent. It is strongly supported for the six canonical
> mechanisms in Qwen and Gemma, but only partially supported in Llama,
> where the competing-objectives group lacks internal cohesion."

## 2. Scope (frozen: 21,600 forwards, not 14,400)

- **Models**: Qwen2.5-7B-Instruct, Meta-Llama-3.1-8B-Instruct,
  gemma-2-9b-it -- independent per-model runs.
- **Instructions**: all 300 `direction_ids`. **Known non-uniqueness**
  (Sec 6): 298 unique normalized texts among the 300 ids (2 duplicate
  pairs: `p139`/`p318`, `p255`/`p410`) -- IDs are kept as-is for
  extraction (comparability with existing R/H/Exp1 results), but
  **all bootstrap resampling is by normalized-text cluster, not by
  raw instruction_id** (Sec 7).
- **Conditions**: 16 context (4 families x (3 positive + 1
  family-specific neutral)) + 8 canonical (`plain`, `placebo`, and all 6
  active mechanisms) = **24 conditions/instruction**.
- **Total forward passes: 3 x 300 x 24 = 21,600.** Canonical directions
  are **re-extracted in this same pipeline**, never reused from
  Experiment 1 -- confirmed incompatible (Experiment 1's
  `paired_diffs_*_full572_corrected.pt` was built with
  `position=-1` = `t_post`, per `scripts/18_extract_paired_diffs.py`
  line 85's default; this formal run's primary position is `t_inst`).
  This is computed dynamically from the real condition/model/id lists at
  run time (already true of `scripts/54`'s existing asserts) -- never a
  hardcoded "must equal 21,600" as the *sole* check, only as a sanity
  cross-check against the dynamic count.
- **Primary token position: `t_inst`.** `t_post` is captured for every
  row too, as a **format-position sensitivity analysis only** -- never a
  co-primary or substitute primary result.
- Same model, tokenizer/chat template, hook position, layer indexing,
  dtype, and aggregation logic across context AND canonical conditions
  within one model's run (both come from the same
  `scripts/54_extract_formal_context_activations.py` invocation --
  already true of the existing script, confirmed by re-inspection: it
  builds both condition sets from the same `instructions_by_id` and the
  same forward-pass loop).
- **Per-instruction paired differences are saved** (not just per-family
  means) -- already true of `scripts/54`'s existing
  `context_paired_diffs_FORMAL_direction300.pt` /
  `canonical_activations_FORMAL_direction300.pt` schema.
- `validation_ids`/`test_ids` remain forbidden for this experiment.

## 3. Direction formula (unchanged from the pilot)

```
d_{f,v}^{(l)} = mean_i[ h^{(l)}(T+_{f,v}(x_i)) - h^{(l)}(T-_f(x_i)) ]
```
Positive and neutral use the SAME source instruction `x_i`. Primary
representation remains the family-specific 4-tuple `C_m = (C_persona,
C_authority, C_fictional, C_continuation)` -- **no single shared
direction is primary; `k` (any k) is exploratory only** (pilot protocol
Sec 15, unchanged).

## 4. CO/MG prototype (frozen definition)

```
tilde_d_m = d_m - d_placebo                    # per canonical mechanism m
u_m = tilde_d_m / ||tilde_d_m||                # normalize BEFORE averaging
p_CO = normalize(mean(u_prefix_injection, u_refusal_suppression, u_persona_roleplay))
p_MG = normalize(mean(u_encoding_obfuscation, u_payload_splitting, u_distractors_negated))
```
Normalizing each mechanism individually before averaging prevents a
large-norm mechanism from dominating the prototype. **Individual
mechanism comparisons are always reported alongside the prototype
comparison -- the prototype never replaces them.**

**Fail-fast rule (tightened 2026-09-09, second revision)**: if
`||tilde_d_m||` (or any intermediate norm in the above chain) falls at or
below the dtype-aware machine epsilon scaled to the vector's own
magnitude (e.g. `torch.finfo(dtype).eps * scale`, not a hand-picked
"effect size" threshold), that mechanism's `u_m` is marked
`UNDEFINED_NEAR_ZERO_NORM`. **All 3 constituent mechanisms of a prototype
must be well-defined.** If even ONE of the 3 is `UNDEFINED_NEAR_ZERO_NORM`,
the whole prototype for that `(model, layer)` is marked
`UNDEFINED_INCOMPLETE_CATEGORY` -- it is **never** built from the
remaining 2 mechanisms, and never imputed or skipped silently. (The
first revision of this protocol incorrectly tolerated 1-of-3 undefined
mechanisms; that tolerance is removed -- a prototype is a claim about
what CO, resp. MG, look like as a *category* of exactly 3 canonical
mechanisms, and dropping a member changes what is being measured.)

**Token position (tightened 2026-09-09)**: `tilde_d_m` for the CO/MG
prototype AND for the canonical attack-profile projection (Sec 8-C) is
computed at **t_inst** (primary), the SAME position as the context
directions `C_m` they are compared against -- comparing vectors measured
at different token positions would confound "geometric similarity" with
"position mismatch." `t_post` versions of both are still computed and
reported, but ONLY as an explicitly, separately labeled sensitivity lens
(e.g. `canonical_attack_profiles_tpost_sensitivity`), never merged into,
averaged with, or silently substituted for the t_inst primary result.

## 5. Random 2D subspace null (frozen)

Per `(model, layer)`:
1. Derive a deterministic sub-seed from `(model_alias, layer_index,
   global_seed=20260908)` (e.g. `seed = hash_to_int(f"{model_alias}|{layer_index}|20260908")`
   truncated to a valid `torch.Generator` seed range) -- recorded in
   output metadata so it is independently reproducible.
2. Draw 2 iid Gaussian vectors in `R^hidden_dim` (same hidden_dim as the
   model/layer being tested).
3. Orthonormalize via QR (or SVD) to get an orthonormal 2D basis.
4. Compute `E_random = ||Proj_{span}(d)||^2 / ||d||^2` for the SAME
   observed `d` (the direction being tested; `d` is never resampled --
   only the random subspace is).
5. Repeat **1000 times** (`n_random_subspaces=1000`) per `(model, layer,
   d)` to build the null distribution.

**Reporting** (all required, per finding): `observed E_RH`, `observed
q(d)`, `null median`, `null 95th percentile`, `observed percentile`
(where `observed E_RH` falls in the null distribution), `unadjusted p`
(one-sided: fraction of null draws >= observed), `Holm-adjusted p`
(across the 12 primary-layer variant tests), and the theoretical sanity
check `E[E_random] ~= 2/hidden_dim` (closed-form expectation for a
random 2D subspace in a high-dim isotropic Gaussian space, compared
against the empirical null mean as a basic correctness check on the
null-generation code itself, not a substitute for the empirical null).

**Interpretation** (frozen, all three pieces required together --
significance alone is not sufficient):
- `observed E_RH` significantly above the null's 95th percentile
  (one-sided) -> R/H has a special (non-random) association with `d`.
- `observed E_RH` within the null's typical range -> R/H does not
  specially explain `d`.
- Significant AND `q(d)` still substantial -> R/H explains only part of
  the effect; **both the significance AND the absolute size of `E_RH`
  must be reported together** -- "significant but small `E_RH`" is a
  valid, nameable outcome (special association, limited explanatory
  share), not evidence to round up to "R/H explains context."
- All-layer results are exploratory only; the primary-layer test (with
  Holm correction across the 12 variants) is the only confirmatory claim
  this protocol permits.

**Dual-position definition (made explicit 2026-09-09, second revision --
must never be described as "R, H, and C are all extracted at the same
token position", which is false)**:
- `R` (`refusal_dir_v3_en.pt`) is constructed at **t_post** (its own
  `semantic_position` metadata field; independently asserted, not
  assumed, by `assert_dual_position_design()` before any `E_RH`
  computation).
- `H` (`harmfulness_dir_v2_en.pt`) is constructed at **t_inst** (same
  assertion mechanism).
- The context direction `C` (this experiment's primary quantity) is
  measured at **t_inst**.
- `E_RH(d)`/`q(d)` therefore describe `d`'s (t_inst) relationship to the
  2D subspace `span(R, H)`, where `R` and `H` were themselves each built
  at their OWN (different, pre-existing) token positions. This is the
  pre-existing dual-position R/H design from the R/H rebuild work, not
  something this experiment introduces or resolves -- it is a caveat on
  every `E_RH`/`q` result, not a defect this protocol can silently paper
  over by re-describing all three as "the same position."

## 6. 572-pool duplicate-text audit (read-only, completed 2026-09-09)

Via `utils.axis_manifest.normalized_text_hash` (already-existing,
license-safe utility) applied to `data/sampled_prompts.json`'s
`instruction_en` field for IDs in `data/splits.json`:

- `direction_ids` (n=300): **298 unique normalized texts**, 2 duplicate
  pairs: `{p139, p318}`, `{p255, p410}`.
- `validation_ids` (n=72): 72 unique normalized texts, 0 duplicates.
- `direction_ids` x `validation_ids` normalized-text overlap: **0**
  (confirmed by direct set intersection on hashes, not just IDs).
- `direction_ids`/`validation_ids`/`test_ids` are pairwise ID-disjoint
  (0/0/0 intersections) -- `test_ids` content was never read, only the
  ID set, per the standing prohibition.
- The full 572-pool itself has only 562 unique normalized texts (10
  pool-internal duplicates) -- noted for the record; not investigated
  further this round (out of scope, no impact on direction_ids exclusion
  decisions since none of the pool-internal duplicate pairs beyond the
  2 named above fall inside `direction_ids`... **caveat**: this was
  checked only for `direction_ids`/`validation_ids` themselves, not
  exhaustively for how the other 562-vs-572 duplicates distribute across
  `direction_ids`/`test_ids` boundaries; if this matters later, it needs
  a dedicated follow-up).

**Processing rule (frozen, per user decision)**: `direction_ids` is
**not** modified or de-duplicated for extraction -- the frozen ID list
is preserved for comparability with existing R/H/Exp1 results. Instead,
**every bootstrap resample in this experiment's analysis draws whole
normalized-text clusters, not raw instruction_ids** -- `{p139, p318}`
and `{p255, p410}` are each resampled as a single unit (both members
included or excluded together), so they contribute at most 2 independent
data points to any bootstrap, not 4. A deduplicated-instructions
sensitivity analysis may be reported additionally but never replaces the
pre-registered 300-id primary analysis.

## 7. Instruction-cluster bootstrap (implementation contract, corrected 2026-09-09)

**Equal-weight aggregation, not just "resampled together" (this round's
correction)**: the first revision of this protocol required only that a
duplicate-text pair's 2 ids move together in a resample -- that is
necessary but **not sufficient**. If the resampled ids are then collapsed
with a flat, unweighted mean, a 2-member cluster still counts twice as
much as a 1-member cluster in the resulting point estimate. The correct,
now-frozen two-step procedure:

```
clusters = group_instruction_ids_by_normalized_text_hash(direction_ids)   # 298 clusters for the 300 ids

# STEP 1 -- collapse each cluster to its OWN within-cluster mean FIRST,
# before any cross-cluster aggregation:
cluster_means = [mean(delta[i] for i in cluster) for cluster in clusters]  # 298 vectors

# STEP 2 -- every point estimate, bootstrap resample, and split-half
# partition operates on `cluster_means` (298 rows), NEVER on the raw
# 300-row `delta` directly:
point_estimate = mean(cluster_means)                                       # equal weight per cluster
resample = [cluster_means[i] for i in random_choices(range(298), k=298)]   # bootstrap over CLUSTER rows
resampled_estimate = mean(resample)
```
This guarantees a duplicate-text cluster and a singleton cluster
contribute EXACTLY the same 1/298 weight to every reported statistic --
in the point estimate itself, not only in which ids move together during
resampling. This generalizes the pilot's "resample by instruction" rule
(pilot protocol Sec 4) to "resample by equal-weighted normalized-text
cluster" -- for the 296 singleton clusters step 1 is a no-op (mean of 1
element), so this reduces to ordinary per-instruction resampling there;
only the 2 duplicate pairs are affected, and now correctly so.

When the same instruction axis is pooled across multiple repeats (e.g.
concatenating all 3 variants of a family before a pooled-centroid CI),
the same 298 clusters are extended to include every repeat's copy of each
cluster's members (a duplicate pair pooled across 3 variants becomes one
6-member pooled cluster, not 3 separate 2-member ones) -- the cluster
identity is about the underlying TEXT, not about which tensor axis or
condition it appears under.

## 8. Third-category determination (three layers, per family)

**A. Behavioural validity** (Experiment 3, not this experiment): positive
context must raise ASR or harmful-compliance relative to that family's
neutral. Without this, a family is a "contextual manipulation," not an
"attack category" -- naming discipline enforced in all outputs.

**B. Representational reliability** (this experiment): each family must
show stable direction estimates via instruction-cluster bootstrap
(Sec 7), repeated split-half, and leave-one-variant-out -- same metrics
as the pilot's v2 analysis, generalized to 300 ids / instruction
clusters / 3 models, reported per model (never averaged across models).

**C. Non-reducibility to CO/MG or R/H**: for each family, report cosine
(and instruction-cluster-bootstrap CI) against `p_CO`, `p_MG`, each of
the 6 individual canonical mechanism directions, AND `E_RH`/`q` against
`span(R_m, H_m)` for that model. A family is "reducible" if it aligns
strongly and reliably with one of these; "non-reducible" if it does not.

**No family is force-fit into the third category.** If a family reduces
to CO or MG, it is reported as such and dropped from the "candidate
third category" set. The four families are **not** presupposed to form
one geometric cluster -- a family can pass A+B+C independently of the
others (per-family decision, not an all-or-nothing category vote,
consistent with pilot protocol's already-recorded open outcomes:
"family-specific effects with no shared structure" remains a legitimate
result, not a failure mode).

## 9. Output schema and fail-fast

Extends `scripts/54`'s existing schema (Sec 9 of
`EXPERIMENT2_FORMAL_C_DIRECTION_PROTOCOL.md`, unchanged) with the new
analysis-side artifact from `scripts/55`:
```
output/context_activations_formal/formal_c_direction_analysis.json
```
`result_status: "FORMAL_DIRECTION_ESTIMATION"` throughout; CO/MG
prototype fields carry `UNDEFINED_NEAR_ZERO_NORM` (per-mechanism) /
`UNDEFINED_INCOMPLETE_CATEGORY` (whole prototype) where triggered
(Sec 4); random-null fields never emit a bare `NaN` (reuses the pilot's
`{"value": ..., "status": ...}` convention). Canonical-mechanism fields
are split into `*_tinst_primary` and `*_tpost_sensitivity` (Sec 4);
never a single ambiguously-named field mixing the two positions.

**Fail-fast additions** (on top of `scripts/54`'s existing 17 rules):
18. Any R or H direction file fails `verify_direction_file()` (hash/
    shape/dtype mismatch against its own metadata) -> halt before any
    `E_RH`/`q` computation for that model.
19. `assert_dual_position_design()` fails (R's `semantic_position` !=
    `t_post`, or H's != `t_inst`) -> halt before any `E_RH`/`q`
    computation for that model (Sec 5's dual-position definition).
20. A CO or MG prototype has even 1 of its 3 constituent mechanisms
    `UNDEFINED_NEAR_ZERO_NORM` -> the WHOLE prototype is
    `UNDEFINED_INCOMPLETE_CATEGORY` for that `(model, layer)`, never
    built from the remaining 2; the run continues for other
    models/layers.
21. Instruction-cluster construction (Sec 7) must yield exactly 298
    clusters from 300 `direction_ids` -- a different count halts the
    analysis (something about the text pool or id list changed
    unexpectedly).

## 10. Authorization gate (current)

Authorized this round: cluster R/H tensor read-only verification
(Sec 11), protocol authoring, `scripts/54`/`55` modification, dry-run/
synthetic CPU tests, `git commit`/`push` for code and protocol.
**Not authorized**: the real 21,600-forward extraction, any generation,
any judge call, reading `test_ids` content, deleting/reverting any
commit, or reporting pilot results as formal results. The real GPU
extraction requires a separate, explicit go-ahead after this round's
report is reviewed.

## 11. R/H tensor field-verification checklist (to run on cluster)

For each of the 6 files (`refusal_dir_v3_en.pt` x3 models,
`harmfulness_dir_v2_en.pt` x3 models), `verify_direction_file()` plus
manual extraction of: actual SHA-256, metadata SHA-256, match/mismatch,
shape, dtype, `torch.isfinite().all()`, layer count (`shape[0]`), hidden
dimension (`shape[1]`), `semantic_position` (from metadata; expected
`t_post` for every `refusal_dir_v3_en.pt`, `t_inst` for every
`harmfulness_dir_v2_en.pt` -- Sec 5's dual-position definition), and
`status`/backfill-note (from metadata). H-axis keeps its
`tensor_hash_backfilled_note` regardless of verification outcome
(Sec 6 of the user's decision) -- verification passing does not erase
the "hash only attests to post-backfill integrity" caveat from the paper
or protocol text.

**Stop condition (2026-09-09, second revision)**: the verification script
must halt immediately (not continue to the next file) on the FIRST of:
a hash mismatch, a non-finite (NaN/Inf) value anywhere in the tensor, a
shape/dtype mismatch against the file's own metadata, a `primary_layer`
index out of range for the tensor's layer count, or a `semantic_position`
that doesn't match the expected value above. `verify_direction_file()`
already raises for the hash/shape/dtype cases by construction; the other
checks (finiteness, layer range, semantic position) are explicit
additional asserts in the same script, also raising immediately.
