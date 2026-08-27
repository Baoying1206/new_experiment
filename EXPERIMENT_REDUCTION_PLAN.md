# Experiment Reduction Plan

Audit + implementation plan for the reduced thesis scope (Experiments 1–3:
taxonomy geometry test, dual-axis failure diagnosis, continuous profile
evaluation; steering pilot deferred/conditional). **No code changes or GPU
runs have been made in producing this document.** Written for review before
any implementation begins.

**Update: cross-lingual replication is out of scope for now.** The thesis's
main line is English-only (Experiments 1–3 on the 572-instruction English
`direction_ids`/`validation_ids`/`test_ids` split). The 6-language
confirmatory config (`scripts/_lang_config.py`) and the already-generated
`generation_input_{lang}_xling.json` files remain in the repo, unused for
now -- not deleted, available if cross-lingual replication is revisited
later. Any in-flight `generate_and_label_xling.sh` SLURM jobs are the
user's call whether to let finish or cancel (cancelling early wastes
already-spent compute; letting a near-complete job finish is fine since the
data is harmless to have on hand even if unused in the current plan).

---

## 1. Which existing script already covers each analysis

| Experiment | Covered by | Coverage |
|---|---|---|
| Exp 1 — pairwise cosine matrix (6 templates) | `04_extract_directions_and_analyze.py` → `pilot_results.json['same_language_cross_mechanism']` | **Partial.** The 15 pairwise cosine values already exist, but only on the old 75-instruction data, all-layer-averaged (not a specific layer/token position field), single estimator (arithmetic mean only), no robust-estimator variants. |
| Exp 1 — within/between-category gap, exact permutation, bootstrap, leave-one-template-out | `19_taxonomy_robustness.py` | **Mostly covers it**, already implements exact enumeration over the 10 possible 3-3 partitions, bootstrap CI, split-half, leave-one-template-out, placebo-calibrated re-test, per-model (already run once, on old data). Missing: robust-estimator (trimmed mean / geometric median) variants — explicitly deferred in its own docstring ("NOT implemented here"). Output JSON field names don't match this plan's requested schema (`within_similarity`, `between_similarity`, `taxonomy_gap`, etc.) — needs either renaming or a thin wrapper. |
| Exp 2 — δR/δH per template, independent token positions | **Nothing.** | `09_refusal_geometry.py` computes a similar projection (`frac_along_refusal`) but at a **single, undifferentiated token position** (see §7) — not the independently-verified `t_inst` vs `t_post_inst` split this plan requires. `21_component_causal_decomposition.py` does a related Gram-Schmidt decomposition but for **causal injection testing**, not correlational profiling, and also inherits the same single-position limitation. **New script needed.** |
| Exp 2 — variance decomposition (taxonomy_label vs template_identity vs model/language) | **Nothing.** | No regression/ANOVA-style variance-explained code exists anywhere in the repo. **New code needed** (in the shared statistics module). |
| Exp 3 — continuous profile prediction, leave-one-out CV, leak-free | **Nothing.** | No script combines δR/δH with behavioral labels for prediction. `data/splits.json`'s direction/validation/test partition is directly reusable as the leak-free scaffold, but nothing currently consumes it this way. **New script needed.** |
| Steering pilot (conditional) | `10a_calibrate_injection_alpha.py`, `10b_phase1_injection_experiment.py`, `21_component_causal_decomposition.py` | Injection/WildGuard infrastructure is directly reusable; alpha calibration logic needs the validation-only fix noted in §8 before reuse. |

---

## 2. Scripts that only need modification (not full rewrites)

- **`04_extract_directions_and_analyze.py`** — the pairwise-cosine computation logic is correct and reusable; needs to be re-run on the 572/200-instruction data (via `--suffix`, which it does not currently support — same gap as 18/19 had before that was added) and needs per-layer (not only all-layer-averaged) output to satisfy the `layer`/`token_position` metadata field this plan requires.
- **`19_taxonomy_robustness.py`** — add trimmed-mean/geometric-median estimator variants; rename/restructure output JSON to match the field list in §5; already supports `--suffix`/reads `paired_diffs_{lang}{suffix}.pt`, so re-running at 572-scale is a matter of first running `18_extract_paired_diffs.py --ids_key direction_ids` (already supported) then this script — no new capability needed for that part.
- **`10a_calibrate_injection_alpha.py`** — needs to read prompts from `validation_ids` only (currently samples from whatever `completions_{lang}.json` it's given, no split awareness at all — see §8).
- **`14_find_safety_layer.py`** — same validation-only gap as above; also uses a peak-norm heuristic already flagged in this project's own history as confounded by residual-stream depth growth (fixed once for the "exclude last 20% of layers" case, but the underlying criterion still has no validation-set restriction).

## 3. Scripts that need to be newly created

1. **δR/δH extraction script** (Exp 2) — computes per-(instruction, template) projections onto refusal_direction and harmfulness_direction, respecting the independently-verified token positions once §7's blocker is resolved.
2. **Dual-axis diagnosis / variance-decomposition script** (Exp 2) — consumes the above, reports within/between-taxonomy overlap in (δR, δH) space, template-identity vs taxonomy-label vs model/language variance explained.
3. **Continuous profile prediction script** (Exp 3) — leave-one-template-out (and optionally leave-one-language-out) cross-validated comparison of `Wei label` vs `δR only` vs `δR+δH+magnitude` against held-out behavioral outcomes, with strict split enforcement (test labels never touch profile construction or hyperparameter selection).
4. **Shared statistics module** (`src/statistical_tests.py` or similar, per your §八 instruction) — permutation test, bootstrap-by-source-ID, trimmed mean, geometric median, variance-explained decomposition. Currently every script that needs any of these (06, 07, 19) reimplements its own version inline; this plan is the first time enough distinct callers exist to justify extracting a shared module rather than a third copy-paste.
5. **Direction re-extraction with dual token positions** (see §7) — only if you decide not to proceed with the current single-position directions as an approximation.

## 4. Existing results that can be reused as-is

- `data/splits.json` (300/72/200 English split, 100/30/70 cross-lingual subset) — directly reusable, no changes needed, already leak-checked (24/25 automated checks passed, see `output_572_split_v1/six_language_validation.json`).
- `scripts/_lang_config.py` — **the 6-language confirmatory config you asked me to check is already implemented**, matches your spec exactly (`en, zh, ar, th, yo, am`; `de, ko, sw` excluded but not deleted).
- `data/generation_input_en_full572.json`, `data/generation_input_{zh,ar,th,yo,am}_xling.json` — already built, already validated (25 CPU-only checks, 24 pass + 1 informational).
- `completions_en_full572.json` (Qwen only, currently) — already generated (54 min, 4576 rows), bypass-rate summary already sanity-checked against prior causal findings.
- `experiment_thesis`'s existing `refusal_dir_{lang}.pt`/`harmfulness_dir_{lang}.pt` — reusable **only** as an interim/approximate baseline (see §7's caveat); not reusable if the dual-token-position requirement is treated as a hard constraint.

## 5. Results invalidated by the rescoping (572 restart / 6-language reduction / split changes)

- **All 75-instruction-pilot analysis outputs remain valid as supplementary evidence** (per `DATA_MANIFEST.md`, this was a deliberate, already-communicated decision — not deleted, not silently superseded).
- **`pilot_results.json`'s `same_language_cross_mechanism`** (the existing 6-template pairwise cosine matrix) is **not** directly substitutable for Exp 1's requirement — it's on the old 75-instruction sample, so a fresh computation on the 572-instruction `direction_ids` set is needed for the primary-line claim; the old value remains usable as a "does the pattern replicate at smaller-n" comparison point, not as the headline number.
- **Any `10a`/`14` output that used the full (unsplit) 75-instruction data to pick an alpha or layer** is not compliant with this plan's leak-free requirement (§8) and should not be treated as validation-only calibration for Experiment 3's threshold-freezing step, even though it remains valid as "the strength/layer that worked in Phase 0/1's less strict earlier design."

## 6. Possible data leakage

Three concrete, already-identified risks, ranked by severity:

1. **(Confirmed, not yet fixed) `10a`/`14` use the full unsplit dataset for alpha/layer selection** — grep confirms zero references to `validation_ids`/`splits.json`/`ids_key` in either script. If Experiment 3's profile-guided steering pilot (§六 in your spec) reuses these calibration values, that constitutes a leak: alpha/layer would have been implicitly chosen using signal from what are now `direction_ids`/`test_ids`-overlapping instructions, not a clean validation-only slice.
2. **(Structural, low risk but unverified) `refusal_direction`/`harmfulness_direction` construction data vs `new_experiment`'s splits** — these come from PolyRefuse's **train** split; `new_experiment`'s 572-instruction pool comes from PolyRefuse's **test** split (`01_sample_prompts.py` reads `harmful_test_translated_{lang}.json`). Structurally disjoint source pools, so unlikely to overlap — but the exact 128 instructions drawn for each direction were never logged (`random.sample` with a seed, no saved ID list), so this **cannot be verified**, only argued as structurally implausible.
3. **(Design decision, not yet a leak because not yet implemented) if δR/δH extraction ends up reusing `direction_ids` for both the reference-direction construction AND the template-profile construction**, that would be a within-experiment leak (using the same instructions to build the yardstick and to measure against it). Needs an explicit decision: should refusal_direction/harmfulness_direction be rebuilt from a further-split subset of `direction_ids`, or is reusing the *same* `direction_ids` for both acceptable because they're different *conditions* (harmful/harmless vs template/plain) on principle grounds? This is a real methodological choice, not something I should decide unilaterally.

## 7. Token position / layer selection / direction construction consistency

**Confirmed inconsistent with this plan's requirement.** Read `experiment_thesis/scripts/extract_jailbreak_vectors.py` and the underlying `pipeline/submodules/generate_directions.py`:

- Both `refusal_direction` and `harmfulness_direction` are extracted via `get_mean_activations(..., positions=[-1])` — **the same single last-token position of the chat-template-rendered prompt**, functionally closer to `t_post_inst` than a distinct `t_inst`.
- **There is currently no code path that extracts a direction at the raw instruction's own last token (`t_inst`), separate from the post-template `t_post_inst` position.** Your instruction "do not assume `t_inst` and `t_post_inst` are the same position" is correct to flag — they currently *are* conflated in the existing pipeline, not just assumed equal.
- Saved `.pt` files contain only the raw tensor (`torch.save(refusal_dir.cpu(), save_path)`) — no embedded model/language/layer/token-position/estimator/source-ID metadata, confirmed by direct inspection of the save call.

**This means Experiment 2 cannot proceed on the existing `refusal_dir_{lang}.pt`/`harmfulness_dir_{lang}.pt` files without either (a) accepting the single-position conflation as a documented approximation/limitation, or (b) rebuilding both directions with a genuine `t_inst` vs `t_post_inst` distinction, which requires a new GPU extraction pass and a unit test verifying the two positions are located correctly per model's chat template (as your §十 already anticipates). This is the single largest open decision blocking Experiment 2 — needs your explicit call before any new extraction code is written.**

## 8. Has the current pipeline used the test set to select alpha, layer, or threshold?

**Yes, for alpha (10a) and layer (14) — this is a real, confirmed issue, not a hypothetical one.** Neither script has any split-awareness (grep confirms zero references to `validation_ids`/`splits.json`). Every calibration run so far (Qwen `refusal_suppression` α-sweep, Llama/Gemma layer-1.6 calibration, etc.) drew its 10–20 sample prompts from the full, unsplit `completions_{lang}.json` — which, for the OLD 75-instruction pilot, predates `data/splits.json`'s existence entirely, so "did it touch test_ids" isn't even a well-posed question for that data (the split didn't exist yet). For the NEW 572-instruction data, if this pattern is reused as-is, it **would** constitute a leak, since `direction_ids`/`validation_ids`/`test_ids` now exist and calibration should be validation-only per this plan.

**Wei's taxonomy label itself was never selected/tuned on any data** — the `CO`/`MG` mapping is a fixed, preregistered assignment from the literature (per §四 of this plan, correctly labeled as *operational*, not *Wei et al.'s own validated* mapping), not something the pipeline learned or fit.

## 9. Recommended minimal execution order

1. **Decision checkpoint (you, not GPU work):** resolve §6 point 3 and §7 — accept single-position directions as a documented limitation, or commit to rebuilding with dual positions. This gates everything else in Experiment 2/3.
2. **CPU-only, no GPU:** add validation-only sampling to `10a`/`14` (code change, testable without a cluster).
3. **GPU, cheap (forward-pass only, already-proven infrastructure):** re-run `18_extract_paired_diffs.py --ids_key direction_ids` for English/Qwen (already done per your prior message) and the other 2 models; extend `04`'s pairwise-cosine computation to accept `--suffix`/`--ids_key` and re-run on the 300-id direction set, all 3 models.
4. **CPU-only:** re-run `19_taxonomy_robustness.py` (already suffix-aware) on the new 572-scale `paired_diffs`; add robust-estimator variants.
5. **GPU (new code, per §7's decision):** build δR/δH extraction — either on existing single-position directions (documented limitation) or on freshly-rebuilt dual-position directions.
6. **CPU-only:** dual-axis diagnosis / variance decomposition (new script), continuous profile prediction with leave-one-out CV (new script).
7. **Conditional, GPU, only if step 6 shows the continuous profile outperforms the Wei-label baseline:** scope and run the small steering pilot.

## 10. Estimated CPU vs GPU work

- **CPU-only, no cluster GPU needed:** `19` reruns, robust-estimator additions, dual-axis diagnosis script, profile-prediction script, all unit tests, the shared statistics module.
- **GPU needed (English-only scope):** completing the 572-instruction English generation for the remaining 2 models (Llama, Gemma — Qwen done), `18` reruns for those 2 models on `direction_ids`, `04` rerun at 572-scale (English only), and — **only if you decide to rebuild directions with dual token positions** — a new extraction pass for `refusal_direction`/`harmfulness_direction` themselves (this is the one substantial new GPU cost not already accounted for in the currently-running generation jobs). Cross-lingual generation (`generate_and_label_xling.sh`) is no longer required for this plan; any already-running jobs are the user's call to finish or cancel (see note at top of this document).

## 11. Suggested unified config / output schema

Given `scripts/_lang_config.py` already exists as a working precedent for "shared constants instead of re-hardcoding," the same pattern extends naturally to a `configs/thesis_reduced.yaml` (per your §八) covering: confirmatory language list (already exists as `_lang_config.py`, could be mirrored into YAML or left as Python — your call), Wei taxonomy operational mapping (currently only exists inline in `09_refusal_geometry.py` and `21_component_causal_decomposition.py` as a `MECH_CATEGORY` dict — **should be centralized**, since two independent copies already exist and are a real drift risk), split-file path, default estimator, default layer-selection policy (once §7 is resolved).

Output schema: your §5/§6 field lists (model, language, layer, token_position, estimator, sample_count, split_half_reliability, ..., random_seed) are more complete than any current script's output — none of 04/09/19/21 currently emit git-commit/config-hash/model-revision/tokenizer-revision fields. Adding these is new work across every script touched, not just the 3 new ones.

---

## Summary of what's genuinely uncertain and needs your decision before implementation

1. **§7 (biggest blocker):** accept single-position refusal/harmfulness directions as a documented approximation, or commit to a full dual-position rebuild (GPU cost, new unit tests for chat-template position-finding per model)?
2. **§6 point 3:** should direction-construction data and template-profile-construction data be further split within `direction_ids`, or is reusing the same `direction_ids` for both acceptable given they're different conditions?
3. Given §8's confirmed leak in `10a`/`14`: do you want these fixed now (before any new calibration is run on 572-scale data), or is this acceptable to defer since the steering pilot itself is conditional/optional?

No code has been changed and no GPU jobs have been run in producing this plan, per your instructions. Waiting for your confirmation before implementing.

---

## 12. Audit round (Decisions 1/2/3 resolved by user; this section reports the audit results)

The three open questions in the Summary above were resolved by the user as:
**Decision 1** — rebuild both directions with genuine `t_inst`/`t_post` positions (never `positions=[-1]` for both again). **Decision 2** — defer steering (`10a` untouched), but fix `14`'s layer-selection leakage, prioritizing split-half reliability / template-placebo separation / reference-direction reliability on the validation set, plus a pre-registered `floor(0.6*n_layers)` sensitivity check. **Decision 3** — reference axes must not overlap validation/test; prefer an independent PolyRefuse train split, fall back to 5-fold cross-fitting on `direction_ids` if independence can't be confirmed.

This round is audit + infrastructure only — no full GPU direction-extraction job has been submitted, per the explicit instruction to wait for token-audit confirmation first.

### 12.1 Token-position audit

Cannot be run locally — no tokenizer/model access on this machine. Built:
- `scripts/utils/token_positions.py` — `get_instruction_end_position` (t_inst, via subsequence search of the raw instruction's token ids inside the fully-rendered prompt) and `get_post_instruction_position` (t_post, the last token of the rendered prompt — i.e. what `positions=[-1]` already computed, now given an explicit name and metadata). Both return a `PositionResult` with semantic name, token id, decoded token, chat-template hash, and context — never a bare int.
- `scripts/audits/audit_token_positions.py` — the real-tokenizer audit script, to be run on the cluster against Qwen2.5-7B-Instruct/Meta-Llama-3.1-8B-Instruct/gemma-2-9b-it with 5 sample instructions each. Writes `output/audits/token_position_audit.{json,md}`, explicitly flagged for **required human review** of the decoded tokens before trusting the positions.
- `scripts/utils/test_token_positions.py` — 7 CPU-only unit tests against a synthetic mock tokenizer (Qwen/im_start, Llama/header, Gemma/start_of_turn conventions), all passing locally. Proves the subsequence-search *algorithm* is correct given a template; does **not** prove real BPE tokenizers segment the instruction/template boundary the same way the mock assumes — that is exactly what `audit_token_positions.py` must still verify on the cluster.
- **Do not submit a full GPU direction-extraction job until `audit_token_positions.py` has been run on the cluster and its output reviewed.**

### 12.2 Source-overlap audit (ran locally, partial result)

`scripts/audits/audit_source_overlap.py` — run against local `ployrefuse_Enhanced/`, output in `output/audits/axis_source_overlap.{json,md}`.

**Critical finding: local PolyRefuse files carry no native per-item ID** (only `{'instruction': str, 'category': str|None}`) — "overlap by ID" as originally specified is not computable locally; all checks are by normalised instruction text instead.

**Critical finding: English `harmful_train`/`harmless_train`/`harmless_val` are missing from the local `ployrefuse_Enhanced/` mirror.** All 15 other languages have train+val locally; only English is missing it (and, symmetrically, only English has `harmless_test`). **This blocks direct verification of Decision 3 for English** — cannot confirm from local data whether English's would-be independent-train-split axis data overlaps the 572-instruction pool.

Confirmed instead, as indirect evidence: for all 15 non-English languages checkable locally, `harmful_train` (260 items) vs `harmful_test` (572 items) normalised-text overlap = 0, `harmful_val` (39 items) vs `harmful_test` overlap = 0, `harmful_train` vs `harmful_val` overlap = 0 — consistent with PolyRefuse maintaining genuine train/val/test disjointness by design. Also confirmed: `data/sampled_prompts.json` (572 items) is **exactly** `ployrefuse_Enhanced/harmful_test_translated_en.json` (same 572 rows, 562 unique texts, 10 internal duplicates) — exact text-set equality, not assumed.

**Recommendation for Decision 3, pending cluster verification:** locate English `harmful_train`/`harmless_train`/`harmless_val` via the `_orig` (`dataset.load_dataset`) path referenced in `experiment_thesis/scripts/extract_jailbreak_vectors.py::load_dataset_split` (this is the only path that could plausibly reach data absent from the local mirror), and run the same normalised-text overlap check directly. **If that data cannot be located or turns out to overlap, use 5-fold cross-fitting on the 300 `direction_ids` instead** (build refusal/harmfulness directions from 4 folds, compute out-of-fold δR/δH on the held-out fold, repeat ×5, merge) — this has zero dependency on locating an independent English train split, at the cost of ~5x direction-extraction compute and slightly smaller per-fold training data (240 vs 300).

### 12.3 Layer-selection leakage audit (done, no cluster needed)

Full report: `output/audits/layer_selection_leakage.md`. Key correction to §8 above: `14_find_safety_layer.py`'s criterion is peak `refusal_direction` L2-norm within the first 80% of layers — **not** based on classification accuracy, ASR, or test outcomes (that specific fear is not what's happening). The actual leak is structural: `refusal_dir_{lang}.pt` (14's input) was built by `extract_jailbreak_vectors.py` from the original unsplit 75-instruction pilot, predating `data/splits.json` — so the layer-norms 14 maximizes over come from a pool undifferentiated with respect to the current direction/validation/test partition, not a genuine validation-only selection. Fix required: rebuild directions with an explicit `--split direction`, add `--split validation` to `14`, add the `floor(0.6*n_layers)` sensitivity check, and — per the user's explicit instruction — do not unilaterally lock a final selection rule; output candidate rules and their effect on `validation_ids` first.

**Outputs marked stale:** `output/safety_layer_identification.json` (`stale` — built from un-partitioned pre-split directions). No `16_single_layer_geometry.py` output currently exists (confirmed via `ls output/`), so nothing downstream needs retroactive staleness marking yet — but any future run using the current `safety_layer_identification.json` would inherit the same staleness.

### 12.4 New infrastructure built this round

- `scripts/utils/token_positions.py` + `scripts/utils/test_token_positions.py` (7/7 passing)
- `scripts/utils/direction_metadata.py` (`build_direction_metadata`/`save_direction_metadata`/`load_direction_metadata`, enforcing the required-field schema in §12.5) + `scripts/utils/test_direction_metadata.py` (7/7 passing)
- `scripts/audits/audit_source_overlap.py` (run, output committed) + `scripts/audits/audit_token_positions.py` (cluster-only, not yet run) + `scripts/audits/test_smoke.py` (3/3 passing, re-runs the source-overlap audit end-to-end and checks known findings haven't silently changed)
- `output/audits/axis_source_overlap.{json,md}`, `output/audits/layer_selection_leakage.md`

Leakage-prevention CLI flags (`--split`, `--ids-file`, `--axis-source`, `--position`) and the disjointness assertions are **not yet added** to `04`/`10a`/`14`/`18` — flagged as the first GPU-adjacent code change to make once Decision 3's axis-dataset question is resolved (§12.2), since the flag design depends on which of independent-train-split vs cross-fitting is chosen.

### 12.5 Direction metadata schema (implemented)

Every new direction `.pt` must ship a same-named `.json` built via `scripts/utils/direction_metadata.py`, containing: `direction_type`, `model`, `model_revision`, `tokenizer_revision`, `chat_template_hash`, `semantic_position` (`t_inst`|`t_post`), `layer`, `source_partition`, `source_ids_hash`, `sample_count`, `construction_contrast`, `git_commit`, `random_seed`. `save_direction_metadata` refuses to write if any required field is missing.

### 12.6 Recommended next GPU tasks (not submitted)

In dependency order:
1. Run `scripts/audits/audit_token_positions.py` on the cluster for all 3 models; human-review the output; only then trust `token_positions.py` for real extraction.
2. On the cluster, attempt to locate English `harmful_train`/`harmless_train`/`harmless_val` via the `_orig` loader path; re-run the overlap check for English specifically. This determines whether Decision 3 uses independent-train-split or cross-fitting.
3. Once 1–2 are resolved: rebuild `refusal_direction`/`harmfulness_direction` for English at `t_inst`/`t_post` respectively, using whichever axis-source design §2 selected, saving full metadata per §12.5.
4. Add `--split`/`--ids-file`/`--axis-source`/`--position` CLI flags + disjointness assertions to `04`/`14`/`18` (and `10a` only if/when steering is un-deferred).
5. Re-run `14_find_safety_layer.py` validation-only, output candidate layer-selection rules (not a single locked-in choice) for the user to pick from.
6. Only after 1–5: resume the Experiment 1/2/3 GPU pipeline (English 572-instruction generation completion for Llama/Gemma, `18`/`04` reruns on `direction_ids` with the new dual-position directions).

No GPU jobs have been submitted in producing this round's deliverables.
