# Data Manifest

Documents what each file in `output/` is, how it was produced, and what depends on what — so the file structure stays interpretable without needing a database. No schema changes, no migration; this is documentation only.

---

## Scope (two-phase design, current)

Pilot restarted from a 75-instruction sample to the full 572-instruction
`harmful_test` pool per language (all already translated). Given the full
572 x 9 languages x 3 models x 8 conditions scale (123,552 generations)
is disproportionate for a master's thesis, scope is split:

- **English core experiment** (`data/generation_input_en_full572.json`):
  full 572 instructions x 8 conditions x 3 models = 4,576 rows/model.
  Answers the primary question (does Wei et al.'s taxonomy correspond to
  separable activation geometry?) with maximum statistical power. Submit
  `03_generate_and_label.py --lang en --suffix _full572` -> writes
  `completions_en_full572.json`, distinct from the original pilot's
  `completions_en.json` (75 instructions) so nothing gets overwritten.
- **Cross-lingual validation, CONFIRMATORY scope** (`data/generation_input_{lang}_xling.json`
  for `zh/ar/th/yo/am` -- see `scripts/_lang_config.py`): a shared
  200-instruction subset x 8 conditions x 3 models = 1,600 rows/model/language.
  Tests whether the English finding generalizes, without the full-pool cost.
  Submit `03_generate_and_label.py --lang {lang} --suffix _xling` -> writes
  `completions_{lang}_xling.json`, again distinct from the original
  9-language pilot's `completions_{lang}.json` (75 instructions).
  Total confirmatory scale: English 572×8×3=13,728 + 5 languages ×200×8×3=24,000
  = **37,728** model-input combinations.
- **`de`/`ko`/`sw` are excluded from the confirmatory run** (kept 2
  languages/resource-tier instead of 3, to keep total scale proportionate to
  a master's thesis) but **not deleted**: their `generation_input_{lang}_xling.json`
  files (already built, 1,600 rows each) remain on disk, and all their
  original 9-language 75-instruction pilot results are untouched. See
  `scripts/_lang_config.py` for `CONFIRMATORY_LANGUAGES` vs `PILOT_LANGUAGES`.
- **English has two direction estimates, not one** -- do not compare them
  interchangeably: `english_full_direction` (built from the 300-id English
  `direction_ids`, used for the English-only core taxonomy experiment) vs
  `english_xling_direction` (built from the 100-id `cross_lingual_direction_ids`
  subset, the *only* one valid for comparison against zh/ar/th/yo/am's
  directions, which are necessarily built from the same 100 shared ids).

`data/splits.json` defines the leak-free partition: `direction_ids` (300),
`validation_ids` (72), `test_ids` (200) for English, each with a strict
subset (`cross_lingual_*_ids`, 100/30/70) used for the 8 cross-lingual
languages -- so a cross-lingual test-set id is guaranteed to also be an
English test-set id, never a direction or validation id. Built by
`01b_define_splits.py`; consumed by `02_build_templated_data.py --ids_key`.

The original 9-language x 3-model x 75-instruction pilot results (README,
sections 01-14 of the published summary artifact) remain valid as
supplementary evidence and are not being redone at full scale.

## Directory structure

```
new_experiment/
├── templates/templates_{lang}.json       # 9 languages, 6 mechanisms + placebo, translated + back-translation reviewed
├── data/
│   ├── resource_tiers.json               # high/medium/low language tier assignments
│   ├── sampled_prompts.json              # the 75 category-stratified source instructions (shared across all languages/models)
│   └── generation_input_{lang}.json      # 600 rows/language (75 instructions x 8 conditions), input to 03
├── ployrefuse_Enhanced/                  # local copy of the 16-language PolyRefuse dataset
├── scripts/00.py .. 19.py                # numbered pipeline, see "Script -> output" below
├── slurm/*.sh                            # one submission script per script, MODEL_IDX=0/1/2 pattern
└── output/
    ├── {model_alias}/completions_{lang}.json         # per-model, per-language: 600 generations + WildGuard labels (from 03)
    ├── {model_alias}/pilot_results.json               # core cross-language/cross-mechanism cosine similarity (from 04)
    ├── {model_alias}/split_half_reliability.json       # noise-floor ceiling (from 06)
    ├── {model_alias}/leave_one_category_out.json       # robustness to harm-category composition (from 07)
    ├── {model_alias}/magnitude_vs_behavior.json        # ||template_direction|| vs behavioral ΔASR (from 08)
    ├── {model_alias}/refusal_geometry.json             # template_direction vs refusal/harmfulness_direction (from 09)
    ├── {model_alias}/calibration_{lang}_{mech}[_layer{L}].json  # Phase-0 alpha calibration (from 10a); layer suffix only present if --layer was overridden
    ├── {model_alias}/phase1_injection_results.json     # Phase-1 causal injection results (from 10b)
    ├── {model_alias}/single_layer_geometry_L{layer}.json  # 09's geometry recomputed at one fixed layer (from 16)
    ├── {model_alias}/outlier_diagnosis_layer{layer}.json  # per-instruction diff-norm distribution check (from 17)
    ├── {model_alias}/paired_diffs_{lang}.pt             # raw per-instruction, all-layer diff tensors (from 18) -- prerequisite for 19
    ├── {model_alias}/taxonomy_robustness_{lang}.json    # Stage-1 clustering-robustness diagnostics (from 19)
    ├── {model_alias}/cross_mechanism_defense_pilot.json # pairwise mechanism defense-transfer pilot (from 15, not yet run)
    ├── encoding_obfuscation_audit.json                  # cross-model, cross-language genuine-decode audit (from 05)
    ├── layerwise_profile.json                           # same_language_cross_mechanism per-layer curves (from 12)
    ├── layerwise_cross_language.json                    # same_mechanism_cross_language per-layer curves (from 13)
    ├── safety_layer_identification.json                 # per-language peak-norm layer, l<0.8L cutoff (from 14)
    └── translation_surface_similarity.json              # template-text surface-similarity confound check (from 11)
```

## Script → output (source of truth for what generated what)

| Script | Reads | Writes | Needs GPU? |
|---|---|---|---|
| 00_translate_templates.py | templates_en.json | templates_{lang}_draft.json | No |
| 01_sample_prompts.py | ployrefuse_Enhanced/ | data/sampled_prompts.json | No |
| 02_build_templated_data.py | sampled_prompts.json, templates | data/generation_input_{lang}.json | No |
| 03_generate_and_label.py | generation_input_{lang}.json | completions_{lang}.json | Yes (+WildGuard) |
| 04_extract_directions_and_analyze.py | completions_{lang}.json (all 9) | pilot_results.json | Yes |
| 05_audit_encoding_obfuscation.py | completions_{lang}.json | encoding_obfuscation_audit.json | No |
| 06_split_half_reliability.py | completions_{lang}.json | split_half_reliability.json | Yes |
| 07_leave_one_category_out.py | completions_{lang}.json | leave_one_category_out.json | Yes |
| 08_magnitude_vs_behavior.py | completions_{lang}.json, pilot_results.json | magnitude_vs_behavior.json | Yes |
| 09_refusal_geometry.py | completions_{lang}.json + **experiment_thesis** refusal_dir/harmfulness_dir .pt | refusal_geometry.json | Yes |
| 10a_calibrate_injection_alpha.py | completions_{lang}.json | calibration_{lang}_{mech}[_layer{L}].json | Yes (+WildGuard) |
| 10b_phase1_injection_experiment.py | completions_{lang}.json (6 langs) | phase1_injection_results.json | Yes (+WildGuard) |
| 11_translation_surface_similarity.py | templates/*.json | translation_surface_similarity.json | No |
| 12_layerwise_profile.py | pilot_results.json (all 3 models) | layerwise_profile.json | No |
| 13_layerwise_cross_language.py | pilot_results.json (all 3 models) | layerwise_cross_language.json | No |
| 14_find_safety_layer.py | **experiment_thesis** refusal_dir_{lang}.pt | safety_layer_identification.json | No (needs venv's torch) |
| 15_cross_mechanism_defense_pilot.py | completions_{lang}.json | cross_mechanism_defense_pilot.json | Yes (+WildGuard) |
| 16_single_layer_geometry.py | completions_{lang}.json + experiment_thesis refusal_dir/harmfulness_dir | single_layer_geometry_L{layer}.json | Yes |
| 17_diagnose_outlier_influence.py | completions_{lang}.json | outlier_diagnosis_layer{layer}.json | Yes |
| 18_extract_paired_diffs.py | completions_{lang}.json | paired_diffs_{lang}.pt | Yes |
| 19_taxonomy_robustness.py | paired_diffs_{lang}.pt | taxonomy_robustness_{lang}.json | No (CPU-only) |

## Cross-repository dependency

`refusal_direction`/`harmfulness_direction` are **not** produced in this repo — they come from `experiment_thesis/scripts/extract_jailbreak_vectors.py`, saved as `refusal_dir_{lang}.pt` / `harmfulness_dir_{lang}.pt` under `experiment_thesis/output/jailbreak_analysis/{model_alias}/`. Scripts 09, 14, 16 read them via `--refusal_dir_root`. If those files are stale (e.g. missing sw/am, or predate the `extract_jailbreak_vectors.py` fix), 09/14/16's results downstream are stale too — check the source repo's git log, not just this repo's.

## Known gaps / not-yet-run (as of this writing)

- `paired_diffs_{lang}.pt` / `taxonomy_robustness_{lang}.json`: only run for Qwen/en so far. Needs Llama, Gemma, and (per the staged rollout) more languages before the taxonomy-robustness conclusion can be called cross-model/cross-language robust.
- `cross_mechanism_defense_pilot.json`: script written, not yet executed on the cluster.
- `single_layer_geometry_L{layer}.json`: depends on `safety_layer_identification.json`'s peak-norm layer, which is a proxy (not Arditi et al.'s full induce/kl-score criteria) — treat as a sensitivity check, not a definitive "correct layer."
- `calibration_{lang}_{mech}.json` (no layer suffix) exists for `refusal_suppression`/`en` on all 3 models at the default middle layer; layer-suffixed versions only exist where a re-check was explicitly run (see conversation history for which alpha/layer combinations were tested and why).

## Audit round (token position / axis independence / layer-selection leakage)

New, read-only-so-far audit infrastructure — full findings in
`EXPERIMENT_REDUCTION_PLAN.md` §12. Summary:

- `scripts/utils/token_positions.py` — explicit `t_inst`/`t_post` position
  finding (replaces the implicit `positions=[-1]` convention), with
  `scripts/utils/test_token_positions.py` (7/7 passing, mock-tokenizer only).
  `scripts/audits/audit_token_positions.py` still needs to run **on the
  cluster** (real tokenizer required) before any new direction extraction —
  not yet run.
- `scripts/utils/direction_metadata.py` — required-field schema for every
  new direction `.pt`'s companion `.json` (`scripts/utils/test_direction_metadata.py`,
  7/7 passing).
- `scripts/audits/audit_source_overlap.py` — run against local data, output
  in `output/audits/axis_source_overlap.{json,md}`. **English
  `harmful_train`/`harmless_train`/`harmless_val` are absent from
  `ployrefuse_Enhanced/`** (present for all 15 other languages) — but
  **resolved via `output/audits/english_axis_data_followup.json`**: they
  live at `related_work/Multilingual-Refusal/dataset/splits/{harmtype}_{split}.json`
  instead (English never needed translation), confirmed present locally.
  `harmful_train` (260) overlaps the 572-instruction pool by exactly 1
  instruction (`p457`), which falls in `direction_ids`, not
  `validation_ids`/`test_ids` — so the independent-train-split design for
  Decision 3 is usable as long as `p457` is excluded from `harmful_train`
  when building `refusal_direction`/`harmfulness_direction`. Local
  PolyRefuse files have no native ID field, so all overlap checks are by
  normalised text, not ID.
- `output/audits/layer_selection_leakage.md` — `output/safety_layer_identification.json`
  is marked **`stale`**: `14_find_safety_layer.py`'s peak-norm criterion is
  not itself test-outcome-based, but its input (`refusal_dir_{lang}.pt`) was
  built pre-`splits.json`, from an unpartitioned pool. No downstream
  `single_layer_geometry_L{layer}.json` currently exists, so nothing else
  needs retroactive marking yet.
- `scripts/audits/test_smoke.py` (3/3 passing) — CPU-only, re-runs the
  source-overlap audit and checks its known findings (English gap, zero
  train/test overlap for the 15 checkable languages, `sampled_prompts.json`
  == `harmful_test_translated_en.json`) haven't silently changed.

**Old `positions=[-1]`-based `refusal_dir_{lang}.pt`/`harmfulness_dir_{lang}.pt`
files (in `experiment_thesis/output/jailbreak_analysis/`) remain usable for
Experiment 1** (the binary-taxonomy geometry test doesn't depend on the
t_inst/t_post distinction) **but are `legacy_same_position` and
`stale_for_dual_axis_claim = true` for Experiment 2/3**, which require
genuinely separate refusal/harmfulness reference positions. Any new
dual-position direction extraction should write to a new
`output_v2_dual_position/` directory rather than overwriting these —
not yet created, since no extraction has run yet (blocked on the
token-position audit, per `EXPERIMENT_REDUCTION_PLAN.md` §12.6).

## Exp3 defence-protocol infrastructure (benign data + hook, no GPU run yet)

- `scripts/36_build_benign_data.py` — CPU-only. Samples the disjoint benign
  validation/test sets for the defence evaluation from the SAME
  `related_work/Multilingual-Refusal/dataset/splits/` pool already confirmed
  in `output/audits/english_axis_data_followup.json`, but from
  `harmless_val.json`/`harmless_test.json` (not `harmless_train.json`, which
  remains reserved for the refusal_direction/harmfulness_direction axis
  construction elsewhere in the project) — a third, independent slice of the
  same already-audited pool, so no new overlap risk with `direction_ids`/
  `validation_ids`/`test_ids` is introduced.
  - Source pools: `harmless_val.json` (6264 rows), `harmless_test.json` (6266 rows) —
    confirmed pairwise disjoint from each other and from the 572-instruction
    harmful pool by normalized-instruction-text comparison (same convention
    as `audit_source_overlap.py`), not by ID (PolyRefuse-derived files have
    no native ID field).
  - `data/benign_validation_80.json` — 80 rows, seeded (seed=20260830) sample
    of `harmless_val.json`.
  - `data/benign_test_100.json` — 100 rows, seeded (seed=20260831) sample of
    `harmless_test.json`.
  - `data/benign_data_manifest.json` — records both source file paths, both
    pool sizes, both seeds, both sample sizes, the normalization method used
    for the disjointness checks (`.strip().lower()`), the three pairwise
    overlap counts (all 0), and the SHA-256 of each output file's exact JSON
    content (re-verify by re-running the script and diffing the hash, not by
    eyeballing the file).
- `scripts/37_defence_directions_and_hooks.py` — GPU-free (torch only, no
  `pipeline` import), unit-tested via a mock `nn.Module` in
  `scripts/audits/audit_defence_hooks_dry_run.py`. Provides:
  - `build_c_G` / `build_c_placebo` / `build_c_template_specific` implementing
    the frozen `g_G = normalize(mean(normalize(dtilde_m)))`, `s_G = median(||dtilde_m||)`,
    `c_G = s_G * g_G` recipe (Placebo is the one exception — it uses its own
    raw paired-diff direction unscaled, described here only as a
    **content-neutral control wrapper direction**, not as a direction already
    proven free of jailbreak-relevant signal; that is an empirical question
    the defence evaluation itself is meant to help answer, not an assumption
    baked into its construction).
  - `FROZEN_ADAPTIVE_GROUPING` — the per-model template-specific/subgroup
    membership decided from `output/canonical_v2/experiment3_common_direction_bootstrap.json`'s
    real (non-synthetic) results: a mechanism is `template_specific` iff (a)
    its existing split-half reliability check passes (all 18 mechanism x
    model combinations already do, per `experiment1_taxonomy_geometry.json`'s
    `reliability_at_fixed_layer`) and (b) its leave-one-template-out
    prototype-affinity `A_k`'s 2000-rep source-level bootstrap 95% CI upper
    bound is `< 0` (seed 20260830). Every group's frozen membership was
    verified against those real CI values before being hardcoded here;
    module-level asserts additionally check each model's grouping still
    exactly partitions all 6 active mechanisms with no overlap/gap.
  - `make_prefill_last_token_hook` — additive `h' = h - alpha*c_G` at the
    last prefill token position only, single-fire per `generate()` call via
    a `has_intervened` flag (NOT a `seq_len>1` hard assertion — that check is
    a non-fatal audit-log warning only, since a future input construction
    could legitimately violate that assumption without it being a real bug).
    **Contract for the not-yet-written generation driver**: after each
    `generate()` call, it must call `assert_single_intervention(state)`; if
    a batch's `intervention_count != 1`, that batch's output must be marked
    invalid and excluded, never silently saved as if the intervention had
    applied normally.
  - `assert_left_padded` — cheap pre-generation check that
    `attention_mask[:, -1]` is all 1s (confirms `tokenizer.padding_side='left'`
    is actually in effect for the batch about to be generated).
- **Not yet written**: the actual generation driver (wiring the hook into
  `pipeline`'s existing `add_hooks`/`generate_completions`, confirmed to
  already left-pad and already accept `fwd_hooks=[...]` — see
  `pipeline/model_utils/model_base.py`) and the unified WildGuard re-judge
  script. No GPU job has been run for Exp3's defence protocol yet.

### Known metadata defect in `completions_en_full572_corrected.json`, and its repair

`scripts/audits/audit_corrected_completions.py`'s real run against all 3
models' `completions_en_full572_corrected.json` (before repair) found
exactly one failing check, identically in all 3 files: every
`condition=='persona_roleplay'` record's `mechanism` field still read
`'mismatched_generalization'` (572 rows/model) instead of
`'competing_objectives'` — a leftover from before the taxonomy correction
that generated this file's content was never updated. **The template text,
generated `response`, and `generation_tokens` for `persona_roleplay` are
correct and unaffected** — only this one per-record metadata annotation was
stale. All other checks (record count, id/condition completeness, no
duplicates, split coverage, no missing responses, no stale
`instruction_hierarchy`/`fictional_framing` conditions, cross-model
agreement) passed on the first run.

This field has never been read by any geometry script in this project —
`18_extract_paired_diffs.py`, `33_canonical_taxonomy_geometry.py`,
`34_llama_co_divergence_diagnostic.py`, and
`35_common_direction_coverage_audit.py` all resolve CO/MG membership
dynamically via `_taxonomy_v2_loader.py` reading `templates/templates_en.json`,
never from a completion record's `mechanism` field — so **Exp1/Exp2/Exp3's
existing results are unaffected by this defect and required no re-run**.

- `scripts/38_repair_corrected_mechanism_metadata.py` — the deterministic
  fix. Default is dry run; only writes with `--apply`, via a temp-file +
  `os.replace` atomic swap. Refuses to run (raises, writes nothing) unless
  pre-flight checks confirm the defect is exactly this known shape (572 or
  0 stale `persona_roleplay` rows, all stale values exactly
  `'mismatched_generalization'`, zero mismatches on any other
  active-mechanism condition, 572 unique ids each with 8 conditions, no
  duplicate `(id, condition)` pairs) — a differently-shaped defect will not
  be silently patched by this script. After repair, independently verifies
  that removing the `mechanism` key from every record leaves before/after
  records identical, proving only that one field changed. Tested against
  synthetic fixtures in `scripts/audits/audit_repair_mechanism_metadata_dry_run.py`
  (dry run writes nothing; apply changes exactly 572/model; a second apply
  is idempotent — 0 rows changed; an injected non-`persona_roleplay`
  mismatch causes refusal even under `--apply`; independently re-verified
  field-level equivalence; the real `audit_corrected_completions.py`
  reports `OVERALL_PASS: True` on the repaired synthetic fixture).
- `output/canonical_v2/corrected_completions_metadata_repair.json` — the
  real repair run's report (per model: input path, sha256 before/after,
  total/changed row counts, taxonomy config sha256, git commit, timestamp).
- `output/canonical_v2/experiment3_corrected_completions_audit.json` — the
  post-repair re-run of the audit; must show `OVERALL_PASS: true` before
  any defence-generation driver is implemented.
- `scripts/39_defence_pilot.py` — the first script in this project that
  actually touches a real model/GPU for the defence protocol. Deliberately
  tiny and NOT the driver: 1 model (Llama), 4 real `validation_ids`
  instructions, 1 template (`persona_roleplay`), 1 condition (Global),
  1 alpha (1.0). Confirms the hook fires exactly once per batch on a real
  `model.model.layers[19]` forward call; reports a real GPU name (sacct/
  sinfo report no GRES info on this cluster — see
  `output/canonical_v2/experiment3_throughput_pilot.json`'s contents once
  run); measures isolated target-generation-only and WildGuard-only
  throughput (historical `slurm/logs/generate_en572_*.out` fuse both into
  one timestamp with no boundary marker, so only a combined rate could be
  computed from those). Reuses `03_generate_and_label.py`'s exact
  `WILDGUARD_PROMPT`/`_parse_wildguard` rather than a new prompt, and
  `pipeline`'s existing `generate_completions`/`add_hooks` machinery rather
  than a hand-rolled generation loop. No output from this pilot is a
  reportable defence-efficacy result — it is a timing/correctness check
  only, run before committing to the full validation-sweep GPU budget.
- `scripts/40_defence_generation_driver.py` — the actual driver framework,
  superseding the ad-hoc `39_defence_pilot.py` (which stays as-is, not
  reused further). Implements `--phase timing-pilot` (the only phase
  authorized to run so far) and the full `--phase validation` code path
  (implemented for completeness/testability but its CLI entry point raises
  `NotImplementedError` on purpose — running a full 72x6x4x4 sweep needs
  separate explicit authorization, not just this script existing).
  - `record_key(...)` — deterministic sha256 over
    `{model, split, instruction_id, benign_or_harmful, template, method,
    alpha, direction_config_hash, generation_config_hash}`; `--resume`
    checks this key (via `load_jsonl`/`append_jsonl` on the output
    `.jsonl`), never instruction ID alone.
  - `judge_cache_key(...)` — sha256 over
    `request + response + judge_prompt_version + judge_model_version`
    (never instruction ID alone), so identical (instruction, response)
    pairs across different conditions/methods can share one WildGuard call
    without silently caching on the wrong key.
  - `run_condition(...)` reuses `pipeline`'s `generate_completions`/
    `add_hooks` (never a hand-rolled generation loop) and
    `37_defence_directions_and_hooks.py`'s hook; `hook_alpha_zero`
    genuinely registers and fires the hook with `alpha=0` rather than
    skipping hook registration as an "equivalent" optimization, so the
    measured overhead is real.
  - `compare_determinism(...)` — hard per-record equality check
    (`generation_tokens`, decoded `response`, `stop_reason`) between
    `no_hook` and `hook_alpha_zero`; `run_timing_pilot` prints and stops
    short of WildGuard judging/comparison analysis if it fails.
  - `parse_wildguard_strict(...)` — unlike `03_generate_and_label.py`'s
    `_parse_wildguard` (which silently defaults a missing/malformed line to
    0), this returns `(parsed_fields, parse_success, parse_error_reason)`
    and the caller excludes failed rows from all metrics rather than
    counting them as a valid "no".
  - Timing-pilot prompt set: 5 harmful `validation_ids` (seed 20260901) +
    5 benign ids from `data/benign_validation_80.json` (seed 20260902),
    each rendered under all 6 V2 templates via `02_build_templated_data.py`'s
    real `build_condition` (not a re-derived copy) = 60 prompts, 180 target
    generations across the 3 conditions. `test_ids`/`benign_test_100.json`
    are never read by this script.
  - Outputs (filenames include `model_alias` -- a shared name would let a
    later model's run silently overwrite an earlier model's already-passed
    results, an actual bug caught and fixed before the cross-model audit):
    `output/canonical_v2/experiment3_timing_pilot_60_{model_alias}.json`
    (metadata + determinism check + per-condition generation/judge metrics
    + the two required comparisons; the summary files ARE committed --
    `..._generations_{model_alias}.jsonl`/`..._judgements_{model_alias}.jsonl`
    (the full 180-row-per-model raw generation/judge logs) are NOT committed,
    per explicit instruction, only produced locally on the cluster run).
  - `slurm/defence_timing_pilot_60.sh` submits this phase only; `MODEL_IDX`
    (via `--export`) selects which model, doubling as the cross-model hook
    audit since the phase's own checks (layer correctness via
    `FIXED_LAYERS`, single-fire-per-batch, determinism, GPU memory) are
    model-generic.
  - `CONDITION_BATCH_SIZE_OVERRIDE = {'gemma-2-9b-it': 15}` in
    `40_defence_generation_driver.py` -- Gemma2's HF forward computes
    float32 logits over the FULL padded sequence length for every row
    during prefill (not just the last position), so `batch_size=60` OOMs
    (~17GB just for logits; confirmed on job 4979, "Tried to allocate 16.82
    GiB"). `run_condition` was fixed to genuinely chunk into sub-batches
    with a **fresh hook (fresh `has_intervened` state) per chunk** --
    reusing one hook object across `generate_completions()` calls would
    have silently skipped the intervention on every chunk after the first.
    `intervention_count_distribution` (one entry per batch, each asserted
    `==1` immediately, not deferred to an aggregate check) replaced the
    original single-scalar `intervention_count` field.
  - **Cross-model hook audit: PASSED for all 3 models** (jobs 4977=Qwen,
    4978=Llama, 4980=Gemma) -- correct fixed layer per model (16/19/25),
    `intervention_count_distribution` all-1s, alpha=0 determinism check
    passed with 0 mismatches, 0 warnings, 0 WildGuard parse failures, no
    OOM. Real GPU: NVIDIA L40S (46GB). This clears the gate for `--phase
    validation` implementation/execution.

## Exp3 validation-phase implementation (code complete, not yet run)

`40_defence_generation_driver.py --phase validation` is now implemented
(previously raised `NotImplementedError`). Gate for running it for real:
Step 1 (cross-model timing-pilot hook audit) and Step 2 (versioning) both
passed -- see the timing-pilot section above.

- `build_validation_harmful_prompts`/`build_validation_benign_prompts` --
  72 `validation_ids` x 6 templates = 432, and 80 `benign_validation_80.json`
  rows x 6 templates = 480, asserted exactly. `test_ids`/`benign_test_100.json`
  are never read anywhere in this phase.
- `direction_vector_for(conds, method, template)` -- routes `placebo`/`global`
  to their single shared vector, `fixed_wei`/`adaptive` to the per-template
  vector (Adaptive's boundary members get their own `c_m`; subgroup members
  share their reduced-group `c_G`, per `FROZEN_ADAPTIVE_GROUPING`); raises on
  `no_defence` (handled by separate functions, has no direction).
- **Resume is integrity-checked, not line-count-checked**: `record_is_valid`
  requires every field in `REQUIRED_GENERATION_FIELDS` present and
  non-null/non-empty (a record truncated by a killed job -- e.g. missing
  `generation_tokens` because the process died mid-write -- is treated as
  absent and regenerated, never silently counted as done).
  `load_valid_existing_keys` reports both the valid-key set and how many
  existing rows were invalid.
- `run_validation_intervention_method` -- one (model, method) job. Computes
  the full to-do list (4 alphas x 912 prompts minus already-valid records)
  BEFORE loading the model (so a fully-resumed job doesn't pay model-load
  cost). Uses `CONDITION_BATCH_SIZE_OVERRIDE` (same Gemma fix as the
  timing-pilot) and creates a **fresh hook per sub-batch** (never one hook
  reused across `generate_completions()` calls), asserting
  `intervention_count==1` immediately after each sub-batch -- an
  `AssertionError` propagates and stops the job rather than being caught
  and logged. Each sub-batch's records are appended to the `.jsonl`
  immediately (crash-safe incremental writes).
- `run_no_defence_benign` -- fresh generation (no hook at all) for the 480
  benign validation prompts per model.
- `run_no_defence_harmful_rejudge` -- generates NOTHING; reads the already-
  repaired `completions_en_full572_corrected.json`, filters to
  `validation_ids` x the 6 active mechanisms (432 rows/model, asserted),
  and re-judges them with the strict judge only. This is the "reuse
  completions but re-judge with the new pipeline" requirement -- it is a
  WildGuard-only job, no target model / no `pipeline` import needed for
  this specific path.
- **Bug found and fixed after the first real run** (job 4982, Llama x
  global, 3648/3648 records generated successfully): `run_validation_intervention_method`
  only generated -- there was no corresponding judge step wired up, and
  `run_judge`'s judgement-building loop hardcoded `r['condition']`, which
  validation records don't have (they use `method`/`benign_or_harmful`
  instead of the timing-pilot schema's `condition`) -- would have KeyError'd
  the first time judging was attempted. Fixed by generalizing `run_judge`
  to pass through whichever of a fixed field list (`instruction_id`,
  `template`, `condition`, `method`, `benign_or_harmful`, `alpha`, `model`,
  `split`, `record_key`) each record actually has, and adding
  `run_validation_judge` (new `--stage generate|judge` CLI flag; `generate`
  is the default, matching the already-run jobs' behavior) plus
  `_validation_gen_and_judge_paths` for consistent per-(model,method)
  filenames. `run_no_defence_harmful_rejudge` also now sets `r['method']`
  (previously only `r['condition']`) for consistency across all judgement
  files. Verified with `scripts/audits/audit_run_judge_schema_dry_run.py`
  against both schemas using a faked WildGuard tokenizer/model (no GPU).
- **Second bug found and fixed, same real run** (job 4983, STAGE=judge on
  the 3648 records job 4982 had generated): `run_judge` had no internal
  batching at all -- it tried one single WildGuard forward pass over all
  3648 new records and OOM'd ("Tried to allocate 36.68 GiB"). Fixed by
  adding `WILDGUARD_JUDGE_BATCH_SIZE=16` (matches `03_generate_and_label.py`'s
  own `wg_batch` default) and an internal chunking loop, plus an optional
  `on_new_batch(list_of_merged_judgement_dicts)` callback so callers can
  persist incrementally instead of only writing at the very end (crash-
  safety at thousands-of-records scale) -- `run_validation_judge` and
  `run_no_defence_harmful_rejudge` both now use this; `run_timing_pilot`
  (max 60 records/condition) still writes once at the end, unaffected.
  Verified with `scripts/audits/audit_run_judge_batching_dry_run.py`: 100
  fake records batch into 7 WildGuard calls of size ≤16 (never one big
  call), `on_new_batch` fires once per chunk with fully-merged persistable
  dicts, and a fully-cached re-run makes zero new WildGuard calls.
- `compute_template_asr`/`compute_template_frr`/`compute_macro_asr`/
  `compute_macro_frr` -- pure functions; both explicitly exclude
  `parse_success=False` rows and (for ASR) `request_harmful=0` rows from
  the denominator, reporting `valid_denominator`/`n_excluded` alongside the
  rate so a low denominator is visible, never silently averaged over.
- `select_alpha` -- the frozen rule: minimize macro-ASR among alphas whose
  macro-FRR does not exceed `no_defence_macro_frr + 5 percentage points`;
  ties broken by smallest alpha; if no non-zero alpha is eligible, freezes
  `alpha=0.0` with an explicit reason string (never silently picks
  something arbitrary).
- 21 GPU-free tests across `scripts/audits/audit_validation_phase_dry_run.py`
  (prompt counts, direction routing, integrity-checked resume x4, ASR/FRR
  computation, macro aggregation, alpha selection incl. tie-break and the
  no-eligible-alpha fallback) and the earlier chunking-dry-run script all
  pass.
- `slurm/defence_validation.sh` -- one job per `(MODEL_IDX, METHOD)` (or
  `(MODEL_IDX, no_defence, NO_DEFENCE_TARGET)`), independently resumable,
  matching the "at least model x method" job-splitting requirement.
- **Not yet run**: no validation generation job has been submitted. Full
  expected scope once run: 43,776 intervention records (3 models x 4
  methods x (72x6x4 harmful + 80x6x4 benign)) + 1,440 no-defence-benign
  generations + 1,296 no-defence-harmful re-judgements (reused, not
  regenerated). Test phase remains explicitly unauthorized until validation
  completes and its frozen-config audit is reviewed.
- **Real run status (Llama x global)**: job 4982 generated all 3,648
  records successfully. Job 4983 (judge stage) OOM'd -- see the WildGuard
  batching fix above. Job 4984 (re-run after the fix) judged all 3,648
  successfully, 0 parse failures; only 2,051 unique judgement rows were
  written because ~44% of the 3,648 records were byte-identical-content
  duplicates across the 4 alphas (the content-cache design working exactly
  as authorized, not data loss -- see `41_join_and_summarize_defence_validation.py`
  below for how the full 3,648-row table gets reconstructed from this).

### `_defence_metrics.py` (torch-free) and the join/summarize script

`scripts/_defence_metrics.py` -- pulled the pure functions (`judge_cache_key`,
`record_key`, `load_jsonl`/`append_jsonl`, `sha256_hex`/`sha256_of_file`/
`git_commit_hash`, `compute_template_asr`/`compute_template_frr`/
`compute_macro_asr`/`compute_macro_frr`, `select_alpha`, plus the
`MODEL_PATHS`/`JUDGE_MODEL_VERSION`/`JUDGE_PROMPT_VERSION`/`VALIDATION_ALPHAS`
constants) out of `40_defence_generation_driver.py` into a standalone module
with NO dependency on torch/pipeline. `40_defence_generation_driver.py` now
imports these names from `_defence_metrics` (single source of truth,
re-exported for backward compatibility with existing call sites in that
file) instead of redefining them. Reason: importing script 40 for its pure
functions alone still pulled in `torch` (its own top-level `import torch`,
plus `35_common_direction_coverage_audit.py`'s and
`37_defence_directions_and_hooks.py`'s, both imported eagerly by script 40
at module level) -- unavailable when running with the cluster's bare system
`python3` rather than the GPU venv, discovered when the join/summarize
script below was first run on the cluster.

`scripts/41_join_and_summarize_defence_validation.py` -- joins one (model,
method)'s 3,648-row generation JSONL with its content-deduplicated
judgement JSONL back into a full 3,648-row expanded table. Imports ONLY
`_defence_metrics` (confirmed torch-free via an import-time `sys.meta_path`
blocker that makes `import torch` raise, both directly and via
`scripts/audits/audit_join_and_summarize_dry_run.py`). Recomputes
`judge_cache_key` per generation record via the canonical imported function
(never reimplemented); hard-stops (`JudgeKeyCollisionError`) if the same
key ever maps to different `(instruction_en, response)` content.
`direction_config_hash`/`generation_config_hash` are read off the
generation records themselves (already embedded per-record at generation
time) rather than recomputed -- doubles as a consistency check that all
3,648 records agree on exactly one value each. Reports missing/orphan
keys, duplicate keys, parse failures, and an `OVERALL_PASS` gate -- the
summary must not be used if `False`. Per-`(alpha, template)` stats
(ASR/compliance/`response_harmful_rate` from the harmful subset,
`benign_FRR` from the benign subset, generation length/EOS/cache stats
from the combined subset) plus per-alpha **macro** (mean of the 6
per-template rates) are kept strictly separate from per-alpha **pooled**
(all records merged first) -- both reported, never conflated. Does not
select alpha (needs all 4 methods + the No-defence benign baseline for the
same model first). 7 synthetic-data test scenarios in
`scripts/audits/audit_join_and_summarize_dry_run.py`: shared-key join
restores the full row count with correctly-copied labels (own
alpha/instruction_id preserved per record), missing-key handling,
duplicate-judgement-key detection, a forced real collision raises
immediately, macro vs pooled differ on deliberately imbalanced synthetic
data (proving no accidental mixing), parse failures excluded and counted.

### Critical bug found via real data: instruction_en was the wrapped prompt

Running `41_join_and_summarize_defence_validation.py` on the real Llama x
global data (job 4982/4984, `OVERALL_PASS: True` after the earlier fix)
showed `encoding_obfuscation`'s `asr_denom=0` across all 4 alphas (288/288
records) -- traced to `pipeline`'s `generate_completions()` setting
`instructions_en = [x['instruction'] for x in dataset]` **unconditionally**
(never reading our own `instruction_en` field). Since our prompt dicts set
`item['instruction']` = the rendered/wrapped text (correct model input) and
`item['instruction_en']` = the true plain original (intended for judging),
every returned `c['instruction_en']` was simply a duplicate of the wrapped
prompt. WildGuard was judging against the wrapped prompt for every record,
not just `encoding_obfuscation` -- that template is just where the effect
is total and unmistakable (base64 gibberish has no legible harmful content,
so `request_harmful` reads 0 every time); the other 5 templates likely have
a subtler, not-yet-quantified distortion since the harmful content usually
stays legible even wrapped.

Fixed in `40_defence_generation_driver.py` (commit `d4ed608`): both
`_build_generation_record` and `run_condition`'s per-completion
re-attachment loop now read `item['instruction_en']`, never
`c['instruction_en']`. `run_no_defence_harmful_rejudge` was unaffected --
it reads `completions_en_full572_corrected.json`, which
`03_generate_and_label.py` already re-attached `instruction_en` correctly
for. Regression test: `audit_defence_driver_chunking_dry_run.py` Test A2.

**Responses themselves are valid** (the target model received the correct
wrapped `instruction`) -- only the judging reference text was wrong, so
repair (not regeneration) is possible: `scripts/42_repair_validation_instruction_en.py`
re-derives the true plain text per record from `(instruction_id,
benign_or_harmful)` alone (harmful -> `data/sampled_prompts.json`, benign
-> `data/benign_validation_80.json`), dry-run by default, atomic write
under `--apply`, refuses to run if any instruction_id can't be resolved,
verifies all non-`instruction_en` fields stay byte-identical. Because
`judge_cache_key` depends on `instruction_en`, repairing it changes every
record's key -- **the existing judgement file for a repaired (model,
method) becomes entirely stale and must be re-judged from scratch**
(the script prints this as an explicit next step, does not do it itself).
4 tests in `scripts/audits/audit_instruction_en_repair_dry_run.py` (using
real `sampled_prompts.json`/`benign_validation_80.json` ids against a
temp output dir): dry run writes nothing, apply fixes exactly the wrong
rows, non-`instruction_en` fields stay identical, a second apply is
idempotent.

**Recovery sequence for the real Llama x global data** (not yet run):
```
python3 scripts/42_repair_validation_instruction_en.py --model_alias Meta-Llama-3.1-8B-Instruct --method global --output_path output          # dry run
python3 scripts/42_repair_validation_instruction_en.py --model_alias Meta-Llama-3.1-8B-Instruct --method global --output_path output --apply   # apply
mv output/canonical_v2/experiment3_validation_judgements_Meta-Llama-3.1-8B-Instruct_global.jsonl{,.stale_pre_instruction_en_fix}
# re-run STAGE=judge for this (model, method), then re-run 41_join_and_summarize_defence_validation.py
```
The timing-pilot outputs (all 3 models) have the same defect but are not
a reportable result (see the timing-pilot section above) -- no repair
priority unless they're needed again later.

## Conventions

- **Model alias** is always the HuggingFace-style directory name: `Qwen2.5-7B-Instruct`, `Meta-Llama-3.1-8B-Instruct`, `gemma-2-9b-it` — matches `MODEL_ALIASES` arrays in every `slurm/*.sh`.
- **Language codes**: en, zh, de (high) / ko, ar, th (medium) / yo, sw, am (low) — see `data/resource_tiers.json`.
- **`id` field** (e.g. `p000`): assigned once in `data/sampled_prompts.json` (a single list of 75 items, each holding all 9 languages' translations under `instructions`), then propagated unchanged through script 02 into every `generation_input_{lang}.json` / `completions_{lang}.json`. It **is** stable both within a language (pairing `templated` against `plain`) and **across** languages/models (the same `id` refers to the same source instruction everywhere) — cross-language joins can use `id` directly, `instruction_en` is redundant for this purpose (kept mainly for WildGuard prompting and human-readability).
