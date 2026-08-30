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

## Conventions

- **Model alias** is always the HuggingFace-style directory name: `Qwen2.5-7B-Instruct`, `Meta-Llama-3.1-8B-Instruct`, `gemma-2-9b-it` — matches `MODEL_ALIASES` arrays in every `slurm/*.sh`.
- **Language codes**: en, zh, de (high) / ko, ar, th (medium) / yo, sw, am (low) — see `data/resource_tiers.json`.
- **`id` field** (e.g. `p000`): assigned once in `data/sampled_prompts.json` (a single list of 75 items, each holding all 9 languages' translations under `instructions`), then propagated unchanged through script 02 into every `generation_input_{lang}.json` / `completions_{lang}.json`. It **is** stable both within a language (pairing `templated` against `plain`) and **across** languages/models (the same `id` refers to the same source instruction everywhere) — cross-language joins can use `id` directly, `instruction_en` is redundant for this purpose (kept mainly for WildGuard prompting and human-readability).
