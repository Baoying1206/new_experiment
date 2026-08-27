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

## Conventions

- **Model alias** is always the HuggingFace-style directory name: `Qwen2.5-7B-Instruct`, `Meta-Llama-3.1-8B-Instruct`, `gemma-2-9b-it` — matches `MODEL_ALIASES` arrays in every `slurm/*.sh`.
- **Language codes**: en, zh, de (high) / ko, ar, th (medium) / yo, sw, am (low) — see `data/resource_tiers.json`.
- **`id` field** (e.g. `p000`): assigned once in `data/sampled_prompts.json` (a single list of 75 items, each holding all 9 languages' translations under `instructions`), then propagated unchanged through script 02 into every `generation_input_{lang}.json` / `completions_{lang}.json`. It **is** stable both within a language (pairing `templated` against `plain`) and **across** languages/models (the same `id` refers to the same source instruction everywhere) — cross-language joins can use `id` directly, `instruction_en` is redundant for this purpose (kept mainly for WildGuard prompting and human-readability).
