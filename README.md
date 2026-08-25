# Summary

---

## Research Overview

This pilot extends **Wang et al. (2026)**, *"Refusal Direction is Universal Across Safety-Aligned Languages"*, and complements the sibling `experiment_thesis` repository. That paper shows the *defense* representation (refusal direction, harmful vs. harmless) is shared across languages. `experiment_thesis`'s `jailbreak_vector` asks the same question of the *attack* side, but constructs it behaviorally (bypassed vs. refused activations) — a construction with a content confound, since harm categories are refused at very different rates.

This repository removes that confound by design: instead of contrasting two *different* instructions selected post hoc by model behavior, it holds the harmful instruction **fixed** and manipulates only whether — and by which of six jailbreak mechanisms — it is wrapped in an attack template. The resulting `template_direction` is purely exogenous: it does not depend on model behavior or WildGuard labels at all, only on which condition (plain vs. templated) a fixed instruction was run under.

The core question: is the representational shift a jailbreak template induces shared across languages, across mechanisms, and across model families — and is that sharing causally load-bearing, or only a geometric coincidence?

---

## Research Design

| Construct | Definition | Depends on model behavior? |
|---|---|---|
| `template_direction[lang][mech]` | mean(act(templated) − act(plain)), paired per instruction | No |
| `refusal_direction[lang]` | mean(harmful) − mean(harmless), PolyRefuse train split | No |
| `harmfulness_direction[lang]` | mean(bypassed harmful) − mean(harmless) |  |
| `placebo_direction[lang]` | mean(act(placebo-wrapped) − act(plain)), content-neutral wrapper | No |

`template_direction` and `placebo_direction` are the exogenous constructs unique to this repository; `refusal_direction`/`harmfulness_direction` are recomputed from `experiment_thesis`'s pipeline for direct geometric comparison (Section "Direction Geometry" below).

---

## Experimental Setup

### Models
| Model | Layers | Phase 1 α (calibrated) |
|---|---|---|
| Qwen2.5-7B-Instruct | 28 | 2.0 (fictional_framing only — refusal_suppression has no causal window at any α∈[0.1,2.0]) |
| Meta-Llama-3.1-8B-Instruct | 32 | 2.0 (clean on en/zh; ko/ar/yo/am need per-language recalibration) |
| gemma-2-9b-it | 42 | 1.6 (narrow window: α<1.1 too weak, α≥1.8 generation degrades) |

### Languages (9, three resource tiers)
- **High:** en, zh, de
- **Medium:** ko, ar, th
- **Low:** yo, sw, am

### Templates (6 real mechanisms + placebo control)
| Mechanism | Wei et al. (2023) category |
|---|---|
| prefix_injection | competing_objectives |
| refusal_suppression | competing_objectives |
| instruction_hierarchy | competing_objectives |
| persona_roleplay | mismatched_generalization |
| fictional_framing | mismatched_generalization |
| encoding_obfuscation | mismatched_generalization (⚠ see caveats) |
| placebo | control — content-neutral wrapper |

Templates authored in English, machine-translated to the other 8 languages, reviewed via back-translation.

### Data
75 harmful instructions, stratified by 8 coarse harm categories, sampled from the shared `ployrefuse_Enhanced` dataset. 9 languages × 8 conditions (plain + 6 mechanisms + placebo) = 600 generations/language/model.

---

## Results

### 1. Core geometry — resource-tier continuum (complete, all 3 models)

Cross-language cosine similarity of `template_direction`, averaged across the 6 real mechanisms, by resource-tier pair:

| Tier pair | Qwen | Llama | Gemma |
|---|---|---|---|
| HH | 0.606 | 0.578 | 0.798 |
| MM | 0.676 | 0.599 | 0.753 |
| LL | 0.543 | 0.375 | 0.571 |
| HM | 0.566 | 0.551 | 0.744 |
| **HL** | **0.258** | **0.286** | **0.519** |
| **LM** | **0.326** | **0.353** | **0.582** |

**Finding:** HL < LM in all 3 models (p<0.01, permutation test; robust to split-half normalization and leave-one-category-out). Resource-tier proximity predicts sharing better than a binary high/low split.

### 2. Real templates vs. placebo (complete, all 3 models)

Sign test across the 6 tier-pair comparisons: real templates > placebo in 6/6 for all 3 models (p=0.0156).

### 3. Split-half reliability ceiling (complete, all 3 models)

| Model | Ceiling |
|---|---|
| Qwen | 0.946 |
| Llama | 0.943 |
| Gemma | 0.957 |

Near-identical across models — raw similarity differences between models reflect real signal, not measurement noise.

### 4. Direction geometry vs. refusal / harmfulness (complete, all 3 models)

`cos(template_direction, refusal_direction)` weakens monotonically H→M→L; `cos(template_direction, harmfulness_direction)` flips sign from negative (high-resource) to positive (low-resource) — in all 3 models, including Gemma, which diverges from Qwen/Llama on several other measures.

| Tier | Qwen cos(ref) | Qwen cos(harm) | Llama cos(ref) | Llama cos(harm) | Gemma cos(ref) | Gemma cos(harm) |
|---|---|---|---|---|---|---|
| H | −0.265 | −0.126 | −0.222 | −0.098 | −0.303 | −0.166 |
| M | −0.205 | −0.028 | −0.167 | −0.090 | −0.247 | −0.121 |
| L | −0.166 | **+0.124** | −0.104 | **+0.082** | −0.105 | **+0.029** |

### 5. Mechanism taxonomy — geometric test (complete, all 3 models)

Wei et al.'s competing_objectives / mismatched_generalization split is **not supported** at the geometric level: full pairwise clustering of all 6 mechanisms shows `encoding_obfuscation` isolated (matches its known hallucination artifact — see caveats), `prefix_injection`–`refusal_suppression` clustering as predicted, but `instruction_hierarchy`–`persona_roleplay` (different categories) clustering just as tightly, in all 3 models.

### 6. Causal validation — activation injection (mostly complete)

Phase 0/1 (`scripts/10a`, `10b`): inject raw (non-unit-normalized) `template_direction`/`placebo_direction` at a single pre-registered middle layer, no template text shown to the model, WildGuard-relabel, compare induced bypass rate against a placebo control at matched magnitude.

**Placebo bypass rate by resource tier** (the causal counterpart to Finding 1 — refusal itself is more fragile to generic perturbation at lower resource tiers):

| Tier | Llama (α=2.0) | Gemma (α=1.6) |
|---|---|---|
| H (en/zh) | 0.025 / 0.100 | 0.025 / 0.100 |
| M (ko/ar) | 0.875 / 0.800 | 0.225 / 0.250 |
| L (yo/am) | 0.800 / 0.455 | 1.000 / 0.800 |

**Mechanism-level causal effect** (template − placebo induced bypass rate):

| Mechanism | Qwen (en) | Llama (en/zh) | Gemma (en/zh/ko/ar) |
|---|---|---|---|
| refusal_suppression | fails at every α∈[0.1,2.0] | +0.875 / +0.850 | +0.525 / +0.725 / +0.275 / +0.200 |
| fictional_framing | +0.650 | +0.725 / +0.700 | +0.925 / +0.550 / +0.550 / +0.425 |
| persona_roleplay | fails | −0.025 / −0.100 | +0.075 / +0.050 / −0.075 / −0.125 |

**`fictional_framing` is the only mechanism causally effective across all 3 models and every language tested.** `refusal_suppression` is model-specific — a genuine architectural divergence, not a calibration artifact (confirmed by exhaustive α-sweep on Qwen).

---

## Repository Structure

```
new_experiment/
├── templates/                              # templates_{lang}.json, 9 languages, 6 mechanisms + placebo
├── data/                                   # resource_tiers.json, sampled_prompts.json, generation_input_{lang}.json
├── ployrefuse_Enhanced/                    # local copy of the 16-language PolyRefuse dataset
├── scripts/
│   ├── 00_translate_templates.py           # template translation + back-translation QC
│   ├── 01_sample_prompts.py                # category-stratified 75-prompt sample
│   ├── 02_build_templated_data.py          # builds the 8-condition generation input per language
│   ├── 03_generate_and_label.py            # generation + WildGuard labeling
│   ├── 04_extract_directions_and_analyze.py# core template_direction extraction + cosine analysis
│   ├── 05_audit_encoding_obfuscation.py    # genuine-decode audit
│   ├── 06_split_half_reliability.py        # noise-floor calibration
│   ├── 07_leave_one_category_out.py        # robustness to harm-category composition
│   ├── 08_magnitude_vs_behavior.py         # ||template_direction|| vs ΔASR correlation
│   ├── 09_refusal_geometry.py              # template_direction vs refusal/harmfulness_direction
│   ├── 10a_calibrate_injection_alpha.py    # Phase 0: alpha calibration with placebo comparison
│   ├── 10b_phase1_injection_experiment.py  # Phase 1: causal injection test, 6 langs x 3 mechs
│   ├── 11_translation_surface_similarity.py# translation-confound check (no model needed)
│   ├── 12_layerwise_profile.py             # per-layer same_language_cross_mechanism curves
│   └── 13_layerwise_cross_language.py      # per-layer same_mechanism_cross_language curves
├── slurm/                                  # SLURM job scripts, MODEL_IDX=0/1/2 pattern
└── output/{model_alias}/                   # completions, pilot_results.json, calibration/phase1 results
```
