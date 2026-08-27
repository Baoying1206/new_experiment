# Axis Source-Overlap Audit

**Critical finding: English harmful_train / harmless_train / harmless_val are missing from the local ployrefuse_Enhanced mirror.** All 15 other languages have train+val data locally; only English is missing it (and only English has harmless_test). This blocks direct verification of Decision 3 for English.

## sampled_prompts.json provenance
- `data/sampled_prompts.json` (572 items) == `ployrefuse_Enhanced/harmful_test_translated_en.json` (572 items, 562 unique) by exact normalised-text-set equality: True, overlap=562/572.
- Consequence: `data/splits.json` (direction_ids/validation_ids/test_ids, all drawn from sampled_prompts.json) is entirely sourced from PolyRefuse's English **harmful_test** split.

## Per-language train/val vs test overlap (normalised instruction text)

| lang | harmful_train∩test | harmful_val∩test | harmful_train∩val | provenance_status |
|---|---|---|---|---|
| en | None | None | None | unknown_missing_local_data |
| zh | 0 | 0 | 0 | checked_locally |
| ar | 0 | 0 | 0 | checked_locally |
| th | 0 | 0 | 0 | checked_locally |
| yo | 0 | 0 | 0 | checked_locally |
| am | 0 | 0 | 0 | checked_locally |
| de | 0 | 0 | 0 | checked_locally |
| ko | 0 | 0 | 0 | checked_locally |
| sw | 0 | 0 | 0 | checked_locally |
| es | 0 | 0 | 0 | checked_locally |
| fr | 0 | 0 | 0 | checked_locally |
| it | 0 | 0 | 0 | checked_locally |
| ja | 0 | 0 | 0 | checked_locally |
| nl | 0 | 0 | 0 | checked_locally |
| pl | 0 | 0 | 0 | checked_locally |
| ru | 0 | 0 | 0 | checked_locally |

All 15 non-English languages checkable locally show exactly 0 overlap on every pairwise comparison -- indirect evidence for train/val/test disjointness by design, but NOT a direct check of English data (English row above will show `unknown_missing_local_data`).

## Recommended next step for Decision 3

Locate English harmful_train_translated_en.json / harmless_train_translated_en.json / harmless_val_translated_en.json on the cluster (likely reachable via the _orig dataset.load_dataset path referenced in extract_jailbreak_vectors.py::load_dataset_split), and run this same normalised-text overlap check directly against ployrefuse_Enhanced/harmful_test_translated_en.json (== data/sampled_prompts.json). Until this is done, do not assume the independent-train-split design is safe for English.

**Fallback if English train data cannot be located/verified on the cluster:**
5-fold cross-fitting on data/splits.json direction_ids (300 English ids): split into 5 folds, build refusal_direction/harmfulness_direction from 4 folds at a time, compute out-of-fold delta_R/delta_H on the held-out fold only, repeat 5x and merge. This has zero dependency on locating/verifying an independent English train split, at the cost of the axis directions being built from slightly less data per fold (240 vs 300 instructions) and requiring 5x the direction-extraction compute.

## Note on ID-based overlap
Local PolyRefuse files carry no native per-item ID field (only `instruction` and `category`) -- overlap-by-ID as originally specified is not computable from local data; all checks above are by normalised instruction text. If the cluster-side `dataset.load_dataset` loader exposes native IDs, re-run this check there for a stronger guarantee.
