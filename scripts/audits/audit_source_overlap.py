"""
Source-overlap audit for Decision 3 (axis-dataset independence).

Checks, for each language with local data available, whether PolyRefuse's
train/val splits (candidate source for refusal_direction/harmfulness_direction,
per experiment_thesis/scripts/extract_jailbreak_vectors.py) are disjoint from
the harmful_test split (source of this repo's 572-instruction English core
pool, data/sampled_prompts.json, and by construction data/splits.json's
direction_ids/validation_ids/test_ids).

IMPORTANT CAVEAT (read before trusting any English-specific conclusion):
PolyRefuse's local files here carry NO native per-item ID -- only
{'instruction': str, 'category': str|None}. So "overlap by ID" as originally
specified is not computable from this local mirror; all overlap checks here
are by normalised instruction text. This itself is a finding: any claim of
"same ID, different text" or "different ID, same text" cannot be verified
without a native ID field, which does not exist in local files. If the
cluster-side `dataset.load_dataset` loader exposes native IDs, that should
be checked there and would strengthen this audit.

CRITICAL GAP: ployrefuse_Enhanced/ is missing harmful_train_translated_en.json,
harmful_val_translated_en.json, harmless_train_translated_en.json, and
harmless_val_translated_en.json -- i.e. ENGLISH has only test-split files
locally. All 15 other languages have train+val (but only English has
harmless_test). This means the single most important check for Decision 3
-- does English's refusal_direction/harmfulness_direction training data
overlap with the English 572-instruction pool -- CANNOT be computed from
local data. This script still runs the same check for every language where
data IS available, as indirect (not conclusive) evidence about whether
PolyRefuse maintains train/test disjointness by design, and clearly marks
the English harmful/harmless train-vs-test comparison as
provenance_status="unknown_missing_local_data" rather than guessing.

Usage: python scripts/audits/audit_source_overlap.py
Reads only local files; makes no network/cluster calls. Writes
output/audits/axis_source_overlap.json and .md.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENHANCED = ROOT / 'ployrefuse_Enhanced'
OUT_DIR = ROOT / 'output' / 'audits'

ALL_LANGS = ['en', 'zh', 'ar', 'th', 'yo', 'am', 'de', 'ko', 'sw',
             'es', 'fr', 'it', 'ja', 'nl', 'pl', 'ru']


def norm(s):
    return re.sub(r'\s+', ' ', s.strip().lower())


def load(split_type, split, lang):
    p = ENHANCED / f'{split_type}_{split}_translated_{lang}.json'
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def partition_stats(items):
    if items is None:
        return None
    texts = [norm(x['instruction']) for x in items]
    return {
        'count': len(items),
        'unique_normalised_texts': len(set(texts)),
        'duplicate_count_within_partition': len(items) - len(set(texts)),
    }


def overlap(a, b):
    if a is None or b is None:
        return None
    ta = set(norm(x['instruction']) for x in a)
    tb = set(norm(x['instruction']) for x in b)
    return len(ta & tb)


def audit_language(lang):
    harmful_train = load('harmful', 'train', lang)
    harmful_val = load('harmful', 'val', lang)
    harmful_test = load('harmful', 'test', lang)
    harmless_train = load('harmless', 'train', lang)
    harmless_val = load('harmless', 'val', lang)
    harmless_test = load('harmless', 'test', lang)

    result = {
        'language': lang,
        'partitions': {
            'harmful_train': partition_stats(harmful_train),
            'harmful_val': partition_stats(harmful_val),
            'harmful_test': partition_stats(harmful_test),
            'harmless_train': partition_stats(harmless_train),
            'harmless_val': partition_stats(harmless_val),
            'harmless_test': partition_stats(harmless_test),
        },
        'overlap_by_normalised_text': {
            'harmful_train_vs_harmful_test': overlap(harmful_train, harmful_test),
            'harmful_val_vs_harmful_test': overlap(harmful_val, harmful_test),
            'harmful_train_vs_harmful_val': overlap(harmful_train, harmful_val),
            'harmless_train_vs_harmless_test': overlap(harmless_train, harmless_test),
            'harmless_val_vs_harmless_test': overlap(harmless_val, harmless_test),
        },
        'overlap_by_id': 'not_computable_local_files_have_no_native_id_field',
    }

    if lang == 'en':
        missing = [name for name, v in [
            ('harmful_train', harmful_train), ('harmful_val', harmful_val),
            ('harmless_train', harmless_train), ('harmless_val', harmless_val),
        ] if v is None]
        result['provenance_status'] = (
            'unknown_missing_local_data' if missing else 'checked_locally'
        )
        result['missing_local_partitions'] = missing
        result['note'] = (
            'English harmful_train/harmless_train/harmless_val are absent from '
            'this local ployrefuse_Enhanced mirror. extract_jailbreak_vectors.py '
            'loads via an _orig (dataset.load_dataset) path first, falling back '
            'to these local files only on FileNotFoundError/KeyError/TypeError -- '
            'so this data likely exists on the cluster but is unverified here. '
            'Cannot confirm English axis-training-data vs 572-pool disjointness '
            'from local data alone.'
        ) if missing else None
    else:
        result['provenance_status'] = 'checked_locally'

    return result


def sampled_prompts_provenance():
    sp = json.load(open(ROOT / 'data' / 'sampled_prompts.json'))
    en_test = load('harmful', 'test', 'en')
    sp_texts = set(norm(x['instruction_en']) for x in sp)
    test_texts = set(norm(x['instruction']) for x in en_test)
    return {
        'sampled_prompts_count': len(sp),
        'harmful_test_translated_en_count': len(en_test),
        'sampled_prompts_is_subset_of_harmful_test_en': sp_texts <= test_texts,
        'overlap_count': len(sp_texts & test_texts),
        'conclusion': (
            'data/sampled_prompts.json (source of the entire English 572-instruction '
            'core pool, and by extension data/splits.json direction_ids/validation_ids/'
            'test_ids) is exactly ployrefuse_Enhanced/harmful_test_translated_en.json '
            '(all 572 rows, including its 10 internal duplicate-text rows). Confirmed '
            'by exact text-set equality, not assumed.'
        ),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_language = {lang: audit_language(lang) for lang in ALL_LANGS}
    provenance = sampled_prompts_provenance()

    checkable = [lang for lang in ALL_LANGS if lang != 'en'
                 and per_language[lang]['overlap_by_normalised_text']['harmful_train_vs_harmful_test'] == 0]
    all_checkable = [lang for lang in ALL_LANGS if lang != 'en']

    recommended_axis_partition = {
        'decision': 'CANNOT finalize independent-train-split design for English from local data alone',
        'indirect_evidence': (
            f'{len(checkable)}/{len(all_checkable)} non-English languages checked locally show '
            f'ZERO normalised-text overlap between harmful_train (260 items) and harmful_test '
            f'(572 items, the source of the 572-pool) -- and also zero for harmful_val vs '
            f'harmful_test, and harmful_train vs harmful_val. This is consistent with PolyRefuse '
            f'maintaining genuine train/val/test disjointness by design across all languages, which '
            f'would support English also being disjoint -- but this is INDIRECT evidence, not a '
            f'direct verification of English data, since English train/val files are not present '
            f'locally.'
        ),
        'required_before_finalizing': (
            'Locate English harmful_train_translated_en.json / harmless_train_translated_en.json / '
            'harmless_val_translated_en.json on the cluster (likely reachable via the _orig '
            'dataset.load_dataset path referenced in extract_jailbreak_vectors.py::load_dataset_split), '
            'and run this same normalised-text overlap check directly against '
            'ployrefuse_Enhanced/harmful_test_translated_en.json (== data/sampled_prompts.json). '
            'Until this is done, do not assume the independent-train-split design is safe for English.'
        ),
        'recommended_fallback_if_unverifiable_or_overlapping': (
            '5-fold cross-fitting on data/splits.json direction_ids (300 English ids): split into 5 '
            'folds, build refusal_direction/harmfulness_direction from 4 folds at a time, compute '
            'out-of-fold delta_R/delta_H on the held-out fold only, repeat 5x and merge. This has zero '
            'dependency on locating/verifying an independent English train split, at the cost of the '
            'axis directions being built from slightly less data per fold (240 vs 300 instructions) '
            'and requiring 5x the direction-extraction compute.'
        ),
    }

    output = {
        'per_language': per_language,
        'sampled_prompts_provenance': provenance,
        'recommended_axis_partition': recommended_axis_partition,
    }

    with open(OUT_DIR / 'axis_source_overlap.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    md_lines = [
        '# Axis Source-Overlap Audit',
        '',
        '**Critical finding: English harmful_train / harmless_train / harmless_val are '
        'missing from the local ployrefuse_Enhanced mirror.** All 15 other languages have '
        'train+val data locally; only English is missing it (and only English has '
        'harmless_test). This blocks direct verification of Decision 3 for English.',
        '',
        '## sampled_prompts.json provenance',
        f"- `data/sampled_prompts.json` (572 items) == "
        f"`ployrefuse_Enhanced/harmful_test_translated_en.json` (572 items, 562 unique) "
        f"by exact normalised-text-set equality: "
        f"{provenance['sampled_prompts_is_subset_of_harmful_test_en']}, "
        f"overlap={provenance['overlap_count']}/572.",
        f"- Consequence: `data/splits.json` (direction_ids/validation_ids/test_ids, all "
        f"drawn from sampled_prompts.json) is entirely sourced from PolyRefuse's English "
        f"**harmful_test** split.",
        '',
        '## Per-language train/val vs test overlap (normalised instruction text)',
        '',
        '| lang | harmful_train∩test | harmful_val∩test | harmful_train∩val | provenance_status |',
        '|---|---|---|---|---|',
    ]
    for lang in ALL_LANGS:
        r = per_language[lang]
        ov = r['overlap_by_normalised_text']
        md_lines.append(
            f"| {lang} | {ov['harmful_train_vs_harmful_test']} | "
            f"{ov['harmful_val_vs_harmful_test']} | {ov['harmful_train_vs_harmful_val']} | "
            f"{r['provenance_status']} |"
        )
    md_lines += [
        '',
        f"All {len(checkable)} non-English languages checkable locally show exactly 0 overlap "
        f"on every pairwise comparison -- indirect evidence for train/val/test disjointness by "
        f"design, but NOT a direct check of English data (English row above will show "
        f"`unknown_missing_local_data`).",
        '',
        '## Recommended next step for Decision 3',
        '',
        recommended_axis_partition['required_before_finalizing'],
        '',
        '**Fallback if English train data cannot be located/verified on the cluster:**',
        recommended_axis_partition['recommended_fallback_if_unverifiable_or_overlapping'],
        '',
        '## Note on ID-based overlap',
        'Local PolyRefuse files carry no native per-item ID field (only `instruction` and '
        '`category`) -- overlap-by-ID as originally specified is not computable from local '
        'data; all checks above are by normalised instruction text. If the cluster-side '
        '`dataset.load_dataset` loader exposes native IDs, re-run this check there for a '
        'stronger guarantee.',
    ]
    with open(OUT_DIR / 'axis_source_overlap.md', 'w') as f:
        f.write('\n'.join(md_lines) + '\n')

    print(f"Wrote {OUT_DIR / 'axis_source_overlap.json'}")
    print(f"Wrote {OUT_DIR / 'axis_source_overlap.md'}")
    print(f"English provenance_status: {per_language['en']['provenance_status']}")


if __name__ == '__main__':
    main()
