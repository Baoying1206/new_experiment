"""
CPU-only completeness/leakage validation for the 6-language confirmatory
scope (English full-572 + 5-language cross-lingual 200-subset), run BEFORE
any GPU generation. Reads only already-built data/*.json files -- writes no
model output, submits nothing.

Checks (see conversation for the full spec this implements):
  1. English full-scale file has 4,576 rows.
  2. Each confirmatory non-English file has 1,600 rows.
  3. Every source id has exactly 8 conditions, in both the English file and
     each confirmatory xling file.
  4. No duplicate (id, language, condition) rows.
  5. All 5 confirmatory xling languages + English's own xling-subset view
     share exactly the same 200 ids.
  6. cross_lingual_direction/validation/test (100/30/70) are pairwise disjoint.
  7. Each is a strict subset of English's own direction/validation/test (300/72/200).
  8. Harm-category distribution in the 200-id cross-lingual subset stays
     roughly proportional to the full 572-id pool (reported, not asserted --
     stratified sampling with small strata can't guarantee exact proportionality).
  9. No test id appears in direction or validation (both English and xling).
  10. Total confirmatory model-input combination count == 37,728.

Usage:
  python scripts/20_validate_six_language_split.py
"""
import json
import os
from collections import Counter

SCRIPT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
OUT_DIR = os.path.join(SCRIPT_DIR, '..', 'output_572_split_v1')

from _lang_config import CONFIRMATORY_XLING_LANGUAGES  # noqa: E402

EXPECTED_EN_ROWS = 4576
EXPECTED_XLING_ROWS = 1600
N_MODELS = 3


def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def check_row_count(rows, expected, label, findings):
    ok = len(rows) == expected
    findings.append({'check': f'row_count[{label}]', 'pass': ok,
                      'expected': expected, 'actual': len(rows)})
    return ok


def check_conditions_per_id(rows, label, findings):
    by_id = Counter(r['id'] for r in rows)
    n_ids = len(by_id)
    bad = {pid: n for pid, n in by_id.items() if n != 8}
    ok = len(bad) == 0
    findings.append({'check': f'conditions_per_id[{label}]', 'pass': ok,
                      'n_unique_ids': n_ids, 'n_ids_with_wrong_count': len(bad),
                      'examples': dict(list(bad.items())[:5])})
    return ok


def check_no_duplicates(rows, lang, findings):
    keys = [(r['id'], lang, r['condition']) for r in rows]
    dupes = [k for k, n in Counter(keys).items() if n > 1]
    ok = len(dupes) == 0
    findings.append({'check': f'no_duplicate_id_lang_condition[{lang}]', 'pass': ok,
                      'n_duplicates': len(dupes), 'examples': dupes[:5]})
    return ok


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    findings = []
    all_pass = True

    def record(ok):
        nonlocal all_pass
        all_pass = all_pass and ok

    # ── 1/2: row counts ──────────────────────────────────────────────────
    en_rows = load(os.path.join(DATA_DIR, 'generation_input_en_full572.json'))
    record(check_row_count(en_rows, EXPECTED_EN_ROWS, 'en_full572', findings))

    xling_rows_by_lang = {}
    for lang in CONFIRMATORY_XLING_LANGUAGES:
        rows = load(os.path.join(DATA_DIR, f'generation_input_{lang}_xling.json'))
        xling_rows_by_lang[lang] = rows
        record(check_row_count(rows, EXPECTED_XLING_ROWS, f'{lang}_xling', findings))

    # ── 3: conditions per id ────────────────────────────────────────────
    record(check_conditions_per_id(en_rows, 'en_full572', findings))
    for lang, rows in xling_rows_by_lang.items():
        record(check_conditions_per_id(rows, f'{lang}_xling', findings))

    # ── 4: no duplicate (id, lang, condition) ──────────────────────────
    record(check_no_duplicates(en_rows, 'en', findings))
    for lang, rows in xling_rows_by_lang.items():
        record(check_no_duplicates(rows, lang, findings))

    # ── 5: all confirmatory xling languages share exactly the same 200 ids ─
    id_sets = {lang: {r['id'] for r in rows} for lang, rows in xling_rows_by_lang.items()}
    reference = next(iter(id_sets.values()))
    mismatches = {lang: sorted(ids ^ reference) for lang, ids in id_sets.items() if ids != reference}
    ok = len(mismatches) == 0 and len(reference) == 200
    findings.append({'check': 'six_languages_share_same_200_ids', 'pass': ok,
                      'reference_size': len(reference), 'mismatches': mismatches})
    record(ok)

    # ── splits.json-derived checks (6, 7, 9) ───────────────────────────
    splits = load(os.path.join(DATA_DIR, 'splits.json'))
    direction_ids = set(splits['direction_ids'])
    validation_ids = set(splits['validation_ids'])
    test_ids = set(splits['test_ids'])
    xling_direction = set(splits['cross_lingual_direction_ids'])
    xling_validation = set(splits['cross_lingual_validation_ids'])
    xling_test = set(splits['cross_lingual_test_ids'])
    cross_lingual_ids = set(splits['cross_lingual_ids'])

    ok = (xling_direction & xling_validation == set()
          and xling_direction & xling_test == set()
          and xling_validation & xling_test == set())
    findings.append({'check': 'xling_100_30_70_pairwise_disjoint', 'pass': ok,
                      'sizes': {'direction': len(xling_direction), 'validation': len(xling_validation),
                                'test': len(xling_test)}})
    record(ok)

    ok = (xling_direction <= direction_ids and xling_validation <= validation_ids
          and xling_test <= test_ids)
    findings.append({'check': 'xling_subsets_of_english_splits', 'pass': ok,
                      'xling_direction_subset_of_english_direction': xling_direction <= direction_ids,
                      'xling_validation_subset_of_english_validation': xling_validation <= validation_ids,
                      'xling_test_subset_of_english_test': xling_test <= test_ids})
    record(ok)

    ok = test_ids.isdisjoint(direction_ids) and test_ids.isdisjoint(validation_ids)
    findings.append({'check': 'no_test_id_in_direction_or_validation', 'pass': ok})
    record(ok)

    # cross-check: the 200 ids actually used in the xling generation_input files
    # match splits.json's cross_lingual_ids exactly
    ok = reference == cross_lingual_ids
    findings.append({'check': 'xling_files_match_splits_json_cross_lingual_ids', 'pass': ok,
                      'splits_json_size': len(cross_lingual_ids), 'generation_input_size': len(reference)})
    record(ok)

    # ── 8: harm category proportionality (reported, not strict pass/fail) ─
    sampled = load(os.path.join(DATA_DIR, 'sampled_prompts.json'))
    cat_by_id = {item['id']: item['category'] for item in sampled}
    full_cats = Counter(cat_by_id[item['id']] for item in sampled)
    xling_cats = Counter(cat_by_id[pid] for pid in cross_lingual_ids)
    full_total, xling_total = sum(full_cats.values()), sum(xling_cats.values())
    proportion_check = {
        cat: {'full_pct': round(100 * n / full_total, 1),
              'xling_pct': round(100 * xling_cats.get(cat, 0) / xling_total, 1)}
        for cat, n in full_cats.items()
    }
    findings.append({'check': 'harm_category_proportionality (informational, not pass/fail)',
                      'pass': None, 'detail': proportion_check})

    # ── 10: total confirmatory task count ──────────────────────────────
    expected_total = EXPECTED_EN_ROWS * N_MODELS + len(CONFIRMATORY_XLING_LANGUAGES) * EXPECTED_XLING_ROWS * N_MODELS
    actual_total = len(en_rows) * N_MODELS + sum(len(r) for r in xling_rows_by_lang.values()) * N_MODELS
    ok = actual_total == expected_total == 37728
    findings.append({'check': 'total_confirmatory_task_count', 'pass': ok,
                      'expected': 37728, 'computed_from_formula': expected_total,
                      'computed_from_actual_files': actual_total})
    record(ok)

    # ── Report ───────────────────────────────────────────────────────────
    report = {'all_checks_passed': all_pass, 'n_checks': len(findings), 'findings': findings}
    out_path = os.path.join(OUT_DIR, 'six_language_validation.json')
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"{'PASS' if all_pass else 'FAIL'} -- {len(findings)} checks run")
    for item in findings:
        status = 'PASS' if item['pass'] else ('INFO' if item['pass'] is None else 'FAIL')
        print(f"  [{status}] {item['check']}")
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
