"""
Defines and persists a leak-free three-way split of the 572-instruction pool
(data/sampled_prompts.json), plus a smaller cross-lingual subset drawn as a
strict subset of each split -- so no id ever crosses from e.g. the English
test set into the cross-lingual direction set.

  direction_ids   (300): used to construct template_direction / refusal_direction /
                          harmfulness_direction (mean-of-differences).
  validation_ids   (72): used for layer/alpha selection (Phase 0 calibration,
                          safety-layer identification) -- never touches the
                          numbers reported as "the result."
  test_ids        (200): held out for behavioral evaluation, causal injection,
                          and the taxonomy-robustness diagnostics reported in
                          the thesis.

  cross_lingual_direction_ids   (100, subset of direction_ids)
  cross_lingual_validation_ids   (30, subset of validation_ids)
  cross_lingual_test_ids         (70, subset of test_ids)
  cross_lingual_ids             (200, union of the three above)

English uses the full 572-instruction pipeline (all three splits, full size).
The other 8 pilot languages use only cross_lingual_ids (200 total) --
built by 02_build_templated_data.py's --ids_key option.

No model/GPU needed -- pure data wrangling, safe to run locally.

Usage:
  python scripts/01b_define_splits.py
"""
import json
import os
import random

SCRIPT_DIR = os.path.dirname(__file__)
SAMPLED_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'sampled_prompts.json')
OUT_PATH = os.path.join(SCRIPT_DIR, '..', 'data', 'splits.json')

SEED = 0
N_DIRECTION, N_VALIDATION, N_TEST = 300, 72, 200
N_XLING_DIRECTION, N_XLING_VALIDATION, N_XLING_TEST = 100, 30, 70


def main():
    with open(SAMPLED_PATH, encoding='utf-8') as f:
        sampled = json.load(f)
    all_ids = [item['id'] for item in sampled]
    assert len(all_ids) == N_DIRECTION + N_VALIDATION + N_TEST, (
        f"Expected {N_DIRECTION + N_VALIDATION + N_TEST} ids, got {len(all_ids)} -- "
        f"rerun 01_sample_prompts.py or adjust the split sizes above.")

    rng = random.Random(SEED)
    shuffled = all_ids.copy()
    rng.shuffle(shuffled)

    direction_ids = sorted(shuffled[:N_DIRECTION])
    validation_ids = sorted(shuffled[N_DIRECTION:N_DIRECTION + N_VALIDATION])
    test_ids = sorted(shuffled[N_DIRECTION + N_VALIDATION:])

    xling_direction = sorted(rng.sample(direction_ids, N_XLING_DIRECTION))
    xling_validation = sorted(rng.sample(validation_ids, N_XLING_VALIDATION))
    xling_test = sorted(rng.sample(test_ids, N_XLING_TEST))
    cross_lingual_ids = sorted(xling_direction + xling_validation + xling_test)

    # Sanity: no leakage across splits, cross-lingual ids are true subsets.
    assert set(direction_ids) & set(validation_ids) == set()
    assert set(direction_ids) & set(test_ids) == set()
    assert set(validation_ids) & set(test_ids) == set()
    assert set(xling_direction) <= set(direction_ids)
    assert set(xling_validation) <= set(validation_ids)
    assert set(xling_test) <= set(test_ids)

    out = {
        'seed': SEED,
        'direction_ids': direction_ids, 'validation_ids': validation_ids, 'test_ids': test_ids,
        'cross_lingual_direction_ids': xling_direction,
        'cross_lingual_validation_ids': xling_validation,
        'cross_lingual_test_ids': xling_test,
        'cross_lingual_ids': cross_lingual_ids,
    }
    with open(OUT_PATH, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"direction={len(direction_ids)}  validation={len(validation_ids)}  test={len(test_ids)}")
    print(f"cross_lingual: direction={len(xling_direction)}  validation={len(xling_validation)}  "
          f"test={len(xling_test)}  total={len(cross_lingual_ids)}")
    print(f"Saved: {OUT_PATH}")


if __name__ == '__main__':
    main()
