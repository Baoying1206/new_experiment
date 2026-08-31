"""
Synthetic-data tests for scripts/41_join_and_summarize_defence_validation.py's
core join/audit/summary logic. No GPU, no real generation/judgement files --
constructs small in-memory generation and judgement lists directly to exercise
edge cases production-scale data wouldn't conveniently isolate.

Usage:
  python scripts/audits/audit_join_and_summarize_dry_run.py
"""
import os
import sys
from importlib import import_module

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
join_mod = import_module('41_join_and_summarize_defence_validation')
drv = import_module('40_defence_generation_driver')


def make_gen_row(instruction_id, template, alpha, benign_or_harmful, instruction_en, response,
                  generation_length=10, stop_reason='eos', record_key=None):
    return {
        'record_key': record_key or f"{instruction_id}|{template}|{alpha}|{benign_or_harmful}",
        'model': 'Meta-Llama-3.1-8B-Instruct', 'split': 'validation',
        'instruction_id': instruction_id, 'benign_or_harmful': benign_or_harmful,
        'template': template, 'method': 'global', 'alpha': alpha,
        'instruction_en': instruction_en, 'response': response,
        'generation_tokens': '1 2 3', 'generation_length': generation_length, 'stop_reason': stop_reason,
    }


def make_judgement(instruction_en, response, request_harmful, refusal, response_harmful, parse_success=True):
    key = drv.judge_cache_key(instruction_en, response)
    return {
        'judge_cache_key': key, 'raw_judge_output': 'fake',
        'request_harmful': request_harmful, 'refusal': refusal, 'response_harmful': response_harmful,
        'parse_success': parse_success, 'parse_error_reason': None if parse_success else 'fake error',
    }


def main():
    # ---- Test 1: multiple generation records sharing one judge key -> join
    # restores the original generation row count, cache labels correctly
    # copied to ALL matching records ----
    g1 = make_gen_row('p1', 'global', 0.25, 'harmful', 'instrA', 'SAME RESPONSE')
    g2 = make_gen_row('p2', 'global', 0.5, 'harmful', 'instrA', 'SAME RESPONSE')  # same instr+response -> same key
    g3 = make_gen_row('p3', 'global', 1.0, 'harmful', 'instrB', 'different response')
    gen_rows = [g1, g2, g3]
    j1 = make_judgement('instrA', 'SAME RESPONSE', request_harmful=1, refusal=0, response_harmful=1)
    j2 = make_judgement('instrB', 'different response', request_harmful=1, refusal=1, response_harmful=0)
    judge_rows = [j1, j2]  # only 2 unique judgements for 3 generation rows

    key_map = join_mod.check_no_key_collisions(gen_rows)
    assert len(key_map) == 2, "expected 2 unique (instruction_en,response) pairs among 3 generation rows"
    judge_lookup = {j['judge_cache_key']: j for j in judge_rows}
    joined, missing = join_mod.join_generation_and_judgement(gen_rows, judge_lookup)
    assert len(joined) == 3, f"join must restore original generation row count, got {len(joined)}"
    assert len(missing) == 0
    # both p1 and p2 (sharing the same response) must get IDENTICAL judge labels
    j_by_id = {r['instruction_id']: r for r in joined}
    assert j_by_id['p1']['refusal'] == j_by_id['p2']['refusal'] == 0
    assert j_by_id['p1']['response_harmful'] == j_by_id['p2']['response_harmful'] == 1
    assert j_by_id['p1']['judge_cache_key'] == j_by_id['p2']['judge_cache_key']
    # but each keeps its OWN alpha/instruction_id (not overwritten by the shared judgement)
    assert j_by_id['p1']['alpha'] == 0.25 and j_by_id['p2']['alpha'] == 0.5
    print("Test 1 PASSED: shared judge key -> join restores full row count, cache labels copied "
          "identically to all matching records, own alpha/instruction_id preserved per record.")

    # ---- Test 2: missing key must be reported (not silently dropped/crashed) ----
    judge_rows_missing = [j1]  # j2 (for instrB) is missing
    judge_lookup_missing = {j['judge_cache_key']: j for j in judge_rows_missing}
    joined2, missing2 = join_mod.join_generation_and_judgement(gen_rows, judge_lookup_missing)
    assert len(joined2) == 2, "rows with a missing judgement must be excluded from the joined table"
    assert len(missing2) == 1
    print(f"Test 2 PASSED: missing key correctly detected and excluded from join, reported separately: "
          f"{len(missing2)} missing.")

    # ---- Test 3: duplicate judgement key detected by preflight_judgement_checks ----
    dup_judge_rows = [j1, j1]  # exact duplicate key
    jchecks, n_pf = join_mod.preflight_judgement_checks(dup_judge_rows)
    assert not jchecks['no_duplicate_judge_key']['pass']
    assert jchecks['no_duplicate_judge_key']['n_duplicates'] == 1
    print("Test 3 PASSED: duplicate judgement key correctly flagged as a failing check.")

    # ---- Test 4: same key, different content -> hard collision error ----
    g_collide_a = make_gen_row('pX', 'global', 0.25, 'harmful', 'same_instr', 'response ONE')
    g_collide_b = make_gen_row('pY', 'global', 0.5, 'harmful', 'same_instr', 'response ONE')  # identical -> fine
    # Force an artificial collision by monkeypatching judge_cache_key is overkill; instead directly
    # verify check_no_key_collisions raises when we hand it two DIFFERENT contents under a
    # deliberately-collided key by constructing rows whose real (instr,response) truly differ
    # but re-using the same generation dict structure is not enough to force a real SHA-256
    # collision -- so we test the DETECTION LOGIC itself by calling the checker on rows we
    # know share a key only because they're byte-identical, then assert the *absence* of a
    # false positive, and separately unit-test the collision branch directly.
    g_ok = [g_collide_a, g_collide_b]
    join_mod.check_no_key_collisions(g_ok)  # must NOT raise (content is identical, not a collision)
    print("Test 4a PASSED: identical content sharing a key is correctly NOT flagged as a collision.")

    class FakeRow(dict):
        pass

    # Directly exercise the collision branch's logic with two rows whose recomputed keys
    # are forced to collide by monkeypatching judge_cache_key for this call only.
    orig_key_fn = drv.judge_cache_key
    try:
        drv.judge_cache_key = lambda instr, resp: 'FORCED_SAME_KEY'
        colliding_rows = [
            make_gen_row('pA', 'global', 0.25, 'harmful', 'instr1', 'response1'),
            make_gen_row('pB', 'global', 0.5, 'harmful', 'instr2', 'response2'),  # different content, same forced key
        ]
        try:
            join_mod.check_no_key_collisions(colliding_rows)
            raise SystemExit("FAILED: expected JudgeKeyCollisionError, none raised")
        except join_mod.JudgeKeyCollisionError as e:
            print(f"Test 4b PASSED: forced same-key/different-content correctly raises JudgeKeyCollisionError: {str(e)[:100]}")
    finally:
        drv.judge_cache_key = orig_key_fn

    # ---- Test 5: macro vs pooled differ on imbalanced synthetic data ----
    # Template A: 1 harmful record, ASR=1.0 (bypassed). Template B: 9 harmful records, ASR=0.0 (all refused).
    # macro_asr = mean(1.0, 0.0) = 0.5
    # pooled_asr = 1 bypass / 10 total = 0.1  -- these must differ, proving no accidental mixing.
    rows_imbalanced = []
    rows_imbalanced.append(make_gen_row('h1', 'templateA', 0.5, 'harmful', 'i1', 'bypassed response'))
    for i in range(9):
        rows_imbalanced.append(make_gen_row(f'h2_{i}', 'templateB', 0.5, 'harmful', f'i2_{i}', f'refused response {i}'))
    judge_imbalanced = [make_judgement('i1', 'bypassed response', 1, 0, 1)]
    for i in range(9):
        judge_imbalanced.append(make_judgement(f'i2_{i}', f'refused response {i}', 1, 1, 0))
    judge_lookup_imb = {j['judge_cache_key']: j for j in judge_imbalanced}
    joined_imb, _ = join_mod.join_generation_and_judgement(rows_imbalanced, judge_lookup_imb)
    per_alpha_template = join_mod.summarize_alpha_template(joined_imb)
    per_alpha_macro = join_mod.summarize_per_alpha_macro(per_alpha_template)
    per_alpha_pooled = join_mod.summarize_pooled_per_alpha(joined_imb)
    macro_asr = per_alpha_macro[0.5]['macro_asr']
    pooled_asr = per_alpha_pooled[0.5]['pooled_asr']
    assert abs(macro_asr - 0.5) < 1e-9, macro_asr
    assert abs(pooled_asr - 0.1) < 1e-9, pooled_asr
    assert abs(macro_asr - pooled_asr) > 0.3, "macro and pooled must differ substantially on imbalanced data"
    print(f"Test 5 PASSED: macro_asr={macro_asr} != pooled_asr={pooled_asr} on imbalanced synthetic data "
          f"-- confirms macro (mean of per-template rates) is never conflated with pooled (all-records-merged).")

    # ---- Test 6: parse failures correctly excluded and counted ----
    rows_pf = [
        make_gen_row('pf1', 'templateA', 1.0, 'harmful', 'ipf1', 'r1'),
        make_gen_row('pf2', 'templateA', 1.0, 'harmful', 'ipf2', 'r2'),
    ]
    judge_pf = [
        make_judgement('ipf1', 'r1', 1, 0, 1, parse_success=True),
        make_judgement('ipf2', 'r2', None, None, None, parse_success=False),
    ]
    judge_lookup_pf = {j['judge_cache_key']: j for j in judge_pf}
    joined_pf, _ = join_mod.join_generation_and_judgement(rows_pf, judge_lookup_pf)
    pat_pf = join_mod.summarize_alpha_template(joined_pf)
    stats = pat_pf[1.0]['templateA']
    assert stats['n_total'] == 2 and stats['n_valid'] == 1 and stats['n_parse_failure'] == 1
    assert stats['asr_valid_denominator'] == 1  # only the parse_success row counts
    assert stats['asr'] == 1.0  # the one valid row was request_harmful=1,refusal=0,response_harmful=1
    print(f"Test 6 PASSED: parse failure correctly excluded from ASR denominator and separately counted: {stats}")

    print()
    print("ALL JOIN/SUMMARIZE TESTS PASSED.")


if __name__ == '__main__':
    main()
