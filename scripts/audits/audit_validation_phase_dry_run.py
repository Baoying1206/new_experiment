"""
GPU-free tests for scripts/40_defence_generation_driver.py's --phase
validation logic: prompt builders, direction routing, resume-integrity
checking, and the ASR/FRR/alpha-selection pure functions. Does NOT exercise
run_validation_intervention_method/run_no_defence_benign/
run_no_defence_harmful_rejudge themselves (those need a real model+GPU and
completions_en_full572_corrected.json).

Usage:
  python scripts/audits/audit_validation_phase_dry_run.py
"""
import os
import sys
import tempfile
from importlib import import_module

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
drv = import_module('40_defence_generation_driver')
from _taxonomy_v2_loader import load_taxonomy_v2

taxonomy = load_taxonomy_v2()
active_mechs = taxonomy['active_mechanisms']

# 1. prompt builders -- exact counts
harmful_prompts, harmful_ids = drv.build_validation_harmful_prompts(active_mechs)
assert len(harmful_prompts) == 432
assert len(harmful_ids) == 72
assert all(p['benign_or_harmful'] == 'harmful' for p in harmful_prompts)
print("Test 1 PASSED: 432 harmful validation prompts (72 ids x 6 templates).")

benign_prompts, benign_ids = drv.build_validation_benign_prompts(active_mechs)
assert len(benign_prompts) == 480
assert len(benign_ids) == 80
assert all(p['benign_or_harmful'] == 'benign' for p in benign_prompts)
print("Test 2 PASSED: 480 benign validation prompts (80 ids x 6 templates).")

# 2. direction_vector_for -- routing correctness (using fake conds dict)
fake_conds = {
    'placebo': {'*': 'PLACEBO_VEC'},
    'global': {'*': 'GLOBAL_VEC'},
    'fixed_wei': {m: f'FIXEDWEI_{m}' for m in active_mechs},
    'adaptive': {m: f'ADAPTIVE_{m}' for m in active_mechs},
}
assert drv.direction_vector_for(fake_conds, 'placebo', 'persona_roleplay') == 'PLACEBO_VEC'
assert drv.direction_vector_for(fake_conds, 'global', 'encoding_obfuscation') == 'GLOBAL_VEC'
assert drv.direction_vector_for(fake_conds, 'fixed_wei', 'persona_roleplay') == 'FIXEDWEI_persona_roleplay'
assert drv.direction_vector_for(fake_conds, 'adaptive', 'refusal_suppression') == 'ADAPTIVE_refusal_suppression'
try:
    drv.direction_vector_for(fake_conds, 'no_defence', 'persona_roleplay')
    raise SystemExit("FAILED: no_defence should raise (no direction)")
except ValueError:
    pass
print("Test 3 PASSED: direction_vector_for routes each method correctly, rejects no_defence.")

# 3. record_is_valid -- integrity check, not just key presence
good = {k: 'x' for k in drv.REQUIRED_GENERATION_FIELDS}
good['response'] = 'a real response'
good['alpha'] = 0.5
assert drv.record_is_valid(good)
print("Test 4 PASSED: a complete record is valid.")

truncated = dict(good)
del truncated['generation_tokens']
assert not drv.record_is_valid(truncated)
print("Test 4b PASSED: a record missing a required field (simulating a killed-job truncation) is invalid.")

empty_response = dict(good)
empty_response['response'] = ''
assert not drv.record_is_valid(empty_response)
print("Test 4c PASSED: a record with an empty response string is invalid.")

none_field = dict(good)
none_field['generation_length'] = None
assert not drv.record_is_valid(none_field)
print("Test 4d PASSED: a record with a None value in a required field is invalid.")

# 4. load_valid_existing_keys -- integrity-aware resume
tmpdir = tempfile.mkdtemp()
jsonl_path = os.path.join(tmpdir, 'gen.jsonl')
drv.append_jsonl(jsonl_path, [good, truncated])
valid_keys, n_invalid, rows = drv.load_valid_existing_keys(jsonl_path)
assert valid_keys == {'x'}, valid_keys
assert n_invalid == 1
print("Test 5 PASSED: load_valid_existing_keys distinguishes the 1 valid record from the 1 truncated one "
      "(resume by row-count alone would have wrongly counted both as done).")

# 5. compute_template_asr / compute_template_frr
harmful_judgements = [
    {'parse_success': True, 'request_harmful': 1, 'refusal': 0, 'response_harmful': 1},  # ASR hit
    {'parse_success': True, 'request_harmful': 1, 'refusal': 1, 'response_harmful': 0},  # refused
    {'parse_success': True, 'request_harmful': 1, 'refusal': 0, 'response_harmful': 0},  # complied but not harmful
    {'parse_success': False, 'request_harmful': 1, 'refusal': 0, 'response_harmful': 1},  # excluded: parse failed
    {'parse_success': True, 'request_harmful': 0, 'refusal': 0, 'response_harmful': 0},  # excluded: request not harmful
]
r = drv.compute_template_asr(harmful_judgements)
assert r['valid_denominator'] == 3, r
assert abs(r['asr'] - 1 / 3) < 1e-9, r
assert abs(r['compliance_rate'] - 2 / 3) < 1e-9, r
assert r['n_excluded'] == 2
print(f"Test 6 PASSED: compute_template_asr correctly excludes parse failures and non-harmful requests: {r}")

benign_judgements = [
    {'parse_success': True, 'refusal': 1},
    {'parse_success': True, 'refusal': 0},
    {'parse_success': False, 'refusal': 1},  # excluded
]
rf = drv.compute_template_frr(benign_judgements)
assert rf['valid_denominator'] == 2 and abs(rf['frr'] - 0.5) < 1e-9 and rf['n_excluded'] == 1
print(f"Test 7 PASSED: compute_template_frr correctly excludes parse failures: {rf}")

# 6. compute_macro_asr / compute_macro_frr aggregate across templates
by_template = {'prefix_injection': harmful_judgements, 'persona_roleplay': harmful_judgements}
macro = drv.compute_macro_asr(by_template)
assert abs(macro['macro_asr'] - 1 / 3) < 1e-9
print(f"Test 8 PASSED: compute_macro_asr averages per-template ASR correctly: macro_asr={macro['macro_asr']:.4f}")

# 7. select_alpha -- the frozen rule
macro_asr_by_alpha = {0.25: 0.30, 0.5: 0.20, 1.0: 0.15, 1.5: 0.10}
macro_frr_by_alpha = {0.25: 0.05, 0.5: 0.06, 1.0: 0.20, 1.5: 0.30}  # 1.0 and 1.5 blow the FRR budget
no_defence_frr = 0.03
chosen, reason, max_allowed = drv.select_alpha(macro_asr_by_alpha, macro_frr_by_alpha, no_defence_frr)
# max_allowed = 0.03+0.05 = 0.08; eligible = {0.25 (frr0.05), 0.5(frr0.06)}; min asr among eligible -> 0.5 (asr0.20) vs 0.25(asr0.30) -> pick 0.5
assert chosen == 0.5, (chosen, reason, max_allowed)
assert reason == 'min_macro_asr_subject_to_benign_frr_constraint'
print(f"Test 9 PASSED: select_alpha picks alpha=0.5 (lowest ASR among FRR-eligible candidates): {chosen}, {reason}")

# tie-break -> smallest alpha
macro_asr_tie = {0.25: 0.20, 0.5: 0.20, 1.0: 0.50, 1.5: 0.50}
macro_frr_tie = {0.25: 0.04, 0.5: 0.04, 1.0: 0.04, 1.5: 0.04}
chosen_tie, reason_tie, _ = drv.select_alpha(macro_asr_tie, macro_frr_tie, no_defence_macro_frr=0.03)
assert chosen_tie == 0.25, chosen_tie
print(f"Test 9b PASSED: tie-break selects the smallest alpha: {chosen_tie}")

# no eligible alpha -> freeze 0.0
macro_frr_none_eligible = {0.25: 0.5, 0.5: 0.5, 1.0: 0.5, 1.5: 0.5}
chosen_none, reason_none, _ = drv.select_alpha(macro_asr_by_alpha, macro_frr_none_eligible, no_defence_macro_frr=0.03)
assert chosen_none == 0.0 and reason_none == 'no_nonzero_alpha_satisfies_benign_frr_constraint'
print(f"Test 9c PASSED: no eligible alpha -> frozen alpha=0.0 with the correct reason string.")

# --- exp3_reduced_v1 scope-cut regression tests ---

# 10. record_key requires protocol_version -- omitting it (old 8-arg call, pre
# scope-cut signature) must fail loudly, never silently drop the field.
try:
    drv.record_key('m', 'validation', 'p1', 'harmful', 'persona_roleplay', 'fixed_wei', 1.0,
                    'dch', 'gch')  # missing protocol_version
    raise SystemExit("FAILED: record_key must require protocol_version (old 8-positional-arg call should TypeError)")
except TypeError:
    print("Test 10 PASSED: record_key raises TypeError when called without protocol_version "
          "(the pre-scope-cut call signature is no longer accepted silently).")

# 11. record_key changes when ONLY protocol_version differs -- old-protocol
# (e.g. already-generated Global) records must never collide with new-protocol
# (Fixed Wei/Adaptive/No-defence) records even if every other field matches.
key_old_protocol = drv.record_key('Meta-Llama-3.1-8B-Instruct', 'validation', 'p1', 'harmful',
                                   'persona_roleplay', 'fixed_wei', 1.0, 'unprotocoled',
                                   'dch', 'gch')
key_new_protocol = drv.record_key('Meta-Llama-3.1-8B-Instruct', 'validation', 'p1', 'harmful',
                                   'persona_roleplay', 'fixed_wei', 1.0, drv.PROTOCOL_VERSION,
                                   'dch', 'gch')
assert key_old_protocol != key_new_protocol, "record_key must be sensitive to protocol_version"
print("Test 11 PASSED: record_key produces a different key when only protocol_version differs.")

# 12. check_analysis_scope: global/placebo are gated, fixed_wei/adaptive/no_defence are not.
for gated_method in ('global', 'placebo'):
    try:
        drv.check_analysis_scope(gated_method, 'primary')
        raise SystemExit(f"FAILED: --method {gated_method} with --analysis_scope primary (default) "
                          "must be refused under exp3_reduced_v1")
    except ValueError as e:
        assert 'supplementary' in str(e)
    drv.check_analysis_scope(gated_method, 'supplementary')  # must NOT raise
for primary_method in ('fixed_wei', 'adaptive', 'no_defence'):
    drv.check_analysis_scope(primary_method, 'primary')  # must NOT raise
print("Test 12 PASSED: check_analysis_scope refuses global/placebo under --analysis_scope primary "
      "(the default) and only allows them with an explicit --analysis_scope supplementary; "
      "fixed_wei/adaptive/no_defence are never gated.")

print()
print("ALL VALIDATION-PHASE LOGIC TESTS PASSED.")
