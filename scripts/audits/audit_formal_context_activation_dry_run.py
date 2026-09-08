"""
Synthetic + real-repo-integration dry-run checks for
scripts/54_extract_formal_context_activations.py and
scripts/55_analyze_formal_context_activations.py. NO torch model
weights, NO transformers model loading, NO GPU -- torch (if available)
is used only for tensor math in synthetic fixtures, mirroring the
pilot's dry-run suite (scripts/audits/audit_context_activation_pilot_dry_run.py).

Run: python scripts/audits/audit_formal_context_activation_dry_run.py
"""
import importlib.util
import inspect
import os
import sys

SCRIPT_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
EXTRACT_SCRIPT_PATH = os.path.join(REPO_ROOT, 'scripts', '54_extract_formal_context_activations.py')
ANALYZE_SCRIPT_PATH = os.path.join(REPO_ROOT, 'scripts', '55_analyze_formal_context_activations.py')
# scripts/54 and 55 do `from utils.<module> import ...` relying on
# scripts/ being on sys.path (true automatically when invoked directly as
# `python scripts/54_....py`, NOT true when loaded via importlib from a
# different directory like this file's) -- add it explicitly.
sys.path.insert(0, os.path.join(REPO_ROOT, 'scripts'))


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    failed = 0

    def check(name, condition, detail=''):
        nonlocal failed
        if condition:
            print(f"PASS {name}")
        else:
            failed += 1
            print(f"FAIL {name}: {detail}")

    check('0_extract_script_exists', os.path.exists(EXTRACT_SCRIPT_PATH))
    check('0b_analyze_script_exists', os.path.exists(ANALYZE_SCRIPT_PATH))
    if not (os.path.exists(EXTRACT_SCRIPT_PATH) and os.path.exists(ANALYZE_SCRIPT_PATH)):
        print(f"\n{failed} CHECK(S) FAILED (scripts missing, cannot continue).")
        return failed

    mod = _load(EXTRACT_SCRIPT_PATH, 'formal_extract')

    # ---- 1. condition counts derived from real repo files, not hardcoded ----
    context_conditions, context_data = mod.load_context_conditions()
    canonical_conditions = mod.load_canonical_conditions()
    n_context, n_canonical = len(context_conditions), len(canonical_conditions)
    n_conditions = n_context + n_canonical
    check('1_context_conditions_16', n_context == 16, f"got {n_context}")
    check('1b_canonical_conditions_8_plain_placebo_plus_6_mechanisms', n_canonical == 8, f"got {n_canonical}")
    check('1c_total_conditions_24', n_conditions == 24, f"got {n_conditions}")

    access_log = mod.AccessLog()
    ids = mod.load_formal_instruction_ids(access_log)
    instrs = mod.load_source_instructions(ids, access_log)
    access_log.assert_clean()
    n_ids = len(ids)
    check('2_n_ids_is_300', n_ids == 300, f"got {n_ids}")
    check('2b_n_instructions_loaded_matches', len(instrs) == n_ids)
    n_forward_per_model = n_ids * n_conditions
    check('2c_forward_passes_per_model_7200', n_forward_per_model == 7200, f"got {n_forward_per_model}")
    check('2d_forward_passes_all_3_models_21600', n_forward_per_model * 3 == 21600)
    check('2e_access_log_clean_direction_ids_only', True)  # assert_clean() above raises on failure

    # ---- 2. gate checks pass against the real, currently-approved repo state ----
    try:
        attestation, checklist = mod.run_gate_checks(context_data)
        gate_ok = attestation.get('template_texts_identical') is True
    except Exception as e:
        gate_ok = False
        print(f"  run_gate_checks raised {type(e).__name__}: {e}")
    check('3_gate_checks_pass_against_real_repo', gate_ok)

    # ---- 3. special-mechanism rendering does not raise, and produces text
    # that does NOT contain the literal raw instruction as a substring
    # (confirms the base64/split path actually engaged, not a silent
    # pass-through) ----
    sample_instr = "do a specific test thing right now"
    special_ok = True
    for name in mod.SPECIAL_CANONICAL_MECHANISMS:
        if name not in canonical_conditions:
            special_ok = False
            print(f"  {name} missing from canonical_conditions")
            continue
        try:
            rendered = mod.render_canonical(name, canonical_conditions[name], sample_instr)
        except Exception as e:
            special_ok = False
            print(f"  render_canonical({name!r}) raised {type(e).__name__}: {e}")
            continue
        if sample_instr in rendered:
            special_ok = False
            print(f"  render_canonical({name!r}) output still contains the raw instruction verbatim "
                  f"-- special encoding did not engage")
    check('4_special_canonical_mechanisms_render_without_raising', special_ok)

    # non-special canonical mechanisms (e.g. prefix_injection) DO contain
    # the raw instruction literally
    non_special_ok = True
    for name, text in canonical_conditions.items():
        if name in mod.SPECIAL_CANONICAL_MECHANISMS or name == 'plain':
            continue
        rendered = mod.render_canonical(name, text, sample_instr)
        if sample_instr not in rendered:
            non_special_ok = False
            print(f"  render_canonical({name!r}) does NOT contain the raw instruction -- unexpected")
    check('4b_non_special_canonical_mechanisms_contain_raw_instruction', non_special_ok)

    # ---- 4. Phase 0 audit runs cleanly (mock tokenizer) for all 3 model
    # families, with the 2 special mechanisms correctly using the
    # structural (not literal-substring) position method ----
    class MockEncoding:
        def __init__(self, ids):
            self.input_ids = ids

    class MockTokenizer:
        def __init__(self, family):
            from utils.token_positions import EOT_TOKEN_STR_BY_FAMILY_SUBSTRING
            self.vocab = {}
            self.chat_template = 'mock'
            self.eot_str = None
            for substr, tok in EOT_TOKEN_STR_BY_FAMILY_SUBSTRING:
                if substr in family:
                    self.eot_str = tok
            self._id(self.eot_str)

        def _id(self, w):
            return self.vocab.setdefault(w, len(self.vocab))

        def __call__(self, text, add_special_tokens=True):
            ids = [self._id(w) for w in text.split()]
            ids.append(self._id(self.eot_str))
            return MockEncoding(ids)

        def encode(self, text, add_special_tokens=False):
            return [self._id(w) for w in text.split()]

        def decode(self, ids):
            rev = {v: k for k, v in self.vocab.items()}
            return ' '.join(rev[i] for i in ids)

        def convert_tokens_to_ids(self, tok):
            return self.vocab.get(tok)

    audit_ok = True
    real_sample_instr = next(iter(instrs.values()))
    for fam in ('qwen', 'llama', 'gemma'):
        tok = MockTokenizer(fam)
        rows, anomalies = mod.audit_token_positions(tok, real_sample_instr, context_data,
                                                      canonical_conditions, fam)
        if len(rows) != 16 or anomalies:
            audit_ok = False
            print(f"  {fam}: n_rows={len(rows)} (expected 16), anomalies={anomalies}")
        special_rows = [r for r in rows if r.get('special_encoding')]
        if len(special_rows) != 2:
            audit_ok = False
            print(f"  {fam}: expected 2 special-encoding rows, got {len(special_rows)}")
        for r in special_rows:
            if 'structural_end_of_turn_boundary' not in r.get('t_inst_method', ''):
                audit_ok = False
                print(f"  {fam}: {r['sample_id']} did not use the structural t_inst method: "
                      f"{r.get('t_inst_method')}")
    check('5_phase0_audit_16_samples_0_anomalies_all_3_families', audit_ok)

    # ---- 5. no generation/judge-model-loading symbols in either formal
    # script (AutoModelForCausalLM IS expected/allowed; only .generate(
    # calls and an actual lowercase 'wildguard' reference are forbidden) ----
    forbidden = ['.' + 'generate(', 'wild' + 'guard']
    scan_ok = True
    for path in (EXTRACT_SCRIPT_PATH, ANALYZE_SCRIPT_PATH):
        with open(path, encoding='utf-8') as f:
            src = f.read()
        hits = [s for s in forbidden if s in src]
        if hits:
            scan_ok = False
            print(f"  {path}: found {hits}")
    check('6_no_generate_or_judge_model_usage_in_formal_scripts', scan_ok)

    # ---- 6. validation_ids/test_ids never referenced as a literal access
    # expression in the extraction script's own source ----
    extract_src = inspect.getsource(mod)
    key1 = 'test' + '_ids'
    key2 = 'validation' + '_ids'
    forbidden_access = [f"splits['{key1}']", f'splits["{key1}"]', f"splits['{key2}']", f'splits["{key2}"]']
    access_scan_hits = [s for s in forbidden_access if s in extract_src]
    check('7_no_validation_or_test_ids_literal_access_in_extraction_script', not access_scan_hits,
          f"found: {access_scan_hits}")

    print()
    if failed == 0:
        print("ALL FORMAL CONTEXT ACTIVATION DRY-RUN CHECKS PASSED.")
    else:
        print(f"{failed} CHECK(S) FAILED.")
    return failed


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
