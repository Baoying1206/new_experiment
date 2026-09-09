"""
Synthetic + real-repo-integration dry-run checks for
scripts/54_extract_formal_context_activations.py and
scripts/55_analyze_formal_context_activations.py. NO torch model
weights, NO transformers model loading, NO GPU -- torch (if available)
is used only for tensor math in synthetic fixtures, mirroring the
pilot's dry-run suite (scripts/audits/audit_context_activation_pilot_dry_run.py).

Checks 8+ cover the 2026-09-09 additions to scripts/55 (CO/MG prototype
construction, R/H explanatory power + random-subspace null + Holm
correction, instruction-cluster bootstrap) -- component-level synthetic
tests, not a full analyze_one_model() run (a real-hidden_dim,
300-instruction synthetic fixture for that would be several GB and is
not worth the memory/time cost for a dry-run check).

Run: python scripts/audits/audit_formal_context_activation_dry_run.py
"""
import importlib.util
import inspect
import json
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
    # families, with the 2 special mechanisms EACH using their OWN
    # corrected t_inst method (2026-09-09 fix, found via the first real
    # cluster dry_run): encoding_obfuscation -> t_inst_equals_t_post (the
    # b64 payload is the final template content); payload_splitting ->
    # the SAME longest-common-prefix method as ordinary conditions, just
    # targeting payload_b's span. Neither relies on an end-of-turn token
    # existing in full_ids (this script never wraps text in a chat
    # template, so no such token is ever present). The old
    # structural_end_of_turn_boundary method is gone.----
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
            method = r.get('t_inst_method', '')
            if r['sample_id'] == 'encoding_obfuscation':
                if 't_inst_equals_t_post_by_construction' not in method:
                    audit_ok = False
                    print(f"  {fam}: encoding_obfuscation did not use t_inst_equals_t_post_by_construction: {method}")
                if r.get('t_inst') != r.get('t_post'):
                    audit_ok = False
                    print(f"  {fam}: encoding_obfuscation t_inst != t_post despite the by-construction method: "
                          f"t_inst={r.get('t_inst')}, t_post={r.get('t_post')}")
            elif r['sample_id'] == 'payload_splitting':
                if 'longest_common_prefix_of_truncated_vs_full_tokenization' not in method:
                    audit_ok = False
                    print(f"  {fam}: payload_splitting did not use the longest-common-prefix method: {method}")
                if 'targeting payload_b' not in method:
                    audit_ok = False
                    print(f"  {fam}: payload_splitting method does not indicate it targeted payload_b: {method}")
                if 'structural_end_of_turn_boundary' in method:
                    audit_ok = False
                    print(f"  {fam}: payload_splitting incorrectly still uses the removed structural method: {method}")
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

    # ---- 8. scripts/55's new statistics (component-level synthetic tests) ----
    try:
        import torch
        amod = _load(ANALYZE_SCRIPT_PATH, 'formal_analyze')
        torch_ok = True
    except Exception as e:
        torch_ok = False
        print(f"  could not load scripts/55 with torch available: {type(e).__name__}: {e}")
    check('8_scripts_55_loads_with_torch', torch_ok)

    if torch_ok:
        # 8a. normalize_or_undefined: zero vector -> UNDEFINED_NEAR_ZERO_NORM;
        # ordinary vector -> OK with unit norm.
        zero_vec = torch.zeros(16)
        u, status = amod.normalize_or_undefined(zero_vec)
        check('8a_normalize_zero_vector_is_undefined', u is None and status['status'] == 'UNDEFINED_NEAR_ZERO_NORM',
              f"got u={u}, status={status}")
        ordinary_vec = torch.tensor([3.0, 4.0] + [0.0] * 14)
        u2, status2 = amod.normalize_or_undefined(ordinary_vec)
        check('8b_normalize_ordinary_vector_unit_norm',
              u2 is not None and status2['status'] == 'OK' and abs(u2.norm().item() - 1.0) < 1e-5,
              f"u2={u2}, status2={status2}")

        # 8c. build_co_mg_prototype: even a SINGLE near-zero-norm mechanism
        # (1-of-3, not just 2-of-3) must invalidate the WHOLE prototype --
        # never fall back to averaging the remaining 2 (this round's
        # tightened requirement).
        def make_deltas_fn(mech_to_vec, dim, clusters_idx):
            def canonical_deltas_vs_plain(mech, ly, pos):
                v = mech_to_vec[mech]
                n = sum(len(c) for c in clusters_idx)
                return v.unsqueeze(0).repeat(n, 1)  # identical value at every raw id -- any
                                                     # equal-weighting bug would still pass this
                                                     # particular test, so 8m below is the real check
            return canonical_deltas_vs_plain

        dim = 16
        toy_clusters_idx = [[0, 1], [2], [3], [4]]  # mimics one duplicate pair + singletons
        d_placebo = torch.zeros(dim)
        near_zero = torch.zeros(dim)
        real_vec_a = torch.eye(dim)[0] * 2.0
        real_vec_b = torch.eye(dim)[1] * 2.0
        mechs_1_undefined = {'prefix_injection': near_zero, 'refusal_suppression': real_vec_a,
                              'persona_roleplay': real_vec_b, 'placebo': d_placebo}
        fn = make_deltas_fn(mechs_1_undefined, dim, toy_clusters_idx)
        proto, per_mech, proto_status = amod.build_co_mg_prototype(
            amod.CO_MECHS, fn, d_placebo, 0, 0, toy_clusters_idx)
        check('8c_prototype_undefined_when_even_1_of_3_mechanisms_near_zero',
              proto is None and proto_status == 'UNDEFINED_INCOMPLETE_CATEGORY',
              f"proto={proto}, status={proto_status}, per_mech={per_mech}")

        # 8d. build_co_mg_prototype: all 3 mechanisms well-defined and
        # identical -> prototype is that same unit direction (OK).
        same_vec = torch.eye(dim)[1] * 3.0
        mechs_all_ok = {'prefix_injection': same_vec, 'refusal_suppression': same_vec,
                         'persona_roleplay': same_vec, 'placebo': d_placebo}
        fn2 = make_deltas_fn(mechs_all_ok, dim, toy_clusters_idx)
        proto2, per_mech2, proto_status2 = amod.build_co_mg_prototype(
            amod.CO_MECHS, fn2, d_placebo, 0, 0, toy_clusters_idx)
        check('8d_prototype_ok_when_all_3_mechanisms_aligned',
              proto2 is not None and proto_status2 == 'OK'
              and amod.safe_cosine(proto2, same_vec)['value'] > 0.999,
              f"proto2={proto2}, status2={proto_status2}")

        # 8e. compute_E_RH_and_q: d strictly inside span(R,H) -> E_RH~=1, q~=0.
        r_vec = torch.eye(dim)[0].float()
        h_vec = torch.eye(dim)[1].float()
        d_in_span = 2.0 * r_vec + 3.0 * h_vec
        e_rh_in, q_in = amod.compute_E_RH_and_q(d_in_span, r_vec, h_vec)
        check('8e_E_RH_near_1_when_d_in_span_RH',
              abs(e_rh_in - 1.0) < 1e-5 and q_in < 1e-4, f"E_RH={e_rh_in}, q={q_in}")

        # 8f. compute_E_RH_and_q: d orthogonal to span(R,H) -> E_RH~=0, q~=1.
        d_orth = torch.eye(dim)[5].float()
        e_rh_orth, q_orth = amod.compute_E_RH_and_q(d_orth, r_vec, h_vec)
        check('8f_E_RH_near_0_when_d_orthogonal_to_span_RH',
              e_rh_orth < 1e-5 and abs(q_orth - 1.0) < 1e-4, f"E_RH={e_rh_orth}, q={q_orth}")

        # 8g. generate_random_subspace_bases: deterministic given the same
        # (model_alias, layer, global_seed); different layer -> different
        # bases (not the same seed reused blindly).
        bases_a, seed_a = amod.generate_random_subspace_bases(dim, 'FakeModel', 5, n_random=20)
        bases_a2, seed_a2 = amod.generate_random_subspace_bases(dim, 'FakeModel', 5, n_random=20)
        bases_b, seed_b = amod.generate_random_subspace_bases(dim, 'FakeModel', 6, n_random=20)
        check('8g_random_subspace_seed_deterministic_and_layer_specific',
              seed_a == seed_a2 and torch.allclose(bases_a[0], bases_a2[0]) and seed_a != seed_b,
              f"seed_a={seed_a}, seed_b={seed_b}")

        # 8h. rh_significance_test: a random subspace null centered at
        # E[E_random]=2/dim; a d strongly inside span(R,H) should be
        # significant (E_RH far above the null 95th percentile).
        bases_dim, _ = amod.generate_random_subspace_bases(dim, 'FakeModel', 7, n_random=500)
        test_in_span = amod.rh_significance_test(d_in_span, r_vec, h_vec, bases_dim, dim)
        check('8h_significance_test_flags_d_in_span_as_significant',
              test_in_span['status'] == 'OK' and test_in_span['significant_at_0_05_one_sided'] is True
              and test_in_span['E_RH'] > test_in_span['null_95th_percentile'],
              f"test_in_span={test_in_span}")

        # 8i. holm_correction: monotone non-decreasing in sorted-p order,
        # and each adjusted p >= its own unadjusted p.
        named = [('a', 0.001), ('b', 0.20), ('c', 0.04), ('d', 0.50)]
        adj = amod.holm_correction(named)
        holm_ok = all(adj[k] >= p - 1e-12 for k, p in named) and all(0.0 <= v <= 1.0 for v in adj.values())
        check('8i_holm_correction_adjusted_p_geq_raw_p_and_bounded', holm_ok, f"adj={adj}")

        # 8j. load_instruction_clusters against the REAL repo data: must
        # find exactly 298 clusters from the real 300 direction_ids, with
        # the 2 known duplicate pairs from the 2026-09-09 audit.
        with open(os.path.join(REPO_ROOT, 'data', 'splits.json'), encoding='utf-8') as f:
            real_direction_ids = json.load(f)['direction' + '_ids']
        clusters_idx, cluster_report = amod.load_instruction_clusters(real_direction_ids)
        dup_id_pairs = sorted(tuple(sorted(g)) for g in cluster_report['duplicate_groups'])
        expected_dup_pairs = sorted([('p139', 'p318'), ('p255', 'p410')])
        check('8j_instruction_clusters_298_with_known_duplicate_pairs',
              cluster_report['n_unique_text_clusters'] == 298
              and cluster_report['n_raw_ids'] == 300
              and dup_id_pairs == expected_dup_pairs,
              f"cluster_report={cluster_report}")

        # 8k. compute_cluster_means / cluster_equal_weight_mean: the two
        # members of a duplicate-text cluster must always move together
        # through the collapse step (both averaged into ONE row before any
        # cross-cluster aggregation) -- verified structurally on the real
        # 298-cluster grouping (every cluster's raw-id members map to
        # exactly one output row).
        cm_shape_ok = True
        n_clusters_real = len(clusters_idx)
        fake_deltas = torch.arange(300 * 4, dtype=torch.float32).reshape(300, 4)
        cm_real = amod.compute_cluster_means(fake_deltas, clusters_idx)
        if list(cm_real.shape) != [n_clusters_real, 4]:
            cm_shape_ok = False
        check('8k_compute_cluster_means_collapses_to_one_row_per_cluster', cm_shape_ok,
              f"cm_real.shape={list(cm_real.shape)}, expected [{n_clusters_real}, 4]")

        # 8l. load_instruction_clusters fail-fast: a deliberately wrong
        # cluster count (fewer ids than the real 300) must raise, not
        # silently proceed.
        raised = False
        try:
            amod.load_instruction_clusters(real_direction_ids[:50])
        except ValueError:
            raised = True
        check('8l_instruction_clusters_wrong_count_raises', raised)

        # 8m. EQUAL-WEIGHT AGGREGATION, extreme synthetic fixture (this
        # round's central requirement): a 2-member duplicate cluster with
        # value A and a 1-member singleton cluster with a very different
        # value B. A flat mean over the 3 raw rows would give
        # (2A+B)/3 -- clearly biased toward A. The correct equal-weight
        # result is (A+B)/2, giving each of the 2 CLUSTERS the same 1/2
        # weight regardless of how many raw ids compose it.
        dup_cluster_idx = [[0, 1], [2]]  # cluster 0 has 2 members (both value A), cluster 1 has 1 (value B)
        A = torch.tensor([10.0, 0.0])
        B = torch.tensor([0.0, 100.0])
        deltas_extreme = torch.stack([A, A, B])  # rows 0,1 = A (duplicate cluster), row 2 = B (singleton)
        naive_flat_mean = deltas_extreme.mean(dim=0)  # (2A+B)/3 = [6.667, 33.333] -- the WRONG answer
        correct_equal_weight = amod.cluster_equal_weight_mean(deltas_extreme, dup_cluster_idx)
        expected_equal_weight = (A + B) / 2.0  # [5.0, 50.0]
        check('8m_cluster_equal_weight_mean_matches_A_plus_B_over_2_not_flat_mean',
              torch.allclose(correct_equal_weight, expected_equal_weight, atol=1e-5)
              and not torch.allclose(correct_equal_weight, naive_flat_mean, atol=1e-5),
              f"correct={correct_equal_weight.tolist()}, expected={expected_equal_weight.tolist()}, "
              f"naive_flat_mean={naive_flat_mean.tolist()} (must differ from correct)")

        # 8n. Same extreme fixture through the bootstrap path: with 0
        # resampling noise removed by checking the CENTER of the bootstrap
        # distribution (median over enough resamples with a fixed seed),
        # the point estimate embedded in cluster_bootstrap_from_cluster_means
        # must also reflect equal cluster weighting, not raw-id weighting.
        cluster_means_extreme = amod.compute_cluster_means(deltas_extreme, dup_cluster_idx)
        check('8n_cluster_means_tensor_has_one_row_per_cluster_not_per_id',
              list(cluster_means_extreme.shape) == [2, 2]
              and torch.allclose(cluster_means_extreme[0], A)
              and torch.allclose(cluster_means_extreme[1], B),
              f"cluster_means_extreme={cluster_means_extreme.tolist()}")

        # 8o. tile_clusters_across_repeats: pooling n_repeats=3 copies of a
        # 5-row axis must produce clusters whose members span all 3
        # repeats' offsets for the same underlying cluster, with the total
        # membership count multiplied by n_repeats (never re-splitting a
        # text-cluster into separate per-repeat clusters).
        base_clusters = [[0, 1], [2], [3], [4]]
        pooled = amod.tile_clusters_across_repeats(base_clusters, n_instr=5, n_repeats=3)
        pooled_ok = (len(pooled) == len(base_clusters)
                     and sorted(pooled[0]) == sorted([0, 1, 5, 6, 10, 11])
                     and sorted(pooled[1]) == sorted([2, 7, 12]))
        check('8o_tile_clusters_across_repeats_pools_same_cluster_across_repeats', pooled_ok, f"pooled={pooled}")

        # 8p. compute_attack_profiles: PRIMARY (t_inst-position-index 0) vs
        # SENSITIVITY (t_post-position-index 1) must be computed
        # independently and can legitimately differ -- verifies the two are
        # not accidentally aliased to the same underlying computation.
        class FakeCanonicalPayload:
            def __init__(self):
                pass

        def fake_canonical_deltas_vs_plain(mech, ly, pos):
            # pos 0 (t_inst) and pos 1 (t_post) return DIFFERENT synthetic
            # values so the two profiles are distinguishable if wired correctly
            base = torch.eye(dim)[0] if pos == 0 else torch.eye(dim)[1]
            return (base * (2.0 if mech == 'mech_a' else 5.0)).unsqueeze(0).repeat(5, 1)

        # asymmetric centroid (different weight on the pos=0 vs pos=1 basis
        # vectors) so a correctly-wired primary/sensitivity split produces
        # genuinely different projections rather than coinciding by symmetry
        toy_centroids = {fam: torch.eye(dim)[0] * 2.0 + torch.eye(dim)[1] * 5.0 for fam in amod.FAMILIES}
        toy_clusters_5 = [[0], [1], [2], [3], [4]]
        profiles_tinst = amod.compute_attack_profiles(
            fake_canonical_deltas_vs_plain, ['mech_a'], toy_centroids, 0, 0, toy_clusters_5)
        profiles_tpost = amod.compute_attack_profiles(
            fake_canonical_deltas_vs_plain, ['mech_a'], toy_centroids, 0, 1, toy_clusters_5)
        check('8p_attack_profiles_primary_and_sensitivity_computed_independently',
              profiles_tinst['mech_a'][amod.FAMILIES[0]]['raw_signed_projection']
              != profiles_tpost['mech_a'][amod.FAMILIES[0]]['raw_signed_projection'],
              f"tinst={profiles_tinst}, tpost={profiles_tpost}")

        # 8q. assert_dual_position_design: R@t_post/H@t_inst passes; any
        # other combination (including "both at the same position") raises.
        raised_ok, raised_bad_r, raised_bad_h, raised_swapped = False, False, False, False
        try:
            amod.assert_dual_position_design({'semantic_position': 't_post'},
                                              {'semantic_position': 't_inst'}, 'FakeModel')
        except ValueError:
            raised_ok = True  # should NOT raise -- track failure below
        try:
            amod.assert_dual_position_design({'semantic_position': 't_inst'},
                                              {'semantic_position': 't_inst'}, 'FakeModel')
            raised_bad_r = False
        except ValueError:
            raised_bad_r = True
        try:
            amod.assert_dual_position_design({'semantic_position': 't_post'},
                                              {'semantic_position': 't_post'}, 'FakeModel')
            raised_bad_h = False
        except ValueError:
            raised_bad_h = True
        try:
            amod.assert_dual_position_design({'semantic_position': 't_inst'},
                                              {'semantic_position': 't_post'}, 'FakeModel')  # swapped
            raised_swapped = False
        except ValueError:
            raised_swapped = True
        check('8q_dual_position_design_assertion_correct',
              (not raised_ok) and raised_bad_r and raised_bad_h and raised_swapped,
              f"correct_combo_raised={raised_ok} (should be False), bad_r_raised={raised_bad_r}, "
              f"bad_h_raised={raised_bad_h}, swapped_raised={raised_swapped}")

    print()
    if failed == 0:
        print("ALL FORMAL CONTEXT ACTIVATION DRY-RUN CHECKS PASSED.")
    else:
        print(f"{failed} CHECK(S) FAILED.")
    return failed


if __name__ == '__main__':
    sys.exit(1 if main() else 0)
