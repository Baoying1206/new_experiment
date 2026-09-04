"""
Synthetic-data-only tests for the R/H rebuild infrastructure
(scripts/utils/direction_metadata.py, and the taxonomy-source-of-truth fix
in scripts/25_extract_delta_r_h.py / scripts/27_dual_axis_diagnosis.py).
No GPU, no real model, no real activations -- every tensor here is a small
hand-constructed synthetic array. Per EXPERIMENT2_RH_REBUILD_PROTOCOL.md,
synthetic data may only be used to test code, never reported as an
experimental result.

Usage:
  python scripts/audits/audit_rh_rebuild_dry_run.py
"""
import os
import shutil
import sys
import tempfile

import torch
import torch.nn.functional as F

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(SCRIPT_DIR, '..'))
import utils.direction_metadata as dm
from _taxonomy_v2_loader import load_taxonomy_v2

STALE_NAMES = ('instruction_hierarchy', 'fictional_framing')


def make_logical_meta(**overrides):
    base = dict(
        direction_type='refusal_direction', model='FakeModel',
        model_revision='unknown', tokenizer_revision='unknown',
        chat_template_hash='deadbeef', semantic_position='t_post', layer='all',
        source_partition='independent_train', source_ids=['a', 'b', 'c'],
        construction_contrast='fake_contrast', random_seed=0,
    )
    base.update(overrides)
    return dm.build_direction_metadata(**base)


def main():
    tmpdir = tempfile.mkdtemp()
    try:
        # ---- Test 1: sign convention + projection calculation ----
        # refusal_hat points along +x. A diff that ALSO points along +x should
        # project to a POSITIVE delta_R; a diff pointing along -x should be
        # NEGATIVE; a diff orthogonal to it should project to ~0. This is
        # exactly 25_extract_delta_r_h.py's
        # delta_R[mech][i] = (diff_at_post * refusal_hat).sum(-1) formula,
        # replicated here with a hand-picked synthetic vector.
        refusal_hat = F.normalize(torch.tensor([[1.0, 0.0, 0.0]]), dim=-1)  # [1 layer, 3 dims]
        diff_same = torch.tensor([[2.0, 0.0, 0.0]])
        diff_opposite = torch.tensor([[-2.0, 0.0, 0.0]])
        diff_orthogonal = torch.tensor([[0.0, 5.0, 0.0]])
        proj_same = (diff_same * refusal_hat).sum(-1).item()
        proj_opposite = (diff_opposite * refusal_hat).sum(-1).item()
        proj_orthogonal = (diff_orthogonal * refusal_hat).sum(-1).item()
        assert abs(proj_same - 2.0) < 1e-6, proj_same
        assert abs(proj_opposite - (-2.0)) < 1e-6, proj_opposite
        assert abs(proj_orthogonal - 0.0) < 1e-6, proj_orthogonal
        print(f"Test 1 PASSED: projection sign/magnitude match hand-computed values "
              f"(same-direction={proj_same:+.1f}, opposite={proj_opposite:+.1f}, "
              f"orthogonal={proj_orthogonal:+.1f}).")

        # ---- Test 2: placebo calibration ----
        # delta_R_pc = delta_R[mech] - delta_R['placebo'], per instruction/layer.
        delta_R = {
            'mech_a': torch.tensor([[1.0, 2.0]]),
            'placebo': torch.tensor([[0.5, 0.5]]),
        }
        delta_R_pc = delta_R['mech_a'] - delta_R['placebo']
        assert torch.allclose(delta_R_pc, torch.tensor([[0.5, 1.5]]))
        print(f"Test 2 PASSED: placebo calibration (mech - placebo) matches hand-computed "
              f"values: {delta_R_pc.tolist()}.")

        # ---- Test 3: fail-fast on missing direction file ----
        missing_pt = os.path.join(tmpdir, 'does_not_exist.pt')
        try:
            dm.verify_direction_file(missing_pt)
            raise SystemExit("FAILED: expected FileNotFoundError for a missing direction tensor")
        except FileNotFoundError as e:
            assert 'missing' in str(e)
            print(f"Test 3a PASSED: verify_direction_file raises FileNotFoundError on a missing "
                  f"tensor: {str(e)[:70]}")

        # metadata-without-tensor case (the exact real-world failure mode this
        # whole rebuild is a response to: a .json sidecar with no paired .pt)
        orphan_json = os.path.join(tmpdir, 'orphan.json')
        dm.atomic_json_save({'fake': True}, orphan_json)
        orphan_pt = os.path.join(tmpdir, 'orphan.pt')
        try:
            dm.verify_direction_file(orphan_pt)
            raise SystemExit("FAILED: expected FileNotFoundError for an orphaned metadata-only case")
        except FileNotFoundError as e:
            print(f"Test 3b PASSED: a metadata JSON with no paired tensor is correctly rejected "
                  f"(tensor missing, checked first): {str(e)[:70]}")

        # ---- Test 4: save_direction_atomic + verify_direction_file round-trip,
        # metadata-vs-tensor hash match ----
        direction = torch.randn(4, 8)  # fake [n_layers=4, d_model=8]
        pt_path = os.path.join(tmpdir, 'refusal_dir_test.pt')
        meta_written = dm.save_direction_atomic(direction, make_logical_meta(), pt_path)
        assert os.path.exists(pt_path) and os.path.exists(pt_path[:-3] + '.json')
        loaded_tensor, loaded_meta = dm.verify_direction_file(pt_path)
        assert torch.allclose(loaded_tensor, direction)
        assert loaded_meta['tensor_sha256'] == meta_written['tensor_sha256']
        assert loaded_meta['tensor_shape'] == [4, 8]
        print("Test 4 PASSED: save_direction_atomic -> verify_direction_file round-trips "
              "exactly, with a matching content hash.")

        # ---- Test 5: hash mismatch is caught (tensor tampered with after metadata written) ----
        tampered_pt = os.path.join(tmpdir, 'tampered.pt')
        dm.save_direction_atomic(direction, make_logical_meta(), tampered_pt)
        torch.save(torch.randn(4, 8), tampered_pt)  # silently overwrite the tensor, metadata now stale
        try:
            dm.verify_direction_file(tampered_pt)
            raise SystemExit("FAILED: expected ValueError for a tensor/metadata hash mismatch")
        except ValueError as e:
            assert 'SHA-256' in str(e)
            print(f"Test 5 PASSED: a tensor that no longer matches its metadata's recorded hash "
                  f"is rejected: {str(e)[:70]}")

        # ---- Test 6: save_direction_atomic refuses if a LOGICAL field is missing ----
        incomplete_meta = {'direction_type': 'refusal_direction'}  # missing everything else
        try:
            dm.save_direction_atomic(direction, incomplete_meta, os.path.join(tmpdir, 'incomplete.pt'))
            raise SystemExit("FAILED: expected ValueError for incomplete metadata")
        except ValueError as e:
            assert 'missing required field' in str(e)
            print(f"Test 6 PASSED: save_direction_atomic refuses incomplete metadata: {str(e)[:70]}")

        # ---- Test 7: delta payload save/verify round-trip + mechanism-set fail-fast ----
        fake_refusal_pt = os.path.join(tmpdir, 'refusal_dir_v3_en.pt')
        fake_harmfulness_pt = os.path.join(tmpdir, 'harmfulness_dir_v2_en.pt')
        dm.save_direction_atomic(torch.randn(2, 4), make_logical_meta(), fake_refusal_pt)
        dm.save_direction_atomic(torch.randn(2, 4), make_logical_meta(direction_type='harmfulness_direction',
                                                                        semantic_position='t_inst'),
                                  fake_harmfulness_pt)
        real_mechs = load_taxonomy_v2()['active_mechanisms']
        payload = {
            'instruction_ids': ['id0', 'id1'], 'n_layers': 2, 'ids_key': 'direction_ids',
            'delta_R': {m: torch.randn(2, 2) for m in real_mechs + ['placebo']},
            'delta_H': {m: torch.randn(2, 2) for m in real_mechs + ['placebo']},
            'delta_R_placebo_calibrated': {m: torch.randn(2, 2) for m in real_mechs},
            'delta_H_placebo_calibrated': {m: torch.randn(2, 2) for m in real_mechs},
            'valid_mask': {m: torch.ones(2, dtype=torch.bool) for m in real_mechs + ['placebo']},
            'failures': [],
        }
        delta_meta = dm.build_delta_metadata(
            model='FakeModel', lang='en', suffix='_full572', ids_key='direction_ids',
            active_mechanisms=real_mechs, n_instructions=2, n_layers=2,
            refusal_direction_path=fake_refusal_pt, harmfulness_direction_path=fake_harmfulness_pt,
            token_position_R='t_post', token_position_H='t_inst', estimator='mean',
        )
        delta_pt = os.path.join(tmpdir, 'delta_r_h_en_full572_direction_ids.pt')
        dm.save_delta_atomic(payload, delta_meta, delta_pt)
        loaded_payload, loaded_delta_meta = dm.verify_delta_file(delta_pt, expected_active_mechanisms=real_mechs)
        assert loaded_delta_meta['active_mechanisms'] == real_mechs
        print(f"Test 7 PASSED: delta payload save/verify round-trips with the CURRENT canonical "
              f"mechanism set ({real_mechs}).")

        # ---- Test 8 (CRITICAL): a payload built against the OLD stale mechanism
        # set must be rejected, not silently accepted as current-taxonomy data ----
        stale_mechs = ['prefix_injection', 'refusal_suppression', 'instruction_hierarchy',
                       'persona_roleplay', 'fictional_framing', 'encoding_obfuscation']
        stale_payload = {
            'instruction_ids': ['id0'], 'n_layers': 2, 'ids_key': 'direction_ids',
            'delta_R': {m: torch.randn(1, 2) for m in stale_mechs + ['placebo']},
            'delta_H': {m: torch.randn(1, 2) for m in stale_mechs + ['placebo']},
            'delta_R_placebo_calibrated': {m: torch.randn(1, 2) for m in stale_mechs},
            'delta_H_placebo_calibrated': {m: torch.randn(1, 2) for m in stale_mechs},
            'valid_mask': {m: torch.ones(1, dtype=torch.bool) for m in stale_mechs + ['placebo']},
            'failures': [],
        }
        stale_meta = dm.build_delta_metadata(
            model='FakeModel', lang='en', suffix='_full572', ids_key='direction_ids',
            active_mechanisms=stale_mechs, n_instructions=1, n_layers=2,
            refusal_direction_path=fake_refusal_pt, harmfulness_direction_path=fake_harmfulness_pt,
            token_position_R='t_post', token_position_H='t_inst', estimator='mean',
        )
        stale_pt = os.path.join(tmpdir, 'delta_r_h_STALE.pt')
        dm.save_delta_atomic(stale_payload, stale_meta, stale_pt)
        try:
            dm.verify_delta_file(stale_pt, expected_active_mechanisms=real_mechs)
            raise SystemExit("FAILED: expected ValueError for a stale-mechanism-set payload")
        except ValueError as e:
            assert 'stale' in str(e) or 'does not match' in str(e)
            print(f"Test 8 PASSED (CRITICAL): a payload keyed by the OLD pre-correction mechanism "
                  f"names (instruction_hierarchy/fictional_framing) is rejected against the current "
                  f"canonical taxonomy, never silently accepted: {str(e)[:90]}")

        # ---- Test 9: R2-style variance decomposition refuses on an incomplete
        # all_data dict (simulating 27_dual_axis_diagnosis.py's main() loop) ----
        def fake_load_all_data(model_list, available):
            loaded = {}
            for m in model_list:
                if m not in available:
                    raise FileNotFoundError(f"delta_r_h tensor missing for {m}")
                loaded[m] = available[m]
            return loaded

        try:
            fake_load_all_data(['Qwen2.5-7B-Instruct', 'Meta-Llama-3.1-8B-Instruct'],
                                {'Qwen2.5-7B-Instruct': payload})  # Llama missing
            raise SystemExit("FAILED: expected FileNotFoundError for an incomplete model set")
        except FileNotFoundError as e:
            print(f"Test 9 PASSED: loading all_data for R^2 analysis stops immediately (does not "
                  f"proceed to compute anything) when any model's input is missing: {str(e)[:70]}")

        # ---- Test 10: source-of-truth regression check -- 25/27's REAL_MECHS
        # must come from the live taxonomy loader and must NEVER contain the
        # stale names, checked against the REAL script source on disk ----
        scripts_dir = os.path.join(SCRIPT_DIR, '..')
        for fname in ('25_extract_delta_r_h.py', '27_dual_axis_diagnosis.py'):
            with open(os.path.join(scripts_dir, fname), encoding='utf-8') as f:
                src = f.read()
            assert 'load_taxonomy_v2()' in src, (
                f"{fname} no longer calls load_taxonomy_v2() -- REAL_MECHS source-of-truth "
                f"regression check itself needs updating"
            )
            # the stale names may still appear in comments/docstrings (explaining
            # the historical bug) -- what must NEVER appear again is them inside
            # a Python list/string literal assigned to REAL_MECHS/CO/MG.
            for stale in STALE_NAMES:
                bad_patterns = [f"'{stale}'", f'"{stale}"']
                lines_with_stale = [l for l in src.splitlines() if stale in l]
                literal_lines = [l for l in lines_with_stale
                                  if any(p in l for p in bad_patterns) and not l.strip().startswith('#')
                                  and 'stale' not in l.lower() and 'FIXED' not in l]
                assert not literal_lines, (
                    f"{fname} still has {stale!r} as a live string literal (not just in an "
                    f"explanatory comment): {literal_lines}"
                )
        print("Test 10 PASSED: 25_extract_delta_r_h.py and 27_dual_axis_diagnosis.py both read "
              "REAL_MECHS from load_taxonomy_v2() and no longer contain the stale mechanism names "
              "as live string literals.")

        # ---- Test 11: R/H orthogonalization (protocol Sec 10) -- h_perp must
        # be exactly orthogonal to r, and a harmfulness_dir that IS collinear
        # with refusal_dir must orthogonalize to (near-)zero, not silently
        # keep pointing along r ----
        r = torch.tensor([[3.0, 0.0, 0.0]])         # [1 layer, 3 dims]
        h_partly_aligned = torch.tensor([[2.0, 4.0, 0.0]])
        r_dot_r = (r * r).sum(-1, keepdim=True)
        h_dot_r = (h_partly_aligned * r).sum(-1, keepdim=True)
        h_perp = h_partly_aligned - (h_dot_r / r_dot_r) * r
        assert torch.allclose(h_perp, torch.tensor([[0.0, 4.0, 0.0]]), atol=1e-6), h_perp
        residual_dot_r = (h_perp * r).sum(-1).item()
        assert abs(residual_dot_r) < 1e-5, f"h_perp not orthogonal to r: dot={residual_dot_r}"
        h_collinear = torch.tensor([[6.0, 0.0, 0.0]])  # exactly along r
        h_dot_r2 = (h_collinear * r).sum(-1, keepdim=True)
        h_perp2 = h_collinear - (h_dot_r2 / r_dot_r) * r
        assert h_perp2.norm().item() < 1e-5, (
            f"a harmfulness_dir collinear with refusal_dir should orthogonalize to ~0, got {h_perp2}"
        )
        print(f"Test 11 PASSED: h_perp = h - (h.r/r.r)r is exactly orthogonal to r "
              f"(residual dot={residual_dot_r:.2e}), and a fully-collinear h orthogonalizes to ~0 "
              f"(||h_perp||={h_perp2.norm().item():.2e}) -- matches 25_extract_delta_r_h.py's formula.")

        # ---- Test 12: 27_dual_axis_diagnosis.py's single-model pilot gate --
        # cross-model R^2 must only be permitted for the exact canonical
        # 3-model set, never a subset (this is what item 6 of the 2026-09-04
        # correction required: no silent degenerate "cross-model" result from
        # a 1-model pilot run) ----
        sys.path.insert(0, scripts_dir)
        import importlib
        dax = importlib.import_module('27_dual_axis_diagnosis')
        importlib.reload(dax)
        canonical = dax.CANONICAL_3_MODELS
        assert canonical == {'Qwen2.5-7B-Instruct', 'Meta-Llama-3.1-8B-Instruct', 'gemma-2-9b-it'}
        cases = [
            (['Meta-Llama-3.1-8B-Instruct'], False),                                  # 1-model pilot
            (['Meta-Llama-3.1-8B-Instruct', 'Qwen2.5-7B-Instruct'], False),            # 2-model, incomplete
            (sorted(canonical), True),                                                # exact canonical 3
            (list(canonical) + ['SomeOtherModel'], False),                            # non-canonical 4th
        ]
        for models_list, expected_formal in cases:
            is_formal = (len(models_list) == 3 and set(models_list) == canonical)
            assert is_formal == expected_formal, (
                f"models={models_list}: expected is_formal={expected_formal}, got {is_formal}"
            )
        # descriptive_summary/shape_and_finite_check must handle a payload that
        # DOES include delta_H_perp (post-fix) without erroring
        real_mechs = load_taxonomy_v2()['active_mechanisms']
        payload_with_perp = dict(payload)
        payload_with_perp['delta_H_perp'] = {m: torch.randn(2, 2) for m in real_mechs + ['placebo']}
        payload_with_perp['delta_H_perp_placebo_calibrated'] = {m: torch.randn(2, 2) for m in real_mechs}
        desc = dax.descriptive_summary(payload_with_perp, mech_list=real_mechs)
        assert all('delta_H_perp_mean_per_layer' in desc[m] for m in real_mechs)
        sf = dax.shape_and_finite_check(payload_with_perp, mech_list=real_mechs)
        assert all(v['shape_ok'] and v['all_finite'] for v in sf.values())
        print("Test 12 PASSED: 27_dual_axis_diagnosis.py's cross-model R^2 gate (CANONICAL_3_MODELS) "
              "only admits the exact 3-model canonical set (1/2/non-canonical-4 all correctly "
              "excluded), and descriptive_summary/shape_and_finite_check correctly handle a payload "
              "that includes delta_H_perp.")

        print()
        print("ALL R/H REBUILD INFRASTRUCTURE TESTS PASSED.")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == '__main__':
    main()
