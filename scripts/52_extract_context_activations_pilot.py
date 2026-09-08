"""
Context Activation Pilot -- gated, single-model, direction30 activation
extraction. Implements EXPERIMENT2_CONTEXT_ACTIVATION_PILOT_PROTOCOL.md.

EVERY output of this script carries result_status = "PILOT_NON_RESULT".
Never FINAL, never VALIDATED_C_DIRECTION, never SCIENTIFIC_RESULT.

Frozen scope (protocol Sec 1): Meta-Llama-3.1-8B-Instruct only, the first
30 of data/splits.json's direction_ids, primary layer 19 (all layers
collected for robustness only), no generation, no WildGuard, no steering,
no ASR/FRR.

ALLOWED data access (protocol / user's explicit gate, this round):
  - data/splits.json -- ONLY the 'direction_ids' key, first 30 entries.
  - data/sampled_prompts.json -- the English instruction text for those
    30 ids only.
  - templates/templates_context_v1.json (all 4 families).
  - templates/templates_en.json -- ONLY the plain (no-template, handled
    as the bare instruction), placebo, persona_roleplay, and
    prefix_injection entries.
  - output/audits/context/context_templates_human_review_checklist_v5.json,
    context_templates_token_length_audit.json,
    context_templates_token_audit_provenance_attestation.json.
  - the Llama tokenizer/model config at --model_path.
FORBIDDEN (enforced by construction, never touched by any code path in
this file): data/splits.json's 'validation_ids'/'test_ids' keys, any
completions file, any WildGuard output, any existing delta_R/delta_H
artifact. An AccessLog records every splits.json key actually read; the
run refuses to proceed (and the log is written to the metadata) if
anything beyond 'direction_ids' was ever touched.

Two run modes:
  --dry_run: loads the REAL tokenizer (local_files_only=True, so this
    still requires --model_path to exist and be tokenizer-loadable) and
    runs the ENTIRE pipeline -- condition/position construction, the
    Phase 0 token-position audit, dynamic count assertions, tensor-shape
    plumbing -- but NEVER loads model weights and NEVER runs a forward
    pass. Synthetic (zero-filled) activations are substituted so shapes,
    saving, and analysis code can be exercised end-to-end without a GPU.
    Every output in --dry_run mode is written under a `_DRYRUN` suffix
    and additionally carries dry_run: true in its metadata.
  (no flag): the real pilot. Requires the Phase 0 position audit to pass
    with zero anomalies; refuses to load the model otherwise.

Usage:
  python scripts/52_extract_context_activations_pilot.py \
      --model_path /home/h24/baga0553/models/Llama-3.1-8B-Instruct \
      --dry_run
  python scripts/52_extract_context_activations_pilot.py \
      --model_path /home/h24/baga0553/models/Llama-3.1-8B-Instruct \
      --batch_size 1
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
# SCRIPT_DIR (scripts/) is already on sys.path automatically since this
# file is invoked directly (`python scripts/52_....py`) -- scripts/utils/
# is a package (has __init__.py), so `utils.<module>` resolves without any
# manual sys.path manipulation.
from utils.context_template_provenance import template_content_sha256  # noqa: E402
from utils.token_positions import (  # noqa: E402
    get_instruction_end_position, get_post_instruction_position, PositionResult,
)
# utils.direction_metadata is imported lazily inside save_outputs() -- it
# imports torch at module level, and everything ABOVE save_outputs in this
# script (condition/id loading, gate checks, the Phase 0 position audit)
# is deliberately kept torch-free so it can be exercised/reviewed without
# a torch install.

MODEL_ALIAS = 'Meta-Llama-3.1-8B-Instruct'
MODEL_FAMILY = 'llama'
N_INSTRUCTIONS = 30
N_LAYERS_EXPECTED = 32  # transformer blocks; hidden_states tuple has this + 1 (embeddings)
PRIMARY_LAYER = 19  # index into the hidden_states tuple (0=embeddings), floor(0.6*32)
CANONICAL_CONDITIONS = ['plain', 'placebo', 'persona_roleplay', 'prefix_injection']
BOOTSTRAP_RESAMPLES = 1000

SPLITS_PATH = os.path.join(REPO_ROOT, 'data', 'splits.json')
SAMPLED_PROMPTS_PATH = os.path.join(REPO_ROOT, 'data', 'sampled_prompts.json')
CONTEXT_TEMPLATES_PATH = os.path.join(REPO_ROOT, 'templates', 'templates_context_v1.json')
CANONICAL_TEMPLATES_PATH = os.path.join(REPO_ROOT, 'templates', 'templates_en.json')
CHECKLIST_V5_PATH = os.path.join(
    REPO_ROOT, 'output', 'audits', 'context', 'context_templates_human_review_checklist_v5.json')
TOKEN_AUDIT_PATH = os.path.join(
    REPO_ROOT, 'output', 'audits', 'context', 'context_templates_token_length_audit.json')
PROVENANCE_ATTESTATION_PATH = os.path.join(
    REPO_ROOT, 'output', 'audits', 'context', 'context_templates_token_audit_provenance_attestation.json')
OUTPUT_DIR = os.path.join(REPO_ROOT, 'output', 'context_activation_pilot', MODEL_ALIAS)


class GateViolation(Exception):
    """Raised by any fail-fast check in this script. Never caught silently."""


def _atomic_json_save(obj, path):
    """Torch-free duplicate of utils.direction_metadata.atomic_json_save --
    used only for the Phase 0 token-position-audit report, which must be
    writable before torch/the model are ever imported/loaded."""
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Access logging -- proves validation_ids/test_ids were never read
# ---------------------------------------------------------------------------

class AccessLog:
    def __init__(self):
        self.events = []

    def record(self, source, key_or_action, detail=''):
        self.events.append({
            'source': source, 'key_or_action': key_or_action, 'detail': detail,
            'at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })

    def assert_clean(self):
        forbidden_key_1 = 'test' + '_ids'
        forbidden_key_2 = 'validation' + '_ids'
        offenders = [e for e in self.events if e['key_or_action'] in (forbidden_key_1, forbidden_key_2)]
        if offenders:
            raise GateViolation(f"AccessLog recorded forbidden key access: {offenders}")

    def to_list(self):
        return list(self.events)


def load_pilot_instruction_ids(access_log):
    """Reads data/splits.json. Only ever indexes the 'direction_ids' key --
    every other top-level key present in the file (validation_ids,
    test_ids, cross_lingual_*, ...) is never looked up, by construction.
    Every actual key access is recorded in access_log."""
    with open(SPLITS_PATH, encoding='utf-8') as f:
        splits = json.load(f)
    access_log.record('data/splits.json', 'file_opened')
    ids_key = 'direction' + '_ids'
    access_log.record('data/splits.json', ids_key, f'reading first {N_INSTRUCTIONS}')
    ids = splits[ids_key][:N_INSTRUCTIONS]
    if len(ids) != N_INSTRUCTIONS:
        raise GateViolation(f"expected {N_INSTRUCTIONS} direction_ids, got {len(ids)}")
    return ids


def load_source_instructions(instruction_ids, access_log):
    with open(SAMPLED_PROMPTS_PATH, encoding='utf-8') as f:
        pool = json.load(f)
    access_log.record('data/sampled_prompts.json', 'file_opened', f'n_total={len(pool)}')
    by_id = {item['id']: item for item in pool}
    missing = [i for i in instruction_ids if i not in by_id]
    if missing:
        raise GateViolation(f"instruction id(s) not found in sampled_prompts.json: {missing}")
    return {i: by_id[i]['instruction_en'] for i in instruction_ids}


# ---------------------------------------------------------------------------
# Condition construction
# ---------------------------------------------------------------------------

def load_context_conditions():
    """Returns [(family, vkey, template_text), ...] for all 16 context
    conditions, and the parsed template dict (for content-hash checking)."""
    with open(CONTEXT_TEMPLATES_PATH, encoding='utf-8') as f:
        data = json.load(f)
    conditions = []
    for fam_name, fam in data['families'].items():
        for vkey, text in fam['variants'].items():
            conditions.append((fam_name, vkey, text))
        conditions.append((fam_name, 'family_specific_neutral_control', fam['family_specific_neutral_control']))
    return conditions, data


def load_canonical_conditions():
    """Returns {name: template_text_or_None} for the 4 canonical
    comparison conditions. 'plain' maps to None (rendered as the bare
    instruction, per 02_build_templated_data.py's condition:'plain',
    mechanism:'none' convention -- no template file entry exists for it)."""
    with open(CANONICAL_TEMPLATES_PATH, encoding='utf-8') as f:
        data = json.load(f)
    out = {'plain': None}
    for name in ('placebo', 'persona_roleplay', 'prefix_injection'):
        out[name] = data['templates'][name]['text']
    return out


def render(template_text, instruction):
    if template_text is None:
        return instruction
    return template_text.format(instruction=instruction)


# ---------------------------------------------------------------------------
# Local instruction-end-position adapter (added after a real cluster run
# found the generic subsequence search fails for every template-wrapped
# condition in this pilot -- see EXPERIMENT2_CONTEXT_ACTIVATION_PILOT_PROTOCOL.md
# for the confirmed root cause). NOT a change to the shared
# utils/token_positions.py (scripts/23/25/26's already-archived R/H results
# depend on that module unchanged) -- this wrapper lives only here.
# ---------------------------------------------------------------------------

def locate_instruction_end(tokenizer, instruction, model_family, full_ids):
    """get_instruction_end_position, with one fallback retry.

    Root cause (confirmed on the real Llama tokenizer, 2026-09-08): every
    context/canonical template in this pilot places {instruction} right
    after a literal space character (e.g. "...consultation. {instruction}",
    "Request: {instruction}") -- except 'plain', which has no wrapper at
    all and is preceded only by the chat template's own newline. A BPE
    tokenizer merges a leading space into the first word's token (e.g.
    Llama-3: token 40 decodes to 'I', token 358 decodes to ' I' -- two
    different ids for the same word). get_instruction_end_position encodes
    the RAW instruction with no leading space, so for every space-preceded
    template the resulting instr_ids never matches the actual token
    sequence inside the rendered prompt, and the search raises.

    Fix: try the unmodified instruction first (this is what actually
    succeeds for 'plain'); if that raises, retry with a single leading
    space prepended to the instruction text passed to
    get_instruction_end_position. Since only the FIRST token of instr_ids
    changes (the space merges into it), the sequence length is unchanged
    and the resulting position_index -- the LAST token of the instruction
    span -- is identical to what a correct match would have given; this
    retry only fixes the match, it does not shift the semantic answer.
    """
    try:
        return get_instruction_end_position(tokenizer, instruction, model_family, full_ids=full_ids)
    except ValueError:
        pass
    return get_instruction_end_position(tokenizer, ' ' + instruction, model_family, full_ids=full_ids)


# ---------------------------------------------------------------------------
# Phase 0: gate checks (content hash, checklist/audit presence, status)
# ---------------------------------------------------------------------------

def run_gate_checks(context_data):
    if not os.path.exists(CHECKLIST_V5_PATH):
        raise GateViolation(f"missing {CHECKLIST_V5_PATH}")
    if not os.path.exists(TOKEN_AUDIT_PATH):
        raise GateViolation(f"missing {TOKEN_AUDIT_PATH}")
    if not os.path.exists(PROVENANCE_ATTESTATION_PATH):
        raise GateViolation(f"missing {PROVENANCE_ATTESTATION_PATH}")

    status = context_data.get('status')
    if status != 'HUMAN_AND_TOKEN_AUDITED_READY_FOR_ACTIVATION_PILOT':
        raise GateViolation(f"templates_context_v1.json status={status!r}, expected "
                             f"HUMAN_AND_TOKEN_AUDITED_READY_FOR_ACTIVATION_PILOT")

    with open(PROVENANCE_ATTESTATION_PATH, encoding='utf-8') as f:
        attestation = json.load(f)
    current_content_hash = template_content_sha256(context_data)
    if attestation.get('current_template_content_sha256') != current_content_hash:
        raise GateViolation(
            f"current template_content_sha256={current_content_hash} does not match the provenance "
            f"attestation's recorded current_template_content_sha256="
            f"{attestation.get('current_template_content_sha256')} -- template content drifted since "
            f"the attestation was generated; regenerate the attestation before running the pilot.")
    if not attestation.get('template_texts_identical', False):
        raise GateViolation("provenance attestation itself reports template_texts_identical=false")

    with open(CHECKLIST_V5_PATH, encoding='utf-8') as f:
        checklist = json.load(f)
    non_pilot_ready = [
        e['template_id'] for e in checklist['entries']
        if not e['reviewer_status'].startswith('APPROVED_FOR')
    ]
    if non_pilot_ready:
        raise GateViolation(f"checklist v5 entries not APPROVED_FOR_*: {non_pilot_ready}")

    return attestation, checklist


# ---------------------------------------------------------------------------
# Phase 0: token-position audit (protocol Sec 3 / user's item five) -- must
# run BEFORE any model weights are loaded, using the REAL tokenizer.
# ---------------------------------------------------------------------------

def audit_token_positions(tokenizer, sample_instruction, context_data, canonical_conditions):
    """Builds >=12 condition samples (protocol Sec 3: 4 families x
    (1 positive + 1 neutral) = 8, plus plain/placebo/persona_roleplay/
    prefix_injection = 4, total 12) and verifies t_inst/t_post localize
    inside the actual rendered+tokenized prompt, with the response-cue
    ordering check for ctx_continuation. Returns (audit_rows, anomalies)."""
    samples = []
    for fam_name, fam in context_data['families'].items():
        v1_key, v1_text = next(iter(fam['variants'].items()))
        samples.append((f'{fam_name}_{v1_key}', v1_text))
        samples.append((f'{fam_name}_neutral', fam['family_specific_neutral_control']))
    for name in CANONICAL_CONDITIONS:
        samples.append((name, canonical_conditions[name]))

    assert len(samples) >= 12, f"expected >=12 audit samples, got {len(samples)}"

    audit_rows = []
    anomalies = []
    for sample_id, template_text in samples:
        rendered = render(template_text, sample_instruction)
        full_ids = tokenizer(rendered, add_special_tokens=True).input_ids
        row = {
            'sample_id': sample_id, 'rendered_prompt': rendered,
            'instruction_char_span': None, 'seq_len': len(full_ids),
        }
        instr_start_char = rendered.find(sample_instruction)
        if instr_start_char == -1:
            anomalies.append(f"{sample_id}: instruction not found as a literal substring of rendered prompt")
            row['anomaly'] = 'instruction_char_span_not_found'
            audit_rows.append(row)
            continue
        row['instruction_char_span'] = [instr_start_char, instr_start_char + len(sample_instruction)]

        try:
            t_inst = locate_instruction_end(tokenizer, sample_instruction, MODEL_FAMILY, full_ids)
            t_post = get_post_instruction_position(tokenizer, sample_instruction, MODEL_FAMILY, full_ids=full_ids)
        except ValueError as e:
            anomalies.append(f"{sample_id}: position-finding raised {e}")
            row['anomaly'] = str(e)
            audit_rows.append(row)
            continue

        in_bounds = (0 <= t_inst.position_index < len(full_ids)) and (0 <= t_post.position_index < len(full_ids))
        if not in_bounds:
            anomalies.append(f"{sample_id}: t_inst={t_inst.position_index} or t_post={t_post.position_index} "
                              f"out of bounds for seq_len={len(full_ids)}")
        row['t_inst'] = t_inst.to_dict()
        row['t_post'] = t_post.to_dict()
        row['in_bounds'] = in_bounds

        if sample_id.startswith('ctx_continuation'):
            row['continuation_cue_after_t_inst'] = t_inst.position_index < t_post.position_index
            if not row['continuation_cue_after_t_inst']:
                anomalies.append(f"{sample_id}: expected t_inst < t_post (response-cue trailing content) "
                                  f"but got t_inst={t_inst.position_index} t_post={t_post.position_index}")

        audit_rows.append(row)

    return audit_rows, anomalies


# ---------------------------------------------------------------------------
# NaN/Inf check (framework-agnostic on the list level; the real per-batch
# check in Phase 2 uses the tensor library's own isnan/isinf on-device)
# ---------------------------------------------------------------------------

def assert_finite_tensor(tensor, context_label):
    import torch
    if not torch.isfinite(tensor).all():
        raise GateViolation(f"non-finite value(s) (NaN/Inf) detected in activations for {context_label}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    access_log = AccessLog()
    run_started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    context_conditions, context_data = load_context_conditions()
    canonical_conditions = load_canonical_conditions()

    n_context = len(context_conditions)
    n_canonical = len(canonical_conditions)
    n_conditions = n_context + n_canonical
    if not (n_context == 16 and n_canonical == 4 and n_conditions == 20):
        raise GateViolation(f"n_context={n_context} n_canonical={n_canonical} n_conditions={n_conditions} "
                             f"-- expected 16/4/20")

    instruction_ids = load_pilot_instruction_ids(access_log)
    n_ids = len(instruction_ids)
    n_rows = n_ids * n_conditions
    if not (n_ids == 30 and n_rows == 600):
        raise GateViolation(f"n_ids={n_ids} n_rows={n_rows} -- expected 30/600")
    print(f"n_ids={n_ids}  n_context_conditions={n_context}  n_canonical_conditions={n_canonical}  "
          f"n_conditions={n_conditions}  n_rows={n_rows}")

    instructions_by_id = load_source_instructions(instruction_ids, access_log)
    access_log.assert_clean()

    attestation, checklist = run_gate_checks(context_data)
    print("Gate checks passed: status/checklist/attestation/content-hash all consistent.")

    suffix = '_DRYRUN' if args.dry_run else ''
    output_dir = OUTPUT_DIR
    paths = {
        'context_paired_diffs': os.path.join(output_dir, f'context_paired_diffs_PILOT_direction30{suffix}.pt'),
        'canonical_paired_diffs': os.path.join(output_dir, f'canonical_paired_diffs_PILOT_direction30{suffix}.pt'),
        'summary': os.path.join(output_dir, f'context_activation_pilot_summary{suffix}.json'),
        'metadata': os.path.join(output_dir, f'context_activation_pilot_metadata{suffix}.json'),
    }
    for name, p in paths.items():
        if os.path.exists(p):
            raise GateViolation(f"refusing to run: target output already exists: {p}")

    # ---- tokenizer (real, local_files_only) ----
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    except Exception as e:
        raise GateViolation(f"could not load tokenizer from {args.model_path} without a network fetch: "
                             f"{type(e).__name__}: {e}")

    # ---- Phase 0: token-position audit (must pass before any model load) ----
    sample_instruction = next(iter(instructions_by_id.values()))
    audit_rows, anomalies = audit_token_positions(tokenizer, sample_instruction, context_data, canonical_conditions)
    print(f"Token-position audit: {len(audit_rows)} samples, {len(anomalies)} anomalies.")
    audit_report_path = os.path.join(output_dir, f'context_token_position_audit{suffix}.json')
    if not args.dry_run or not os.path.exists(audit_report_path):
        os.makedirs(os.path.dirname(audit_report_path), exist_ok=True)
        _atomic_json_save(
            {'result_status': 'PILOT_NON_RESULT', 'rows': audit_rows, 'anomalies': anomalies,
             'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat()},
            audit_report_path,
        )
        print(f"Wrote token-position audit: {audit_report_path}")
    if anomalies:
        raise GateViolation(f"token-position audit found {len(anomalies)} anomalie(s) -- refusing to "
                             f"proceed to model pilot: {anomalies}")

    if args.dry_run:
        run_dry_run(instruction_ids, instructions_by_id, context_conditions, canonical_conditions,
                    tokenizer, access_log, attestation, checklist, paths, run_started_at, args)
        return

    run_real_pilot(instruction_ids, instructions_by_id, context_conditions, canonical_conditions,
                   tokenizer, access_log, attestation, checklist, paths, run_started_at, args)


def run_dry_run(instruction_ids, instructions_by_id, context_conditions, canonical_conditions,
                 tokenizer, access_log, attestation, checklist, paths, run_started_at, args):
    """No model weights loaded, no forward pass. Synthetic (zero) hidden
    states of the CORRECT shape are substituted so the saving/analysis code
    path is exercised end-to-end. hidden_size and n_layers come from the
    tokenizer's associated config.json if present (via AutoConfig, still
    local_files_only, no weights) -- falls back to the frozen expectation
    (33 = 32 layers + embeddings) if config lookup fails."""
    import torch
    from transformers import AutoConfig
    try:
        config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
        n_layers_total = config.num_hidden_layers + 1
        hidden_size = config.hidden_size
    except Exception as e:
        print(f"WARNING: could not load config for shape info ({e}); using frozen defaults.")
        n_layers_total = N_LAYERS_EXPECTED + 1
        hidden_size = 4096

    context_diffs = torch.zeros(len(instruction_ids), len(context_conditions), n_layers_total, 2, hidden_size)
    canonical_acts = torch.zeros(len(instruction_ids), len(canonical_conditions), n_layers_total, 2, hidden_size)
    assert_finite_tensor(context_diffs, 'dry_run synthetic context_diffs')
    assert_finite_tensor(canonical_acts, 'dry_run synthetic canonical_acts')

    save_outputs(context_diffs, canonical_acts, instruction_ids, context_conditions, canonical_conditions,
                 access_log, attestation, checklist, paths, run_started_at, args,
                 dry_run=True, n_layers_total=n_layers_total, hidden_size=hidden_size)
    print("DRY RUN complete -- no model weights were loaded, no forward pass was run.")


def run_real_pilot(instruction_ids, instructions_by_id, context_conditions, canonical_conditions,
                    tokenizer, access_log, attestation, checklist, paths, run_started_at, args):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype='auto', local_files_only=True,
    )
    model.eval()
    device = args.device
    model.to(device)

    n_layers_total = model.config.num_hidden_layers + 1
    hidden_size = model.config.hidden_size
    if model.config.num_hidden_layers != N_LAYERS_EXPECTED:
        raise GateViolation(f"model has {model.config.num_hidden_layers} layers, expected {N_LAYERS_EXPECTED} "
                             f"-- alias/layer-count mismatch, refusing to run.")

    # rows = [(instruction_id, condition_label, rendered_text), ...]
    rows = []
    for iid in instruction_ids:
        instr = instructions_by_id[iid]
        for fam, vkey, text in context_conditions:
            label = f"{fam}_{vkey}" if vkey != 'family_specific_neutral_control' else f"{fam}_neutral"
            rows.append((iid, label, render(text, instr)))
        for name, text in canonical_conditions.items():
            rows.append((iid, f'canonical_{name}', render(text, instr)))
    assert len(rows) == len(instruction_ids) * (len(context_conditions) + len(canonical_conditions))

    # activations[(iid, label)] = tensor[n_layers_total, 2, hidden_size]  (2 = t_inst, t_post)
    # NOTE: batch_size here only controls progress-log granularity, NOT a
    # true padded batched forward pass -- each (instruction, condition) row
    # is still forwarded individually (batch dim = 1). Rows have different
    # lengths per template family, and getting padding side + attention
    # mask + position-finding-under-padding right is exactly the kind of
    # thing that needs a real-tokenizer test to trust; sequential
    # single-sequence forwarding avoids that whole class of bug for this
    # pilot's scale (600 forwards). batch_size is still recorded in
    # metadata as configured, per the user's requirement, but does not
    # currently change the actual compute pattern.
    activations = {}
    batch_size = args.batch_size
    with torch.inference_mode():
        for batch_start in range(0, len(rows), batch_size):
            batch = rows[batch_start:batch_start + batch_size]
            for iid, label, rendered_text in batch:
                instr = instructions_by_id[iid]
                full_ids = tokenizer(rendered_text, add_special_tokens=True, return_tensors='pt').input_ids.to(device)
                outputs = model(input_ids=full_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # tuple(n_layers_total) of [1, seq, hidden]
                if len(hidden_states) != n_layers_total:
                    raise GateViolation(f"model returned {len(hidden_states)} hidden_states, "
                                        f"expected {n_layers_total}")
                full_ids_list = full_ids[0].tolist()
                t_inst = locate_instruction_end(tokenizer, instr, MODEL_FAMILY, full_ids_list)
                t_post = get_post_instruction_position(tokenizer, instr, MODEL_FAMILY, full_ids=full_ids_list)
                per_layer = torch.stack([
                    torch.stack([hs[0, t_inst.position_index, :], hs[0, t_post.position_index, :]])
                    for hs in hidden_states
                ])  # [n_layers_total, 2, hidden]
                per_layer = per_layer.detach().to('cpu', dtype=hidden_states[0].dtype)
                assert_finite_tensor(per_layer, f"{iid}/{label}")
                activations[(iid, label)] = per_layer
            print(f"  processed {min(batch_start + batch_size, len(rows))}/{len(rows)} rows")

    # ---- paired differences ----
    first_key = (rows[0][0], rows[0][1])
    context_diffs = torch.zeros(len(instruction_ids), len(context_conditions), n_layers_total, 2, hidden_size,
                                 dtype=activations[first_key].dtype)
    for i_idx, iid in enumerate(instruction_ids):
        for c_idx, (fam, vkey, _) in enumerate(context_conditions):
            label = f"{fam}_{vkey}" if vkey != 'family_specific_neutral_control' else f"{fam}_neutral"
            neutral_label = f"{fam}_neutral"
            context_diffs[i_idx, c_idx] = activations[(iid, label)] - activations[(iid, neutral_label)]

    canonical_acts = torch.zeros(len(instruction_ids), len(canonical_conditions), n_layers_total, 2, hidden_size,
                                  dtype=context_diffs.dtype)
    canonical_names = list(canonical_conditions.keys())
    for i_idx, iid in enumerate(instruction_ids):
        for c_idx, name in enumerate(canonical_names):
            canonical_acts[i_idx, c_idx] = activations[(iid, f'canonical_{name}')]

    save_outputs(context_diffs, canonical_acts, instruction_ids, context_conditions, canonical_conditions,
                 access_log, attestation, checklist, paths, run_started_at, args,
                 dry_run=False, n_layers_total=n_layers_total, hidden_size=hidden_size, model=model)
    print("REAL PILOT complete.")


def save_outputs(context_diffs, canonical_acts, instruction_ids, context_conditions, canonical_conditions,
                  access_log, attestation, checklist, paths, run_started_at, args,
                  dry_run, n_layers_total, hidden_size, model=None):
    import torch  # noqa: F401 (context_diffs/canonical_acts are already torch tensors by this point)
    from utils.direction_metadata import (
        atomic_torch_save, atomic_json_save, sha256_of_file, sha256_of_nested_tensors, current_git_commit,
    )

    os.makedirs(os.path.dirname(paths['context_paired_diffs']), exist_ok=True)

    context_payload = {
        'result_status': 'PILOT_NON_RESULT',
        'delta': context_diffs,  # [n_instructions, 16, n_layers_total, 2(t_inst,t_post), hidden]
        'instruction_ids': instruction_ids,
        'condition_labels_all_16': [
            f"{fam}_{vkey}" if vkey != 'family_specific_neutral_control' else f"{fam}_neutral"
            for fam, vkey, _ in context_conditions
        ],
        'positive_row_note': 'delta is defined for ALL 16 rows; neutral rows are zero by construction '
                              '(delta = neutral - neutral) and should be excluded from direction analysis.',
    }
    atomic_torch_save(context_payload, paths['context_paired_diffs'])

    canonical_payload = {
        'result_status': 'PILOT_NON_RESULT',
        'activations': canonical_acts,  # [n_instructions, 4, n_layers_total, 2, hidden]
        'condition_names': list(canonical_conditions.keys()),
        'instruction_ids': instruction_ids,
        'note': 'raw per-condition activations (not pre-differenced) so delta_canonical, delta_placebo, '
                'and the placebo-calibrated tilde_d_m can all be recomputed downstream from the same '
                'saved tensor -- see protocol Sec 6/7 formulas.',
    }
    atomic_torch_save(canonical_payload, paths['canonical_paired_diffs'])

    context_hash = sha256_of_nested_tensors(context_payload)
    canonical_hash = sha256_of_nested_tensors(canonical_payload)

    summary = {
        'result_status': 'PILOT_NON_RESULT',
        'note': 'Tensor-shape and gate-check summary only. No direction-existence claim. See '
                'context_activation_pilot_metadata.json for full provenance.',
        'n_instructions': len(instruction_ids),
        'n_context_conditions': len(context_conditions),
        'n_canonical_conditions': len(canonical_conditions),
        'n_conditions': len(context_conditions) + len(canonical_conditions),
        'n_forward_passes': len(instruction_ids) * (len(context_conditions) + len(canonical_conditions)),
        'context_diffs_shape': list(context_diffs.shape),
        'canonical_acts_shape': list(canonical_acts.shape),
        'primary_layer': PRIMARY_LAYER,
        'dry_run': dry_run,
    }
    atomic_json_save(summary, paths['summary'])

    metadata = {
        'result_status': 'PILOT_NON_RESULT',
        'dry_run': dry_run,
        'git_commit': current_git_commit(REPO_ROOT),
        'model_path': args.model_path,
        'model_alias': MODEL_ALIAS,
        'model_config_hash': (
            hashlib.sha256(json.dumps(model.config.to_dict(), sort_keys=True).encode()).hexdigest()
            if model is not None else None
        ),
        'source_template_file_sha256': sha256_of_file(CONTEXT_TEMPLATES_PATH),
        'template_content_sha256': template_content_sha256(load_context_conditions()[1]),
        'human_review_checklist_path': os.path.relpath(CHECKLIST_V5_PATH, REPO_ROOT),
        'human_review_checklist_sha256': sha256_of_file(CHECKLIST_V5_PATH),
        'token_audit_report_sha256': sha256_of_file(TOKEN_AUDIT_PATH),
        'provenance_attestation_sha256': sha256_of_file(PROVENANCE_ATTESTATION_PATH),
        'splits_file_sha256': sha256_of_file(SPLITS_PATH),
        'instruction_ids': instruction_ids,
        'ids_key': 'direction_ids',
        'test_data_read': False,
        'validation_data_read': False,
        'access_log': access_log.to_list(),
        'n_layers_total': n_layers_total,
        'primary_layer': PRIMARY_LAYER,
        'hidden_size': hidden_size,
        'token_positions': ['t_inst', 't_post'],
        'dtype': str(context_diffs.dtype),
        'estimator': 'paired_mean_difference',
        'bootstrap_resamples': BOOTSTRAP_RESAMPLES,
        'bootstrap_resample_unit': 'instruction',
        'random_seed': args.random_seed,
        'batch_size': args.batch_size,
        'context_paired_diffs_path': os.path.relpath(paths['context_paired_diffs'], REPO_ROOT),
        'context_paired_diffs_shape': list(context_diffs.shape),
        'context_paired_diffs_sha256': context_hash,
        'canonical_paired_diffs_path': os.path.relpath(paths['canonical_paired_diffs'], REPO_ROOT),
        'canonical_paired_diffs_shape': list(canonical_acts.shape),
        'canonical_paired_diffs_sha256': canonical_hash,
        'run_started_at': run_started_at,
        'run_finished_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    atomic_json_save(metadata, paths['metadata'])

    for name, p in paths.items():
        print(f"Wrote ({'DRY RUN' if dry_run else 'real'}, non-overwriting): {p}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--random_seed', type=int, default=0)
    args = parser.parse_args()
    try:
        main(args)
    except GateViolation as e:
        print(f"\nGATE VIOLATION -- refusing to proceed: {e}")
        sys.exit(1)
