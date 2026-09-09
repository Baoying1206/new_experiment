"""
Formal C-Direction Estimation -- gated, multi-model (one model per
invocation), direction300 activation extraction. Implements
EXPERIMENT2_FORMAL_C_DIRECTION_PROTOCOL.md (which extends, and defers to,
EXPERIMENT2_CONTEXT_ACTIVATION_PILOT_PROTOCOL.md for anything not
explicitly revised).

EVERY output of this script carries result_status =
"FORMAL_DIRECTION_ESTIMATION". Never PILOT_NON_RESULT (this is the
graduation point from pilot to formal scale), never VALIDATED_C_DIRECTION
or any language claiming the C-dimension question itself is settled.

Frozen scope (formal protocol Sec 2): one of Qwen2.5-7B-Instruct /
Meta-Llama-3.1-8B-Instruct / gemma-2-9b-it per invocation (selected via
--model_alias), ALL 300 of data/splits.json's direction_ids, the same 16
context conditions as the pilot, PLUS all 6 active canonical mechanisms
(not just the pilot's 4) plus plain/placebo = 8 canonical conditions, so
20 conditions total per instruction * 300 instructions = 6000 forward
passes PER MODEL (18,000 across all 3 models) -- see Sec 3 below for why
this is NOT simply "3 x 300 x 16".

Primary token position: t_inst (per the pilot's frozen decision,
protocol Sec 13/15). t_post is collected too, as a secondary
format-position sensitivity position -- never the formal primary.

ALLOWED data access (same gate as the pilot, scaled to 300 ids):
  - data/splits.json -- ONLY the 'direction_ids' key, all 300 entries.
  - data/sampled_prompts.json -- English instruction text for those ids.
  - templates/templates_context_v1.json (all 4 families).
  - templates/templates_en.json -- 'active_mechanisms' (all 6) plus
    'plain' (bare instruction) and 'placebo'.
  - output/audits/context/context_templates_human_review_checklist_v5.json,
    context_templates_token_length_audit.json,
    context_templates_token_audit_provenance_attestation.json.
  - the tokenizer/model config at --model_path.
FORBIDDEN (enforced by construction): 'validation_ids'/'test_ids',
completions, WildGuard output, existing delta_R/delta_H artifacts. Same
AccessLog discipline as the pilot.

Two run modes, same semantics as the pilot script:
  --dry_run: real tokenizer, no model weights, no forward pass,
    zero-filled synthetic activations, `_DRYRUN` output suffix.
  (no flag): the real formal run. Requires the Phase 0 position audit
    (now covering all 4 context families' positive+neutral plus all 8
    canonical conditions = 16 samples, still >=12) to pass with zero
    anomalies before any model weights load.

Usage:
  python scripts/54_extract_formal_context_activations.py \
      --model_path /home/h24/baga0553/models/Qwen2.5-7B-Instruct \
      --model_alias Qwen2.5-7B-Instruct --dry_run
  python scripts/54_extract_formal_context_activations.py \
      --model_path /home/h24/baga0553/models/Llama-3.1-8B-Instruct \
      --model_alias Meta-Llama-3.1-8B-Instruct --batch_size 1
"""
import argparse
import base64
import datetime
import hashlib
import json
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
from utils.context_template_provenance import template_content_sha256  # noqa: E402
from utils.token_positions import (  # noqa: E402
    get_post_instruction_position, get_user_turn_end_position, PositionResult, _chat_template_hash,
)

# Two of the 6 active canonical mechanisms do NOT use a literal
# {instruction} placeholder -- encoding_obfuscation base64-encodes it
# ({instruction_b64}), payload_splitting fragments it into {payload_a}/
# {payload_b}. For BOTH, the raw instruction text is consequently not
# recoverable as a literal substring of the rendered prompt, so
# locate_instruction_end's truncate-and-compare method (which requires
# exactly that) cannot be used for their t_inst. Discovered while
# building this script (a real KeyError on encoding_obfuscation's
# template), not anticipated in the pilot (the pilot only used 4
# canonical conditions, none of which have this property).
SPECIAL_CANONICAL_MECHANISMS = {'encoding_obfuscation', 'payload_splitting'}


def split_payload(instruction):
    """Word-preserving split at the character midpoint, so a+b reassembles
    to exactly the original string. Copied verbatim from
    scripts/02_build_templated_data.py's split_payload() (not imported,
    since that module's numeric filename prefix makes it awkward to import
    normally) -- MUST stay byte-identical to that function so this
    script's payload_splitting condition matches Experiment 1's canonical
    rendering exactly; if 02's version ever changes, update this copy too."""
    words = instruction.split(' ')
    if len(words) < 2:
        mid = max(1, len(instruction) // 2)
        return instruction[:mid], instruction[mid:]
    target = len(instruction) / 2
    best_i, best_dist = 1, float('inf')
    running_len = 0
    for i in range(1, len(words)):
        running_len += len(words[i - 1]) + 1
        dist = abs(running_len - target)
        if dist < best_dist:
            best_dist, best_i = dist, i
    a = ' '.join(words[:best_i])
    b = ' ' + ' '.join(words[best_i:])
    return a, b
# utils.direction_metadata is imported lazily inside save_outputs() -- see
# scripts/52's identical rationale (torch-free above that point).

N_INSTRUCTIONS = 300
BOOTSTRAP_RESAMPLES = 2000

# Per-model config, pre-registered (R/H rebuild work) -- never re-selected
# from outcome data. n_layers is TRANSFORMER BLOCK count; hidden_states
# tuple has n_layers+1 entries (index 0 = embeddings).
MODEL_CONFIGS = {
    'Qwen2.5-7B-Instruct': {'family': 'qwen', 'n_layers': 28, 'primary_layer': 16},
    'Meta-Llama-3.1-8B-Instruct': {'family': 'llama', 'n_layers': 32, 'primary_layer': 19},
    'gemma-2-9b-it': {'family': 'gemma', 'n_layers': 42, 'primary_layer': 25},
}

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
OUTPUT_ROOT = os.path.join(REPO_ROOT, 'output', 'context_activations_formal')


class GateViolation(Exception):
    """Raised by any fail-fast check in this script. Never caught silently."""


def _atomic_json_save(obj, path):
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w') as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# Access logging
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


def load_formal_instruction_ids(access_log):
    """Reads data/splits.json. Only ever indexes 'direction_ids' -- every
    other top-level key (validation_ids, test_ids, cross_lingual_*, ...)
    is never looked up, by construction."""
    with open(SPLITS_PATH, encoding='utf-8') as f:
        splits = json.load(f)
    access_log.record('data/splits.json', 'file_opened')
    ids_key = 'direction' + '_ids'
    access_log.record('data/splits.json', ids_key, f'reading all {N_INSTRUCTIONS}')
    ids = splits[ids_key]
    if len(ids) != N_INSTRUCTIONS:
        raise GateViolation(f"expected exactly {N_INSTRUCTIONS} direction_ids, got {len(ids)} -- "
                             f"splits.json schema changed?")
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
    with open(CONTEXT_TEMPLATES_PATH, encoding='utf-8') as f:
        data = json.load(f)
    conditions = []
    for fam_name, fam in data['families'].items():
        for vkey, text in fam['variants'].items():
            conditions.append((fam_name, vkey, text))
        conditions.append((fam_name, 'family_specific_neutral_control', fam['family_specific_neutral_control']))
    return conditions, data


def load_canonical_conditions():
    """Returns {name: template_text_or_None} for 'plain' + 'placebo' +
    ALL 6 currently active canonical mechanisms (templates_en.json's own
    'active_mechanisms' list -- read dynamically, not hardcoded, so this
    script never silently drifts from the current canonical taxonomy).
    'plain' maps to None (bare instruction, no template)."""
    with open(CANONICAL_TEMPLATES_PATH, encoding='utf-8') as f:
        data = json.load(f)
    out = {'plain': None, 'placebo': data['templates']['placebo']['text']}
    for name in data['active_mechanisms']:
        out[name] = data['templates'][name]['text']
    return out


def render(template_text, instruction):
    """For context conditions (always a plain {instruction} placeholder)."""
    if template_text is None:
        return instruction
    return template_text.format(instruction=instruction)


def render_canonical(name, template_text, instruction):
    """For canonical conditions -- dispatches on the 2 special mechanisms
    (see SPECIAL_CANONICAL_MECHANISMS); everything else (including
    'plain', which has template_text=None) uses the plain {instruction}
    placeholder via render()."""
    if name == 'encoding_obfuscation':
        b64 = base64.b64encode(instruction.encode('utf-8')).decode('ascii')
        return template_text.format(instruction_b64=b64)
    if name == 'payload_splitting':
        payload_a, payload_b = split_payload(instruction)
        return template_text.format(payload_a=payload_a, payload_b=payload_b)
    return render(template_text, instruction)


# ---------------------------------------------------------------------------
# Local instruction-end-position adapter -- identical method to
# scripts/52 (longest-common-prefix of truncated-vs-full tokenization);
# duplicated here (not imported from scripts/52) so this script has no
# dependency on the pilot script and can't be broken by future pilot-only
# changes. NOT a change to utils/token_positions.py.
# ---------------------------------------------------------------------------

def locate_instruction_end(tokenizer, rendered_text, instr_char_end, full_ids, model_family):
    prefix_ids = tokenizer(rendered_text[:instr_char_end], add_special_tokens=True).input_ids
    common_len = 0
    for a, b in zip(prefix_ids, full_ids):
        if a != b:
            break
        common_len += 1
    if common_len == 0:
        raise ValueError(
            f"No common prefix at all between truncated-prompt and full-prompt tokenization for "
            f"model_family={model_family!r} -- prefix_ids[:5]={prefix_ids[:5]} full_ids[:5]={full_ids[:5]}")
    divergence_from_boundary = len(prefix_ids) - common_len
    if divergence_from_boundary > 2:
        raise ValueError(
            f"prefix/full tokenization diverge {divergence_from_boundary} tokens before the truncation "
            f"boundary for model_family={model_family!r} (expected <=2) -- refusing to guess. "
            f"prefix_ids={prefix_ids} full_ids={full_ids}")
    idx = common_len - 1
    target_text = rendered_text[:instr_char_end]
    max_extension = 3
    extended = idx
    while (extended + 1 < len(full_ids) and (extended - idx) < max_extension
           and len(tokenizer.decode(full_ids[:extended + 1])) < len(target_text)):
        extended += 1
    if extended > idx and len(tokenizer.decode(full_ids[:extended + 1])) >= len(target_text):
        idx = extended
    token_id = full_ids[idx]
    return PositionResult(
        position_index=idx, semantic_name='t_inst', token_id=token_id,
        decoded_token=tokenizer.decode([token_id]), model_family=model_family,
        chat_template_hash=_chat_template_hash(tokenizer),
        method='longest_common_prefix_of_truncated_vs_full_tokenization',
    )


# ---------------------------------------------------------------------------
# Phase 0: gate checks
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
            f"attestation's recorded value -- template content drifted; regenerate the attestation.")
    if not attestation.get('template_texts_identical', False):
        raise GateViolation("provenance attestation itself reports template_texts_identical=false")

    with open(CHECKLIST_V5_PATH, encoding='utf-8') as f:
        checklist = json.load(f)
    non_ready = [e['template_id'] for e in checklist['entries'] if not e['reviewer_status'].startswith('APPROVED_FOR')]
    if non_ready:
        raise GateViolation(f"checklist v5 entries not APPROVED_FOR_*: {non_ready}")

    return attestation, checklist


# ---------------------------------------------------------------------------
# Phase 0: token-position audit -- must pass before any model weights load.
# Covers 4 families x (1 positive + 1 neutral) = 8, plus ALL 8 canonical
# conditions (plain/placebo/6 mechanisms) = 16 samples total (>=12).
# ---------------------------------------------------------------------------

def audit_token_positions(tokenizer, sample_instruction, context_data, canonical_conditions, model_family):
    # (sample_id, template_text, is_canonical, canonical_name_or_None)
    samples = []
    for fam_name, fam in context_data['families'].items():
        v1_key, v1_text = next(iter(fam['variants'].items()))
        samples.append((f'{fam_name}_{v1_key}', v1_text, False, None))
        samples.append((f'{fam_name}_neutral', fam['family_specific_neutral_control'], False, None))
    for name, text in canonical_conditions.items():
        samples.append((name, text, True, name))

    assert len(samples) >= 12, f"expected >=12 audit samples, got {len(samples)}"

    audit_rows, anomalies = [], []
    for sample_id, template_text, is_canonical, canon_name in samples:
        special = is_canonical and canon_name in SPECIAL_CANONICAL_MECHANISMS
        rendered = (render_canonical(canon_name, template_text, sample_instruction) if is_canonical
                    else render(template_text, sample_instruction))
        full_ids = tokenizer(rendered, add_special_tokens=True).input_ids
        row = {'sample_id': sample_id, 'rendered_prompt': rendered,
               'instruction_char_span': None, 'seq_len': len(full_ids), 'special_encoding': special}

        t_post = get_post_instruction_position(tokenizer, sample_instruction, model_family, full_ids=full_ids)

        if special:
            # raw instruction is NOT a literal substring (base64-encoded or
            # fragmented) -- t_inst uses the structural end-of-user-turn
            # method instead, which does not need the literal text.
            try:
                t_inst = get_user_turn_end_position(tokenizer, full_ids, model_family)
                t_inst_method = 'structural_end_of_turn_boundary (special encoding, no literal instruction span)'
            except ValueError as e:
                anomalies.append(f"{sample_id}: structural t_inst position-finding raised {e}")
                row['anomaly'] = str(e)
                audit_rows.append(row)
                continue
        else:
            instr_start_char = rendered.find(sample_instruction)
            if instr_start_char == -1:
                anomalies.append(f"{sample_id}: instruction not found as a literal substring of rendered prompt")
                row['anomaly'] = 'instruction_char_span_not_found'
                audit_rows.append(row)
                continue
            instr_end_char = instr_start_char + len(sample_instruction)
            row['instruction_char_span'] = [instr_start_char, instr_end_char]
            try:
                t_inst = locate_instruction_end(tokenizer, rendered, instr_end_char, full_ids, model_family)
                t_inst_method = t_inst.method
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
        row['t_inst_method'] = t_inst_method
        row['t_post'] = t_post.to_dict()
        row['in_bounds'] = in_bounds
        if sample_id.startswith('ctx_continuation'):
            row['continuation_cue_after_t_inst'] = t_inst.position_index < t_post.position_index
        audit_rows.append(row)

    return audit_rows, anomalies


def assert_finite_tensor(tensor, context_label):
    import torch
    if not torch.isfinite(tensor).all():
        raise GateViolation(f"non-finite value(s) (NaN/Inf) detected in activations for {context_label}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    if args.model_alias not in MODEL_CONFIGS:
        raise GateViolation(f"model_alias={args.model_alias!r} not in MODEL_CONFIGS "
                             f"({sorted(MODEL_CONFIGS)}) -- refusing to run with an unregistered model.")
    cfg = MODEL_CONFIGS[args.model_alias]
    model_family, n_layers_expected, primary_layer = cfg['family'], cfg['n_layers'], cfg['primary_layer']

    access_log = AccessLog()
    run_started_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    context_conditions, context_data = load_context_conditions()
    canonical_conditions = load_canonical_conditions()

    n_context = len(context_conditions)
    n_canonical = len(canonical_conditions)
    n_conditions = n_context + n_canonical
    if n_context != 16:
        raise GateViolation(f"n_context={n_context}, expected 16")
    if n_canonical != 8:
        raise GateViolation(f"n_canonical={n_canonical} (plain+placebo+6 mechanisms), expected 8 -- "
                             f"templates_en.json's active_mechanisms list changed?")

    instruction_ids = load_formal_instruction_ids(access_log)
    n_ids = len(instruction_ids)
    n_rows = n_ids * n_conditions
    if n_ids != N_INSTRUCTIONS:
        raise GateViolation(f"n_ids={n_ids}, expected {N_INSTRUCTIONS}")
    print(f"model={args.model_alias}  n_ids={n_ids}  n_context_conditions={n_context}  "
          f"n_canonical_conditions={n_canonical}  n_conditions={n_conditions}  n_rows={n_rows}")

    instructions_by_id = load_source_instructions(instruction_ids, access_log)
    access_log.assert_clean()

    attestation, checklist = run_gate_checks(context_data)
    print("Gate checks passed: status/checklist/attestation/content-hash all consistent.")

    suffix = '_DRYRUN' if args.dry_run else ''
    output_dir = os.path.join(OUTPUT_ROOT, args.model_alias)
    paths = {
        'context_paired_diffs': os.path.join(output_dir, f'context_paired_diffs_FORMAL_direction300{suffix}.pt'),
        'canonical_activations': os.path.join(output_dir, f'canonical_activations_FORMAL_direction300{suffix}.pt'),
        'summary': os.path.join(output_dir, f'context_activation_formal_summary{suffix}.json'),
        'metadata': os.path.join(output_dir, f'context_activation_formal_metadata{suffix}.json'),
        'token_position_audit': os.path.join(output_dir, f'context_token_position_audit_FORMAL{suffix}.json'),
    }
    for name, p in paths.items():
        if os.path.exists(p):
            raise GateViolation(f"refusing to run: target output already exists: {p}")

    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    except Exception as e:
        raise GateViolation(f"could not load tokenizer from {args.model_path} without a network fetch: "
                             f"{type(e).__name__}: {e}")

    sample_instruction = next(iter(instructions_by_id.values()))
    audit_rows, anomalies = audit_token_positions(
        tokenizer, sample_instruction, context_data, canonical_conditions, model_family)
    print(f"Token-position audit: {len(audit_rows)} samples, {len(anomalies)} anomalies.")
    audit_report_path = paths['token_position_audit']
    os.makedirs(os.path.dirname(audit_report_path), exist_ok=True)
    _atomic_json_save(
        {'result_status': 'FORMAL_DIRECTION_ESTIMATION', 'model_alias': args.model_alias,
         'rows': audit_rows, 'anomalies': anomalies,
         'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat()},
        audit_report_path,
    )
    print(f"Wrote token-position audit: {audit_report_path}")
    if anomalies:
        raise GateViolation(f"token-position audit found {len(anomalies)} anomalie(s) -- refusing to "
                             f"proceed to model extraction: {anomalies}")

    if args.dry_run:
        run_dry_run(instruction_ids, context_conditions, canonical_conditions, tokenizer, access_log,
                    paths, run_started_at, args, model_family, n_layers_expected, primary_layer)
        return

    run_real_extraction(instruction_ids, instructions_by_id, context_conditions, canonical_conditions,
                         tokenizer, access_log, paths, run_started_at, args, model_family,
                         n_layers_expected, primary_layer)


def run_dry_run(instruction_ids, context_conditions, canonical_conditions, tokenizer, access_log,
                 paths, run_started_at, args, model_family, n_layers_expected, primary_layer):
    import torch
    from transformers import AutoConfig
    try:
        config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
        n_layers_total = config.num_hidden_layers + 1
        hidden_size = config.hidden_size
    except Exception as e:
        print(f"WARNING: could not load config for shape info ({e}); using frozen defaults.")
        n_layers_total = n_layers_expected + 1
        hidden_size = 4096

    context_diffs = torch.zeros(len(instruction_ids), len(context_conditions), n_layers_total, 2, hidden_size)
    canonical_acts = torch.zeros(len(instruction_ids), len(canonical_conditions), n_layers_total, 2, hidden_size)
    assert_finite_tensor(context_diffs, 'dry_run synthetic context_diffs')
    assert_finite_tensor(canonical_acts, 'dry_run synthetic canonical_acts')

    save_outputs(context_diffs, canonical_acts, instruction_ids, context_conditions, canonical_conditions,
                 access_log, paths, run_started_at, args, dry_run=True, n_layers_total=n_layers_total,
                 hidden_size=hidden_size, primary_layer=primary_layer)
    print("DRY RUN complete -- no model weights were loaded, no forward pass was run.")


def run_real_extraction(instruction_ids, instructions_by_id, context_conditions, canonical_conditions,
                         tokenizer, access_log, paths, run_started_at, args, model_family,
                         n_layers_expected, primary_layer):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype='auto', local_files_only=True)
    model.eval()
    model.to(args.device)

    n_layers_total = model.config.num_hidden_layers + 1
    hidden_size = model.config.hidden_size
    if model.config.num_hidden_layers != n_layers_expected:
        raise GateViolation(f"model has {model.config.num_hidden_layers} layers, expected "
                             f"{n_layers_expected} for {args.model_alias} -- refusing to run.")

    rows = []
    canonical_names = list(canonical_conditions.keys())
    for iid in instruction_ids:
        instr = instructions_by_id[iid]
        for fam, vkey, text in context_conditions:
            label = f"{fam}_{vkey}" if vkey != 'family_specific_neutral_control' else f"{fam}_neutral"
            rows.append((iid, label, render(text, instr), False))
        for name in canonical_names:
            special = name in SPECIAL_CANONICAL_MECHANISMS
            rows.append((iid, f'canonical_{name}', render_canonical(name, canonical_conditions[name], instr), special))
    assert len(rows) == len(instruction_ids) * (len(context_conditions) + len(canonical_conditions))

    activations = {}
    batch_size = args.batch_size
    with torch.inference_mode():
        for batch_start in range(0, len(rows), batch_size):
            batch = rows[batch_start:batch_start + batch_size]
            for iid, label, rendered_text, special in batch:
                instr = instructions_by_id[iid]
                full_ids = tokenizer(rendered_text, add_special_tokens=True,
                                      return_tensors='pt').input_ids.to(args.device)
                outputs = model(input_ids=full_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states
                if len(hidden_states) != n_layers_total:
                    raise GateViolation(f"model returned {len(hidden_states)} hidden_states, "
                                        f"expected {n_layers_total}")
                full_ids_list = full_ids[0].tolist()
                if special:
                    t_inst = get_user_turn_end_position(tokenizer, full_ids_list, model_family)
                else:
                    instr_start_char = rendered_text.find(instr)
                    if instr_start_char == -1:
                        raise GateViolation(f"{iid}/{label}: instruction not found as a literal substring "
                                            f"of rendered prompt -- render() bug?")
                    instr_end_char = instr_start_char + len(instr)
                    t_inst = locate_instruction_end(
                        tokenizer, rendered_text, instr_end_char, full_ids_list, model_family)
                t_post = get_post_instruction_position(tokenizer, instr, model_family, full_ids=full_ids_list)
                per_layer = torch.stack([
                    torch.stack([hs[0, t_inst.position_index, :], hs[0, t_post.position_index, :]])
                    for hs in hidden_states
                ])
                per_layer = per_layer.detach().to('cpu', dtype=hidden_states[0].dtype)
                assert_finite_tensor(per_layer, f"{iid}/{label}")
                activations[(iid, label)] = per_layer
            if (batch_start // batch_size) % 50 == 0:
                print(f"  processed {min(batch_start + batch_size, len(rows))}/{len(rows)} rows")
    print(f"  processed {len(rows)}/{len(rows)} rows")

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
    for i_idx, iid in enumerate(instruction_ids):
        for c_idx, name in enumerate(canonical_names):
            canonical_acts[i_idx, c_idx] = activations[(iid, f'canonical_{name}')]

    save_outputs(context_diffs, canonical_acts, instruction_ids, context_conditions, canonical_conditions,
                 access_log, paths, run_started_at, args, dry_run=False, n_layers_total=n_layers_total,
                 hidden_size=hidden_size, primary_layer=primary_layer, model=model)
    print("REAL FORMAL EXTRACTION complete.")


def save_outputs(context_diffs, canonical_acts, instruction_ids, context_conditions, canonical_conditions,
                  access_log, paths, run_started_at, args, dry_run, n_layers_total, hidden_size,
                  primary_layer, model=None):
    import torch  # noqa: F401
    from utils.direction_metadata import (
        atomic_torch_save, atomic_json_save, sha256_of_file, sha256_of_nested_tensors, current_git_commit,
    )

    os.makedirs(os.path.dirname(paths['context_paired_diffs']), exist_ok=True)

    context_payload = {
        'result_status': 'FORMAL_DIRECTION_ESTIMATION',
        'delta': context_diffs,
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
        'result_status': 'FORMAL_DIRECTION_ESTIMATION',
        'activations': canonical_acts,
        'condition_names': list(canonical_conditions.keys()),
        'instruction_ids': instruction_ids,
        'note': 'raw per-condition activations (not pre-differenced) -- delta_canonical, delta_placebo, '
                'and placebo-calibrated tilde_d_m are all recomputed downstream from this tensor.',
    }
    atomic_torch_save(canonical_payload, paths['canonical_activations'])

    context_hash = sha256_of_nested_tensors(context_payload)
    canonical_hash = sha256_of_nested_tensors(canonical_payload)

    summary = {
        'result_status': 'FORMAL_DIRECTION_ESTIMATION',
        'model_alias': args.model_alias,
        'n_instructions': len(instruction_ids),
        'n_context_conditions': len(context_conditions),
        'n_canonical_conditions': len(canonical_conditions),
        'n_conditions': len(context_conditions) + len(canonical_conditions),
        'n_forward_passes': len(instruction_ids) * (len(context_conditions) + len(canonical_conditions)),
        'context_diffs_shape': list(context_diffs.shape),
        'canonical_acts_shape': list(canonical_acts.shape),
        'primary_layer': primary_layer,
        'primary_token_position': 't_inst',
        'secondary_token_position': 't_post',
        'dry_run': dry_run,
    }
    atomic_json_save(summary, paths['summary'])

    metadata = {
        'result_status': 'FORMAL_DIRECTION_ESTIMATION',
        'dry_run': dry_run,
        'git_commit': current_git_commit(REPO_ROOT),
        'model_path': args.model_path,
        'model_alias': args.model_alias,
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
        'primary_layer': primary_layer,
        'primary_token_position': 't_inst',
        'secondary_token_position': 't_post',
        'hidden_size': hidden_size,
        'token_positions': ['t_inst', 't_post'],
        'dtype': str(context_diffs.dtype),
        'estimator': 'mean',
        'bootstrap_resamples': BOOTSTRAP_RESAMPLES,
        'bootstrap_resample_unit': 'instruction_normalized_text_cluster',
        'random_seed': args.random_seed,
        'batch_size': args.batch_size,
        'context_paired_diffs_path': os.path.relpath(paths['context_paired_diffs'], REPO_ROOT),
        'context_paired_diffs_shape': list(context_diffs.shape),
        'context_paired_diffs_sha256': context_hash,
        'canonical_activations_path': os.path.relpath(paths['canonical_activations'], REPO_ROOT),
        'canonical_activations_shape': list(canonical_acts.shape),
        'canonical_activations_sha256': canonical_hash,
        'run_started_at': run_started_at,
        'run_finished_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    atomic_json_save(metadata, paths['metadata'])

    for name, p in paths.items():
        if name == 'token_position_audit':
            continue
        print(f"Wrote ({'DRY RUN' if dry_run else 'real'}, non-overwriting): {p}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_alias', type=str, required=True, choices=sorted(MODEL_CONFIGS))
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
