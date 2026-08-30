"""
Core, GPU-free building blocks for the Exp3 defence protocol: direction
construction (reusing 35_common_direction_coverage_audit.py's exact
formulas) and the prefill-only, last-token, single-fire steering hook.

This module has NO dependency on `pipeline`/transformers -- it only needs
torch -- so it is importable and unit-testable on a laptop. The actual
generation driver (a later script) will import `make_prefill_last_token_hook`
and pass it into `pipeline`'s existing `add_hooks`/`generate_completions`
machinery (pipeline/utils/hook_utils.py, pipeline/model_utils/model_base.py)
rather than reimplementing batching/tokenization/generation from scratch --
confirmed by reading that code: it already left-pads
(tokenizer.padding_side='left' in all 3 model_utils/*_model.py files) and
already accepts fwd_hooks=[(module, fn)] into `model.generate(...)`. What it
does NOT already provide is a hook that (a) fires only once per generate()
call and (b) does additive (not projection-ablation) steering only at the
last prefill position -- pipeline/utils/hook_utils.py's existing
get_direction_ablation_*_hook fires on every forward call (prefill AND every
decode step), which is a different intervention semantics than this
protocol's single-shot h' = h - alpha*c_G and must not be reused as-is.

Direction definitions (frozen, from the Exp3 protocol):
  g_G = normalize(mean_{m in G}(normalize(dtilde_m)))
  s_G = median_{m in G}(||dtilde_m||)
  c_G = s_G * g_G
  Global:   G = all 6 active mechanisms
  Fixed Wei: G = CO or G = MG (full 3-member groups, as in script 35)
  Adaptive: G = the model-specific REDUCED subgroup (frozen membership below);
            template-specific ("boundary") members use c_m = dtilde_m directly
  Placebo:  c_P = d_placebo (raw, unscaled -- not put through the g/s recipe)
  Template-specific (oracle, NOT in the main experiment): c_m = dtilde_m

Frozen adaptive grouping (from script 35's real bootstrap results, A_k 95%
CI upper bound < 0 test; verified consistent with
output/canonical_v2/experiment3_common_direction_bootstrap.json):
  Qwen2.5-7B-Instruct:        template_specific=[prefix_injection]
                              subgroups={CO_reduced:[refusal_suppression,persona_roleplay],
                                         MG_full:[encoding_obfuscation,payload_splitting,distractors_negated]}
  Meta-Llama-3.1-8B-Instruct: template_specific=[refusal_suppression,distractors_negated]
                              subgroups={CO_reduced:[prefix_injection,persona_roleplay],
                                         MG_reduced:[encoding_obfuscation,payload_splitting]}
  gemma-2-9b-it:              template_specific=[prefix_injection]
                              subgroups={CO_reduced:[refusal_suppression,persona_roleplay],
                                         MG_full:[encoding_obfuscation,payload_splitting,distractors_negated]}

Usage contract for the hook (IMPORTANT -- read before wiring into generation):
  `make_prefill_last_token_hook(...)` must be called ONCE PER `generate()`
  INVOCATION, producing a fresh hook closure + state dict each time. Reusing
  one hook object across multiple generate() calls will carry over
  `has_intervened=True` from the previous call and silently skip the
  intervention on the next one.
"""
import torch


ACTIVE_MECHANISMS = ['prefix_injection', 'refusal_suppression', 'persona_roleplay',
                      'encoding_obfuscation', 'payload_splitting', 'distractors_negated']
CO_MECHS = ['prefix_injection', 'refusal_suppression', 'persona_roleplay']
MG_MECHS = ['encoding_obfuscation', 'payload_splitting', 'distractors_negated']

FROZEN_ADAPTIVE_GROUPING = {
    'Qwen2.5-7B-Instruct': {
        'template_specific': ['prefix_injection'],
        'subgroups': {
            'CO_reduced': ['refusal_suppression', 'persona_roleplay'],
            'MG_full': ['encoding_obfuscation', 'payload_splitting', 'distractors_negated'],
        },
    },
    'Meta-Llama-3.1-8B-Instruct': {
        'template_specific': ['refusal_suppression', 'distractors_negated'],
        'subgroups': {
            'CO_reduced': ['prefix_injection', 'persona_roleplay'],
            'MG_reduced': ['encoding_obfuscation', 'payload_splitting'],
        },
    },
    'gemma-2-9b-it': {
        'template_specific': ['prefix_injection'],
        'subgroups': {
            'CO_reduced': ['refusal_suppression', 'persona_roleplay'],
            'MG_full': ['encoding_obfuscation', 'payload_splitting', 'distractors_negated'],
        },
    },
}

for _model, _cfg in FROZEN_ADAPTIVE_GROUPING.items():
    _members = list(_cfg['template_specific'])
    for _sg in _cfg['subgroups'].values():
        _members.extend(_sg)
    assert sorted(_members) == sorted(ACTIVE_MECHANISMS), (
        f"{_model}: adaptive grouping does not partition all 6 active mechanisms exactly once: {sorted(_members)}"
    )


def normalize(v):
    n = v.norm()
    return v / n if n > 0 else v


def build_c_G(dtilde, group):
    """g_G = normalize(mean(normalize(dtilde_m))); s_G = median(||dtilde_m||); c_G = s_G * g_G."""
    us = [normalize(dtilde[m]) for m in group]
    g_G = normalize(torch.stack(us, 0).mean(0))
    norms = torch.tensor([dtilde[m].norm().item() for m in group])
    s_G = norms.median()
    return s_G * g_G


def build_c_placebo(d_placebo):
    """Placebo uses its own raw paired-diff direction -- no unit/median rescaling."""
    return d_placebo


def build_c_template_specific(dtilde, mechanism):
    """c_m = dtilde_m directly (oracle upper bound; also used for Adaptive's boundary members)."""
    return dtilde[mechanism]


def build_all_condition_directions(dtilde, model_alias):
    """Returns {condition_name: {mechanism_or_'*': c_vector}} for the 5 main conditions
    (Template-specific oracle intentionally excluded -- not in the main experiment)."""
    grouping = FROZEN_ADAPTIVE_GROUPING[model_alias]
    out = {}
    out['placebo'] = {'*': build_c_placebo(dtilde['placebo'])}
    out['global'] = {'*': build_c_G(dtilde, ACTIVE_MECHANISMS)}
    out['fixed_wei'] = {
        **{m: build_c_G(dtilde, CO_MECHS) for m in CO_MECHS},
        **{m: build_c_G(dtilde, MG_MECHS) for m in MG_MECHS},
    }
    adaptive = {}
    for m in grouping['template_specific']:
        adaptive[m] = build_c_template_specific(dtilde, m)
    for sg_members in grouping['subgroups'].values():
        c_sub = build_c_G(dtilde, sg_members)
        for m in sg_members:
            adaptive[m] = c_sub
    assert sorted(adaptive.keys()) == sorted(ACTIVE_MECHANISMS)
    out['adaptive'] = adaptive
    return out


# ── the hook itself ────────────────────────────────────────────────────

def make_prefill_last_token_hook(per_row_c_vectors, audit_log=None):
    """
    per_row_c_vectors: [batch_size, d_model] tensor, row i already equals
        alpha * c_G for prompt i in the CURRENT batch (scaling and per-row
        routing must be done by the caller before this call -- the hook
        itself is routing-agnostic).
    audit_log: optional list; hook appends short strings to it for
        non-fatal anomalies (e.g. firing before any multi-token call seen).

    Returns (hook_fn, state). state['has_intervened'] / state['intervention_count']
    must be inspected by the caller after generate() returns; call this
    factory fresh for every generate() invocation (see module docstring).
    """
    state = {'has_intervened': False, 'intervention_count': 0, 'prefill_seq_len_seen': None}

    def hook_fn(module, input, output):
        if state['has_intervened']:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        seq_len = hidden.shape[1]
        if seq_len == 1:
            if audit_log is not None:
                audit_log.append(
                    "WARNING: hook fired with seq_len==1 before any multi-token "
                    "(prefill) call was observed -- skipping intervention this call."
                )
            return output
        vec = per_row_c_vectors.to(dtype=hidden.dtype, device=hidden.device)
        hidden = hidden.clone()
        hidden[:, -1, :] = hidden[:, -1, :] - vec
        state['has_intervened'] = True
        state['intervention_count'] += 1
        state['prefill_seq_len_seen'] = seq_len
        if isinstance(output, tuple):
            return (hidden, *output[1:])
        return hidden

    return hook_fn, state


def assert_single_intervention(state):
    assert state['intervention_count'] == 1, (
        f"Expected exactly one intervention per generate() call, got "
        f"intervention_count={state['intervention_count']} (has_intervened={state['has_intervened']}). "
        f"This means either the hook never fired (direction not applied) or fired more than "
        f"once (decode-step leakage) -- both are protocol violations."
    )


def assert_left_padded(attention_mask):
    """attention_mask[:, -1] must be all 1s under left-padding (the last column is
    always a real token, never padding) -- cheap sanity check before generation."""
    last_col = attention_mask[:, -1]
    assert torch.all(last_col == 1), (
        f"attention_mask[:, -1] is not all 1s -- tokenizer.padding_side is not "
        f"'left', or an unexpected all-padding row exists: {last_col.tolist()}"
    )
