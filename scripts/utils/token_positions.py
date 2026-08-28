"""
Explicit, semantically-named token position finding for direction extraction.

Replaces the implicit `positions=[-1]` convention (used identically for both
refusal_direction and harmfulness_direction in
experiment_thesis/scripts/extract_jailbreak_vectors.py via
pipeline/submodules/generate_directions.py's get_mean_activations) with two
distinct, explicitly-verified positions:

  t_inst  -- the last token of the user's raw instruction text itself,
             before any chat-template closing/turn-end markers. Intended
             for harmfulness_direction: "does the model recognize this
             content as harmful" should be readable right where the
             instruction itself ends, before any assistant-turn scaffolding.
  t_post  -- the last token of the fully-rendered prompt (chat template +
             generation prompt applied), i.e. the position immediately
             before the model would start generating. This is what the
             existing positions=[-1] convention already computes -- so
             t_post is NOT a new position, it's the existing one, given an
             explicit name and verification. Intended for refusal_direction:
             "will the model refuse" is a property of the point where it's
             about to start answering.

NEVER assume t_inst == t_post. They are computed independently below.

Method (model-family-agnostic, verified per-model by
scripts/audits/audit_token_positions.py before trusting it for any given
model): render the full templated prompt once, tokenize the raw instruction
text alone, and locate the raw instruction's token sequence as a contiguous
subsequence within the full templated sequence (searching from the end,
since the instruction is user content and should appear close to verbatim,
modulo tokenizer boundary effects at the seam). t_inst is the last token of
that matched span. If no exact subsequence match is found (this DOES happen
-- tokenization at a text boundary can merge/split tokens differently
depending on surrounding context), this raises rather than silently
guessing, and the model family needs an explicit adapter (see
MODEL_FAMILY_ADAPTERS below) instead of the generic subsequence search.

Every function here returns a PositionResult, never a bare int -- per the
requirement that a position without semantic metadata is not acceptable
for this project going forward.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PositionResult:
    position_index: int              # index into the full token sequence (can be negative)
    semantic_name: str                # 't_inst' or 't_post'
    token_id: int
    decoded_token: str
    model_family: str
    chat_template_hash: str
    context_before: List[str] = field(default_factory=list)  # decoded tokens, up to 5
    context_after: List[str] = field(default_factory=list)   # decoded tokens, up to 5
    method: str = 'generic_subsequence_search'  # or a named per-model adapter

    def to_dict(self):
        return {
            'position_index': self.position_index, 'semantic_name': self.semantic_name,
            'token_id': self.token_id, 'decoded_token': self.decoded_token,
            'model_family': self.model_family, 'chat_template_hash': self.chat_template_hash,
            'context_before': self.context_before, 'context_after': self.context_after,
            'method': self.method,
        }


def _chat_template_hash(tokenizer):
    import hashlib
    template_str = getattr(tokenizer, 'chat_template', None) or ''
    return hashlib.sha256(template_str.encode('utf-8')).hexdigest()[:12]


def _decode_context(tokenizer, ids, center, window=5):
    lo = max(0, center - window)
    hi = min(len(ids), center + window + 1)
    return [tokenizer.decode([t]) for t in ids[lo:hi]]


def render_full_prompt_ids(tokenizer, instruction, model_family, system=None):
    """Applies the tokenizer's OWN chat template (add_generation_prompt=True) via
    apply_chat_template. Returns token id list.

    WARNING: this is only a reasonable default for a generic, model-agnostic check
    (e.g. scripts/audits/audit_token_positions.py, which has no model_base
    instance to work with, only a bare tokenizer). It is NOT guaranteed to
    produce the same token sequence as this repo's actual generation/extraction
    pipeline (pipeline/model_utils/*_model.py's _get_tokenize_instructions_fn),
    which hand-rolls its own chat-template string per model family instead of
    using apply_chat_template. For any real activation extraction (direction
    building, paired-diff extraction, etc.), build full_ids from
    model_base.tokenize_instructions_fn's actual output and pass it in via the
    full_ids parameter below -- do not rely on this function's output matching
    the pipeline's tokenization without checking."""
    messages = []
    if system:
        messages.append({'role': 'system', 'content': system})
    messages.append({'role': 'user', 'content': instruction})
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    return ids


def get_post_instruction_position(tokenizer, instruction, model_family, system=None, full_ids=None) -> PositionResult:
    """t_post: last token of the fully-rendered prompt (existing positions=[-1] convention,
    given an explicit name here).

    full_ids: if provided, used directly instead of calling apply_chat_template --
    pass in the actual ids produced by whatever tokenization pipeline built the
    activations you're locating a position within (e.g.
    model_base.tokenize_instructions_fn(instructions=[instruction]).input_ids[0]),
    so the position is guaranteed consistent with those activations. Assumes
    full_ids has no trailing padding (i.e. was built with batch_size=1 / no
    padding) -- t_post is defined as literally the last token, so a padded
    sequence would give the wrong index."""
    if full_ids is None:
        full_ids = render_full_prompt_ids(tokenizer, instruction, model_family, system=system)
    idx = len(full_ids) - 1
    return PositionResult(
        position_index=idx, semantic_name='t_post',
        token_id=full_ids[idx], decoded_token=tokenizer.decode([full_ids[idx]]),
        model_family=model_family, chat_template_hash=_chat_template_hash(tokenizer),
        context_before=_decode_context(tokenizer, full_ids, idx)[:-1],
        context_after=[], method='last_token_of_rendered_prompt',
    )


def get_instruction_end_position(tokenizer, instruction, model_family, system=None, full_ids=None) -> PositionResult:
    """t_inst: last token of the raw instruction text, located via subsequence search
    within the full rendered prompt. Raises ValueError if no exact match is found --
    callers must not fall back to guessing; that model family needs an explicit adapter.

    full_ids: see get_post_instruction_position -- pass in the pipeline's actual
    tokenized output for real extraction; only omit for a generic apply_chat_template-based check."""
    if full_ids is None:
        full_ids = render_full_prompt_ids(tokenizer, instruction, model_family, system=system)
    instr_ids = tokenizer.encode(instruction, add_special_tokens=False)

    if not instr_ids:
        raise ValueError(f"Empty instruction token sequence for model_family={model_family}")

    # search from the end (instruction is the last user-content block before the
    # assistant turn begins) for the last occurrence of instr_ids as a contiguous
    # subsequence of full_ids
    match_start = None
    for start in range(len(full_ids) - len(instr_ids), -1, -1):
        if full_ids[start:start + len(instr_ids)] == instr_ids:
            match_start = start
            break

    if match_start is None:
        raise ValueError(
            f"Could not locate instruction token sequence as an exact contiguous "
            f"subsequence of the rendered prompt for model_family={model_family}. "
            f"This model's chat template likely alters tokenization at the "
            f"instruction/template boundary -- needs a model-specific adapter in "
            f"MODEL_FAMILY_ADAPTERS instead of the generic subsequence search. "
            f"instr_ids={instr_ids[:10]}...  full_ids={full_ids}"
        )

    idx = match_start + len(instr_ids) - 1
    return PositionResult(
        position_index=idx, semantic_name='t_inst',
        token_id=full_ids[idx], decoded_token=tokenizer.decode([full_ids[idx]]),
        model_family=model_family, chat_template_hash=_chat_template_hash(tokenizer),
        context_before=_decode_context(tokenizer, full_ids, idx)[:-1],
        context_after=_decode_context(tokenizer, full_ids, idx)[len(_decode_context(tokenizer, full_ids, idx)) // 2 + 1:],
        method='generic_subsequence_search',
    )


# Populated only if the audit (scripts/audits/audit_token_positions.py) finds the
# generic subsequence search fails validation for a given model family -- do not
# pre-populate with unverified guesses.
MODEL_FAMILY_ADAPTERS = {}


# Confirmed via scripts/audits/audit_token_positions.py (real tokenizers, all 3
# models, output/audits/token_position_audit.md) and scripts/23_extract_reference_directions.py's
# dry runs against the real pipeline: the token immediately AFTER t_inst is
# always this one, closing the user's turn. Used by get_user_turn_end_position
# below for a position-finding method that works even when the raw instruction
# text is not recoverable as a literal substring (e.g. the encoding_obfuscation
# mechanism transforms the instruction text itself) -- unlike
# get_instruction_end_position's subsequence search, which requires the literal
# instruction text to still be present.
EOT_TOKEN_STR_BY_FAMILY_SUBSTRING = [
    ('qwen', '<|im_end|>'),
    ('llama', '<|eot_id|>'),
    ('gemma', '<end_of_turn>'),
]


def _eot_token_str_for_family(model_family):
    fam_lower = model_family.lower()
    for substr, tok in EOT_TOKEN_STR_BY_FAMILY_SUBSTRING:
        if substr in fam_lower:
            return tok
    raise ValueError(
        f"No known end-of-turn token for model_family={model_family!r} -- add an "
        f"entry to EOT_TOKEN_STR_BY_FAMILY_SUBSTRING after confirming it via "
        f"scripts/audits/audit_token_positions.py for this model, don't guess."
    )


def get_user_turn_end_position(tokenizer, full_ids, model_family) -> PositionResult:
    """t_inst, located structurally (last token before the user-turn's closing
    special token) rather than via subsequence search. Works on ANY user-turn
    content, including template-wrapped or encoded/obfuscated instructions
    where the raw instruction text is no longer recoverable as a literal
    substring -- use this (not get_instruction_end_position) for templated
    conditions in delta_R/delta_H extraction. full_ids must be the actual
    tokenized sequence the activations were extracted from (e.g. from
    model_base.tokenize_instructions_fn), batch_size=1 / no padding.

    Assumes a single-turn conversation (exactly one occurrence of the
    end-of-turn token, closing the user's message) -- true for this project's
    prompts (system=None, one user turn, then the assistant generation
    prompt). Raises ValueError if that assumption doesn't hold (0 or >1
    occurrences), rather than silently picking one."""
    eot_str = _eot_token_str_for_family(model_family)
    eot_id = tokenizer.convert_tokens_to_ids(eot_str)
    if eot_id is None or eot_id == getattr(tokenizer, 'unk_token_id', None):
        raise ValueError(f"Could not resolve end-of-turn token {eot_str!r} to a valid id "
                          f"for model_family={model_family!r}")

    occurrences = [i for i, tid in enumerate(full_ids) if tid == eot_id]
    if len(occurrences) != 1:
        raise ValueError(
            f"Expected exactly 1 occurrence of the end-of-turn token {eot_str!r} "
            f"(id={eot_id}) in full_ids, found {len(occurrences)} -- this method "
            f"assumes a single-turn conversation; investigate before using it on "
            f"multi-turn or differently-structured prompts. full_ids={full_ids}"
        )
    idx = occurrences[0] - 1
    if idx < 0:
        raise ValueError(f"End-of-turn token found at position 0 -- no content before it "
                          f"to treat as t_inst.")
    return PositionResult(
        position_index=idx, semantic_name='t_inst',
        token_id=full_ids[idx], decoded_token=tokenizer.decode([full_ids[idx]]),
        model_family=model_family, chat_template_hash=_chat_template_hash(tokenizer),
        context_before=_decode_context(tokenizer, full_ids, idx)[:-1],
        context_after=_decode_context(tokenizer, full_ids, idx)[len(_decode_context(tokenizer, full_ids, idx)) // 2 + 1:],
        method='structural_end_of_turn_boundary',
    )
