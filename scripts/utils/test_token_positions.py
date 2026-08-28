"""
CPU-only, no-model-required unit tests for token_positions.py's core logic
(the subsequence-search algorithm and PositionResult metadata contract).
Uses a synthetic mock tokenizer that simulates chat-template rendering for
three model-family-like conventions (Qwen/im_start style, Llama/header
style, Gemma/start_of_turn style) so the generic subsequence-search
approach can be checked WITHOUT cluster access. This does NOT replace
audit_token_positions.py's real-tokenizer audit -- it only proves the
matching algorithm is correct given a template; it cannot prove real
tokenizers segment text the way these mocks assume.

Run: python -m pytest scripts/utils/test_token_positions.py -v
  or: python scripts/utils/test_token_positions.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from token_positions import (
    get_instruction_end_position, get_post_instruction_position, PositionResult,
)


class MockTokenizer:
    """Word-level toy tokenizer with a template's special tokens as atomic
    'words'. Good enough to exercise the subsequence-search logic; not a
    stand-in for real BPE tokenization boundary effects (that's exactly why
    audit_token_positions.py must still run on real tokenizers)."""

    def __init__(self, template_style):
        self.template_style = template_style
        self.vocab = {}
        self.rev_vocab = {}
        self.all_special_ids = set()
        self.chat_template = f'mock-{template_style}-template'
        for tok in self._special_tokens():
            self._id_for(tok, special=True)

    def _special_tokens(self):
        if self.template_style == 'qwen':
            return ['<|im_start|>', '<|im_end|>', 'system', 'user', 'assistant', '\n']
        if self.template_style == 'llama':
            return ['<|start_header_id|>', '<|end_header_id|>', '<|eot_id|>', 'user', 'assistant', '\n\n']
        if self.template_style == 'gemma':
            return ['<start_of_turn>', '<end_of_turn>', 'user', 'model', '\n']
        raise ValueError(self.template_style)

    def _id_for(self, word, special=False):
        if word not in self.vocab:
            new_id = len(self.vocab)
            self.vocab[word] = new_id
            self.rev_vocab[new_id] = word
            if special:
                self.all_special_ids.add(new_id)
        return self.vocab[word]

    def _tokenize_words(self, text):
        # naive whitespace split, each word is a token -- fine for a mock
        return text.split(' ')

    def encode(self, text, add_special_tokens=False):
        return [self._id_for(w) for w in self._tokenize_words(text) if w != '']

    def decode(self, ids):
        return ' '.join(self.rev_vocab[i] for i in ids)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        parts = []
        for m in messages:
            if self.template_style == 'qwen':
                parts += ['<|im_start|>', m['role'], '\n'] + self._tokenize_words(m['content']) + ['<|im_end|>', '\n']
            elif self.template_style == 'llama':
                parts += ['<|start_header_id|>', m['role'], '<|end_header_id|>', '\n\n'] + \
                         self._tokenize_words(m['content']) + ['<|eot_id|>']
            elif self.template_style == 'gemma':
                parts += ['<start_of_turn>', m['role'], '\n'] + self._tokenize_words(m['content']) + \
                         ['<end_of_turn>', '\n']
        if add_generation_prompt:
            if self.template_style == 'qwen':
                parts += ['<|im_start|>', 'assistant', '\n']
            elif self.template_style == 'llama':
                parts += ['<|start_header_id|>', 'assistant', '<|end_header_id|>', '\n\n']
            elif self.template_style == 'gemma':
                parts += ['<start_of_turn>', 'model', '\n']
        return [self._id_for(w) for w in parts if w != '']


def _check_basic(style, instruction):
    tok = MockTokenizer(style)
    t_inst = get_instruction_end_position(tok, instruction, style)
    t_post = get_post_instruction_position(tok, instruction, style)
    assert isinstance(t_inst, PositionResult)
    assert isinstance(t_post, PositionResult)
    assert t_inst.semantic_name == 't_inst'
    assert t_post.semantic_name == 't_post'
    last_instruction_word = instruction.split(' ')[-1]
    assert t_inst.decoded_token == last_instruction_word, (
        f"[{style}] t_inst decoded to {t_inst.decoded_token!r}, expected last "
        f"instruction word {last_instruction_word!r}"
    )
    assert t_post.position_index != t_inst.position_index, (
        f"[{style}] t_inst and t_post must never be computed as the same position"
    )
    full_ids = tok.apply_chat_template(
        [{'role': 'user', 'content': instruction}], add_generation_prompt=True)
    assert t_post.position_index == len(full_ids) - 1
    return t_inst, t_post


def test_qwen_style():
    _check_basic('qwen', 'Explain how photosynthesis works')


def test_llama_style():
    _check_basic('llama', 'Explain how photosynthesis works')


def test_gemma_style():
    _check_basic('gemma', 'Explain how photosynthesis works')


def test_t_inst_before_t_post_in_sequence():
    # t_inst must always come strictly before t_post -- the instruction ends
    # before the assistant generation prompt begins.
    for style in ['qwen', 'llama', 'gemma']:
        t_inst, t_post = _check_basic(style, 'What is the capital of France')
        assert t_inst.position_index < t_post.position_index, (
            f"[{style}] t_inst ({t_inst.position_index}) must precede "
            f"t_post ({t_post.position_index})"
        )


def test_varying_instruction_length():
    for style in ['qwen', 'llama', 'gemma']:
        _check_basic(style, 'Short')
        _check_basic(style, 'A much longer instruction with many more words in it than the short one above')


def test_no_bare_int_returned():
    tok = MockTokenizer('qwen')
    result = get_instruction_end_position(tok, 'Explain how photosynthesis works', 'qwen')
    d = result.to_dict()
    for required_field in ['position_index', 'semantic_name', 'token_id', 'decoded_token',
                            'model_family', 'chat_template_hash', 'method']:
        assert required_field in d, f"missing required metadata field: {required_field}"


def test_full_ids_injection_matches_default():
    # a caller passing pre-built full_ids (e.g. from a pipeline's own
    # tokenize_instructions_fn) must get the same positions as the default
    # apply_chat_template path, when the ids are actually identical -- this
    # is the mechanism real direction extraction relies on to avoid a
    # tokenization-pipeline mismatch between position-finding and the
    # activations actually being indexed.
    for style in ['qwen', 'llama', 'gemma']:
        tok = MockTokenizer(style)
        instr = 'What is the capital of France'
        default_ids = tok.apply_chat_template(
            [{'role': 'user', 'content': instr}], add_generation_prompt=True)
        t_inst_default = get_instruction_end_position(tok, instr, style)
        t_post_default = get_post_instruction_position(tok, instr, style)
        t_inst_injected = get_instruction_end_position(tok, instr, style, full_ids=default_ids)
        t_post_injected = get_post_instruction_position(tok, instr, style, full_ids=default_ids)
        assert t_inst_default.position_index == t_inst_injected.position_index
        assert t_post_default.position_index == t_post_injected.position_index


def test_full_ids_injection_uses_injected_not_default():
    # if the injected full_ids differs from what apply_chat_template would
    # produce, the result must reflect the INJECTED sequence, not silently
    # fall back to recomputing via apply_chat_template.
    tok = MockTokenizer('qwen')
    instr = 'What is the capital of France'
    real_ids = tok.apply_chat_template(
        [{'role': 'user', 'content': instr}], add_generation_prompt=True)
    padded_ids = real_ids + [tok._id_for('\n')]  # append one extra token
    t_post_injected = get_post_instruction_position(tok, instr, 'qwen', full_ids=padded_ids)
    assert t_post_injected.position_index == len(padded_ids) - 1
    assert t_post_injected.position_index != len(real_ids) - 1


def test_raises_on_unmatched_instruction():
    # simulate a template that transforms the instruction text so it can no
    # longer be found verbatim -- must raise, not silently guess
    tok = MockTokenizer('qwen')

    def broken_encode(text, add_special_tokens=False):
        return [999999]  # id that will never appear in the rendered prompt

    tok.encode = broken_encode
    try:
        get_instruction_end_position(tok, 'Explain how photosynthesis works', 'qwen')
        assert False, "expected ValueError on unmatched instruction, got none"
    except ValueError:
        pass


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_')]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
