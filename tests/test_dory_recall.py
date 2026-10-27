"""The recall loop's position contract.

run_recall had the same off-by-one as dory_chat.generate: it advanced the
position before feeding the sampled token, so the first generated key was
written one row late and the row the token belongs to stayed zero.  This
backend asserts the row on every decode, so the old loop fails here.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.dory_recall import build_prompt, missing_codes, run_recall  # noqa: E402

VOCAB = 32


class ScriptedBackend:
    """Emits a prepared answer and refuses a decode at the wrong row."""

    def __init__(self, answer, context=512):
        self.answer = list(answer)
        self.context = int(context)
        self.prompt_tokens = None
        self.decoded = []              # (token, pos) in call order
        self.resets = 0

    def reset(self):
        self.resets += 1

    def prefill(self, ids, progress=None):
        self.prompt_tokens = len(ids)
        return np.zeros(VOCAB, dtype=np.float32)

    def decode(self, token, pos):
        expected = int(self.prompt_tokens) + len(self.decoded)
        assert pos == expected, (
            f"decode fed token {token} at row {pos}, but it belongs at row "
            f"{expected} ({self.prompt_tokens} prompt tokens, "
            f"{len(self.decoded)} already decoded)")
        self.decoded.append((int(token), int(pos)))
        logits = np.full(VOCAB, -10.0, dtype=np.float32)
        step = min(len(self.decoded), len(self.answer) - 1)
        logits[self.answer[step]] = 10.0
        return logits

    def close(self):
        pass


def _prompt(tokens=12):
    return {"ids": [7] * int(tokens), "text": "filler", "codes": []}


def test_the_first_generated_token_is_decoded_at_the_prompt_length():
    backend = ScriptedBackend([1, 5, 6, 7])
    prompt = _prompt(12)
    result = run_recall(backend, prompt, max_tokens=4)
    rows = [pos for _, pos in backend.decoded]
    assert rows[0] == len(prompt["ids"])
    assert rows == list(range(len(prompt["ids"]), len(prompt["ids"]) + len(rows)))
    assert len(result["generated"]) == 4
    assert len(backend.decoded) == len(result["generated"]) - 1


def test_a_stop_token_is_never_fed_back():
    backend = ScriptedBackend([1, 5, 6, 7])
    prompt = _prompt(4)
    # the prefill's logits are zeros, so token 0 wins: make it a stop token
    result = run_recall(backend, prompt, max_tokens=4, stop_ids=[0])
    assert result["generated"] == []
    assert backend.decoded == []


def test_a_full_context_stops_without_decoding_past_the_last_row():
    prompt = _prompt(7)
    backend = ScriptedBackend([1, 5, 6, 7], context=len(prompt["ids"]) + 1)
    result = run_recall(backend, prompt, max_tokens=4)
    assert len(result["generated"]) == 1
    assert backend.decoded == []


class _Tokenizer:
    """The tokenizer seam build_prompt needs; one id per word, plus the codes."""

    eos_ids = [11]

    def encode(self, text):
        return [len(word) % 97 + 1 for word in str(text).split()]

    def apply_chat_template(self, messages, reasoning=True):
        return " ".join(str(message.get("content", "")) for message in messages)

    def decode(self, ids):
        # the inverse of encode, well enough for the harness's own bookkeeping
        return " ".join("w" * (int(value) % 97) if value else "" for value in ids)


def test_recall_builds_a_prompt_that_keeps_the_codes_in_order():
    prompt = build_prompt(_Tokenizer(), 256, ["ALPHA1", "BRAVO2"], [0.0, 1.0])
    assert prompt["codes"] == ["ALPHA1", "BRAVO2"]
    assert prompt["fractions"] == [0.0, 1.0]
    assert len(prompt["ids"]) == prompt["prompt_tokens"] > 0
    # the codes sit inside the document, in the order they were planted
    assert prompt["positions"] == sorted(prompt["positions"])
    assert prompt["positions"][0] == 0
    assert prompt["positions"][1] > prompt["positions"][0]
    # the answer check finds a code only where it is written
    assert missing_codes("ALPHA1 and BRAVO2", prompt["codes"]) == []
    assert missing_codes("ALPHA1 only", prompt["codes"]) == ["BRAVO2"]
