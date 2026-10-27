"""Tests for the Dory chat runtime: wiring, loop handling and the chat loop.

The fake components here answer with the shapes the real ones do, so everything
the runtime decides -- how often it calls a component, when it resets the state,
when it stops -- is exercised without a converted model.
"""

from __future__ import annotations

import numpy as np
import pytest

from anemll.ane_converter import dory_converter as dcv
from anemll import dory_chat as chat
from anemll.models.dory_ane_model import DoryANEConfig


VOCAB = 64
HIDDEN = 8


def fake_config():
    return DoryANEConfig(context_length=1024, batch_size=64, vocab_size=VOCAB,
                         hidden_size=HIDDEN)


def fake_meta(**overrides):
    config = fake_config()
    specs = dcv.component_specs(config)
    files = {spec.name: {function: f"{spec.name}_{function}.mlpackage"
                         for function in ("infer", "prefill")}
             for spec in specs}
    sizes = [VOCAB // 2, VOCAB // 2]
    kwargs = dict(config=config, specs=specs, files=files,
                  lut_bits={"embeddings": None, "blocks": 6, "lm_head": 6},
                  lm_head_chunk_sizes=sizes, chat_template=None)
    kwargs.update(overrides)
    return dcv.build_meta(**kwargs)


class FakeState:
    def __init__(self, identifier):
        self.identifier = identifier


class FakeModel:
    """Answers with the shapes of the real component and records every call."""

    def __init__(self, runtime, name, function):
        self.runtime = runtime
        self.name = name
        self.function = function
        self.states = []

    def make_state(self):
        self.runtime.state_counter += 1
        return FakeState(self.runtime.state_counter)

    def predict(self, inputs, state=None):
        self.runtime.calls.append((self.name, self.function, state.identifier))
        self.states.append(state)
        if self.name == "embeddings":
            ids = inputs["input_ids"]
            hidden = np.zeros((1, ids.shape[1], HIDDEN), dtype=np.float16)
            hidden[:] = ids[..., None].astype(np.float16)
            return {"hidden_states": hidden}
        if self.name == "lm_head":
            entry = self.runtime.meta["components"]["lm_head"]
            seq = inputs["hidden_states"].shape[1]
            out = {}
            for index, size in enumerate(self.runtime.params["lm_head_chunk_sizes"], start=1):
                out[entry["outputs"][index - 1]] = np.zeros((1, seq, size), dtype=np.float16)
            return out
        hidden = inputs["hidden_states"]
        # one addition per block in the component, so a test can count passes
        blocks = self.runtime.meta["components"][self.name]["blocks"]
        return {"output_hidden_states": (hidden + (blocks[1] - blocks[0])).astype(np.float16)}


class FakeModels(dict):
    def __init__(self, runtime, meta):
        super().__init__()
        for name in meta["run_order"]:
            entry = meta["components"][name]
            self[name] = {fn: FakeModel(runtime, name, fn) for fn in entry["functions"]}


def make_runtime(meta=None, loops=None):
    meta = meta or fake_meta()
    runtime = chat.DoryRuntime.__new__(chat.DoryRuntime)
    runtime.model_dir = None
    runtime.meta = meta
    runtime.params = meta["parameters"]
    runtime.verbose = False
    runtime.compute_units = None
    runtime.calls = []
    runtime.state_counter = 0
    runtime.models = FakeModels(runtime, meta)
    runtime.load_seconds = 0.0
    runtime.predict_seconds = 0.0
    runtime.predict_calls = 0
    runtime._state = None
    runtime._loops = int(loops if loops is not None else meta["loops"]["default"])
    runtime.reset_state()
    return runtime


# -- loop count semantics (spec D4) ---------------------------------------

def test_changing_the_loop_count_resets_the_state():
    runtime = make_runtime(loops=2)
    before = runtime._state.identifier
    assert runtime.set_loops(4) is True
    assert runtime._state.identifier > before


def test_keeping_the_loop_count_keeps_the_state():
    runtime = make_runtime(loops=2)
    state = runtime._state
    assert runtime.set_loops(2) is False
    assert runtime._state is state


def test_loop_count_outside_the_supported_range_is_refused():
    runtime = make_runtime()
    with pytest.raises(ValueError, match="1..4"):
        runtime.set_loops(5)


def test_recurrent_component_runs_once_per_loop():
    runtime = make_runtime(loops=3)
    runtime.forward([1, 2, 3], 0)
    recurrent = sum(1 for name, _, _ in runtime.calls[2:] if name == "recurrent")
    assert recurrent == 3


def test_block_components_run_in_order_with_the_recurrent_one_in_the_middle():
    runtime = make_runtime(loops=2)
    runtime.forward([7], 0)
    order = [name for name, _, _ in runtime.calls]
    assert order == ["embeddings", "input", "recurrent", "recurrent", "output", "lm_head"]


def test_split_recurrent_runs_both_halves_every_loop():
    config = fake_config()
    specs = dcv.component_specs(config, recurrent_splits=2)
    files = {spec.name: {function: f"{spec.name}_{function}.mlpackage"
                         for function in ("infer", "prefill")}
             for spec in specs}
    meta = dcv.build_meta(config=config, specs=specs, files=files,
                          lut_bits={"embeddings": None, "blocks": 6, "lm_head": 6},
                          lm_head_chunk_sizes=[VOCAB // 2, VOCAB // 2], chat_template=None)
    runtime = make_runtime(meta, loops=2)
    runtime.forward([7], 0)
    order = [name for name, _, _ in runtime.calls]
    assert order == ["embeddings", "input", "recurrent_1of2", "recurrent_2of2",
                     "recurrent_1of2", "recurrent_2of2", "output", "lm_head"]


def test_h_in_is_the_input_slot_output_and_is_reused_across_loops():
    runtime = make_runtime(loops=2)
    captured = {}
    original = runtime.models["recurrent"]["infer"].predict

    def spy(inputs, state=None):
        captured.setdefault("h_in", []).append(inputs["h_in"].copy())
        return original(inputs, state=state)

    runtime.models["recurrent"]["infer"].predict = spy
    runtime.forward([5], 0)
    first, second = captured["h_in"]
    np.testing.assert_array_equal(first, second)
    # the input block adds one per block (16 of them) to the embedded token id
    assert float(first.reshape(-1)[0]) == pytest.approx(5.0 + 16.0)


# -- batching --------------------------------------------------------------

def test_prefill_uses_full_batches_then_single_tokens():
    runtime = make_runtime()
    rows = runtime.prefill(np.arange(130, dtype=np.int32), 0)
    assert rows.shape == (130, VOCAB)
    functions = [fn for name, fn, _ in runtime.calls if name == "input"]
    assert functions.count("prefill") == 2
    assert functions.count("infer") == 2


def test_prefill_of_exactly_one_batch_is_a_single_call():
    runtime = make_runtime()
    runtime.prefill(np.arange(64, dtype=np.int32), 0)
    assert [fn for name, fn, _ in runtime.calls if name == "input"] == ["prefill"]


def test_shorter_than_one_batch_goes_token_by_token():
    runtime = make_runtime()
    runtime.prefill(np.arange(5, dtype=np.int32), 0)
    assert [fn for name, fn, _ in runtime.calls if name == "input"] == ["infer"] * 5


def test_masks_have_the_width_the_meta_declares():
    runtime = make_runtime()
    mask = runtime.build_mask(0, 64)
    assert mask.shape == (1, 1, 64, runtime.params["mask_len"])
    assert mask.dtype == np.float16


def test_all_components_share_one_state_object():
    runtime = make_runtime(loops=1)
    runtime.forward([3], 0)
    assert len({identifier for _, _, identifier in runtime.calls}) == 1


# -- the chat loop ---------------------------------------------------------

class FakeTokenizer:
    truthy = True

    def __init__(self, tokens, texts=None):
        self.tokens = list(tokens)
        self.texts = dict(texts or {0: "zero", 5: "t5", 6: "t6", 7: "t7", 8: "t8"})

    def apply_chat_template(self, messages, reasoning=True):
        return "<|im_start|>assistant\n"

    def encode(self, text):
        return np.asarray([1, 2, 3], dtype=np.int32)

    def decode(self, ids):
        return "".join(self.texts.get(int(i), "") for i in ids)

    @property
    def eos_ids(self):
        return [11]


class ScriptedRuntime:
    """A runtime that emits a prepared token sequence, one per decode step."""

    params = {"context_length": 1024}
    loops = 2

    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.step = 0
        self.resets = 0
        self.prompt_tokens = None
        self.decoded = []              # (token, pos) in call order

    def reset_state(self):
        self.resets += 1

    def check_context(self, prompt_tokens, reserve=1):
        chat.DoryRuntime.check_context(self, prompt_tokens, reserve=reserve)

    def prefill(self, ids, start=0, progress=None):
        self.prompt_tokens = int(start) + len(ids)
        if progress is not None:
            progress(len(ids), len(ids), 0.0)
        return np.zeros((len(ids), VOCAB), dtype=np.float32)

    def decode(self, token, pos):
        # The row is part of the contract, so assert it here: a decode that runs
        # ahead of the sequence must fail a test, not silently write a key one
        # row late.
        expected = int(self.prompt_tokens) + len(self.decoded)
        assert pos == expected, (
            f"decode fed token {token} at row {pos}, but it belongs at row "
            f"{expected} ({self.prompt_tokens} prompt tokens, "
            f"{len(self.decoded)} already decoded)")
        self.decoded.append((int(token), int(pos)))
        self.step += 1
        logits = np.full(VOCAB, -10.0, dtype=np.float32)
        logits[self.tokens[min(self.step, len(self.tokens) - 1)]] = 10.0
        return logits


def test_generate_stops_on_eos():
    # the first token comes from the prefill's logits (all zeros -> token 0),
    # then the scripted decode emits 6 and then EOS (11)
    runtime = ScriptedRuntime([5, 6, 11, 7])
    tokenizer = FakeTokenizer([5, 6, 11, 7])
    result = chat.generate(runtime, tokenizer, "hi", max_tokens=10, stream=False)
    assert result["generated_tokens"] == 2
    assert result["text"] == "zerot6"


def test_generate_feeds_each_token_at_its_own_row():
    """The blocker: the first generated token belongs at row len(prompt).

    The prompt's own last logits produce it, so it is written at the row right
    after the prompt -- not one row later, which would leave a zero key that the
    attention mask admits.
    """
    runtime = ScriptedRuntime([5, 6, 7, 8])
    tokenizer = FakeTokenizer([5, 6, 7, 8])
    result = chat.generate(runtime, tokenizer, "hi", max_tokens=4, stream=False)
    rows = [pos for _, pos in runtime.decoded]
    assert runtime.prompt_tokens is not None
    assert rows, "the loop must feed the generated tokens back"
    assert rows[0] == runtime.prompt_tokens
    assert rows == list(range(runtime.prompt_tokens, runtime.prompt_tokens + len(rows)))
    # the last generated token is never fed back: nothing needs its logits
    assert len(runtime.decoded) == result["generated_tokens"] - 1


def test_generate_respects_max_tokens():
    runtime = ScriptedRuntime([5, 6, 7, 8])
    tokenizer = FakeTokenizer([5, 6, 7, 8])
    result = chat.generate(runtime, tokenizer, "hi", max_tokens=3, stream=False)
    assert result["generated_tokens"] == 3


def test_generate_resets_the_cache_before_the_prompt():
    runtime = ScriptedRuntime([5, 11])
    tokenizer = FakeTokenizer([5, 11])
    chat.generate(runtime, tokenizer, "hi", max_tokens=4, stream=False)
    assert runtime.resets == 1


def test_generate_reports_the_loop_count_it_ran():
    runtime = ScriptedRuntime([5, 11])
    runtime.loops = 4
    tokenizer = FakeTokenizer([5, 11])
    assert chat.generate(runtime, tokenizer, "hi", max_tokens=1,
                         stream=False)["loops"] == 4


def test_generate_streams_the_same_text_it_returns():
    import io

    runtime = ScriptedRuntime([5, 6, 11])
    tokenizer = FakeTokenizer([5, 6, 11])
    buffer = io.StringIO()
    result = chat.generate(runtime, tokenizer, "hi", max_tokens=5, stream=True, out=buffer)
    assert buffer.getvalue().strip() == result["text"]


# -- sampling and templates ------------------------------------------------

def test_greedy_sampling_picks_the_argmax():
    logits = np.zeros(VOCAB, dtype=np.float32)
    logits[42] = 5.0
    assert chat.sample_token(logits, temperature=0.0) == 42


def test_sampling_is_deterministic_for_a_seed():
    logits = np.zeros(VOCAB, dtype=np.float32)
    logits[3] = 4.0
    logits[9] = 3.0
    picks = {chat.sample_token(logits, temperature=1.0, top_p=0.9,
                               rng=np.random.default_rng(0)) for _ in range(20)}
    assert picks <= {3, 9}


def test_chatml_fallback_marks_reasoning_on_and_off():
    messages = [{"role": "user", "content": "hello"}]
    on = chat.render_template(chat.CHATML_FALLBACK, messages, reasoning=True)
    off = chat.render_template(chat.CHATML_FALLBACK, messages, reasoning=False)
    assert "<|im_start|>user\nhello<|im_end|>" in on
    # reasoning on leaves the assistant turn at the thinking cue ...
    assert on.rstrip().endswith("thinking")
    assert " thinking" in on
    # ... and reasoning off does not write the cue at all
    assert "thinking" not in off
    assert off.rstrip().endswith("assistant")


def test_chatml_fallback_carries_a_system_prompt():
    messages = chat.build_messages("hello", system="be brief")
    rendered = chat.render_template(chat.CHATML_FALLBACK, messages)
    assert "<|im_start|>system\nbe brief<|im_end|>" in rendered
    assert "<|im_start|>assistant\n" in rendered


# -- loading ---------------------------------------------------------------

def test_resolve_model_path_falls_back_to_the_package(tmp_path):
    (tmp_path / "dory_input_lut6_infer.mlpackage").mkdir()
    resolved = chat.resolve_model_path(tmp_path, "dory_input_lut6_infer.mlmodelc")
    assert resolved.suffix == ".mlpackage"


def test_resolve_model_path_reports_a_missing_component(tmp_path):
    with pytest.raises(FileNotFoundError, match="dory_input"):
        chat.resolve_model_path(tmp_path, "dory_input_lut6_infer.mlpackage")
