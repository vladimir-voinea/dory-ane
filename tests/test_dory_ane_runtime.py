"""The per-program runtime, driven through a stub instead of Core ML.

What this file is for
---------------------
'anemll/runtime/dory_ane_runtime.py' does three things Core ML cannot help with
and that a real package would only reveal by producing garbage: it decides the
dispatch order across the loop, it threads the gate chain, and it owns the
positions and the mask.  All three are checked here against a manifest the
converter itself builds -- 'dory_per_head.program_specs' plus 'build_meta', so
the meta under test has the shape the real one has -- and against a stub factory
whose contract is exactly the seam the runtime uses: 'make_state()' and
'predict(feed, state=...)'.

The stub is deliberately arithmetic-free.  Every program writes its own tag
(its index in run_order, 1-based) into its gate and contribution outputs, and
adds that tag to the hidden state it passes on.  A consumer can then be checked
by value: the gate it received has to be the producer's tag, the contribution it
received has to be the right head of the right program, and the hidden state it
was fed has to be the accumulated chain up to that point.

The toy plan is three periods (24 physical blocks, zones 0-7 / 8-15 / 16-23):
one period before the loop, one inside it, one after, which is the smallest
manifest that exercises every boundary the real 24K plan has -- including a gate
that crosses from the loop into the output zone (gate_16).
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
import yaml

from anemll.ane_converter import dory_per_head as dph
from anemll.ane_converter.dory_converter import build_meta, validate_meta
from anemll.models import dory_ane_model as dam
from anemll.runtime import dory_ane_runtime as rt
from anemll.utils import dory_caches as dc

# ---------------------------------------------------------------------------
# The toy plan, spelled out: if the runtime's order ever changes, this fails
# ---------------------------------------------------------------------------

PRE_LOOP = ("embeddings", "sliding_attn_L0", "sepilogue_L0", "sliding_attn_L2",
            "sepilogue_L2", "sliding_attn_L4", "sepilogue_L4", "attn_full_0_h0",
            "attn_full_0_h1", "attn_full_0_h2", "attn_full_0_h3", "epilogue_0")

LOOP_ZONE = ("entry_gate", "sliding_attn_L8", "sepilogue_L8", "sliding_attn_L10",
             "sepilogue_L10", "sliding_attn_L12", "sepilogue_L12", "attn_full_1_h0",
             "attn_full_1_h1", "attn_full_1_h2", "attn_full_1_h3", "epilogue_1")

POST_LOOP = ("sliding_attn_L16", "sepilogue_L16", "sliding_attn_L18", "sepilogue_L18",
             "sliding_attn_L20", "sepilogue_L20", "attn_full_2_h0", "attn_full_2_h1",
             "attn_full_2_h2", "attn_full_2_h3", "epilogue_2", "lm_head")

#: gate -> the program that emits it, as the manifest records it
GATE_CHAIN = {
    "gate_0": "embeddings", "gate_2": "sepilogue_L0", "gate_4": "sepilogue_L2",
    "gate_6": "sepilogue_L4", "gate_8": "entry_gate", "gate_10": "sepilogue_L8",
    "gate_12": "sepilogue_L10", "gate_14": "sepilogue_L12", "gate_16": "epilogue_1",
    "gate_18": "sepilogue_L16", "gate_20": "sepilogue_L18", "gate_22": "sepilogue_L20",
}

BLOCKS = 24
HIDDEN = 8
GATE = 8                  # num_attention_heads * head_dim
CONTRIBUTION = 2          # rep * head_dim
LOGIT_SPLIT = 16
BATCH = 4
CONTEXT = 32
SLIDING_WINDOW = 8


# ---------------------------------------------------------------------------
# A conversion output the runtime can read, built by the converter itself
# ---------------------------------------------------------------------------

def _slots_for(blocks: int):
    """dc.cache_slots() over 'blocks' physical blocks instead of the shipped 72."""
    full, sliding = {}, {}
    for block in range(blocks):
        if block % dc.PERIOD == dc.FULL_ATTENTION_OFFSET:
            full[block] = len(full)
        elif block % dc.PERIOD in dc.SLIDING_ATTENTION_OFFSETS:
            sliding[block] = len(sliding)
    return full, sliding


class _ToyConfig(dam.DoryANEConfig):
    """The 24-block plan's config.

    DoryANEConfig derives the two cache COUNTS from the shipped 72-block layout,
    so they are recomputed here from num_hidden_layers; everything else is the
    production config at tiny sizes.
    """

    @property
    def num_sliding_caches(self):
        return sum(1 for block in range(self.num_hidden_layers)
                   if dc.block_is_sliding_attention(block))

    @property
    def num_full_caches(self):
        return sum(1 for block in range(self.num_hidden_layers)
                   if dc.block_is_full_attention(block))


def _config(blocks=BLOCKS, input_layers=8, recurrent_layers=8):
    return _ToyConfig(hidden_size=HIDDEN, intermediate_size=16, num_attention_heads=4,
                      num_key_value_heads=4, head_dim=CONTRIBUTION, vocab_size=32,
                      num_hidden_layers=blocks, input_layer_count=input_layers,
                      recurrent_layer_count=recurrent_layers,
                      output_layer_count=blocks - input_layers - recurrent_layers,
                      context_length=CONTEXT, batch_size=BATCH,
                      sliding_window=SLIDING_WINDOW, heads_per_state=1,
                      slots_per_sliding_carrier=1, rope_tables_as_inputs=True,
                      compute_mode="matmul")


def _build(root, blocks=BLOCKS, input_layers=8, recurrent_layers=8) -> dict:
    """Write a package set the runtime can read: meta.yaml, files, rope tables."""
    config = _config(blocks, input_layers, recurrent_layers)
    with pytest.MonkeyPatch.context() as patch:
        # the plan's state layout is derived from the block pattern; the toy has
        # 24 blocks, so the slot tables are the 24-block ones
        patch.setattr(dc, "cache_slots", lambda: _slots_for(blocks))
        specs = dph.program_specs(config)
        files = {spec.name: {"infer": f"dory_{spec.name}_infer.mlpackage",
                             "prefill": f"dory_{spec.name}_prefill.mlpackage"}
                 for spec in specs}
        meta = build_meta(config=config, specs=specs, files=files, lut_bits={},
                          lm_head_chunk_sizes=[LOGIT_SPLIT, LOGIT_SPLIT], prefix="dory",
                          per_program=True, chat_template=None)
        meta["per_head_programs"] = dph.program_manifest(config, specs)
        meta["kv"]["declared_states_by_program"] = {
            spec.name: list(spec.states) for spec in specs if spec.states}
    for name in meta["run_order"]:
        for file_name in meta["components"][name]["files"].values():
            # resolve_model_path only needs the file to exist; the stub factory
            # never reads it
            (root / file_name).write_bytes(b"")
    rows = CONTEXT + 1
    base = (np.arange(rows, dtype=np.float16).reshape(-1, 1)
            + np.arange(CONTRIBUTION, dtype=np.float16) * np.float16(0.25))
    np.savez_compressed(
        root / "rope_tables.npz",
        cos_sliding=base.astype(np.float16),
        sin_sliding=(base + np.float16(100)).astype(np.float16),
        cos_full=(base + np.float16(200)).astype(np.float16),
        sin_full=(base + np.float16(300)).astype(np.float16),
    )
    (root / "meta.yaml").write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    return meta


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    root = tmp_path_factory.mktemp("dory-program-runtime")
    return root, _build(root)


@pytest.fixture(scope="module")
def full_plan(tmp_path_factory):
    """The shipped plan's shape at tiny sizes: 72 blocks, 102 programs."""
    root = tmp_path_factory.mktemp("dory-program-runtime-full")
    return root, _build(root, blocks=72, input_layers=16, recurrent_layers=40)


# ---------------------------------------------------------------------------
# The stub: tag in, tag out, every call recorded
# ---------------------------------------------------------------------------

class _StubState:
    def __init__(self, program: str, function: str, generation: int):
        self.program = program
        self.function = function
        self.generation = generation


class _Call:
    def __init__(self, name, function, feed, state):
        self.name = name
        self.function = function
        self.feed = feed
        self.state = state
        source = feed.get("input_ids", feed.get("hidden_states"))
        self.seq = int(source.shape[1])

    def __repr__(self):  # pragma: no cover - failure messages only
        return f"<{self.function} {self.name} seq={self.seq}>"


class _StubProgram:
    """The seam, with tags instead of arithmetic."""

    def __init__(self, name, function, outputs, tag, log, omit=None):
        self.name = name
        self.function = function
        self.outputs = tuple(outputs)
        self.tag = float(tag)
        self.log = log
        self.omit = omit
        self.make_state_calls = 0

    def make_state(self):
        self.make_state_calls += 1
        return _StubState(self.name, self.function, self.make_state_calls)

    def predict(self, feed, state=None):
        self.log.append(_Call(self.name, self.function,
                              {key: np.array(value, copy=True)
                               for key, value in feed.items()}, state))
        seq = int(feed.get("input_ids", feed.get("hidden_states")).shape[1])
        out = {}
        for name in self.outputs:
            if name == self.omit:
                continue
            if name.startswith("gate_"):
                out[name] = np.full((1, seq, GATE), self.tag, dtype=np.float16)
            elif name.startswith("logits_"):
                out[name] = np.full((1, seq, LOGIT_SPLIT), self.tag, dtype=np.float16)
            elif name == "head_contribution":
                out[name] = np.full((1, seq, CONTRIBUTION), self.tag, dtype=np.float16)
            elif name.startswith("contribution_h"):
                head = int(name[len("contribution_h"):])
                out[name] = np.full((1, seq, CONTRIBUTION),
                                    self.tag + 0.25 * head, dtype=np.float16)
            elif "hidden_states" in feed:
                out[name] = feed["hidden_states"] + np.float16(self.tag)
            else:
                # the embeddings program starts the stream: there is nothing to
                # add its tag to
                out[name] = np.full((1, seq, HIDDEN), self.tag, dtype=np.float16)
        return out


class _StubFactory:
    """Loads a stub per package file, and records what was loaded."""

    def __init__(self, meta: dict, log: list, omit=None):
        self.tags = {name: float(index + 1)
                     for index, name in enumerate(meta["run_order"])}
        self.by_file = {}
        for name in meta["run_order"]:
            for function, file_name in meta["components"][name]["files"].items():
                self.by_file[file_name] = (name, function)
        self.outputs = {name: meta["components"][name]["outputs"]
                        for name in meta["run_order"]}
        self.log = log
        self.omit = omit
        self.loaded = []
        self.handles = {}

    def load(self, path):
        name, function = self.by_file[path.name]
        omit = self.omit[1] if (self.omit and self.omit[0] == name) else None
        handle = _StubProgram(name, function, self.outputs[name], self.tags[name],
                              self.log, omit)
        self.loaded.append(path.name)
        self.handles[(name, function)] = handle
        return handle

    def tag_of(self, name: str) -> float:
        return self.tags[name]


def _runtime(root, meta=None, *, loops=2, omit=None):
    log = []
    loaded_meta = meta if meta is not None else yaml.safe_load(
        (root / "meta.yaml").read_text(encoding="utf-8"))
    factory = _StubFactory(loaded_meta, log, omit)
    return rt.DoryProgramRuntime(root, factory=factory, loops=loops, meta=meta), factory, log


def _calls(log, name):
    return [call for call in log if call.name == name]


def _value(array) -> float:
    """A stub output is one constant; assert that and return it."""
    assert np.all(array == array.flat[0]), f"expected one constant, got {np.unique(array)[:4]}"
    return float(array.flat[0])


def _pass_of(index: int, loops: int) -> int:
    """Which loop pass a dispatch index belongs to (-1 before, loops-1 after)."""
    if index < len(PRE_LOOP):
        return -1
    if index < len(PRE_LOOP) + loops * len(LOOP_ZONE):
        return (index - len(PRE_LOOP)) // len(LOOP_ZONE)
    return loops - 1


# ---------------------------------------------------------------------------
# The manifest under test is one the converter would write
# ---------------------------------------------------------------------------

def test_the_toy_manifest_is_what_the_converter_would_write(toy):
    _, meta = toy
    assert validate_meta(meta) == []
    assert len(meta["run_order"]) == len(PRE_LOOP) + len(LOOP_ZONE) + len(POST_LOOP)
    assert meta["per_head_programs"]["recurrent_dispatches_per_loop_pass"] == len(LOOP_ZONE)
    assert meta["per_head_programs"]["gate_chain"] == GATE_CHAIN
    stated = meta["kv"]["declared_states_by_program"]
    assert len(stated) == 21, "12 full-attention head programs + 9 sliding carriers"
    for name in stated:
        assert meta["per_head_programs"]["programs"][name]["states"], name


def test_the_plan_splits_into_the_three_zones(toy):
    root, _ = toy
    runtime, _, _ = _runtime(root)
    plan = runtime.plan
    assert plan.pre_loop == PRE_LOOP
    assert plan.loop == LOOP_ZONE
    assert plan.post_loop == POST_LOOP
    assert plan.dispatches_per_token(2) == len(PRE_LOOP) + 2 * len(LOOP_ZONE) + len(POST_LOOP)
    assert plan.batch_size == BATCH
    assert plan.context_length == CONTEXT
    assert plan.mask_len == CONTEXT + plan.sliding_rows


def test_the_shipped_plan_shape_is_102_programs_and_56_dispatches_per_pass(full_plan):
    root, meta = full_plan
    runtime, _, _ = _runtime(root, loops=2)
    plan = runtime.plan
    assert validate_meta(meta) == []
    assert len(plan.order) == 102
    assert len(plan.stated_programs()) == 63, "36 full-attention heads + 27 carriers"
    assert len(plan.loop) == 56
    # 102 programs, 56 of them recurrent (the entry gate is one of the 56)
    assert plan.dispatches_per_token(2) == 102 - 56 + 2 * 56
    assert plan.pre_loop[0] == "embeddings" and plan.post_loop[-1] == "lm_head"
    runtime.decode(1, 0)
    assert len(runtime._program_time) == 102, "every program of the plan was dispatched"


# ---------------------------------------------------------------------------
# Order, loop, gate chain, contributions
# ---------------------------------------------------------------------------

def test_the_dispatch_order_is_the_input_zone_the_loop_twice_then_the_output_zone(toy):
    root, _ = toy
    runtime, _, log = _runtime(root, loops=2)
    runtime.decode(7, 0)
    expected = list(PRE_LOOP) + list(LOOP_ZONE) + list(LOOP_ZONE) + list(POST_LOOP)
    assert [call.name for call in log] == expected
    assert len(log) == runtime.plan.dispatches_per_token(2) == 48


def test_the_loop_entry_runs_first_on_every_pass_and_takes_the_zone_input(toy):
    root, _ = toy
    runtime, _, log = _runtime(root, loops=3)
    runtime.decode(7, 0)
    entries = [index for index, call in enumerate(log) if call.name == "entry_gate"]
    assert entries == [len(PRE_LOOP) + pass_index * len(LOOP_ZONE)
                       for pass_index in range(3)], (
        "entry_gate has to open every pass, and nothing else may precede it there")
    zone_input = log[entries[0]].feed["h_in"]
    for call_index in entries:
        assert np.allclose(log[call_index].feed["h_in"], zone_input), (
            "h_in is the hidden state that entered the zone, and it does not change")
    assert np.allclose(log[entries[0]].feed["hidden_states"], zone_input), (
        "on the first pass the zone's input is also the state the entry norm sees")
    for later in entries[1:]:
        assert not np.allclose(log[later].feed["hidden_states"], zone_input), (
            "a later pass has to see the previous pass's output")
    per_pass = len(LOOP_ZONE)
    for pass_index in range(2):
        last_of_pass = entries[pass_index] + per_pass - 1
        assert log[last_of_pass].name == "epilogue_1"
        assert entries[pass_index + 1] == last_of_pass + 1, (
            "the next pass starts immediately after the previous one ends")


def test_every_gate_arrives_from_the_program_the_manifest_names(toy):
    root, _ = toy
    runtime, factory, log = _runtime(root, loops=2)
    runtime.decode(3, 0)
    seen = 0
    for index, call in enumerate(log):
        if "gate" not in call.feed:
            continue
        program = runtime.plan.programs[call.name]
        expected_gate = program.gate_input
        assert expected_gate is not None, f"{call.name} got a gate it does not declare"
        producer = GATE_CHAIN[expected_gate]
        assert _value(call.feed["gate"]) == factory.tag_of(producer), (
            f"{call.name} took {expected_gate} from someone other than {producer}")
        produced_at = [i for i, earlier in enumerate(log[:index]) if earlier.name == producer]
        assert produced_at, f"{producer} had not run before {call.name}"
        assert _pass_of(produced_at[-1], 2) == _pass_of(index, 2), (
            f"{call.name} took a stale {expected_gate} from another pass")
        seen += 1
    consumers = {name for name in runtime.plan.order
                 if runtime.plan.programs[name].gate_in}
    expected = sum(1 for name in list(PRE_LOOP) + list(LOOP_ZONE) * 2 + list(POST_LOOP)
                   if name in consumers)
    assert seen == expected == 16, (
        "four gate consumers before the loop, four in each pass, four after it")


def test_a_full_attention_epilogue_takes_one_contribution_per_kv_head(toy):
    root, _ = toy
    runtime, factory, log = _runtime(root, loops=2)
    runtime.decode(3, 0)
    epilogue = _calls(log, "epilogue_0")[0]
    for head in range(4):
        producer = f"attn_full_0_h{head}"
        assert _value(epilogue.feed[f"head_{head}"]) == factory.tag_of(producer), (
            f"head_{head} did not come from {producer}")


def test_a_sliding_epilogue_takes_the_four_slices_of_its_attention_program(toy):
    root, _ = toy
    runtime, factory, log = _runtime(root, loops=2)
    runtime.decode(3, 0)
    epilogue = _calls(log, "sepilogue_L8")[0]
    tag = factory.tag_of("sliding_attn_L8")
    for head in range(4):
        assert _value(epilogue.feed[f"contribution_h{head}"]) == tag + 0.25 * head, (
            f"contribution_h{head} is not slice {head} of the program")


def test_the_hidden_state_walks_the_chain_in_order(toy):
    root, _ = toy
    runtime, factory, log = _runtime(root, loops=2)
    runtime.decode(3, 0)
    expected = 0.0
    for call in log:
        program = runtime.plan.programs[call.name]
        if "hidden_states" in call.feed:
            assert np.allclose(call.feed["hidden_states"], expected), (
                f"{call.name} was fed {np.unique(call.feed['hidden_states'])[:3]}, "
                f"expected the accumulated {expected}")
        if program.advances_hidden:
            expected += factory.tag_of(call.name)


# ---------------------------------------------------------------------------
# Positions, the mask and the RoPE rows
# ---------------------------------------------------------------------------

def test_a_decode_step_feeds_the_position_the_ring_offset_and_one_row(toy):
    root, _ = toy
    runtime, _, log = _runtime(root, loops=1)
    runtime.decode(3, 9)
    mask_len = runtime.plan.mask_len
    context = runtime.plan.context_length
    checked = 0
    for call in log:
        if "current_pos" not in call.feed:
            continue
        assert int(call.feed["current_pos"][0]) == 9
        assert int(call.feed["cache_pos"][0]) == 9 % runtime.plan.sliding_rows
        mask = call.feed["causal_mask"]
        assert mask.shape == (1, 1, 1, mask_len)
        assert mask.dtype == np.float16
        assert float(mask[0, 0, 0, 9]) == 0.0
        assert float(mask[0, 0, 0, 10]) == float("-inf")
        assert np.all(mask[0, 0, 0, :10] == 0.0), "every earlier column is visible"
        assert np.all(mask[0, 0, 0, 11:context] == float("-inf"))
        ring = mask[0, 0, 0, context:]
        assert np.count_nonzero(ring == 0.0) == SLIDING_WINDOW, (
            "the ring slice holds exactly the window of visible tokens")
        checked += 1
    assert checked, "no program was fed a position"


def test_a_prefill_batch_feeds_one_mask_row_per_token(toy):
    root, _ = toy
    runtime, _, log = _runtime(root, loops=1)
    runtime.prefill([1, 2, 3, 4])
    embeddings = _calls(log, "embeddings")[0]
    assert embeddings.function == "prefill"
    assert embeddings.feed["input_ids"].shape == (1, BATCH)
    attention = _calls(log, "sliding_attn_L0")[0]
    assert attention.function == "prefill"
    mask = attention.feed["causal_mask"]
    assert mask.shape == (1, 1, BATCH, runtime.plan.mask_len)
    for row in range(BATCH):
        assert np.all(mask[0, 0, row, :row + 1] == 0.0)
        assert float(mask[0, 0, row, row + 1]) == float("-inf")
    assert int(attention.feed["current_pos"][0]) == 0
    assert int(attention.feed["cache_pos"][0]) == 0


def test_the_rope_rows_are_the_position_slice_of_the_exported_tables(toy):
    root, _ = toy
    runtime, _, log = _runtime(root, loops=1)
    tables = np.load(root / "rope_tables.npz")
    runtime.decode(3, 5)
    sliding = _calls(log, "sliding_attn_L0")[0]
    assert np.array_equal(sliding.feed["cos_sliding"],
                          tables["cos_sliding"][5:6].reshape(1, 1, 1, CONTRIBUTION))
    assert np.array_equal(sliding.feed["sin_sliding"],
                          tables["sin_sliding"][5:6].reshape(1, 1, 1, CONTRIBUTION))
    full = _calls(log, "attn_full_0_h0")[0]
    assert "cos_full" in full.feed, "a head program takes the full-attention table"
    assert "cos_sliding" not in full.feed
    assert np.array_equal(full.feed["cos_full"],
                          tables["cos_full"][5:6].reshape(1, 1, 1, CONTRIBUTION))


def test_the_mask_matches_the_model_side_builder():
    pytest.importorskip("torch")
    for start, seq in ((0, 1), (0, 4), (5, 1), (7, 3), (30, 2)):
        ours = rt.build_causal_mask(start, seq, CONTEXT, 256, SLIDING_WINDOW)
        theirs = dc.build_causal_mask(start, seq, CONTEXT, 256, SLIDING_WINDOW).numpy()
        assert ours.dtype == theirs.dtype == np.float16
        assert np.array_equal(ours, theirs), f"mask differs at start={start} seq={seq}"


# ---------------------------------------------------------------------------
# State, packages and the report
# ---------------------------------------------------------------------------

def test_one_state_per_stated_program_created_once_and_kept(toy):
    root, meta = toy
    runtime, factory, log = _runtime(root, loops=1)
    runtime.load()
    stated = set(runtime.plan.stated_programs())
    for name in meta["run_order"]:
        if name not in stated:
            continue
        infer = factory.handles[(name, "infer")]
        assert infer.make_state_calls == 1, name
        assert (name, "prefill") not in factory.handles, (
            "prefill packages are only loaded for their own phase")
    runtime.decode(4, 0)
    for call in log:
        if call.name in stated:
            assert call.state is runtime._states[call.name]
        else:
            assert call.state is None, f"{call.name} declares no state"
    before = dict(runtime._states)
    runtime.reset()
    for name in stated:
        assert runtime._states[name] is not before[name], (
            "reset has to hand out new, empty caches")


def test_every_package_is_loaded_exactly_once(toy):
    root, meta = toy
    runtime, factory, _ = _runtime(root, loops=1)
    runtime.load()
    runtime.load()
    runtime.decode(4, 0)
    expected = sorted(file_name
                      for name in meta["run_order"]
                      for file_name in meta["components"][name]["files"].values())
    # Packages are loaded per PHASE now, so a decode-only run keeps the infer
    # half resident and never touches the prefill half.  What must hold is that
    # the resident set stays at or under one phase's worth of packages -- 204 at
    # once is where Core ML starts silently leaving models unloaded.
    infer_only = sorted(f for f in expected if "_infer" in f)
    assert sorted(set(factory.loaded)) == infer_only
    assert len(runtime._handles) == len(meta["run_order"])


def test_prefill_uses_the_prefill_packages_and_decode_the_infer_ones(toy):
    root, meta = toy
    runtime, _, log = _runtime(root, loops=1)
    runtime.prefill(list(range(BATCH)))
    assert {call.function for call in log} == {"prefill"}
    log.clear()
    runtime.decode(0, BATCH)
    assert {call.function for call in log} == {"infer"}
    log.clear()
    runtime.feed([1, 2, 3, 4, 5, 6], prefill_threshold=0)
    functions = [call.function for call in log]
    assert functions.count("prefill") == len(meta["run_order"]), (
        "one full batch of four tokens goes through the prefill packages")
    assert functions.count("infer") == 2 * len(meta["run_order"]), (
        "the two-token tail goes through the infer packages, one token per call")

    # By default a short prompt avoids the prefill set entirely: packing a batch
    # switches the resident model set, which costs 80-175 s on the real model.
    log.clear()
    runtime.feed([1, 2, 3, 4, 5, 6])
    assert {call.function for call in log} == {"infer"}, (
        "a short prompt stays on the decode side even though it fills a batch")


def test_generate_starts_from_empty_caches_every_time(toy):
    root, _ = toy
    runtime, _, log = _runtime(root, loops=1)
    runtime.generate(list(range(BATCH)), 2)
    log.clear()
    runtime.generate([5, 6, 7, 8], 2)
    positions = [int(call.feed["current_pos"][0]) for call in log if "current_pos" in call.feed]
    assert positions[0] == 0, "the second prompt starts at position 0, not after the first"
    assert max(positions) == BATCH, "four prompt tokens, then one generated token at 4"


def test_the_state_handoff_check_compares_prefill_with_the_one_token_path(toy):
    root, _ = toy
    runtime, _, log = _runtime(root, loops=1)
    # a short prompt is padded to one prefill batch by repeating its last token
    result = runtime.verify_state_handoff([1, 2])
    assert result["tokens"] == BATCH
    assert result["max_abs_diff"] == 0.0
    assert result["cosine"] == pytest.approx(1.0)
    assert result["same_argmax"] is True
    # both paths ran
    assert _calls(log, "embeddings")[0].function == "prefill"
    assert any(call.function == "infer" for call in log)


def test_the_report_counts_tokens_dispatches_and_kinds(toy):
    root, _ = toy
    runtime, _, _ = _runtime(root, loops=2)
    runtime.prefill(list(range(BATCH)))
    runtime.decode(0, BATCH)
    stats = runtime.stats()
    assert stats["dispatches_per_token"] == 48
    assert stats["prefill"]["tokens"] == BATCH
    assert stats["decode"]["tokens"] == 1
    assert stats["prefill"]["dispatches"] == 48
    assert stats["decode"]["dispatches"] == 48
    assert stats["decode"]["dispatches_per_token"] == 48
    assert stats["prefill"]["tokens_per_second"] > 0
    # a single decode is the time to first token, reported on its own rather than
    # diluted into a steady rate it cannot represent
    # the toy factory's first decode is instant, so this only checks the field
    assert stats["decode"]["first_token_seconds"] >= 0.0
    assert stats["peak_rss_mib"] > 0
    assert set(stats["per_kind"]) == {"embeddings", "entry_gate", "sliding_attn", "attn_head",
                                      "sliding_epilogue", "full_epilogue", "lm_head"}
    assert stats["per_kind"]["entry_gate"]["calls"] == 4, (
        "once per loop pass: two passes, two forward calls")
    assert stats["per_kind"]["lm_head"]["calls"] == 2
    report = runtime.report()
    assert "dispatches per token" in report and "peak RSS" in report


def test_the_context_and_the_batch_are_guarded(toy):
    root, _ = toy
    runtime, _, _ = _runtime(root, loops=1)
    with pytest.raises(ValueError, match="exactly 4 tokens"):
        runtime.prefill([1, 2, 3])
    with pytest.raises(rt.ContextOverflowError):
        runtime.decode(1, CONTEXT)
    with pytest.raises(rt.ContextOverflowError):
        runtime.prefill([1, 2, 3, 4], start=CONTEXT - 2)


# ---------------------------------------------------------------------------
# The failures the ticket asks to be loud
# ---------------------------------------------------------------------------

def test_an_input_the_runtime_cannot_supply_names_the_program(toy):
    root, meta = toy
    broken = copy.deepcopy(meta)
    for section in (broken["components"]["sepilogue_L0"],
                    broken["per_head_programs"]["programs"]["sepilogue_L0"]):
        section["inputs"] = list(section["inputs"]) + ["mystery"]
    runtime, _, _ = _runtime(root, meta=broken, loops=1)
    with pytest.raises(rt.ProgramError, match="sepilogue_L0.*mystery"):
        runtime.decode(1, 0)


def test_a_gate_consumed_before_it_is_produced_raises(toy):
    root, meta = toy
    broken = copy.deepcopy(meta)
    # sepilogue_L0 moves after sepilogue_L2: every program still has its
    # contributions, but gate_2 is now emitted after the program that consumes it,
    # which is the one thing only the dispatch can see
    order = list(broken["run_order"])
    for name in ("sepilogue_L0", "sepilogue_L2"):
        order.remove(name)
    at = order.index("sliding_attn_L2") + 1
    order[at:at] = ["sepilogue_L2", "sepilogue_L0"]
    broken["run_order"] = order
    runtime, _, _ = _runtime(root, meta=broken, loops=1)
    with pytest.raises(rt.GateError, match="sepilogue_L2 consumes gate_2 before it is "
                                           "produced"):
        runtime.decode(1, 0)


def test_two_consumers_of_one_gate_are_refused_up_front(toy):
    root, meta = toy
    broken = copy.deepcopy(meta)
    broken["per_head_programs"]["programs"]["sepilogue_L0"]["blocks"] = [4, 5]
    with pytest.raises(rt.PlanError, match="gate_4 is consumed by two programs"):
        rt.Plan.from_meta(broken, root)


def test_a_program_that_drops_a_declared_output_raises(toy):
    root, _ = toy
    runtime, _, _ = _runtime(root, loops=1, omit=("sepilogue_L0", "h_next"))
    with pytest.raises(rt.ProgramError, match="sepilogue_L0.*h_next"):
        runtime.decode(1, 0)


def test_a_meta_without_the_program_section_is_refused(tmp_path):
    meta = {"parameters": {"context_length": 8}, "components": {}, "run_order": []}
    with pytest.raises(rt.PlanError, match="per_head_programs"):
        rt.Plan.from_meta(meta, tmp_path)


def test_a_plan_that_straddles_the_loop_is_refused(toy):
    root, meta = toy
    broken = copy.deepcopy(meta)
    # a once-per-token program that spans the zone boundary cannot be placed
    broken["per_head_programs"]["programs"]["epilogue_0"]["blocks"] = [6, 9]
    with pytest.raises(rt.PlanError, match="straddle"):
        rt.Plan.from_meta(broken, root)
