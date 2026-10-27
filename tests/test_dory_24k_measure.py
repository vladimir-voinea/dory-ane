"""The bench workload and the sliding ring: two of the review findings.

Three things were broken and are pinned here: a bench workload that asked for the
whole context and therefore overflowed before the 24K cell could be measured (and
took the cells already measured with it), and a resumed prefill that could write
past the sliding ring.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from anemll import dory_chat as chat
from anemll.models.dory_ane_model import DoryANEConfig
from tests.test_dory_chat import fake_meta, make_runtime

VOCAB, HIDDEN = 64, 8
REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "dory_eval" / "eval_tokens.txt"


def load_script(name: str):
    """Load a script by path, registered in sys.modules (a dataclass needs it)."""
    path = REPO / "scripts" / f"{name}.py"
    module_name = f"{name}_under_test"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def long_meta():
    config = DoryANEConfig(context_length=24576, batch_size=64,
                           vocab_size=VOCAB, hidden_size=HIDDEN)
    return fake_meta(config=config)


# -- the bench workload (review: "the 24K tok/s cell cannot be measured") -----

def test_the_bench_workload_leaves_room_for_the_decode():
    assert chat.bench_workload(24576, 32) == 24544
    assert chat.bench_workload(1024, 1) == 1023
    assert chat.bench_workload(2048, 64) == 1984


def test_a_context_sized_bench_prompt_is_accepted():
    runtime = make_runtime(meta=long_meta())
    prompt = chat.filler_tokens(chat.bench_workload(24576, 32), None)
    assert len(prompt) == 24544
    runtime.check_context(len(prompt))          # the reservation the old prompt broke


def test_a_context_that_leaves_no_room_is_refused_with_a_reason():
    with pytest.raises(chat.ContextOverflowError, match="no room"):
        chat.bench_workload(32, 64)


# -- the sliding ring (review: "a resumed prefill can write past the ring") ---

def test_the_ring_guard_refuses_a_batch_that_straddles_the_seam():
    runtime = make_runtime(meta=long_meta())
    assert int(runtime.params["sliding_rows"]) == 1280
    with pytest.raises(ValueError, match="straddle"):
        runtime.prefill(np.ones(64, dtype=np.int32), 1270)


def test_the_ring_guard_accepts_a_batch_that_ends_exactly_on_the_seam():
    runtime = make_runtime(meta=long_meta())
    runtime.prefill(np.ones(64, dtype=np.int32), 1216)   # 1216 + 64 == 1280
    runtime.reset_state()
    runtime.prefill(np.ones(64, dtype=np.int32), 1280)   # the seam itself is fine


# -- the sweep keeps what it measured (review: "throws away the other cells") --

class _SweepRuntime:
    def __init__(self, fail_above=None):
        self.fail_above = fail_above

    def set_loops(self, loops):
        self.loops = int(loops)

    def reset_state(self):
        pass

    def prefill(self, ids, start):
        if self.fail_above is not None and len(ids) > self.fail_above:
            raise RuntimeError("this fake refuses a prompt this long")
        return np.zeros((1, len(ids), VOCAB), dtype=np.float32)

    def decode(self, token, pos):
        return np.zeros((1, VOCAB), dtype=np.float32)


def test_one_failing_cell_keeps_the_cells_already_measured():
    sweep = load_script("dory_sweep")
    measured = sweep.measure_speed(
        "model", "cfg", [2048, 24576], [2], tokens=32,
        runtime_factory=lambda _dir: _SweepRuntime(fail_above=20000))
    good = [value for key, value in measured.items()
            if isinstance(value, dict) and "decode_tok_s" in value]
    bad = [value for key, value in measured.items()
           if isinstance(value, dict) and "error" in value]
    assert len(good) == 1 and good[0]["prompt_tokens"] == 2016
    assert len(bad) == 1 and "refuses a prompt this long" in bad[0]["error"]
    assert measured["errors"] == [key for key, value in measured.items()
                                  if isinstance(value, dict) and "error" in value]


def test_a_context_sized_sweep_cell_is_measured_not_overflowed():
    sweep = load_script("dory_sweep")
    measured = sweep.measure_speed(
        "model", "cfg", [24576], [2], tokens=32,
        runtime_factory=lambda _dir: _SweepRuntime())
    cells = [value for key, value in measured.items()
             if isinstance(value, dict) and "decode_tok_s" in value]
    assert len(cells) == 1
    assert cells[0]["prompt_tokens"] == 24544
    assert cells[0]["decode_headroom"] == 32
    assert "errors" not in measured
