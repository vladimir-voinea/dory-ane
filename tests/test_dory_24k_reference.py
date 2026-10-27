"""Tests for the two 24K measurement scripts (no model, no Core ML).

`scripts/dory_score_positioned.py` scores decode-mode log-probabilities at
absolute positions beyond 16K; `scripts/dory_24k_reference.py` builds the
reference those rows are compared against.  Both are importable with numpy alone,
which is deliberate: the reference builder has to run in a python that has no
torch and no coremltools, and the scorer's window selection is pure numpy.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent


def _load(name: str):
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def reference_builder():
    return _load("dory_24k_reference")


@pytest.fixture(scope="module")
def scorer():
    return _load("dory_score_positioned")


_IMPORT_PROBE = (
    "import importlib.util, pathlib, sys;"
    "path = pathlib.Path(sys.argv[1]);"
    "spec = importlib.util.spec_from_file_location('probe', path);"
    "module = importlib.util.module_from_spec(spec);"
    "sys.modules['probe'] = module;"
    "spec.loader.exec_module(module);"
    "bad = [name for name in ('torch', 'coremltools') if name in sys.modules];"
    "print('CLEAN' if not bad else 'IMPURE ' + ','.join(bad))"
)


@pytest.mark.parametrize("script", ["dory_24k_reference", "dory_score_positioned"])
def test_the_scripts_import_without_torch_or_coremltools(script):
    """The builder runs in a numpy-only python, so a fresh interpreter must agree.

    A subprocess, not sys.modules here: the whole suite imports torch and Core ML,
    so an in-process check would only ever measure the suite.
    """
    import subprocess
    import sys

    out = subprocess.run([sys.executable, "-c", _IMPORT_PROBE,
                          str(REPO / "scripts" / f"{script}.py")],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "CLEAN" in out.stdout, out.stdout + out.stderr


def test_the_scripts_are_importable_with_something_to_build(reference_builder, scorer):
    assert hasattr(reference_builder, "build")
    assert hasattr(scorer, "positioned_logprobs")


def test_the_reference_defaults_are_the_24k_gate(reference_builder):
    args = reference_builder.parse_args(["--gguf", "g", "--llama-perplexity", "p",
                                         "--out-dir", "o"])
    assert args.ctx == 24576
    assert args.start == 16384 and args.end == 18432
    assert args.loops == 2
    assert args.name == "bf16_lastloop_fullprefill_abs24k_loop2"
    assert args.mode == "full"


def test_the_long_text_is_at_least_two_contexts(reference_builder, tmp_path):
    """llama-perplexity refuses a file shorter than two contexts."""
    source = tmp_path / "short.txt"
    source.write_text("alpha beta gamma delta. " * 40, encoding="utf-8")
    out = reference_builder.build_long_text(source, 2048, tmp_path / "long.txt")
    text = out.read_text(encoding="utf-8")
    assert text.count("alpha") % 40 == 0       # whole repetitions, not a truncation
    assert text.count("alpha") >= 80           # at least two of them
    assert text.endswith(chr(10))
    assert chr(13) not in text


def test_score_window_slices_the_reference_to_the_window(scorer, monkeypatch):
    """The candidate must see exactly the reference's rows in [start, end)."""
    from anemll.eval import kld

    rows = 8192                                  # 12,288 .. 20,479: covers the window
    positions = np.arange(12288, 12288 + rows, dtype=np.int32)
    tokens = np.arange(12288 + rows + 1, dtype=np.int32)
    logprobs = np.zeros((rows, 8), dtype=np.float16)

    class FakeSource:
        def __init__(self):
            self.positions = positions
            self.tokens = tokens
            self.meta = {"kv_mode": "last_loop"}
            self.calls = []

        def logp_block(self, lo, hi):
            self.calls.append((lo, hi))
            return logprobs[lo:hi]

    fake = FakeSource()
    captured = {}

    def fake_score(sliced, candidate, direction="ref->cand"):
        captured["sliced"] = sliced
        captured["candidate"] = candidate
        captured["direction"] = direction
        return "sentinel"

    monkeypatch.setattr(kld, "read_source", lambda path: fake)
    monkeypatch.setattr(kld, "read_candidate_npz", lambda path: "candidate-object")
    monkeypatch.setattr(kld, "score_sources", fake_score)

    result = scorer.score_window("ref.kld", "cand.npz", 16384, 18432)
    assert result == "sentinel"
    assert captured["direction"] == "ref->cand"
    sliced = captured["sliced"]
    assert sliced.n_rows == 2048
    assert int(sliced.positions[0]) == 16384
    assert int(sliced.positions[-1]) == 18431
    assert np.array_equal(np.asarray(sliced.tokens), tokens)


def test_score_window_refuses_a_window_the_reference_does_not_cover(scorer, monkeypatch):
    from anemll.eval import kld

    class EmptySource:
        positions = np.arange(0, 10, dtype=np.int32)
        tokens = np.arange(11, dtype=np.int32)
        meta = {}

        def logp_block(self, lo, hi):
            return np.zeros((hi - lo, 4), dtype=np.float16)

    monkeypatch.setattr(kld, "read_source", lambda path: EmptySource())
    with pytest.raises(ValueError, match="no rows"):
        scorer.score_window("ref.kld", "cand.npz", 16384, 18432)


# -- the reference builder and the scorer window ------------------------------

# -- the reference builder (review: "name is used before it is assigned") -----

class _StopAfterTheTextBranch:
    """A generator stand-in that stops the run once run_llama_perplexity is called."""

    def run_llama_perplexity(self, *args, **kwargs):
        raise RuntimeError("stop after the text branch")

    def ubatch_for(self, mode, n_batch, ctx):
        return int(n_batch)

    def sha256(self, path):
        return "0" * 64


FIXTURE = REPO / "tests" / "fixtures" / "dory_eval" / "eval_tokens.txt"


def test_the_reference_builder_accepts_an_existing_text_file(
        reference_builder, tmp_path, monkeypatch):
    builder = reference_builder
    text = tmp_path / "text.txt"
    text.write_text("alpha beta gamma. " * 50, encoding="utf-8")
    monkeypatch.setattr(builder, "_generator", lambda: _StopAfterTheTextBranch())
    args = builder.parse_args(["--gguf", "g", "--llama-perplexity", "p",
                               "--out-dir", str(tmp_path), "--text", str(text)])
    with pytest.raises(RuntimeError, match="stop after the text branch"):
        builder.build(args)


def test_the_reference_builder_names_its_batching_mode(reference_builder):
    builder = reference_builder
    args = builder.parse_args(["--gguf", "g", "--llama-perplexity", "p", "--out-dir", "o"])
    assert args.mode == "full"
    assert args.name == "bf16_lastloop_fullprefill_abs24k_loop2"
    assert builder.parse_args(["--gguf", "g", "--llama-perplexity", "p", "--out-dir", "o",
                               "--mode", "decode"]).mode == "decode"


# -- the scorer window --------------------------------------------------------

def test_the_scorer_names_the_largest_window_the_file_allows(scorer, tmp_path, capsys):
    code = scorer.main(["--tokens", str(FIXTURE), "--start", "16384", "--end", "18432",
                        "--out", str(tmp_path / "cand.npz"), "--model-dir", str(tmp_path)])
    assert code == 2
    assert "8,191" in capsys.readouterr().err


def test_the_scorer_refuses_a_window_that_is_empty(scorer, tmp_path, capsys):
    code = scorer.main(["--tokens", str(FIXTURE), "--start", "100", "--end", "100",
                        "--out", str(tmp_path / "cand.npz"), "--model-dir", str(tmp_path)])
    assert code == 2
    assert "does not fit" in capsys.readouterr().err

