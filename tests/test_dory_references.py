"""Integration tests: the real reference artifacts built from the BF16 GGUF.

These are skipped unless the artifacts directory exists (they are 1.07 GB each
and live outside the repository). They run the real scorer against real
llama.cpp dumps - no mocking anywhere.

Rebuild them with:

    python scripts/make_dory_references.py --llama-perplexity <bin/llama-perplexity> \
        --bf16-gguf <Dory1.0-4B-BF16.gguf> --q8-gguf <Dory1.0-4B-Q8_0.gguf>
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from anemll.eval.eval_set import FIXTURE_DIR, TOKENS_FILE, load_eval_set
from anemll.eval.kld import read_llama_cpp_kld, score_files

ARTIFACTS = Path(
    os.environ.get("DORY_ARTIFACTS", Path.home() / "dev" / "dory-ane-artifacts" / "refs")
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not ARTIFACTS.is_dir(), reason=f"{ARTIFACTS} not built"),
]

EXACT = {
    "bf16_exact_loop1": 1,
    "bf16_exact_loop2": 2,
    "bf16_exact_loop4": 4,
}
# name -> (loops, n_batch): decode references exist at every reported loop count
DECODE = {
    "bf16_decode_loop1_batch1": (1, 1),
    "bf16_decode_loop1_batch64": (1, 64),
    "bf16_decode_loop2_batch1": (2, 1),
    "bf16_decode_loop2_batch64": (2, 64),
    "bf16_decode_loop4_batch1": (4, 1),
    "bf16_decode_loop4_batch64": (4, 64),
}


def _meta(name: str) -> dict:
    path = ARTIFACTS / f"{name}.kld"
    if not path.is_file():
        pytest.skip(f"{path} not built")
    return json.loads(path.with_suffix(".kld.meta.json").read_text())


@pytest.mark.parametrize("name,loops", sorted(EXACT.items()))
def test_exact_reference_is_the_documented_artifact(name, loops):
    meta = _meta(name)
    assert meta["mode"] == "exact"
    assert meta["kv_cache_mode"] == "per_loop"
    assert meta["loops"] == loops
    assert meta["chunk_size"] == 4096
    assert meta["n_chunks"] == 2
    assert meta["n_rows"] == load_eval_set(FIXTURE_DIR).n_rows
    # One ubatch covering the whole context: a context split into ubatches is
    # not a full recompute once the 1024-token sliding window is active, so
    # -ub 1024 (the ticket's first build) must never come back.
    assert meta["n_batch"] == meta["chunk_size"]
    assert meta["n_ubatch"] == meta["chunk_size"]
    command = meta["tool"]["command"]
    assert command[command.index("-ub") + 1] == str(meta["chunk_size"])
    assert "recurrent_kv_cache_mode=str:per_loop" in " ".join(command)


@pytest.mark.parametrize(
    "name,loops,batch", sorted((n, loops, batch) for n, (loops, batch) in DECODE.items())
)
def test_decode_reference_is_the_documented_artifact(name, loops, batch):
    meta = _meta(name)
    assert meta["mode"] == "decode"
    assert meta["kv_cache_mode"] == "last_loop"
    assert meta["loops"] == loops
    assert meta["n_batch"] == batch
    assert meta["n_ubatch"] == batch
    assert "recurrent_kv_cache_mode=str:last_loop" in " ".join(meta["tool"]["command"])


def test_every_reference_scores_the_committed_eval_set():
    tokens = load_eval_set(FIXTURE_DIR).token_ids
    for name in list(EXACT) + list(DECODE):
        path = ARTIFACTS / f"{name}.kld"
        if not path.is_file():
            pytest.skip(f"{path} not built")
        source = read_llama_cpp_kld(path)
        assert source.n_rows == 4094
        assert source.vocab_size == 131072
        assert np.array_equal(source.tokens, tokens), f"{name} is on different tokens"
        assert source.positions.tolist() == load_eval_set(FIXTURE_DIR).scored_positions().tolist()


def test_q8_0_exact_loop2_lands_near_the_published_figures():
    candidate = ARTIFACTS / "q8_0_exact_loop2.kld"
    reference = ARTIFACTS / "bf16_exact_loop2.kld"
    if not candidate.is_file():
        pytest.skip(f"{candidate} not built")
    result = score_files(reference, candidate)
    # Measured here: 0.000423 mean KLD / 98.461% top-1 against the -ub 4096
    # reference. The published Q8_0
    # figures are 0.002972 / 98.852% on 32k positions of NVIDIA post-training
    # data; on this prose set the quantisation error shows up smaller because
    # the reference distributions are much flatter (see docs/dory-eval.md).
    # The gate is therefore on top-1, plus "nowhere near Q4_K_M territory".
    assert result.mean_kld < 0.002, result.summary()
    assert result.top1_agreement > 0.98, result.summary()


def test_decode_mode_gap_is_the_last_loop_floor():
    # Each decode reference is scored against the exact reference with the SAME
    # loop count: that is the floor a last_loop candidate can reach, because it
    # cannot beat the batching it runs.  At loop 2 on this set: one token per
    # step 0.006975 KLD / 96.116% top-1, 64-token batches 0.002857 / 97.557%.
    missing = [
        name
        for name, (loops, _batch) in DECODE.items()
        if not (ARTIFACTS / f"{name}.kld").is_file()
        or not (ARTIFACTS / f"bf16_exact_loop{loops}.kld").is_file()
    ]
    if missing:
        pytest.skip(f"not built: {', '.join(sorted(missing))}")
    gaps = {}
    for name, (loops, _batch) in DECODE.items():
        result = score_files(
            ARTIFACTS / f"bf16_exact_loop{loops}.kld", ARTIFACTS / f"{name}.kld"
        )
        gaps[name] = result.mean_kld
        assert result.mean_kld < 0.02, f"{name}: {result.summary()}"
        assert result.top1_agreement > 0.95, f"{name}: {result.summary()}"
    scores_dir = ARTIFACTS / "scores"
    report = scores_dir / "decode_gap.json"
    if report.is_file():
        recorded = json.loads(report.read_text())
        for name, value in gaps.items():
            assert recorded[name]["mean_kld"] == pytest.approx(value, rel=1e-6)
