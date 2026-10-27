"""Tests for the Dory reference scorer (anemll.eval.kld).

The seam under test is the documented candidate file contract plus the metrics
it produces; nothing here reaches into internals.  Expected values come from
hand arithmetic or scipy, never from re-running the implementation.
"""

import json
import math

import numpy as np
import pytest

from anemll.eval import kld
from anemll.eval.kld import (
    ArtifactError,
    array_source,
    read_candidate_npz,
    read_llama_cpp_kld,
    score_arrays,
    score_sources,
    scored_positions,
    write_candidate,
)

VOCAB = 3

# One row, absolute position 0 of a two-token sequence: the row predicts tokens[1].
TOKENS = np.array([9, 0], dtype=np.int32)
POS = np.array([0], dtype=np.int32)


def logprobs(probs):
    return np.log(np.asarray(probs, dtype=np.float64))


P = [0.8, 0.1, 0.1]
Q = [0.4, 0.5, 0.1]

# Worked out by hand:
# KLD(p||q) = 0.8*ln(2) + 0.1*ln(0.2) + 0.1*ln(1) = 0.39357395...
# KLD(q||p) = 0.4*ln(0.5) + 0.5*ln(5) + 0.1*ln(1) = 0.52746009...
KLD_P_Q = 0.8 * math.log(2.0) + 0.1 * math.log(0.2)
KLD_Q_P = 0.4 * math.log(0.5) + 0.5 * math.log(5.0)


def test_kld_of_identical_distributions_is_zero():
    lp = logprobs([P, Q])
    result = score_arrays(lp, lp)
    assert result.mean_kld == pytest.approx(0.0, abs=1e-12)
    assert result.top1_agreement == pytest.approx(1.0)
    assert result.n_rows == 2
    assert result.vocab_size == VOCAB
    assert not result.has_ppl


def test_kld_is_reference_to_candidate_and_asymmetric():
    ref = logprobs([P])
    cand = logprobs([Q])
    forward = score_arrays(ref, cand)
    backward = score_arrays(ref, cand, direction="cand->ref")
    # rel=1e-6 floor: the scorer works in the artifact's dtype (float16 in
    # production, and llama.cpp's own reference is 16-bit fixed point), so
    # chasing float64 purity here would assert nothing about behaviour.
    assert forward.mean_kld == pytest.approx(KLD_P_Q, rel=1e-6)
    assert backward.mean_kld == pytest.approx(KLD_Q_P, rel=1e-6)


def test_kld_matches_scipy_entropy():
    scipy_special = pytest.importorskip("scipy.special")
    rng = np.random.default_rng(7)
    ref = rng.normal(size=(5, 50))
    cand = rng.normal(size=(5, 50))
    ref = ref - scipy_special.logsumexp(ref, axis=1, keepdims=True)
    cand = cand - scipy_special.logsumexp(cand, axis=1, keepdims=True)
    expected = float(np.mean(np.sum(np.exp(ref) * (ref - cand), axis=1)))
    assert score_arrays(ref, cand).mean_kld == pytest.approx(expected, rel=1e-6)


def test_top1_agreement_counts_argmax_matches():
    # argmax ref: 0, 1, 2, 0    argmax cand: 0, 2, 2, 1  ->  2 of 4 agree
    ref = logprobs([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]])
    cand = logprobs([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.1, 0.1, 0.8], [0.3, 0.4, 0.3]])
    result = score_arrays(ref, cand)
    assert result.top1_agreement == pytest.approx(2 / 4)


def test_worst_positions_are_sorted_and_identify_the_top_tokens():
    ref = logprobs([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
    cand = logprobs([[0.8, 0.1, 0.1], [0.8, 0.1, 0.1], [0.4, 0.5, 0.1]])
    # by hand: row 1 = 0.1*ln(1/8) + 0.8*ln(8) = 1.455608...
    #          row 2 = 0.1*ln(1/4) + 0.1*ln(1/5) + 0.8*ln(8) = 1.363978...
    result = score_arrays(ref, cand, worst=2)
    assert [item["row"] for item in result.worst] == [1, 2]
    assert result.worst[0]["ref_top"] == 1
    assert result.worst[0]["cand_top"] == 0
    assert result.worst[0]["kld"] == pytest.approx(
        0.1 * math.log(0.125) + 0.8 * math.log(8), rel=1e-6
    )
    assert result.worst[1]["kld"] == pytest.approx(
        0.1 * math.log(0.25) + 0.1 * math.log(0.2) + 0.8 * math.log(8), rel=1e-6
    )
    assert result.worst[0]["kld"] > result.worst[1]["kld"]
    assert result.kld_max == pytest.approx(result.worst[0]["kld"], rel=1e-9)
    assert result.kld_99 >= result.kld_median


def test_clip_ignores_tokens_below_the_reference_floor():
    # Token 2 carries 1% of the reference mass with logp -4.6: below clip=-1 it
    # must not contribute even though the candidate gives it almost nothing.
    ref = logprobs([[0.5, 0.49, 0.01]])
    cand = logprobs([[0.5, 0.499999, 0.000001]])
    clipped = score_arrays(ref, cand, clip=-1.0)
    unclipped = score_arrays(ref, cand, clip=-40.0)
    assert clipped.mean_kld == pytest.approx(0.49 * math.log(0.49 / 0.499999), rel=1e-4)
    assert unclipped.mean_kld == pytest.approx(
        clipped.mean_kld + 0.01 * math.log(0.01 / 0.000001), rel=1e-4
    )
    assert unclipped.mean_kld - clipped.mean_kld == pytest.approx(0.0921, rel=1e-2)


def test_positions_select_the_predicted_tokens_and_the_perplexity():
    # positions 2 and 3 of a 5-token chunk predict tokens[3] and tokens[4]
    tokens = np.array([0, 0, 0, 1, 2], dtype=np.int32)
    positions = np.array([2, 3], dtype=np.int32)
    ref = logprobs([[0.8, 0.1, 0.1], [0.8, 0.1, 0.1]])
    result = score_arrays(ref, ref, tokens=tokens, positions=positions)
    assert result.has_ppl
    assert result.mean_nll_ref == pytest.approx(-float(np.mean(np.log([0.1, 0.1]))))
    assert result.ppl_ref == pytest.approx(math.exp(result.mean_nll_ref))
    # the report carries each row's absolute position and the token it predicts
    assert {item["position"]: item["target"] for item in result.worst} == {2: 1, 3: 2}


def test_blocking_does_not_mix_up_rows():
    # 5 rows scored with a 2-row block: the classic place for an off-by-block bug
    rng = np.random.default_rng(11)
    probs = rng.random((5, VOCAB))
    probs /= probs.sum(axis=1, keepdims=True)
    cand = probs.copy()
    cand[:, 0] *= 0.5
    cand /= cand.sum(axis=1, keepdims=True)
    tokens = (np.arange(7) % VOCAB).astype(np.int32)
    positions = np.arange(0, 5, dtype=np.int32)
    blocked = score_arrays(
        logprobs(probs), logprobs(cand), tokens=tokens, positions=positions, block=2
    )
    whole = score_arrays(
        logprobs(probs), logprobs(cand), tokens=tokens, positions=positions, block=64
    )
    assert blocked.mean_kld == pytest.approx(whole.mean_kld, rel=1e-6)
    assert blocked.mean_nll_cand == pytest.approx(whole.mean_nll_cand, rel=1e-6)
    assert blocked.worst[0]["row"] == whole.worst[0]["row"]


def test_row_count_mismatch_is_rejected():
    ref = logprobs([[0.8, 0.1, 0.1], [0.8, 0.1, 0.1]])
    cand = logprobs([[0.8, 0.1, 0.1]])
    with pytest.raises(ArtifactError, match="row count"):
        score_arrays(ref, cand)


def test_vocabulary_mismatch_is_rejected():
    ref = logprobs([[0.8, 0.1, 0.1]])
    cand = logprobs([[0.5, 0.5]])
    with pytest.raises(ArtifactError, match="vocabulary"):
        score_arrays(ref, cand)


def test_token_mismatch_is_rejected():
    ref = array_source(logprobs([P]), tokens=np.array([1, 2]), positions=POS, name="ref")
    cand = array_source(logprobs([P]), tokens=np.array([1, 3]), positions=POS, name="cand")
    with pytest.raises(ArtifactError, match="token"):
        score_sources(ref, cand)


def test_position_mismatch_is_rejected():
    tokens = np.array([1, 2, 3, 4], dtype=np.int32)
    ref = array_source(logprobs([P]), tokens=tokens, positions=np.array([0]), name="ref")
    cand = array_source(logprobs([P]), tokens=tokens, positions=np.array([1]), name="cand")
    with pytest.raises(ArtifactError, match="position"):
        score_sources(ref, cand)


def test_write_candidate_roundtrip(tmp_path):
    tokens = np.array([0, 0, 0, 1, 2], dtype=np.int32)
    positions = np.array([2, 3], dtype=np.int32)
    rows = logprobs([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
    path = write_candidate(
        tmp_path / "cand.npz",
        rows,
        positions,
        tokens,
        meta={"model": "unit", "loops": 2, "kv_mode": "last_loop", "n_ctx": 5},
    )
    source = read_candidate_npz(path)
    assert source.n_rows == 2
    assert source.vocab_size == VOCAB
    assert source.meta["model"] == "unit"
    assert source.meta["format"] == "dory-logprobs-1"
    assert source.meta["dtype"] == "float32"
    assert np.allclose(source.logp_block(0, 2), rows, atol=1e-3)
    assert source.positions.tolist() == [2, 3]
    assert source.target_tokens.tolist() == [1, 2]


def test_write_candidate_rejects_positions_that_predict_nothing(tmp_path):
    tokens = np.array([0, 0, 0], dtype=np.int32)
    with pytest.raises(ArtifactError, match="positions"):
        write_candidate(tmp_path / "bad.npz", logprobs([P]), np.array([2]), tokens)
    with pytest.raises(ArtifactError, match="positions"):
        write_candidate(tmp_path / "bad.npz", logprobs([P]), np.array([0, 1]), tokens)


def test_candidate_npz_missing_arrays_is_rejected(tmp_path):
    path = tmp_path / "bad.npz"
    np.savez(path, logprobs=np.zeros((1, 3), dtype=np.float16))
    with pytest.raises(ArtifactError, match="positions"):
        read_candidate_npz(path)


def test_score_files_matches_score_arrays(tmp_path):
    tokens = np.array([0, 1, 2, 0, 1, 2, 1], dtype=np.int32)
    positions = np.array([4, 5], dtype=np.int32)
    ref = logprobs([[0.8, 0.1, 0.1], [0.4, 0.5, 0.1]])
    cand = logprobs([[0.8, 0.1, 0.1], [0.5, 0.4, 0.1]])
    ref_path = write_candidate(tmp_path / "ref.npz", ref, positions, tokens)
    cand_path = write_candidate(tmp_path / "cand.npz", cand, positions, tokens)
    from_files = kld.score_files(ref_path, cand_path)
    from_arrays = score_arrays(ref, cand, tokens=tokens, positions=positions)
    assert from_files.has_ppl
    assert from_arrays.ppl_ref == pytest.approx(from_files.ppl_ref, rel=1e-6)
    assert from_files.mean_kld == pytest.approx(from_arrays.mean_kld, rel=1e-6)
    assert from_files.top1_agreement == pytest.approx(from_arrays.top1_agreement)
    assert from_files.rms_dp >= 0.0


def _write_tiny_kld(path, n_ctx, token_ids, rows, n_vocab):
    """Write a llama-perplexity style dump by hand, following its byte layout."""
    header = b"_logits_" + np.array([n_ctx, n_vocab, 1], dtype="<i4").tobytes()
    header += np.asarray(token_ids, dtype="<i4").tobytes()
    padded = n_vocab + (n_vocab % 2)  # llama.cpp pads odd vocabularies
    body = bytearray()
    for row in rows:
        row = np.asarray(row, dtype=np.float64)
        min_logit = float(np.max(row) - 16.0)
        scale = (float(np.max(row)) - min_logit) / 65535.0
        log_sum_exp = float(np.log(np.sum(np.exp(row - np.max(row)))))
        min_log_prob = min_logit - float(np.max(row)) - log_sum_exp
        codes = np.zeros(padded, dtype="<u2")
        codes[:n_vocab] = np.round((row - min_logit) / scale).astype("<u2")
        body += np.array([scale, min_log_prob], dtype="<f4").tobytes()
        body += codes.tobytes()
    path.write_bytes(header + bytes(body))


def test_read_llama_cpp_kld_decodes_rows(tmp_path):
    # n_ctx 6 -> scored positions 3, 4 of the context; two rows
    n_ctx, n_vocab = 6, 3
    rows = [[0.0, -1.0, -2.0], [-2.0, 0.0, -1.0]]
    tokens = [5, 6, 7, 8, 9, 10]
    path = tmp_path / "tiny.kld"
    _write_tiny_kld(path, n_ctx, tokens, rows, n_vocab)
    source = read_llama_cpp_kld(path)
    assert source.n_rows == len(rows)
    assert source.vocab_size == n_vocab
    assert source.tokens.tolist() == tokens
    logp = source.logp_block(0, 2)
    assert logp.shape == (2, n_vocab)
    # Dequantised log-probabilities must be a normalised distribution.
    assert np.exp(logp).sum(axis=1) == pytest.approx([1.0, 1.0], abs=1e-3)
    assert logp[0, 0] > logp[0, 1] > logp[0, 2]
    assert source.positions.tolist() == [3, 4]
    assert source.meta["chunk_size"] == n_ctx


def test_read_llama_cpp_kld_rejects_bad_size(tmp_path):
    path = tmp_path / "bad.kld"
    path.write_bytes(b"_logits_" + np.zeros(12, dtype="<i4").tobytes())
    with pytest.raises(ArtifactError):
        read_llama_cpp_kld(path)


def test_scored_positions_are_the_second_half_of_every_context():
    # 2 contexts of 6: scored positions 3..4 then 9..10
    assert scored_positions(2, 6).tolist() == [3, 4, 9, 10]
    assert scored_positions(1, 4).tolist() == [2]


def test_cli_prints_metrics_and_enforces_gates(tmp_path, capsys):
    tokens = np.array([0, 1, 2, 0, 1, 2, 1], dtype=np.int32)
    positions = np.array([4, 5], dtype=np.int32)
    ref = logprobs([[0.8, 0.1, 0.1], [0.4, 0.5, 0.1]])
    cand = logprobs([[0.8, 0.1, 0.1], [0.5, 0.4, 0.1]])
    ref_path = write_candidate(tmp_path / "ref.npz", ref, positions, tokens)
    cand_path = write_candidate(tmp_path / "cand.npz", cand, positions, tokens)
    report = tmp_path / "report.json"
    code = kld.main(
        [
            "score",
            "--reference",
            str(ref_path),
            "--candidate",
            str(cand_path),
            "--json",
            str(report),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "mean KLD" in out and "top-1" in out
    payload = json.loads(report.read_text())
    assert payload["n_rows"] == 2
    # rows: identical (0), then KL([0.4,0.5,0.1] || [0.5,0.4,0.1]) = 0.022313...
    kl_second = 0.4 * math.log(0.4 / 0.5) + 0.5 * math.log(0.5 / 0.4)
    assert payload["mean_kld"] == pytest.approx(0.5 * kl_second, rel=1e-6)
    assert payload["top1_agreement"] == pytest.approx(0.5)

    code = kld.main(
        [
            "score",
            "--reference",
            str(ref_path),
            "--candidate",
            str(cand_path),
            "--quiet",
            "--max-kld",
            "0.0001",
        ]
    )
    assert code == 1
