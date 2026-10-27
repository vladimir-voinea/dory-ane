"""Tests for the fixed Dory evaluation set (anemll.eval.eval_set)."""

import glob
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from anemll.eval.eval_set import (
    FIXTURE_DIR,
    PROVENANCE_FILE,
    TEXT_FILE,
    TOKENS_FILE,
    EvalSet,
    load_eval_set,
)


def test_rows_are_the_second_half_of_every_context():
    ids = np.arange(16, dtype=np.int32)
    evalset = EvalSet(token_ids=ids, chunk_size=8)
    assert evalset.n_chunks == 2
    assert evalset.first_scored == 4
    assert evalset.rows_per_chunk == 3
    assert evalset.n_rows == 6
    # absolute positions: the second half of each context, contexts in order
    assert evalset.scored_positions().tolist() == [4, 5, 6, 12, 13, 14]
    assert evalset.scored_target_indices().tolist() == [5, 6, 7, 13, 14, 15]
    assert evalset.scored_target_tokens().tolist() == [5, 6, 7, 13, 14, 15]


def test_partial_context_is_rejected():
    with pytest.raises(ValueError, match="whole number"):
        EvalSet(token_ids=np.arange(10, dtype=np.int32), chunk_size=8)


def test_odd_chunk_size_is_rejected():
    with pytest.raises(ValueError, match="chunk_size"):
        EvalSet(token_ids=np.arange(8, dtype=np.int32), chunk_size=1)


@pytest.mark.skipif(not (FIXTURE_DIR / TOKENS_FILE).is_file(), reason="fixture not built")
def test_committed_fixture_shape():
    evalset = load_eval_set()
    assert evalset.chunk_size == 4096
    assert evalset.n_chunks == 2
    assert evalset.token_ids.size == 8192
    assert evalset.n_rows == 4094
    assert evalset.token_ids.dtype == np.int32
    assert np.all(evalset.token_ids >= 0)
    assert np.all(evalset.token_ids < 131072)


@pytest.mark.skipif(not (FIXTURE_DIR / PROVENANCE_FILE).is_file(), reason="fixture not built")
def test_committed_fixture_checksums_match_provenance():
    provenance = json.loads((FIXTURE_DIR / PROVENANCE_FILE).read_text())
    for key, name in (("text_sha256", TEXT_FILE), ("tokens_sha256", TOKENS_FILE)):
        digest = hashlib.sha256((FIXTURE_DIR / name).read_bytes()).hexdigest()
        assert digest == provenance[key], f"{name} does not match its recorded checksum"
    assert provenance["n_tokens"] == 8192
    assert provenance["n_scored_rows"] == 4094
    assert "public domain" in provenance["source"]["licence"]


def _dory_tokenizer():
    pytest.importorskip("transformers")
    path = os.environ.get("DORY_TOKENIZER")
    if not path:
        hits = glob.glob(
            os.path.expanduser(
                "~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"
                "/snapshots/*/tokenizer.json"
            )
        )
        path = str(Path(hits[0]).parent) if hits else None
    if not path or not Path(path).is_dir():
        pytest.skip("Dory tokenizer.json not available; set DORY_TOKENIZER")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path)


@pytest.mark.skipif(not (FIXTURE_DIR / TEXT_FILE).is_file(), reason="fixture not built")
def test_fixture_text_retokenises_to_the_id_file():
    tokenizer = _dory_tokenizer()
    text = (FIXTURE_DIR / TEXT_FILE).read_text(encoding="utf-8")
    ids = np.loadtxt(FIXTURE_DIR / TOKENS_FILE, dtype=np.int64, ndmin=1)
    assert tokenizer.encode(text, add_special_tokens=False) == ids.tolist()
