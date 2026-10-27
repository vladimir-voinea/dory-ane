"""The fixed held-out evaluation set used for every Dory parity comparison.

The set is a token sequence of exactly n_chunks * chunk_size ids, split into
n_chunks independent contexts of chunk_size tokens.  Following llama.cpp's
llama-perplexity convention (the one the published Dory KL table uses), only
the *second half* of each context is scored: the first chunk_size // 2 tokens
are context only.  A scored row is a distribution over the vocabulary that
predicts the next token, so row r of the second half of context c predicts
token c * chunk_size + p + 1 for p in [chunk_size // 2, chunk_size - 1).

The fixture directory holds:

eval_text.txt
    The exact text, decoded from eval_tokens.txt; re-tokenising it with Dory's
    tokenizer reproduces the id file.
eval_tokens.txt
    One integer token id per line, exactly n_chunks * chunk_size of them.
PROVENANCE.json
    Where the text came from, how it was cut, and the checksums of both files.

Rebuild with:

    python -m anemll.eval.eval_set --source /path/to/pg1342.txt

() see docs/dory-eval.md for the download command and the source licence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .kld import scored_positions

DEFAULT_CHUNK_SIZE = 4096
DEFAULT_N_CHUNKS = 2
DEFAULT_EXCERPTS = 2

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "dory_eval"

TEXT_FILE = "eval_text.txt"
TOKENS_FILE = "eval_tokens.txt"
PROVENANCE_FILE = "PROVENANCE.json"

GUTENBERG_START = "*** START OF THE PROJECT GUTENBERG EBOOK"
GUTENBERG_END = "*** END OF THE PROJECT GUTENBERG EBOOK"


@dataclass(frozen=True)
class EvalSet:
    """A fixed token sequence split into scoring contexts."""

    token_ids: np.ndarray  # int32, shape [n_chunks * chunk_size]
    chunk_size: int
    name: str = "dory-eval"

    def __post_init__(self) -> None:
        if self.chunk_size <= 1:
            raise ValueError(f"chunk_size must be > 1, got {self.chunk_size}")
        if self.token_ids.ndim != 1:
            raise ValueError(f"token_ids must be 1-D, got shape {self.token_ids.shape}")
        n = self.token_ids.shape[0]
        if n % self.chunk_size != 0:
            raise ValueError(
                f"{n} token ids is not a whole number of {self.chunk_size}-token contexts"
            )

    @property
    def n_chunks(self) -> int:
        return int(self.token_ids.shape[0] // self.chunk_size)

    @property
    def first_scored(self) -> int:
        return self.chunk_size // 2

    @property
    def rows_per_chunk(self) -> int:
        return self.chunk_size - 1 - self.first_scored

    @property
    def n_rows(self) -> int:
        return self.n_chunks * self.rows_per_chunk

    def scored_positions(self) -> np.ndarray:
        """Absolute token index of every scored row, contexts in order.

        This is exactly the 'positions' array of the candidate format: row r
        is the distribution over the vocabulary after token scored_positions()[r]
        and predicts scored_positions()[r] + 1.
        """
        return scored_positions(self.n_chunks, self.chunk_size)

    def scored_target_indices(self) -> np.ndarray:
        """Index into token_ids of the token each scored row predicts."""
        return (self.scored_positions() + 1).astype(np.int64)

    def scored_target_tokens(self) -> np.ndarray:
        return self.token_ids[self.scored_target_indices()]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_eval_set(directory: Path | str = FIXTURE_DIR) -> EvalSet:
    """Load the committed token-id fixture from directory."""
    directory = Path(directory)
    tokens_path = directory / TOKENS_FILE
    provenance_path = directory / PROVENANCE_FILE
    if not tokens_path.is_file():
        raise FileNotFoundError(
            f"missing {tokens_path}; rebuild it with 'python -m anemll.eval.eval_set'"
        )
    token_ids = np.loadtxt(tokens_path, dtype=np.int64, ndmin=1).astype(np.int32)
    provenance = {}
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text())
    return EvalSet(
        token_ids=token_ids,
        chunk_size=int(provenance.get("chunk_size", DEFAULT_CHUNK_SIZE)),
        name=str(provenance.get("name", "dory-eval")),
    )


# --------------------------------------------------------------------------- #
# Building the fixtures
# --------------------------------------------------------------------------- #


def gutenberg_body(text: str) -> str:
    """Strip the Project Gutenberg header and footer."""
    start = text.find(GUTENBERG_START)
    if start >= 0:
        start = text.find("\n", start) + 1
    else:
        start = 0
    end = text.find(GUTENBERG_END)
    if end < 0:
        end = len(text)
    return text[start:end].replace("\r\n", "\n")


def chapter_starts(body: str) -> list[int]:
    return [m.start() for m in re.finditer(r"(?m)^Chapter\s+[IVXL]+\b", body)]


def _roundtrip_ids(tokenizer, ids: np.ndarray) -> str:
    """Text that re-tokenises to exactly ids, or raise ValueError."""
    text = tokenizer.decode(ids.tolist(), skip_special_tokens=False)
    back = np.asarray(tokenizer.encode(text, add_special_tokens=False), dtype=np.int32)
    if back.shape != ids.shape or not np.array_equal(back, ids):
        raise ValueError("tokenisation of the decoded excerpt does not round-trip")
    return text


def build_eval_set(
    source: Path | str,
    out_dir: Path | str = FIXTURE_DIR,
    tokenizer_path: Path | str | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    n_chunks: int = DEFAULT_N_CHUNKS,
    excerpts: int = DEFAULT_EXCERPTS,
    source_url: str = "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
) -> dict:
    """Cut the fixtures from a Project Gutenberg text file."""
    from transformers import AutoTokenizer

    source = Path(source)
    raw = source.read_text(encoding="utf-8")
    body = gutenberg_body(raw)
    if tokenizer_path is None:
        tokenizer_path = os.environ.get("DORY_TOKENIZER")
    if tokenizer_path is None:
        raise SystemExit(
            "set DORY_TOKENIZER (or --tokenizer) to the Dory tokenizer directory, e.g. "
            "~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"
            "/snapshots/<rev>"
        )
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    vocab_size = len(tokenizer)

    starts = chapter_starts(body)
    if not starts:
        raise SystemExit(f"{source}: no 'Chapter N' headings found")
    if excerpts == 1:
        chosen = [starts[0]]
    else:
        # Spread the excerpts across the book so the set is not one scene.
        step = max(1, len(starts) // excerpts)
        chosen = [starts[min(i * step, len(starts) - 1)] for i in range(excerpts)]
    heading_of = {offset: body[offset : body.find("\n", offset)] for offset in starts}
    per_excerpt = n_chunks * chunk_size // len(chosen)
    if per_excerpt * len(chosen) != n_chunks * chunk_size:
        raise SystemExit("n_chunks * chunk_size must divide evenly by the excerpt count")

    all_ids: list[np.ndarray] = []
    windows = []
    for k, start in enumerate(chosen):
        excerpt_text = body[start : start + 200_000]
        ids = np.asarray(tokenizer.encode(excerpt_text, add_special_tokens=False), dtype=np.int32)
        taken = None
        for shift in range(0, 40):
            window = ids[shift : shift + per_excerpt]
            if window.size < per_excerpt:
                break
            try:
                text = _roundtrip_ids(tokenizer, window)
            except ValueError:
                continue
            taken = (shift, window, text)
            break
        if taken is None:
            raise SystemExit(f"could not find a round-tripping window at chapter {k + 1}")
        shift, window, text = taken
        all_ids.append(window)
        windows.append(
            {
                "excerpt": k,
                "heading": heading_of.get(start, ""),
                "body_char_offset": int(start),
                "token_shift": int(shift),
            }
        )

    ids = np.concatenate(all_ids)
    text = "".join(_roundtrip_ids(tokenizer, part) for part in all_ids)
    back = np.asarray(tokenizer.encode(text, add_special_tokens=False), dtype=np.int32)
    if not np.array_equal(back, ids):
        # Re-cut from the joined text so text and ids agree exactly.
        raise SystemExit("joined excerpts do not re-tokenise to the same ids")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / TEXT_FILE).write_text(text, encoding="utf-8")
    (out_dir / TOKENS_FILE).write_text("".join(f"{int(t)}\n" for t in ids), encoding="utf-8")
    provenance = {
        "name": "dory-eval",
        "chunk_size": int(chunk_size),
        "n_chunks": int(n_chunks),
        "n_tokens": int(ids.size),
        "n_scored_rows": int(n_chunks * (chunk_size - 1 - chunk_size // 2)),
        "source": {
            "url": source_url,
            "sha256": _sha256(source),
            "licence": "public domain (Project Gutenberg, United States)",
            "edition": "Pride and Prejudice, Project Gutenberg eBook #1342",
        },
        "tokenizer": {
            "name": "Dory1.0-4B tokenizer (tokenizer.json from the NVIDIA BF16 release)",
            "vocab_size": int(vocab_size),
            "path": str(tokenizer_path),
            "add_special_tokens": False,
        },
        "excerpts": windows,
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "tokens_sha256": hashlib.sha256(
            "".join(f"{int(t)}\n" for t in ids).encode("utf-8")
        ).hexdigest(),
    }
    (out_dir / PROVENANCE_FILE).write_text(json.dumps(provenance, indent=2) + "\n")
    return provenance


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m anemll.eval.eval_set",
        description="Build the fixed Dory evaluation set fixtures.",
    )
    parser.add_argument("--source", required=True, help="Project Gutenberg text file")
    parser.add_argument("--out", default=str(FIXTURE_DIR), help="fixture directory")
    parser.add_argument("--tokenizer", default=None, help="Dory tokenizer directory")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--n-chunks", type=int, default=DEFAULT_N_CHUNKS)
    parser.add_argument("--excerpts", type=int, default=DEFAULT_EXCERPTS)
    args = parser.parse_args(argv)
    provenance = build_eval_set(
        args.source,
        out_dir=args.out,
        tokenizer_path=args.tokenizer,
        chunk_size=args.chunk_size,
        n_chunks=args.n_chunks,
        excerpts=args.excerpts,
    )
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
