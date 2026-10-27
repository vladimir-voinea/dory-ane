"""Score candidate next-token log-probabilities against a reference.

This is the yardstick every Dory port is judged by (spec, Testing Decisions):
one seam, next-token log-probabilities, two reference modes.

Reference artifacts
-------------------
Built by scripts/make_dory_references.py from llama.cpp (PR #30306) running
Dory's BF16 GGUF at a chosen loop count.  A reference is the file
llama-perplexity writes for --kl-divergence-base (a "_logits_" dump) plus a
sidecar <name>.kld.meta.json.  Two modes:

exact
    dory.recurrent_kv_cache_mode=per_loop (the default) decoded with the whole
    context in ONE ubatch (-b n_ctx -ub n_ctx): every loop keeps its own KV
    cache and a chunk is a single physical pass, so the result is a full
    recompute.  -ub must equal n_ctx - below the 1024-token sliding window the
    logits stop being batch-independent, so a dump built with -ub < n_ctx is not
    a reference (see ubatch_for() in scripts/make_dory_references.py).
decode
    dory.recurrent_kv_cache_mode=last_loop: one KV cache per recurrent layer,
    overwritten each loop, so the result depends on how the candidate batches
    its calls.  The reference records the batch size it was built with
    (1 = one token per step, 64 = 64-token prefill batches).

Candidate format
----------------
A single .npz file (numpy, uncompressed: log-probabilities are high-entropy)
with these arrays:

logprobs : float32 [n_rows, vocab_size]   (float16 accepted: 3x smaller, ~1e-4 KLD)
    Row r is the log-softmax over the vocabulary for one scored position.
positions : int32 [n_rows]
    The absolute index into tokens of each row's position.  Row r's
    distribution predicts tokens[positions[r] + 1].  The reference's positions
    are the second half of every chunk: chunk_size//2 .. chunk_size-2 inside
    each chunk, contexts in order, so a 2 x 4096 evaluation gives
    2048..4094 then 6144..8190.
tokens : int32 [n_tokens]
    The full evaluation token sequence from tests/fixtures/dory_eval: the
    scorer compares it with the reference's, so a misalignment is an error and
    not a silent score.
meta : 0-d string array holding a JSON object
    Free-form provenance; 'format', 'model', 'loops', 'kv_mode', 'batch_size'
    and 'n_ctx' are expected.

Log-probabilities must be the natural log of a normalised distribution over the
whole vocabulary; rows are compared position by position and never re-ordered.

Metrics
-------
mean KLD is KLD(reference || candidate), the same quantity llama.cpp's
--kl-divergence reports: sum over v of p_ref(v) * (logp_ref(v) - logp_cand(v)),
restricted to tokens with logp_ref(v) > clip (-16 by default; the reference
itself is clipped at max_logit - 16, so scoring without it would not match the
published tables).  top-1 agreement is the share of rows whose argmax token
matches, ties going to the lowest id on both sides.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

KLD_MAGIC = b"_logits_"
DEFAULT_CLIP = -16.0
DEFAULT_ROW_BLOCK = 64
FORMAT_NAME = "dory-logprobs-1"
NPZ_LOGPROBS = "logprobs"
NPZ_POSITIONS = "positions"
NPZ_TOKENS = "tokens"
NPZ_META = "meta"


class ArtifactError(ValueError):
    """A reference or candidate file does not follow the documented format."""


def _decode_meta(value) -> dict:
    if value is None:
        return {}
    try:
        if isinstance(value, np.ndarray):
            value = value.reshape(-1)[0]
        if isinstance(value, bytes):
            value = value.decode()
        meta = json.loads(str(value))
    except Exception as exc:  # noqa: BLE001 - re-raised as ArtifactError
        raise ArtifactError(f"meta is not a JSON object: {exc}") from exc
    if not isinstance(meta, dict):
        raise ArtifactError("meta must decode to a JSON object")
    return meta


def scored_positions(n_chunks: int, chunk_size: int) -> np.ndarray:
    """Absolute positions of the scored rows: the second half of each context.

    Context c covers positions c*chunk_size .. c*chunk_size + chunk_size - 1;
    the scored input positions are c*chunk_size + chunk_size//2 ..
    c*chunk_size + chunk_size - 2 (the last position of a context predicts
    nothing), contexts in order.
    """
    if chunk_size <= 2:
        raise ArtifactError(f"chunk_size must be > 2, got {chunk_size}")
    offsets = np.arange(int(n_chunks), dtype=np.int64) * chunk_size
    within = np.arange(chunk_size // 2, chunk_size - 1, dtype=np.int64)
    return (offsets[:, None] + within[None, :]).reshape(-1).astype(np.int32)


class LogprobSource:
    """Per-position log-probabilities plus the tokens they were built on."""

    def __init__(self, name, n_rows, vocab_size, tokens, positions, meta, reader):
        self.name = name
        self.n_rows = int(n_rows)
        self.vocab_size = int(vocab_size)
        self.tokens = np.asarray(tokens, dtype=np.int64)
        self.positions = np.asarray(positions, dtype=np.int64)
        self.meta = dict(meta)
        self._reader = reader
        # An empty positions array is allowed (in-memory arrays scored for KLD
        # only, where there is no token sequence to align to); a populated one
        # must have exactly one entry per row.
        if self.positions.size and self.positions.shape[0] != self.n_rows:
            raise ArtifactError(
                f"{name}: {self.positions.shape[0]} positions for {self.n_rows} rows"
            )

    def logp_block(self, start: int, stop: int) -> np.ndarray:
        """float32 log-probabilities for rows [start, stop)."""
        return self._reader(start, stop)

    def iter_blocks(self, block: int = DEFAULT_ROW_BLOCK) -> Iterator[tuple[int, np.ndarray]]:
        for start in range(0, self.n_rows, block):
            stop = min(start + block, self.n_rows)
            yield start, self.logp_block(start, stop)

    @property
    def target_tokens(self) -> np.ndarray:
        return self.tokens[self.positions + 1]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"LogprobSource({self.name!r}, rows={self.n_rows}, vocab={self.vocab_size})"


def _array_reader(array: np.ndarray):
    def read(start: int, stop: int) -> np.ndarray:
        return np.asarray(array[start:stop], dtype=np.float32)

    return read


def array_source(logprobs, tokens=None, positions=None, name="array", meta=None):
    """Wrap a dense array (numpy, torch or anything exposing __array__)."""
    array = np.asarray(logprobs)
    if array.ndim != 2:
        raise ArtifactError(f"logprobs must be 2-D [n_rows, vocab], got shape {array.shape}")
    if not np.issubdtype(array.dtype, np.floating):
        raise ArtifactError(f"logprobs must be floating point, got {array.dtype}")
    if positions is None:
        positions = np.zeros(0, dtype=np.int64)
    if tokens is None:
        tokens = np.zeros(0, dtype=np.int64)
    return LogprobSource(
        name,
        array.shape[0],
        array.shape[1],
        tokens,
        positions,
        meta or {},
        _array_reader(array),
    )


class _KldFileReader:
    """Row reader for llama.cpp's --kl-divergence-base file.

    Layout: b"_logits_", int32 n_ctx, int32 n_vocab, int32 n_chunk,
    int32 tokens[n_chunk * n_ctx], then one row per scored position -- the
    second half of every context -- as float32 scale, float32 min_log_prob and
    uint16 codes[n_vocab], with logp = scale * code + min_log_prob.
    """

    def __init__(self, path: Path, n_vocab: int, data_offset: int, row_bytes: int):
        self.path = path
        self.n_vocab = n_vocab
        self.data_offset = data_offset
        self.row_bytes = row_bytes

    def read(self, start: int, stop: int) -> np.ndarray:
        if stop <= start:
            return np.zeros((0, self.n_vocab), dtype=np.float32)
        count = stop - start
        out = np.empty((count, self.n_vocab), dtype=np.float32)
        with open(self.path, "rb") as fh:
            fh.seek(self.data_offset + start * self.row_bytes)
            for i in range(count):
                raw = fh.read(self.row_bytes)
                if len(raw) != self.row_bytes:
                    raise ArtifactError(
                        f"{self.path}: truncated at row {start + i} "
                        f"(read {len(raw)} of {self.row_bytes} bytes)"
                    )
                scale, min_log_prob = np.frombuffer(raw, dtype="<f4", count=2)
                codes = np.frombuffer(raw, dtype="<u2", offset=8, count=self.n_vocab)
                out[i] = codes.astype(np.float32) * scale + min_log_prob
        return out


def read_llama_cpp_kld(path: str | Path) -> LogprobSource:
    """Read a llama-perplexity --kl-divergence-base dump."""
    path = Path(path)
    size = path.stat().st_size
    with open(path, "rb") as fh:
        head = fh.read(20)
        if len(head) < 20 or head[:8] != KLD_MAGIC:
            raise ArtifactError(f"{path}: not a llama.cpp logits dump (bad magic)")
        n_ctx, n_vocab, n_chunk = np.frombuffer(head, dtype="<i4", count=3, offset=8)
        tokens = np.frombuffer(fh.read(4 * int(n_ctx) * int(n_chunk)), dtype="<i4")
    n_ctx, n_vocab, n_chunk = int(n_ctx), int(n_vocab), int(n_chunk)
    if n_ctx < 4 or n_vocab < 1 or n_chunk < 1:
        raise ArtifactError(f"{path}: empty dump (n_ctx={n_ctx}, n_vocab={n_vocab})")
    rows_per_chunk = n_ctx - 1 - n_ctx // 2
    n_rows = n_chunk * rows_per_chunk
    row_bytes = 2 * (n_vocab + (n_vocab % 2) + 4)
    data_offset = 20 + 4 * n_ctx * n_chunk
    expected = data_offset + n_rows * row_bytes
    if size != expected:
        raise ArtifactError(
            f"{path}: size {size} != expected {expected} "
            f"(n_ctx={n_ctx}, n_vocab={n_vocab}, n_chunk={n_chunk})"
        )
    meta = {"format": "llama-cpp-logits", "n_ctx": n_ctx, "chunk_size": n_ctx,
            "n_chunk": n_chunk, "chunks": n_chunk, "rows_per_chunk": rows_per_chunk}
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    if sidecar.is_file():
        meta.update(json.loads(sidecar.read_text()))
    return LogprobSource(
        str(path),
        n_rows,
        n_vocab,
        tokens,
        scored_positions(n_chunk, n_ctx),
        meta,
        _KldFileReader(path, n_vocab, data_offset, row_bytes).read,
    )


def read_candidate_npz(path: str | Path) -> LogprobSource:
    """Read a candidate .npz written by write_candidate()."""
    path = Path(path)
    try:
        data = np.load(path)
    except Exception as exc:  # noqa: BLE001
        raise ArtifactError(f"{path}: cannot open as .npz: {exc}") from exc
    for key in (NPZ_LOGPROBS, NPZ_POSITIONS, NPZ_TOKENS):
        if key not in data.files:
            raise ArtifactError(f"{path}: missing '{key}' (found {sorted(data.files)})")
    logprobs = data[NPZ_LOGPROBS]
    if logprobs.ndim != 2:
        raise ArtifactError(f"{path}: logprobs must be 2-D, got shape {logprobs.shape}")
    if not np.issubdtype(logprobs.dtype, np.floating):
        raise ArtifactError(f"{path}: logprobs must be floating point, got {logprobs.dtype}")
    meta = _decode_meta(data[NPZ_META]) if NPZ_META in data.files else {}
    return LogprobSource(
        str(path),
        logprobs.shape[0],
        logprobs.shape[1],
        data[NPZ_TOKENS],
        data[NPZ_POSITIONS],
        meta,
        _array_reader(logprobs),
    )


def read_source(path: str | Path) -> LogprobSource:
    """Read either a reference dump (.kld) or a candidate (.npz)."""
    path = Path(path)
    if path.name.endswith(".kld"):
        return read_llama_cpp_kld(path)
    return read_candidate_npz(path)


def write_candidate(
    path: str | Path,
    logprobs,
    positions,
    tokens,
    meta: dict | None = None,
    dtype: str = "float32",
) -> Path:
    """Write the documented candidate file. This is the only writer."""
    path = Path(path)
    array = np.asarray(logprobs, dtype=dtype)
    if array.ndim != 2:
        raise ArtifactError(f"logprobs must be 2-D [n_rows, vocab], got {array.shape}")
    positions = np.asarray(positions, dtype=np.int32)
    tokens = np.asarray(tokens, dtype=np.int32)
    if positions.shape != (array.shape[0],):
        raise ArtifactError(
            f"positions must have one entry per row ({array.shape[0]}), got {positions.shape}"
        )
    if tokens.ndim != 1:
        raise ArtifactError(f"tokens must be 1-D, got shape {tokens.shape}")
    if positions.size and (positions.min() < 0 or positions.max() >= tokens.size - 1):
        raise ArtifactError(
            f"positions must index a token and its successor in tokens[0:{tokens.size}]"
        )
    payload = dict(meta or {})
    payload.setdefault("format", FORMAT_NAME)
    payload.setdefault("dtype", str(array.dtype))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        logprobs=array,
        positions=positions,
        tokens=tokens,
        meta=np.array(json.dumps(payload, sort_keys=True)),
    )
    return path


@dataclass
class ScoreResult:
    """Outcome of scoring one candidate against one reference."""

    reference: str
    candidate: str
    n_rows: int
    vocab_size: int
    mean_kld: float
    kld_stderr: float
    kld_median: float
    kld_p90: float
    kld_99: float
    kld_max: float
    top1_agreement: float
    mean_nll_ref: float
    mean_nll_cand: float
    rms_dp: float
    clip: float
    direction: str = "ref->cand"
    worst: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def has_ppl(self) -> bool:
        return not math.isnan(self.mean_nll_ref)

    @property
    def ppl_ref(self) -> float:
        return float("nan") if math.isnan(self.mean_nll_ref) else float(math.exp(self.mean_nll_ref))

    @property
    def ppl_cand(self) -> float:
        return (
            float("nan") if math.isnan(self.mean_nll_cand) else float(math.exp(self.mean_nll_cand))
        )

    def as_dict(self) -> dict:
        out = {
            "reference": self.reference,
            "candidate": self.candidate,
            "n_rows": self.n_rows,
            "vocab_size": self.vocab_size,
            "mean_kld": self.mean_kld,
            "kld_stderr": self.kld_stderr,
            "kld_median": self.kld_median,
            "kld_p90": self.kld_p90,
            "kld_99": self.kld_99,
            "kld_max": self.kld_max,
            "top1_agreement": self.top1_agreement,
            "rms_dp": self.rms_dp,
            "direction": self.direction,
            "clip": self.clip,
            "mean_nll_ref": self.mean_nll_ref,
            "mean_nll_cand": self.mean_nll_cand,
            "worst": list(self.worst),
        }
        if self.has_ppl:
            out["ppl_ref"] = self.ppl_ref
            out["ppl_cand"] = self.ppl_cand
        if self.meta:
            out["meta"] = self.meta
        return out

    def summary(self) -> str:
        lines = [
            f"reference : {self.reference}",
            f"candidate : {self.candidate}",
            f"rows      : {self.n_rows}   vocab: {self.vocab_size}   KLD direction: {self.direction}",
            f"mean KLD  : {self.mean_kld:.6f} +/- {self.kld_stderr:.6f}"
            f"  (median {self.kld_median:.6f}, p90 {self.kld_p90:.4f}, "
            f"p99 {self.kld_99:.4f}, max {self.kld_max:.4f})",
            f"top-1     : {100.0 * self.top1_agreement:.3f}%",
        ]
        if self.has_ppl:
            lines.append(
                f"PPL       : ref {self.ppl_ref:.4f}   cand {self.ppl_cand:.4f}   "
                f"RMS dp {self.rms_dp:.5f}"
            )
        if self.worst:
            lines.append("worst positions:")
            lines.append("     row  position       KLD  ref_top  cand_top   target")
            for item in self.worst:
                target = "-" if item.get("target") is None else str(item["target"])
                lines.append(
                    f"{item['row']:8d} {item['position']:9d} {item['kld']:9.4f} "
                    f"{item['ref_top']:8d} {item['cand_top']:9d} {target:>8s}"
                )
        return "\n".join(lines)


def _check_compatible(ref: LogprobSource, cand: LogprobSource) -> None:
    if ref.vocab_size != cand.vocab_size:
        raise ArtifactError(
            f"vocabulary mismatch: reference has {ref.vocab_size}, candidate has {cand.vocab_size}"
        )
    if ref.n_rows != cand.n_rows:
        raise ArtifactError(
            f"row count mismatch: reference has {ref.n_rows}, candidate has {cand.n_rows}; "
            "a candidate must emit exactly one row per scored position"
        )
    if ref.tokens.size and cand.tokens.size:
        if ref.tokens.shape != cand.tokens.shape or not np.array_equal(ref.tokens, cand.tokens):
            raise ArtifactError(
                "token mismatch: the candidate's tokens are not the reference's evaluation "
                "sequence, so the rows are not comparable"
            )
    if ref.positions.size and cand.positions.size:
        if ref.positions.shape != cand.positions.shape or not np.array_equal(
            ref.positions, cand.positions
        ):
            raise ArtifactError(
                "position mismatch: the candidate must emit exactly the reference's rows"
            )


def score_sources(
    ref: LogprobSource,
    cand: LogprobSource,
    *,
    direction: str = "ref->cand",
    clip: float = DEFAULT_CLIP,
    worst: int = 10,
    block: int = DEFAULT_ROW_BLOCK,
    with_ppl: bool = True,
) -> ScoreResult:
    """Score cand against ref; direction picks which distribution the KL uses."""
    if direction not in ("ref->cand", "cand->ref"):
        raise ValueError("direction must be 'ref->cand' or 'cand->ref'")
    _check_compatible(ref, cand)
    want_ppl = with_ppl and ref.tokens.size > 0
    targets = ref.target_tokens if want_ppl else None

    kld_per_row = np.empty(ref.n_rows, dtype=np.float64)
    nll_ref = np.zeros(ref.n_rows, dtype=np.float64)
    nll_cand = np.zeros(ref.n_rows, dtype=np.float64)
    dp2 = np.zeros(ref.n_rows, dtype=np.float64)
    same_top = 0
    for start, ref_block in ref.iter_blocks(block):
        stop = start + ref_block.shape[0]
        cand_block = cand.logp_block(start, stop)
        if direction == "ref->cand":
            base, other = ref_block, cand_block
        else:
            base, other = cand_block, ref_block
        weights = np.where(base > clip, np.exp(np.minimum(base, 0.0)), 0.0)
        # Element-wise in the artifact's dtype, accumulated in float64: a
        # 131k-term float32 sum would otherwise drift by ~1e-6 per row.
        kld_per_row[start:stop] = np.sum(weights * (base - other), axis=1, dtype=np.float64)
        same_top += int(
            np.count_nonzero(np.argmax(ref_block, axis=1) == np.argmax(cand_block, axis=1))
        )
        if want_ppl:
            local = np.arange(ref_block.shape[0])
            target = targets[start:stop]
            nll_ref[start:stop] = -ref_block[local, target]
            nll_cand[start:stop] = -cand_block[local, target]
            dp2[start:stop] = (np.exp(-nll_cand[start:stop]) - np.exp(-nll_ref[start:stop])) ** 2

    order = np.argsort(kld_per_row)[::-1]
    worst_rows = []
    for row in order[: max(0, int(worst))]:
        row = int(row)
        ref_row = ref.logp_block(row, row + 1)[0]
        cand_row = cand.logp_block(row, row + 1)[0]
        worst_rows.append(
            {
                "row": row,
                "position": int(ref.positions[row]) if ref.positions.size else None,
                "kld": float(kld_per_row[row]),
                "ref_top": int(np.argmax(ref_row)),
                "cand_top": int(np.argmax(cand_row)),
                "target": int(targets[row]) if want_ppl else None,
                "nll_ref": float(nll_ref[row]) if want_ppl else None,
                "nll_cand": float(nll_cand[row]) if want_ppl else None,
            }
        )

    return ScoreResult(
        reference=ref.name,
        candidate=cand.name,
        n_rows=ref.n_rows,
        vocab_size=ref.vocab_size,
        mean_kld=float(np.mean(kld_per_row)),
        kld_stderr=(
            float(np.std(kld_per_row, ddof=1) / math.sqrt(ref.n_rows))
            if ref.n_rows > 1
            else float("nan")
        ),
        kld_median=float(np.median(kld_per_row)),
        kld_p90=float(np.percentile(kld_per_row, 90)),
        kld_99=float(np.percentile(kld_per_row, 99)),
        kld_max=float(np.max(kld_per_row)),
        top1_agreement=(same_top / ref.n_rows) if ref.n_rows else float("nan"),
        mean_nll_ref=float(np.mean(nll_ref)) if want_ppl else float("nan"),
        mean_nll_cand=float(np.mean(nll_cand)) if want_ppl else float("nan"),
        rms_dp=float(math.sqrt(np.mean(dp2))) if want_ppl else float("nan"),
        clip=float(clip),
        direction=direction,
        worst=worst_rows,
        meta={
            "kv_mode": ref.meta.get("kv_mode") or ref.meta.get("kv_cache_mode"),
            "loops": ref.meta.get("loops"),
        },
    )


def score_arrays(ref_logprobs, cand_logprobs, tokens=None, positions=None, **kwargs):
    """Score two in-memory arrays (tests and one-off comparisons)."""
    ref = array_source(ref_logprobs, tokens=tokens, positions=positions, name="reference")
    cand = array_source(cand_logprobs, tokens=tokens, positions=positions, name="candidate")
    return score_sources(ref, cand, **kwargs)


def score_files(reference: str | Path, candidate: str | Path, **kwargs) -> ScoreResult:
    """Score a candidate file against a reference file."""
    return score_sources(read_source(reference), read_source(candidate), **kwargs)


def _cmd_score(args: argparse.Namespace) -> int:
    result = score_files(
        args.reference, args.candidate, direction=args.direction, clip=args.clip, worst=args.worst
    )
    if args.json:
        Path(args.json).write_text(json.dumps(result.as_dict(), indent=2) + "\n")
    if args.quiet:
        print(
            f"mean_kld={result.mean_kld:.6f} top1={100.0 * result.top1_agreement:.3f}% "
            f"rows={result.n_rows}"
        )
    else:
        print(result.summary())
    failures = []
    if args.min_top1 is not None and result.top1_agreement < args.min_top1:
        failures.append(
            f"top-1 {100.0 * result.top1_agreement:.3f}% < {100.0 * args.min_top1:.3f}%"
        )
    if args.max_kld is not None and result.mean_kld > args.max_kld:
        failures.append(f"mean KLD {result.mean_kld:.6f} > {args.max_kld:.6f}")
    if failures:
        print("FAIL: " + "; ".join(failures), file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m anemll.eval.kld",
        description="Score candidate next-token log-probabilities against a reference.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score", help="score one candidate against one reference")
    score.add_argument("--reference", required=True, help="reference .kld or .npz")
    score.add_argument("--candidate", required=True, help="candidate .npz (or .kld)")
    score.add_argument(
        "--direction",
        default="ref->cand",
        choices=("ref->cand", "cand->ref"),
        help="KLD(reference || candidate) by default",
    )
    score.add_argument("--clip", type=float, default=DEFAULT_CLIP, help="reference logp floor")
    score.add_argument("--worst", type=int, default=10, help="how many worst rows to list")
    score.add_argument("--json", default=None, help="write the full result as JSON here")
    score.add_argument("--quiet", action="store_true", help="one-line output")
    score.add_argument("--min-top1", type=float, default=None, help="fail if top-1 is lower")
    score.add_argument("--max-kld", type=float, default=None, help="fail if mean KLD is higher")
    score.set_defaults(func=_cmd_score)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
