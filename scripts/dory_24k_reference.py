"""Build the absolute-position reference for the beyond-16K decode gate.

The gate needs log-probabilities at absolute positions 16,384+ of a ~24K prompt.
A llama-perplexity run split into contexts cannot supply them: each context is
prefilled from position 0, so a chunk "beyond 16K" of the text still sits at
relative positions 0..ctx-1.  The only layout that measures what the gate means is
one context of the whole prompt, whose scored rows are the second half,
12,288..24,574, which contains the window of interest.

WHAT THIS ARTIFACT IS, exactly (the numbers are in the sidecar too):

  kv_cache_mode  last_loop      the cache semantics the decode gate measures
  batching       full (default) one prefill of the whole context, one ubatch
                 decode           one token per step -- the candidate's own shape

A full prefill is exact-mode *batching*, not decode-mode batching, so this is not
the same artifact as DANE-2's 4K bf16_decode_loop2.kld.  It is the right yardstick
anyway: at 4K the gap between the exact and the decode reference is 0.0028-0.0069
KLD at two loops, two orders of magnitude below the 0.05 gate, so the batching
cannot move a verdict.  Pass --mode decode if you ever want the strict decode-batched
reference (24,574 sequential steps on the GPU).

That is DANE-2's own generator, called with ctx = the full text length and
chunks = 1; this script is a thin wrapper so nothing about that code path is
re-implemented:

    python scripts/dory_24k_reference.py --gguf <bf16.gguf> --ctx 24576 \
        --out-dir <artifacts>/refs-24k --loops 2

It writes `<name>.kld` plus its `.meta.json` sidecar (with the positions the
artifact actually covers), and `eval-tokens.txt` beside them: the ids llama.cpp
itself tokenised, which is what the candidate has to be fed so that both sides
score the same tokens at the same absolute positions.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

__all__ = ["build", "parse_args", "main"]

DEFAULT_TEXT = Path("tests/fixtures/dory_eval/eval_text.txt")


def _generator():
    """DANE-2's make_dory_references.py, loaded by path (scripts/ is not a package)."""
    path = Path(__file__).with_name("make_dory_references.py")
    spec = importlib.util.spec_from_file_location("_dory_refs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_long_text(source: Path, ctx: int, out: Path) -> Path:
    """A text long enough to fill one ctx-token context, from the eval fixture.

    The fixture is ~8K tokens, so it is repeated until the file holds at least
    `ctx` tokens' worth of characters; verbatim repetition is harmless here
    because both sides score the same tokens, and the scored window is scored
    against itself, not against a notion of natural language.
    """
    text = source.read_text(encoding="utf-8")
    if not text.endswith("\n"):
        text += "\n"
    pieces = [text]
    # llama-perplexity refuses a file shorter than two contexts, so build twice
    # the length it needs even though only the first context is scored.
    while sum(len(piece) for piece in pieces) < ctx * 9:   # ~4 chars/token, 2x ctx
        pieces.append(text)
    out.write_text(chr(10).join(piece.rstrip(chr(10)) for piece in pieces) + chr(10),
                   encoding="utf-8")
    return out


def build(args) -> dict:
    mkref = _generator()
    from anemll.eval.kld import read_llama_cpp_kld, scored_positions

    ctx = int(args.ctx)
    name = args.name
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text = Path(args.text) if args.text else None
    if text is None or not text.is_file():
        text = build_long_text(Path(args.source_text), ctx, out_dir / "eval-long-text.txt")
    else:
        print(f"[{name}] using {text} ({text.stat().st_size} bytes)", flush=True)

    out = out_dir / f"{name}.kld"
    meta_path = out.with_suffix(out.suffix + ".meta.json")
    if out.is_file() and meta_path.is_file() and not args.force:
        print(f"[{name}] exists, skipping (use --force)", flush=True)
    else:
        batching = "full" if args.mode == "full" else "decode"
        n_batch = ctx if batching == "full" else max(1, int(args.decode_batch))
        info = mkref.run_llama_perplexity(
            Path(args.llama_perplexity), Path(args.gguf), text, out,
            "decode", int(args.loops), n_batch, ctx=ctx, chunks=1)
        source = read_llama_cpp_kld(out)
        tokens = np.asarray(source.tokens, dtype=np.int32)
        rows = scored_positions(1, ctx)
        covered = (int(rows.min()), int(rows.max()))
        if not (covered[0] <= int(args.start) and int(args.end) - 1 <= covered[1]):
            raise SystemExit(
                f"{name}: the scored rows cover {covered[0]}..{covered[1]} and the gate "
                f"window is {args.start}..{int(args.end) - 1}; raise --ctx")
        (out_dir / "eval-tokens.txt").write_text(
            chr(10).join(str(int(t)) for t in tokens) + chr(10), encoding="utf-8")
        meta = {
            "name": name,
            "mode": "decode",
            "loops": int(args.loops),
            "kv_cache_mode": "last_loop",
            "n_batch": n_batch,
            "n_ubatch": mkref.ubatch_for("decode", n_batch, ctx),
            "chunk_size": ctx,
            "n_chunks": 1,
            "batching": batching,
            "layout": ("one context, one full prefill: absolute positions, no restart"
                       if batching == "full" else
                       "one context, one token per step: absolute positions, decode batching"),
            "note": ("kv mode is last_loop (the decode gate's semantics); the batching is "
                     "full-context.  At 4K the exact-vs-decode reference gap is 0.0028-0.0069 "
                     "KLD at two loops, far below the 0.05 gate."),
            "n_rows": int(source.n_rows),
            "scored_positions": [int(covered[0]), int(covered[1])],
            "tokens_file": "eval-tokens.txt",
            "tokens": int(tokens.shape[0]),
            "model": {"gguf": str(args.gguf), "sha256": mkref.sha256(Path(args.gguf))},
            "tool": {"llama_perplexity": str(args.llama_perplexity),
                     "command": info["cmd"]},
            "perplexity": info["ppl"],
            "seconds": info["duration_s"],
            "bytes": out.stat().st_size,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        meta_path.write_text(json.dumps(meta, indent=2) + chr(10))
        Path(info["log"]).unlink(missing_ok=True)
    meta = json.loads(meta_path.read_text())
    print(json.dumps({key: meta[key] for key in
                      ("name", "n_rows", "scored_positions", "tokens", "perplexity",
                       "batching", "n_batch", "n_ubatch", "seconds")}, indent=2))
    return meta


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Absolute-position (single-context) reference for the >16K gate")
    parser.add_argument("--gguf", required=True, help="BF16 GGUF")
    parser.add_argument("--llama-perplexity", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--name", default="bf16_lastloop_fullprefill_abs24k_loop2")
    parser.add_argument("--ctx", type=int, default=24576)
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--text", default=None, help="an existing ~ctx-token text file")
    parser.add_argument("--source-text", default=str(DEFAULT_TEXT),
                        help="text to repeat when --text is not given")
    parser.add_argument("--start", type=int, default=16384)
    parser.add_argument("--end", type=int, default=18432)
    parser.add_argument("--mode", choices=("full", "decode"), default="full",
                        help="'full' prefills the whole context in one ubatch (exact-mode "
                             "batching, the default and the yardstick); 'decode' walks one "
                             "token per step, which is 24,574 sequential steps")
    parser.add_argument("--decode-batch", type=int, default=1,
                        help="ubatch for --mode decode")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    build(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
