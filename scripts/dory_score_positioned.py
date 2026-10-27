"""Score decode-mode log-probabilities at absolute positions beyond 16K.

The DANE-6 gate says the decode-mode comparison must hold "on evaluation windows
positioned beyond 16K tokens".  That is a different measurement from
`anemll.dory_chat score`, which starts every window at position 0: a window at
16,384 must be reached by actually feeding the 16,384 tokens before it, or the
positions (and the RoPE rows) are wrong and the answer means nothing.

So this harness prefills [0, start) in 64-token batches, then walks the window of
interest one token at a time -- decode mode, exactly how the model runs -- and
writes the same candidate artifact the rest of the tooling scores:

    python scripts/dory_score_positioned.py --backend coreml --model-dir <dir> \
        --tokens tests/fixtures/dory_eval/eval_tokens.txt \
        --start 16384 --end 18432 --loops 2 --out beyond16k.npz \
        --reference ~/dev/dory-ane-artifacts/refs/bf16_decode24k_loop2.kld

The reference has to cover those absolute positions too: build it with
llama-perplexity over a ~24K-token text (then its own contexts at 16,384+ are the
positions this harness measures), or take the reference's sidecar positions and
pass --start/--end from it.

Backends are the same seam as everything else: --backend coreml for the gate,
--backend torch for a pre-conversion check of the same window.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

__all__ = ["positioned_logprobs", "parse_args", "main"]


def _progress(done, total, elapsed, quiet=False):
    if quiet:
        return
    rate = done / elapsed if elapsed > 0 else 0.0
    print(f"\r{done:,}/{total:,} tokens positioned ({100.0 * done / max(total, 1):.0f}%, "
          f"{rate:,.0f} tok/s)", end="" if done < total else "\n", file=sys.stderr,
          flush=True)


def positioned_logprobs(backend, ids, start: int, end: int, quiet=False):
    """(logprobs, positions) for the absolute positions [start, end).

    Prefill everything before `start` first -- 64-token batches, the way the
    model runs -- then one token per step through the window.  Row r of the
    result is the log-softmax over the vocabulary at position `positions[r]`,
    predicting `ids[positions[r] + 1]`, which is the convention every reference
    and candidate in this repository uses.
    """
    from anemll.dory_chat import log_softmax_rows

    ids = [int(t) for t in ids]
    start, end = int(start), int(end)
    if not 0 <= start < end <= len(ids) - 1:
        raise ValueError(f"window [{start}, {end}) must sit inside {len(ids)} tokens "
                         f"and leave room for the token it predicts")
    backend.reset()
    started = time.time()
    batch = int(getattr(backend, "batch", 64) or 64)
    done = 0
    while done < start:
        piece = ids[done:min(done + batch, start)]
        backend.prefill(piece, done)
        done += len(piece)
        _progress(done, start + (end - start), time.time() - started, quiet)
    rows, positions = [], []
    for pos in range(start, end):
        logits = backend.decode(ids[pos], pos)
        rows.append(log_softmax_rows(np.asarray(logits).reshape(1, -1))[0])
        positions.append(pos)
        if (pos - start) % 64 == 0 or pos == end - 1:
            _progress(start + (pos - start) + 1, start + (end - start),
                      time.time() - started, quiet)
    return np.asarray(rows, dtype=np.float32), np.asarray(positions, dtype=np.int32)


class _CoreMLBackend:
    def __init__(self, model_dir, loops, batch=None):
        # Two seams live behind --backend coreml.  A converted directory whose
        # meta carries per_head_programs is the 102-program plan and has to be
        # driven by DoryProgramRuntime (one state per program, gates threaded
        # from the manifest); anything else is the older five-component module and
        # goes through DoryRuntime exactly as before.
        import yaml
        from pathlib import Path

        meta_path = Path(model_dir) / "meta.yaml"
        meta = yaml.safe_load(meta_path.read_text()) if meta_path.exists() else {}
        self.per_head = bool(meta.get("per_head_programs"))
        self.loops = int(loops)
        if self.per_head:
            from anemll.runtime.dory_ane_runtime import DoryProgramRuntime

            self.runtime = DoryProgramRuntime(model_dir, loops=int(loops), verbose=False)
            self.runtime.load()
            self.context = int(self.runtime.plan.context_length)
            self.batch = int(batch or self.runtime.plan.batch_size or 64)
        else:
            from anemll.dory_chat import DoryRuntime

            self.runtime = DoryRuntime(model_dir, verbose=False)
            self.runtime.set_loops(int(loops))
            self.context = int(self.runtime.params["context_length"])
            self.batch = int(batch or self.runtime.params["batch_size"] or 64)

    def reset(self):
        if getattr(self, "per_head", False):
            self.runtime.reset()
        else:
            self.runtime.reset_state()

    def prefill(self, ids, start):
        ids = np.asarray(ids, dtype=np.int32)
        batch = int(self.batch)
        # The plan's prefill packages are traced at exactly batch_size rows, so a
        # trailing piece shorter than that cannot go through them.  Feed the short
        # tail as single-token decodes instead: the caches end up identical, only
        # the batching differs, and the harness is filling to a position, not
        # timing anything.
        import os
        force_decode = os.environ.get("DORY_FORCE_DECODE") == "1"
        if getattr(self, "per_head", False) and (force_decode or ids.size != batch):
            done = int(start)
            for token in ids.tolist():
                self.runtime.decode(int(token), done)
                done += 1
            return
        self.runtime.prefill(ids, int(start))

    def decode(self, token, pos):
        return self.runtime.decode(int(token), int(pos))

    def close(self):
        pass


class _TorchBackend:
    def __init__(self, model_path, context, loops, batch=None):
        from anemll.models.dory_ane_model import DoryANEConfig, DoryANEForCausalLM

        self.batch = int(batch or 64)
        self.config = DoryANEConfig(context_length=int(context), batch_size=self.batch)
        self.model = DoryANEForCausalLM(self.config)
        self.model.load_pretrained_weights(str(model_path))
        self.model.eval()
        # DORY_TORCH_DTYPE=float16 makes this reference run at the same precision
        # the ANE graph does. Without it the reference is bf16/fp32 and a KLD
        # against the ANE mixes quantization with runtime error, which is exactly
        # the ambiguity this option exists to remove.
        import os
        dtype = os.environ.get("DORY_TORCH_DTYPE")
        if dtype:
            torch_dtype = {"float16": __import__("torch").float16,
                           "bfloat16": __import__("torch").bfloat16,
                           "float32": __import__("torch").float32}[dtype]
            self.model = self.model.to(torch_dtype)
            print(f"reference dtype: {dtype}", file=__import__("sys").stderr)
        self.model.set_loops(int(loops))
        self.loops = int(loops)
        self.context = int(context)

    def reset(self):
        self.model.reset_caches()

    def prefill(self, ids, start):
        self.model.prefill([int(t) for t in ids], int(start))

    def decode(self, token, pos):
        return self.model.decode_step(int(token), int(pos)).numpy()[0]

    def close(self):
        pass


def score_window(reference_path, candidate_path, start: int, end: int):
    """Score the candidate against the reference rows [start, end).

    A reference built as one full context covers a wide range (12,288..24,574 for
    a 24,576-token text), while the gate names a window beyond 16K.  The scorer
    insists that candidate and reference cover exactly the same rows, so the
    reference is sliced to the window here -- KLD is per row, and a window of rows
    is the measurement the gate asks for.
    """
    import numpy as np

    from anemll.eval.kld import (array_source, read_candidate_npz, read_source,
                                 score_sources)

    reference = read_source(reference_path)
    keep = np.where((reference.positions >= int(start))
                    & (reference.positions < int(end)))[0]
    if keep.size == 0:
        raise ValueError(f"the reference has no rows in [{start}, {end})")
    lo, hi = int(keep.min()), int(keep.max()) + 1
    block = reference.logp_block(lo, hi)
    if keep.size != hi - lo:
        block = block[keep - lo]
    sliced = array_source(block, tokens=reference.tokens,
                          positions=reference.positions[keep],
                          name=f"{Path(str(reference_path)).stem}[{start}:{end}]",
                          meta=dict(getattr(reference, "meta", None) or {}))
    candidate = read_candidate_npz(candidate_path)
    return score_sources(sliced, candidate, direction="ref->cand")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Score decode-mode log-probabilities at absolute positions > 16K")
    parser.add_argument("--backend", choices=("coreml", "torch"), default="coreml")
    parser.add_argument("--model-dir", default=None, help="converted model (coreml)")
    parser.add_argument("--model", default=None, help="BF16 safetensors dir (torch)")
    parser.add_argument("--tokens", default="tests/fixtures/dory_eval/eval_tokens.txt")
    parser.add_argument("--context", type=int, default=24576)
    parser.add_argument("--start", type=int, default=16384, help="absolute first position")
    parser.add_argument("--end", type=int, default=18432, help="absolute last position")
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--out", required=True, help="candidate .npz to write")
    parser.add_argument("--reference", default=None, help="reference .kld/.npz to score against")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    ids = np.loadtxt(Path(args.tokens), dtype=np.int64).reshape(-1).astype(np.int32)
    largest = int(len(ids)) - 1
    if not 0 <= int(args.start) < int(args.end) <= largest:
        print(
            f"error: the window [{args.start}, {args.end}) does not fit {len(ids):,} tokens. "
            f"A window must satisfy 0 <= start < end <= {largest:,}: end - 1 is the last "
            f"position a next token can be predicted for. Pass --tokens with a longer file "
            f"(the reference's eval-tokens.txt is ~24K ids and allows end <= "
            f"{24575:,} for a 24,576-token prompt), or a window inside this one.",
            file=sys.stderr)
        return 2
    if args.backend == "coreml":
        if not args.model_dir:
            print("error: --model-dir is required for the coreml backend", file=sys.stderr)
            return 2
        backend = _CoreMLBackend(args.model_dir, args.loops, args.batch_size)
    else:
        if not args.model:
            print("error: --model is required for the torch backend", file=sys.stderr)
            return 2
        backend = _TorchBackend(args.model, args.context, args.loops, args.batch_size)
    started = time.time()
    logprobs, positions = positioned_logprobs(backend, ids, args.start, args.end, args.quiet)
    from anemll.eval.kld import score_files, write_candidate

    payload = {
        "model": f"dory-{args.backend}",
        "loops": int(args.loops),
        "kv_mode": "last_loop",
        "batch_size": int(args.batch_size),
        "n_ctx": int(args.end),
        "first_scored_position": int(args.start),
        "trace": "decode-mode, absolute positions",
        "seconds": round(time.time() - started, 1),
    }
    candidate = write_candidate(args.out, logprobs, positions, ids, payload)
    print(f"wrote {candidate}", file=sys.stderr)
    if args.reference:
        result = score_window(args.reference, candidate, args.start, args.end)
        print(result.summary())
        print(json.dumps(result.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
