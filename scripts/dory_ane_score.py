"""Score the ANE-safe FP16 Dory module against DANE-2's reference artifacts.

The seam is DANE-2's documented candidate format: one row of vocabulary
log-probabilities per scored position of the fixed evaluation set, written with
`anemll.eval.write_candidate` and scored with `anemll.eval.score_sources`.

Two things to know before reading a number out of this script:

* **exact vs decode.** `--mode prefill --batch full` feeds one batch that holds
  the whole prefix, which makes `last_loop` bit-identical to `per_loop`
  (math-dory.md 6.3), i.e. a full recompute. `--mode prefill` with 64-token
  batches and `--mode decode` run the port's real `last_loop` batching and must
  be compared against the matching `-ub 64` / `-ub 1` decode reference.
* **context length.** `context_length` is only the row capacity of the
  full-attention caches; the math for a position does not depend on it. The
  default here is the smallest 256-aligned capacity that covers the scored rows
  (`first_scored + rows`), which keeps a gate run to a few minutes instead of
  processing the whole 4,096-token context. Positions and rows are identical to
  a full-context run because attention is causal.

Example (decode mode, 128 scored rows of the first context, 2 loops):

    python scripts/dory_ane_score.py \
        --reference ~/dev/dory-ane-artifacts/refs/bf16_decode_loop2_batch1.kld \
        --mode decode --loops 2 --rows 128 --chunk 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from anemll.eval import load_eval_set, write_candidate
from anemll.eval.kld import (
    LogprobSource,
    read_llama_cpp_kld,
    read_candidate_npz,
    score_sources,
)
from anemll.models.dory_ane_model import DoryANEConfig, DoryANEForCausalLM

DEFAULT_WEIGHTS = os.path.expanduser(
    "~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16/"
    "snapshots/b366b87f762ea334ba3eb55912031baba1515810/model.safetensors")


#: The ticket's gates (Testing Decisions): a scoring run that cannot fail is not
#: a gate.  --max-kld / --min-top1 override them for a diagnostic run; the exit
#: status still says whether the numbers passed whatever thresholds were used.
MAX_MEAN_KLD = 0.01
MIN_TOP1 = 0.98


def judge(mean_kld, top1, max_kld=MAX_MEAN_KLD, min_top1=MIN_TOP1):
    """(passed, [failures]) for one scored result.

    NaN or infinite numbers fail: a gate has to treat "no usable measurement"
    as a failure, never as a pass.
    """
    failures = []
    if not np.isfinite(mean_kld) or float(mean_kld) > float(max_kld):
        failures.append(f"mean KLD {float(mean_kld):.5f} > {float(max_kld):g}")
    if not np.isfinite(top1) or float(top1) < float(min_top1):
        failures.append(f"top-1 {100.0 * float(top1):.2f} % < {100.0 * float(min_top1):.2f} %")
    return (not failures), failures


def judge_text(passed, failures):
    """The one-line verdict a script prints next to a gate's numbers."""
    return "PASS" if passed else "FAIL (" + "; ".join(failures) + ")"


def log_softmax(logits: torch.Tensor) -> np.ndarray:
    # logits may live on Metal; numpy() needs a host copy first
    return torch.log_softmax(logits.float().to("cpu"), dim=-1).numpy()


def run_candidate(model, ids, first_scored, rows, mode, batch_size):
    """Log-probabilities for scored positions first_scored .. first_scored+rows-1."""
    want = range(first_scored, first_scored + rows)
    out = np.empty((rows, model.config.vocab_size), dtype=np.float32)
    model.reset_caches()
    t0 = time.time()
    done = 0
    if mode == "decode":
        for pos, tok in enumerate(ids):
            logits = model.decode_step(int(tok), pos)
            if pos in want:
                out[done] = log_softmax(logits[0, 0])
                done += 1
            if done % 64 == 0 and done and pos in want:
                print(f"  {done}/{rows} rows, {time.time() - t0:.0f}s", flush=True)
    else:
        pos = 0
        while pos < len(ids):
            take = len(ids) - pos if batch_size is None else min(batch_size, len(ids) - pos)
            if batch_size is not None and take < batch_size:
                # the traced form has a static batch; top up by decoding instead
                for step in range(take):
                    logits = model.decode_step(int(ids[pos + step]), pos + step)
                    if pos + step in want:
                        out[done] = log_softmax(logits[0, 0])
                        done += 1
                pos = len(ids)
                break
            logits = model.prefill(torch.tensor(ids[pos:pos + take]), pos)
            for i in range(take):
                if pos + i in want:
                    out[done] = log_softmax(logits[0, i])
                    done += 1
            pos += take
            print(f"  prefill {pos}/{len(ids)} tokens, {time.time() - t0:.0f}s", flush=True)
    if done != rows:
        raise RuntimeError(f"produced {done} rows, wanted {rows}")
    return out


def build_parser():
    """The command line, kept out of the work so it can be read and tested."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True, help="DANE-2 reference .kld artifact")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--mode", choices=("decode", "prefill"), default="decode")
    ap.add_argument("--loops", type=int, default=2)
    ap.add_argument("--rows", type=int, default=128, help="scored rows to evaluate")
    ap.add_argument("--chunk", type=int, default=0, help="evaluation context to use")
    ap.add_argument("--batch", type=int, default=64, help="prefill batch (0 = whole prefix)")
    ap.add_argument("--out", default="cand_dory_ane.npz")
    ap.add_argument("--meta-out", default=None, help="JSON sidecar for the score result")
    ap.add_argument("--threads", type=int, default=14)
    ap.add_argument("--max-kld", type=float, default=MAX_MEAN_KLD,
                    help=f"mean KLD gate (default {MAX_MEAN_KLD}); the exit status follows it")
    ap.add_argument("--min-top1", type=float, default=MIN_TOP1,
                    help=f"top-1 gate (default {MIN_TOP1}); the exit status follows it")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)

    torch.set_num_threads(args.threads)
    evalset = load_eval_set()
    tokens = np.asarray(evalset.token_ids, dtype=np.int32)
    first_scored = evalset.first_scored
    chunk_base = args.chunk * evalset.chunk_size
    limit = first_scored + args.rows
    ids = tokens[chunk_base:chunk_base + limit]
    context_length = ((limit + 63) // 64) * 64

    cfg = DoryANEConfig(context_length=context_length, batch_size=args.batch or 64,
                        fold_scales=True, dtype=torch.float16, compute_mode="matmul")
    model = DoryANEForCausalLM(cfg)
    t0 = time.time()
    report = model.load_pretrained_weights(args.weights)
    print(f"loaded {report['consumed']} tensors ({len(report['missing'])} missing) "
          f"in {time.time() - t0:.0f}s; context_length={context_length}", flush=True)
    model.set_loops(args.loops)

    logprobs = run_candidate(model, ids, first_scored, args.rows, args.mode,
                             None if args.batch == 0 else args.batch)

    positions = evalset.scored_positions()
    row_start = args.chunk * evalset.rows_per_chunk
    row_slice = slice(row_start, row_start + args.rows)
    meta = {"model": "dory-ane-fp16", "loops": args.loops,
            "kv_mode": "last_loop",
            "batch_size": None if args.mode == "decode" else args.batch,
            "n_ctx": context_length, "mode": args.mode,
            "rows": args.rows, "chunk": args.chunk}
    write_candidate(args.out, logprobs, positions[row_slice], tokens, meta=meta)
    print(f"wrote {args.out}", flush=True)

    ref = read_llama_cpp_kld(args.reference)
    ref_rows = ref.n_rows
    sliced = LogprobSource(
        f"{ref.name}[{row_slice.start}:{row_slice.stop}]",
        args.rows, ref.vocab_size, ref.tokens, ref.positions[row_slice], ref.meta,
        lambda start, stop: ref.logp_block(row_slice.start + start, row_slice.start + stop))
    cand = read_candidate_npz(args.out)
    result = score_sources(sliced, cand)
    print(result.summary())
    passed, failures = judge(result.mean_kld, result.top1_agreement,
                             args.max_kld, args.min_top1)
    print(f"GATE: {judge_text(passed, failures)} "
          f"(mean KLD <= {args.max_kld:g}, top-1 >= {100.0 * args.min_top1:.2f} %)")
    payload = result.as_dict()
    payload["candidate_meta"] = meta
    payload["reference"] = args.reference
    payload["gate"] = {"passed": passed, "failures": failures,
                       "max_kld": args.max_kld, "min_top1": args.min_top1}
    if args.meta_out:
        with open(args.meta_out, "w") as fh:
            json.dump(payload, fh, indent=2)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())