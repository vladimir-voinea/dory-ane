"""All ANE-safe Dory gates in ONE process, so the 8 GB of weights stay hot.

Loading the model once per gate pays the load and the page-in cost five times
over; on this shared machine that is what made a decode step cost 8.6 s.
Gates are run cheapest first and every number is written next to the row range,
the wall time and the load average that produced it.

    python scripts/dory_ane_gate_suite.py --rows-mode window --window-rows 256
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
from anemll.eval.kld import (LogprobSource, array_source, read_candidate_npz,
                             read_llama_cpp_kld, score_sources)
from anemll.models.dory_ane_model import DoryANEConfig, DoryANEForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.dory_ane_score import (  # noqa: E402
    DEFAULT_WEIGHTS, MAX_MEAN_KLD, MIN_TOP1, judge, judge_text, log_softmax)

REFS = os.path.expanduser("~/dev/dory-ane-artifacts/refs")
ARTIFACTS = os.path.expanduser("~/dev/dory-ane-artifacts/DANE-4")


def loadavg():
    return os.getloadavg()[0]


def summarize(results, max_kld=MAX_MEAN_KLD, min_top1=MIN_TOP1):
    """Print PASS/FAIL per gate and return the exit code (1 if any gate failed).

    It reads only what the run recorded, so the exit status cannot disagree with
    the numbers in the JSON: a gate with no score (missing reference) and a gate
    over the threshold are both failures.
    """
    failed = []
    print("", flush=True)
    print(f"{'gate':16s} {'rows':>6s} {'mean KLD':>10s} {'top-1':>8s}  verdict", flush=True)
    for entry in results:
        name = f"{entry['gate']}chunk{entry.get('chunk', 0)}"
        if entry.get("skipped"):
            failed.append(f"{name}: {entry['skipped']}")
            print(f"{name:16s} {entry['rows']:6d} {'-':>10s} {'-':>8s}  "
                  f"FAIL ({entry['skipped']})", flush=True)
            continue
        if "mean_kld" not in entry:
            failed.append(f"{name}: no score recorded")
            print(f"{name:16s} {entry['rows']:6d} {'-':>10s} {'-':>8s}  FAIL (no score)",
                  flush=True)
            continue
        passed, failures = judge(entry["mean_kld"], entry["top1"], max_kld, min_top1)
        entry["gate_passed"] = passed
        entry["gate_failures"] = failures
        if not passed:
            failed.append(f"{name}: " + "; ".join(failures))
        print(f"{name:16s} {entry['rows']:6d} {entry['mean_kld']:10.5f} "
              f"{100.0 * entry['top1']:7.2f}%  {judge_text(passed, failures)}", flush=True)
    print("", flush=True)
    if failed:
        print(f"GATE SUITE: FAILED ({len(failed)} of {len(results)} gate rows)", flush=True)
        for line in failed:
            print("  -", line, flush=True)
        return 1
    print(f"GATE SUITE: PASSED (every row within mean KLD <= {max_kld:g}, "
          f"top-1 >= {100.0 * min_top1:.2f} %)", flush=True)
    return 0


def gate_rows(mode, window):
    """(chunk, first_row, n_rows) for every context this mode scores."""
    if mode == "window":
        return [(0, 0, window)]
    return [(0, 0, 2047), (1, 0, 2047)]


def run_prefill_gate(model, evalset, rows_spec, batch):
    """Prefill in fixed batches; one part per scored context.

    A part is scored and written as soon as its context finishes, so a machine
    or disk failure costs one context, never the whole gate.
    """
    parts = []
    for chunk, first_row, n_rows in rows_spec:
        t0 = time.time()
        base = chunk * evalset.chunk_size
        first_scored = evalset.first_scored
        want_start = first_scored + first_row
        want_stop = want_start + n_rows
        # the reference processed full 64-token ubatches; round the prefix up so
        # the scored rows sit in the same ubatch boundaries as llama.cpp's
        limit = min(((want_stop + batch - 1) // batch) * batch, evalset.chunk_size)
        ids = evalset.token_ids[base:base + limit]
        model.reset_caches()
        out = []
        pos = 0
        while pos < len(ids):
            take = min(batch, len(ids) - pos)
            if take < batch:
                for step in range(take):
                    lg = model.decode_step(int(ids[pos + step]), pos + step)
                    if want_start <= pos + step < want_stop:
                        out.append(keep_row(lg[0, 0]))
                pos = len(ids)
                break
            lg = model.prefill(torch.tensor(ids[pos:pos + take]), pos)
            for i in range(take):
                if want_start <= pos + i < want_stop:
                    out.append(keep_row(lg[0, i]))
            pos += take
            if (pos // take) % 16 == 0:
                print(f"    chunk {chunk} prefill {pos}/{len(ids)} tokens, "
                      f"{time.time() - t0:.0f}s", flush=True)
        positions = evalset.scored_positions()[chunk * evalset.rows_per_chunk + first_row:
                                               chunk * evalset.rows_per_chunk + first_row + n_rows]
        parts.append({"chunk": chunk, "logprobs": np.stack(out), "positions": positions,
                      "wall_s": time.time() - t0})
    return parts


def run_decode_gate(model, evalset, rows_spec):
    """One token per step; one part per scored context."""
    parts = []
    for chunk, first_row, n_rows in rows_spec:
        t0 = time.time()
        base = chunk * evalset.chunk_size
        first_scored = evalset.first_scored
        limit = first_scored + first_row + n_rows
        ids = evalset.token_ids[base:base + limit]
        model.reset_caches()
        out = []
        for pos, tok in enumerate(ids):
            lg = model.decode_step(int(tok), pos)
            if pos >= first_scored + first_row:
                out.append(keep_row(lg[0, 0]))
            if pos and pos % 256 == 0:
                print(f"    chunk {chunk} decode {pos}/{len(ids)} tokens, "
                      f"{time.time() - t0:.0f}s", flush=True)
        positions = evalset.scored_positions()[chunk * evalset.rows_per_chunk + first_row:
                                               chunk * evalset.rows_per_chunk + first_row + n_rows]
        parts.append({"chunk": chunk, "logprobs": np.stack(out), "positions": positions,
                      "wall_s": time.time() - t0})
    return parts


def keep_row(logits_row):
    """One scored row, as float16 (the reference itself is 16-bit)."""
    return log_softmax(logits_row).astype(np.float16)


def score(logprobs, positions, tokens, ref_path, name, out_dir, save_candidate=False):
    """Score in memory: the candidate never touches the disk unless asked.

    A 4,094-row candidate is 1.07 GB of npz and this machine shares its disk
    with three other agents; a transient full disk killed a completed prefill
    gate once.  array_source wraps the array without copying, so the scorer
    streams it exactly like it streams the reference.
    """
    ref = read_llama_cpp_kld(ref_path)
    want = positions
    idx = np.searchsorted(ref.positions, want)
    if idx.size == 0 or not np.array_equal(ref.positions[idx], want):
        raise SystemExit(f"{name}: candidate positions are not a contiguous row range "
                         f"of {ref_path}")
    start = int(idx[0])
    stop = start + len(want)
    sliced = LogprobSource(f"{ref.name}[{start}:{stop}]", len(want), ref.vocab_size,
                           ref.tokens, ref.positions[start:stop], ref.meta,
                           lambda a, b: ref.logp_block(start + a, start + b))
    if save_candidate:
        cand_path = os.path.join(out_dir, f"cand_{name}.npz")
        write_candidate(cand_path, logprobs, positions, tokens,
                        meta={"model": "dory-ane-fp16", "kv_mode": "last_loop", "gate": name},
                        dtype="float16")
        cand = read_candidate_npz(cand_path)
    else:
        cand = array_source(logprobs, tokens=tokens, positions=positions,
                            name=f"{name}_candidate", meta={"gate": name})
    return score_sources(sliced, cand)


def build_parser():
    """The command line, kept out of the work so it can be read and tested."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--rows-mode", choices=("window", "full"), default="window",
                    help="default rows mode; a gate may override it as name:mode")
    ap.add_argument("--window-rows", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--lm-head-parts", type=int, default=8,
                    help="spec D6 packaging; MPS needs >= 4 (65,536-channel cap)")
    ap.add_argument("--device", default="cpu", choices=("cpu", "mps"),
                    help="mps runs the same module through Metal; CPU dispatch still applies")
    ap.add_argument("--gates", default="prefill2,prefill1,prefill4,decode2,decode1,decode4")
    ap.add_argument("--out-json", default=os.path.join(ARTIFACTS, "gate_suite.json"))
    ap.add_argument("--save-candidates", action="store_true",
                    help="also write the candidate npz (1 GB each; off by default)")
    ap.add_argument("--artifacts", default=ARTIFACTS)
    ap.add_argument("--max-kld", type=float, default=MAX_MEAN_KLD,
                    help=f"mean KLD gate (default {MAX_MEAN_KLD}); the exit status follows it")
    ap.add_argument("--min-top1", type=float, default=MIN_TOP1,
                    help=f"top-1 gate (default {MIN_TOP1}); the exit status follows it")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)

    torch.set_num_threads(args.threads)
    os.makedirs(args.artifacts, exist_ok=True)
    evalset = load_eval_set()
    tokens = np.asarray(evalset.token_ids, dtype=np.int32)
    rows_spec = gate_rows(args.rows_mode, args.window_rows)

    cfg = DoryANEConfig(context_length=evalset.chunk_size, batch_size=args.batch,
                        fold_scales=True, dtype=torch.float16,
                        compute_mode="matmul" if args.device == "cpu" else "conv",
                        lm_head_parts=args.lm_head_parts)
    model = DoryANEForCausalLM(cfg)
    t0 = time.time()
    report = model.load_pretrained_weights(args.weights)
    if args.device != "cpu":
        model = model.to(torch.device(args.device))
    print(f"[{time.strftime('%H:%M:%S')}] loaded {report['consumed']} tensors in "
          f"{time.time() - t0:.0f}s; loadavg={loadavg():.0f}", flush=True)

    wanted = [g.strip() for g in args.gates.split(",") if g.strip()]
    results = []
    for spec in wanted:
        name, _, mode = spec.partition(":")
        mode = mode or args.rows_mode
        this_spec = gate_rows(mode, args.window_rows)
        kind = "prefill" if name.startswith("prefill") else "decode"
        loops = int(name[-1])
        model.set_loops(loops)
        t0 = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] gate {name} rows_mode={mode} "
              f"threads={args.threads} loadavg={loadavg():.0f}", flush=True)
        if kind == "prefill":
            parts = run_prefill_gate(model, evalset, this_spec, args.batch)
        else:
            parts = run_decode_gate(model, evalset, this_spec)
        ref_name = (f"bf16_decode_loop{loops}_batch{args.batch if kind == 'prefill' else 1}.kld")
        ref_path = os.path.join(REFS, ref_name)
        for part in parts:
            lp, pos = part["logprobs"], part["positions"]
            n = int(pos.size)
            entry = {"gate": name, "kind": kind, "loops": loops, "rows": n,
                 "device": args.device, "lm_head_parts": args.lm_head_parts,
                     "chunk": part["chunk"],
                     "row_range": f"{int(pos[0])}..{int(pos[-1])}",
                     "wall_s": round(part["wall_s"], 1),
                     "loadavg": round(loadavg(), 1), "reference": ref_name,
                     "threads": args.threads, "rows_mode": mode,
                     "finite": bool(np.isfinite(lp).all())}
            if os.path.exists(ref_path):
                res = score(lp, pos, tokens, ref_path, name, args.artifacts,
                            save_candidate=args.save_candidates)
                entry.update({"mean_kld": res.mean_kld, "top1": res.top1_agreement,
                              "kld_max": res.kld_max, "kld_p90": res.kld_p90})
                part_passed, part_failures = judge(res.mean_kld, res.top1_agreement,
                                                   args.max_kld, args.min_top1)
                entry["gate_passed"] = part_passed
                entry["gate_failures"] = part_failures
                entry["verdict"] = judge_text(part_passed, part_failures)
                print(f"[{time.strftime('%H:%M:%S')}] {name} chunk{part['chunk']} "
                      f"{entry['row_range']}: rows={n} mean_kld={res.mean_kld:.5f} "
                      f"top1={100 * res.top1_agreement:.2f}% max={res.kld_max:.3f} "
                      f"({entry['wall_s']}s, loadavg {entry['loadavg']:.0f}, "
                      f"finite={entry['finite']})", flush=True)
            else:
                entry["skipped"] = f"missing reference {ref_name}"
                print(f"[{time.strftime('%H:%M:%S')}] {name}: {entry['skipped']}", flush=True)
            results.append(entry)
            with open(args.out_json, "w") as fh:
                json.dump(results, fh, indent=2)
    code = summarize(results, args.max_kld, args.min_top1)
    with open(args.out_json, "w") as fh:
        json.dump(results, fh, indent=2)
    return code


if __name__ == "__main__":
    sys.exit(main())