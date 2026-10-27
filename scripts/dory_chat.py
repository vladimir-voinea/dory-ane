#!/usr/bin/env python3
"""Chat with Dory on the ANE -- the check that matters.

Scores say the distributions track a reference; this asks whether the thing is
usable.  It builds the model's own chat template, keeps the KV cache across
turns, and samples instead of taking the argmax, because greedy decoding on a
small model degenerates into loops that make a working model look broken.

    python scripts/dory_chat.py --model-dir out-24k --turns "Hi!" "What is 2+2?"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--turns", nargs="+", required=True, help="user turns, in order")
    p.add_argument("--system", default="You are a helpful assistant.")
    p.add_argument("--loops", type=int, default=1)
    p.add_argument("--max-new", type=int, default=160)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--raw", action="store_true", help="no chat template, plain completion")
    p.add_argument("--think", action="store_true",
                   help="let the model reason in a  thinking block first (slower, needs budget)")
    return p


def sample(logits: np.ndarray, temperature: float, top_p: float,
           rng: np.random.Generator) -> int:
    x = np.asarray(logits, dtype=np.float64)
    if temperature <= 0:
        return int(np.argmax(x))
    x = x / temperature
    x -= x.max()
    p = np.exp(x)
    p /= p.sum()
    if top_p < 1.0:
        order = np.argsort(-p)
        cumulative = np.cumsum(p[order])
        keep = order[:int(np.searchsorted(cumulative, top_p)) + 1]
        mask = np.zeros_like(p, dtype=bool)
        mask[keep] = True
        p = np.where(mask, p, 0.0)
        p /= p.sum()
    return int(rng.choice(len(p), p=p))


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from anemll.runtime.dory_ane_runtime import DoryProgramRuntime

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    eos = {int(tokenizer.eos_token_id)} if tokenizer.eos_token_id is not None else set()

    runtime = DoryProgramRuntime(args.model_dir, loops=args.loops, verbose=False)
    runtime.load()
    runtime.prepare_decode()
    rng = np.random.default_rng(args.seed)

    position = 0
    history: list[dict] = []
    if args.system and not args.raw:
        history.append({"role": "system", "content": args.system})

    for turn in args.turns:
        history.append({"role": "user", "content": turn})
        if args.raw:
            prompt = turn
        else:
            prompt = tokenizer.apply_chat_template(history, tokenize=False,
                                                   add_generation_prompt=True,
                                                   enable_thinking=args.think)
        ids = tokenizer(prompt, return_tensors=None)["input_ids"]
        label = "assistant" if not args.raw else "completion"
        print(f"\n{'=' * 72}\nUSER: {turn}\n{'-' * 72}\n{label.upper()}: ", end="", flush=True)

        started = time.perf_counter()
        if position:
            # Incremental: the cache already holds the conversation up to and including
            # the generated reply, so only the closing token and the new user turn are
            # fed.  Re-feeding everything instead costs a phase switch each way (~160 s
            # measured) because a 64-token batch takes the prefill packages and the
            # reply takes the infer ones -- worth avoiding for a chat.
            delta = (f"{tokenizer.eos_token}<|im_start|>user\n{turn}"
                     f"<|im_end|>\n<|im_start|>assistant\n")
            delta += "\n" if args.think else " thinking</think>"
            ids = tokenizer(delta, return_tensors=None)["input_ids"]
            logits = runtime.feed(ids, position)
            position += len(ids)
        else:
            logits = runtime.feed(ids, 0)
            position = len(ids)
        produced: list[int] = []
        first = time.perf_counter()
        for _ in range(args.max_new):
            token = sample(logits[-1], args.temperature, args.top_p, rng)
            if token in eos:
                break
            produced.append(token)
            piece = tokenizer.decode([token], skip_special_tokens=False)
            print(piece, end="", flush=True)
            if position + 1 >= runtime.plan.context_length:
                break
            logits = np.stack([runtime.decode(token, position)])
            position += 1
        elapsed = time.perf_counter() - started
        reply = tokenizer.decode(produced, skip_special_tokens=True)
        if "</think>" in reply:
            thinking, reply = reply.split("</think>", 1)
            print(f"\n{'-' * 72}\nTHINKING ({len(thinking.split())} words): "
                  f"{thinking.strip()[:300]}...")
        history.append({"role": "assistant", "content": reply.strip()})
        print(f"\n{'-' * 72}\n{len(produced)} tokens in {elapsed:.1f} s "
              f"({len(produced) / max(elapsed, 1e-9):.2f} tok/s incl. prefill)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
