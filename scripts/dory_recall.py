"""The long-context recall harness: six codes at depths 0 to 24K, one prompt.

Spec gate: "six codes planted at depths from 0 to 24K in one 24K prompt are all
recalled at 2 loops".  The codes are random-looking tokens that do not occur in
the filler text, planted at fixed fractions of the prompt, and the question asks
for them in document order.  A miss at the *deepest* code is the interesting
one: it means the far end of the context is not reaching the answer.

Two backends, the same seam:

    --backend coreml --model-dir <converted>     the gate (Neural Engine)
    --backend torch  --model <bf16 dir>          pre-conversion, ANE-safe FP16

The harness never truncates: a prompt that does not fit is a clear error from
the runtime (see anemll.dory_chat.ContextOverflowError).
"""

from __future__ import annotations

import argparse
import json
import random
import string
import sys
import time
from pathlib import Path

import numpy as np

__all__ = [
    "DEFAULT_QUESTION",
    "build_codes",
    "build_prompt",
    "missing_codes",
    "Recorder",
    "run_recall",
    "main",
]

DEFAULT_QUESTION = ("List the six access codes that appear in the document above, "
                    "in the order they appear, one code per line. "
                    "Output only the codes.")

#: Filler sentences: deliberately dull, and never containing a code.
FILLER = [
    "The notebook lay open on the bench beside a cold cup of tea.",
    "Outside, the harbour slowly filled with the evening's small boats.",
    "Measurements were taken every hour and written down in pencil.",
    "The lighthouse keeper checked the lamp and noted the wind direction.",
    "A long list of ordinary observations follows, none of them important.",
    "The tide came in, then went out again, and the sand was smooth.",
    "Someone had left a crate of rope on the far side of the quay.",
    "The record continues with the same quiet routine as the day before.",
]


def build_codes(count: int = 6, seed: int = 20261024) -> list:
    """Codes that cannot collide with the filler or with each other."""
    alphabet = string.ascii_uppercase + string.digits
    rng = random.Random(seed)
    codes = []
    while len(codes) < int(count):
        code = "".join(rng.choice(alphabet) for _ in range(6))
        if any(ch.isdigit() for ch in code) and code not in codes:
            codes.append(code)
    return codes


def _filler_ids(tokenizer, token_count: int) -> list:
    """Exactly token_count filler tokens (repeat the sentences as needed)."""
    token_count = int(token_count)
    if token_count <= 0:
        return []
    text = " ".join(FILLER)
    ids = list(int(t) for t in tokenizer.encode(text))
    if not ids:
        raise ValueError("the tokenizer produced no filler tokens")
    while len(ids) < token_count:
        ids.extend(int(t) for t in tokenizer.encode(" " + text))
    return ids[:token_count]


def build_prompt(tokenizer, context: int, codes: list, fractions: list) -> dict:
    """One prompt of about `context` tokens with the codes at `fractions`.

    Each code is inserted inside a sentence ("The access code X is CODE.") so it
    sits in ordinary text rather than at a document edge, and the insertion
    token counts are subtracted from the filler so the whole prompt fits the
    context with room for the question and the answer.
    """
    codes = [str(c) for c in codes]
    fractions = [float(f) for f in fractions]
    if len(codes) != len(fractions):
        raise ValueError("one fraction per code")
    if not fractions or min(fractions) < 0.0 or max(fractions) > 1.0:
        raise ValueError("fractions are 0..1 positions in the prompt")
    if sorted(fractions) != fractions:
        raise ValueError("fractions must be in document order (they define the answer order)")

    insertions = [f" The access code {index + 1} is {code}. "
                  for index, code in enumerate(codes)]
    insert_tokens = [list(int(t) for t in tokenizer.encode(text)) for text in insertions]
    reserved = sum(len(tokens) for tokens in insert_tokens)
    question = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}], reasoning=False)
    overhead = len(tokenizer.encode(question))
    filler_count = int(context) - reserved - overhead
    if filler_count < 64:
        raise ValueError(f"a {int(context)}-token context leaves no room for the document")

    ids = _filler_ids(tokenizer, filler_count)
    positions = []
    for fraction, tokens in zip(fractions, insert_tokens):
        position = min(max(int(round(fraction * (len(ids) - 1))), 0), len(ids) - len(tokens))
        ids[position:position + len(tokens)] = tokens
        positions.append(position)

    document = tokenizer.decode(ids)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": document + chr(10) + chr(10) + DEFAULT_QUESTION}],
        reasoning=False)
    prompt_ids = [int(t) for t in tokenizer.encode(text)]
    return {"ids": prompt_ids, "codes": codes,
            "fractions": fractions, "positions": positions,
            "document_tokens": len(ids), "prompt_tokens": len(prompt_ids)}


def missing_codes(answer: str, codes: list) -> list:
    """The codes the answer does not contain, in the order they were planted."""
    upper = str(answer).upper()
    return [code for code in codes if str(code).upper() not in upper]


class Recorder:
    """Collects whatever a backend produces, so a run is one JSON object."""

    def __init__(self):
        self.started = time.time()
        self.events = []

    def note(self, message: str) -> None:
        self.events.append({"at": round(time.time() - self.started, 3), "note": message})


class TorchBackend:
    """The ANE-safe FP16 module on this machine (pre-conversion check)."""

    def __init__(self, model_path, context, loops=2, batch_size=64):
        from anemll.models.dory_ane_model import DoryANEConfig, DoryANEForCausalLM

        self.config = DoryANEConfig(context_length=int(context), batch_size=int(batch_size))
        self.model = DoryANEForCausalLM(self.config)
        self.model.load_pretrained_weights(str(model_path))
        self.model.eval()
        self.model.set_loops(int(loops))
        self.loops = int(loops)
        self.batch = int(batch_size)
        self.context = int(context)

    def reset(self):
        self.model.reset_caches()

    def prefill(self, ids, progress=None):
        ids = [int(t) for t in ids]
        rows = None
        done = 0
        while done + self.batch <= len(ids):
            rows = self.model.prefill(ids[done:done + self.batch], done)
            done += self.batch
            if progress is not None:
                progress(done, len(ids), 0.0)
        while done < len(ids):
            rows = self.model.prefill([ids[done]], done)
            done += 1
            if progress is not None:
                progress(done, len(ids), 0.0)
        return rows[0].numpy()

    def decode(self, token, pos):
        return self.model.decode_step(int(token), int(pos)).numpy()[0]

    def close(self):
        pass


class CoreMLBackend:
    """The converted pipeline on the Neural Engine (the gate)."""

    def __init__(self, model_dir, loops=2, progress=None):
        from anemll.dory_chat import DoryRuntime

        self.runtime = DoryRuntime(model_dir, verbose=False)
        self.runtime.set_loops(int(loops))
        self.loops = int(loops)
        self.context = int(self.runtime.params["context_length"])
        self.progress = progress

    def reset(self):
        self.runtime.reset_state()

    def prefill(self, ids, progress=None):
        rows = self.runtime.prefill(np.asarray(ids, dtype=np.int32), 0,
                                    progress=progress or self.progress)
        return rows[-1]

    def decode(self, token, pos):
        return self.runtime.decode(int(token), int(pos))

    def close(self):
        pass


def run_recall(backend, prompt, max_tokens=64, stop_ids=(), record=None) -> dict:
    """Prefill the prompt, greedily answer, and check every code came back."""
    record = record or Recorder()
    backend.reset()
    started = time.time()
    logits = backend.prefill(prompt["ids"])
    prefill_seconds = time.time() - started
    generated = []
    # pos is the row the token just sampled occupies; the decode that consumes it
    # must use that row and only then advance.  See dory_chat.generate.
    pos = len(prompt["ids"])
    steps = 0
    stop = set(int(t) for t in stop_ids)
    while steps < int(max_tokens):
        token = int(np.argmax(np.asarray(logits).reshape(-1)))
        if token in stop:
            break
        generated.append(token)
        steps += 1
        if steps >= int(max_tokens) or pos >= backend.context - 1:
            break
        logits = backend.decode(token, pos)
        pos += 1
    return {"generated": generated, "prefill_seconds": prefill_seconds,
            "answer_logits_last": True, "record": record}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Long-context recall: six codes in one prompt")
    parser.add_argument("--backend", choices=("coreml", "torch"), default="coreml")
    parser.add_argument("--model-dir", default=None, help="converted model directory (coreml)")
    parser.add_argument("--model", default=None, help="BF16 safetensors directory (torch)")
    parser.add_argument("--context", type=int, default=24576)
    parser.add_argument("--loops", type=int, default=2)
    parser.add_argument("--codes", type=int, default=6)
    parser.add_argument("--depths", default="0,0.2,0.4,0.6,0.8,1.0",
                        help="fractions of the document to plant the codes at")
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--out", default=None, help="write the JSON report here")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    fractions = [float(part) for part in str(args.depths).split(",") if part.strip()]
    codes = build_codes(args.codes, args.seed)
    record = Recorder()

    if args.backend == "coreml":
        if not args.model_dir:
            print("error: --model-dir is required for the coreml backend", file=sys.stderr)
            return 2
        from anemll.dory_chat import ChatTokenizer, read_meta

        meta = read_meta(Path(args.model_dir) / "meta.yaml")
        tokenizer = ChatTokenizer(Path(args.model_dir), meta)
        backend = CoreMLBackend(args.model_dir, args.loops)
        stop = tokenizer.eos_ids
    else:
        if not args.model:
            print("error: --model is required for the torch backend", file=sys.stderr)
            return 2
        from anemll.dory_chat import ChatTokenizer

        tokenizer = ChatTokenizer(Path(args.model),
                                  {"tokenizer": {"eos_token_id": 11,
                                                 "chat_template_fallback": "chatml-think"}})
        backend = TorchBackend(args.model, args.context, args.loops)
        stop = tokenizer.eos_ids

    prompt = build_prompt(tokenizer, min(args.context, backend.context), codes, fractions)
    record.note(f"prompt {prompt['prompt_tokens']} tokens, context {backend.context}, "
                f"loops {backend.loops}")

    def progress(done, total, elapsed):
        if args.quiet:
            return
        rate = done / elapsed if elapsed > 0 else 0.0
        print(f"\rprefill {done:,}/{total:,} ({100.0 * done / max(total, 1):.0f}%, "
              f"{rate:,.0f} tok/s)", end="" if done < total else "\n", file=sys.stderr)

    result = run_recall(backend, prompt, args.max_tokens, stop, record)
    answer = tokenizer.decode(result["generated"])
    missed = missing_codes(answer, codes)
    report = {
        "gate": "six codes at depths 0..24K recalled",
        "passed": not missed,
        "backend": args.backend,
        "context": int(backend.context),
        "loops": int(backend.loops),
        "prompt_tokens": int(prompt["prompt_tokens"]),
        "codes": prompt["codes"],
        "fractions": prompt["fractions"],
        "positions": prompt["positions"],
        "missed": missed,
        "generated_tokens": len(result["generated"]),
        "answer": answer,
        "prefill_seconds": round(result["prefill_seconds"], 3),
        "events": record.events,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + chr(10))
    print(json.dumps({key: report[key] for key in
                      ("passed", "codes", "positions", "missed", "prompt_tokens",
                       "generated_tokens", "prefill_seconds")}, indent=2))
    if not args.quiet:
        print(answer, file=sys.stderr)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

