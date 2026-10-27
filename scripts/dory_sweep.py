#!/usr/bin/env python3
"""DANE-8: the shipping-configuration sweep.

One table decides the default: how well each LUT width matches the reference,
what it costs on disk, and what it buys in tok/s and memory on the Neural
Engine and on the GPU.  This script collects those numbers and renders that
table, and it REFUSES to render anything while a cell has not been measured:
no interpolation, no "assume it is about the same as its neighbour".

    python scripts/dory_sweep.py plan
    python scripts/dory_sweep.py sizes  --config lut6 --model-dir <dir>
    python scripts/dory_sweep.py score  --config lut6 --loops 2 --model-dir <dir> \
        --reference ~/dev/dory-ane-artifacts/refs/bf16_decode_loop2_batch64.kld
    python scripts/dory_sweep.py speed  --config lut6 --model-dir <dir> \
        --contexts 2048,8192,16384,24576 --loops 1,2,4 --tokens 32
    python scripts/dory_sweep.py gpu    --gguf <Q8_0.gguf> --contexts 2048,8192,16384,24576
    python scripts/dory_sweep.py energy-plan
    python scripts/dory_sweep.py energy --build ane --log ane-24k-2loops.txt \
        --tokens 64 --seconds 42
    python scripts/dory_sweep.py report            # writes docs/dory-ane-config.md
    python scripts/dory_sweep.py check             # the same, without writing

Every recorded cell carries its provenance -- the command that produced it, the
load average, the artifact and its sha256 -- so a number in the table can be
traced back to a run.

Scoring is DANE-2's: this script shells out to "python -m anemll.dory_chat
score" to build the candidate and then compares it with
anemll.eval.kld.score_files.  It never reimplements KLD, and it never estimates
a cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from anemll.eval.kld import score_files

NEWLINE = chr(10)

ARTIFACTS = Path(os.path.expanduser(
    os.environ.get("DORY_SWEEP_DIR", "~/dev/dory-ane-artifacts/DANE-8")))
RESULTS_PATH = ARTIFACTS / "sweep.json"
REPO_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = REPO_ROOT / "docs" / "dory-ane-config.md"

BEGIN = "<!-- BEGIN SWEEP TABLE -->"
END = "<!-- END SWEEP TABLE -->"

SCHEMA = 1

#: One entry per LUT configuration the ticket asks about: block width 4/6/8 with
#: a LUT6 LM head (spec D8: the LM head is 131,072 channels and is tested apart).
CONFIGS = (
    ("lut4", 4, 6),
    ("lut6", 6, 6),
    ("lut8", 8, 6),
)
CONFIG_NAMES = tuple(name for name, _, _ in CONFIGS)
GPU_CONFIG = "q8_0"

#: The speed matrix the ticket asks for, in tokens of context.
CONTEXTS = (2048, 8192, 16384, 24576)
LOOPS = (1, 2, 4)

#: Spec gates (Testing Decisions).  GATE_DECODE_TOK_S is the speed gate; a
#: configuration under REASSESS_DECODE_TOK_S is reported but never shipped.
GATE_MEAN_KLD = 0.05
GATE_TOP1 = 0.93
GATE_DECODE_TOK_S = 8.0
REASSESS_DECODE_TOK_S = 6.0
GATE_CONTEXT = 24576
GATE_LOOPS = 2


# ---------------------------------------------------------------------------
# the measurement store
# ---------------------------------------------------------------------------

def load_results(path: Path = RESULTS_PATH) -> dict:
    """The sweep so far, or an empty store."""
    if not path.exists():
        return {"schema": SCHEMA, "configs": {}, "cells": {}}
    data = json.loads(path.read_text())
    if data.get("schema") != SCHEMA:
        raise SystemExit(f"{path} has schema {data.get('schema')}, this script writes {SCHEMA}")
    data.setdefault("configs", {})
    data.setdefault("cells", {})
    return data


def save_results(results: dict, path: Path = RESULTS_PATH) -> Path:
    """Atomic write: a killed run must not corrupt the sweep so far."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(results, indent=2, sort_keys=True))
    temporary.replace(path)
    return path


def sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(command: str, *, artifacts: Sequence[str] = (),
               extra: Optional[dict] = None) -> dict:
    """What produced a number: the command, the machine, the artifacts."""
    try:
        load = round(os.getloadavg()[0], 2)
    except (OSError, AttributeError):                 # pragma: no cover
        load = None
    record = {
        "command": command,
        "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": platform.node(),
        "loadavg": load,
        "artifacts": {str(path): (sha256(path) if os.path.isfile(path) else None)
                      for path in artifacts},
    }
    if extra:
        record.update(extra)
    return record


def record(results: dict, key: str, value: dict, *, command: str,
           artifacts: Sequence[str] = (), extra: Optional[dict] = None) -> dict:
    """Store one measured cell."""
    payload = dict(value)
    payload["provenance"] = provenance(command, artifacts=artifacts, extra=extra)
    results["cells"][key] = payload
    return payload


def cell(results: dict, key: str) -> Optional[dict]:
    return results["cells"].get(key)


# ---------------------------------------------------------------------------
# cell names
# ---------------------------------------------------------------------------

def size_key(config: str) -> str:
    return "size/" + config


def kld_key(config: str, loops: int) -> str:
    return f"kld/{config}/loops{int(loops)}"


def speed_key(build: str, config: str, context: int, loops: int) -> str:
    return f"speed/{build}/{config}/ctx{int(context)}/loops{int(loops)}"


def energy_key(build: str, config: str, context: int, loops: int) -> str:
    return f"energy/{build}/{config}/ctx{int(context)}/loops{int(loops)}"


# ---------------------------------------------------------------------------
# what the sweep requires
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Requirement:
    """One cell the report cannot do without, and the command that fills it."""

    key: str
    what: str
    fill: str
    why: str


def required_cells(results: dict, shipping: Optional[str] = None) -> List[Requirement]:
    """Every cell the ticket needs, in the order the report shows them.

    The shipping configuration defaults to whatever the stored cells justify.
    While the quality cells are missing there is none yet, and the report
    refuses on those first -- the honest order to fill them in.
    """
    needs: List[Requirement] = []
    for name in CONFIG_NAMES:
        needs.append(Requirement(
            size_key(name), f"{name}: package bytes by component",
            f"python scripts/dory_sweep.py sizes --config {name} --model-dir <dir>",
            "spec D8: the disk cost of each configuration (the machine has ~8 GB free)"))
    for name in CONFIG_NAMES:
        for loops in LOOPS:
            needs.append(Requirement(
                kld_key(name, loops), f"{name}: mean KLD and top-1 at {loops} loops",
                f"python scripts/dory_sweep.py score --config {name} --loops {loops} "
                f"--model-dir <dir> --reference <decode reference>.kld",
                "ticket criterion 1 (spec gate: mean KLD <= 0.05, top-1 >= 93 %)"))
    for name in CONFIG_NAMES:
        needs.append(Requirement(
            speed_key("ane", name, GATE_CONTEXT, GATE_LOOPS),
            f"{name}: decode tok/s at {GATE_CONTEXT} tokens / {GATE_LOOPS} loops",
            f"python scripts/dory_sweep.py speed --config {name} "
            f"--contexts {GATE_CONTEXT} --loops {GATE_LOOPS} --model-dir <dir>",
            "spec speed gate: decode >= 8 tok/s at 24K with 2 loops, per candidate"))

    chosen = shipping if shipping is not None else choose_shipping(results)
    if chosen is None:
        needs.append(Requirement(
            "shipping", "the shipping configuration cannot be chosen yet",
            "python scripts/dory_sweep.py plan",
            "the quality and gate-speed cells above decide which configurations "
            "are candidates at all"))
    else:
        for context in CONTEXTS:
            for loops in LOOPS:
                needs.append(Requirement(
                    speed_key("ane", chosen, context, loops),
                    f"{chosen} (shipping): decode and prefill tok/s at "
                    f"{context} tokens / {loops} loops",
                    f"python scripts/dory_sweep.py speed --config {chosen} "
                    f"--contexts {context} --loops {loops} --model-dir <dir>",
                    "ticket criterion 2, plus resident memory"))

    for context in CONTEXTS:
        for loops in LOOPS:
            needs.append(Requirement(
                speed_key("gpu", GPU_CONFIG, context, loops),
                f"GPU (Q8_0, last_loop): decode and prefill tok/s at "
                f"{context} tokens / {loops} loops",
                f"python scripts/dory_sweep.py gpu --gguf <Q8_0.gguf> "
                f"--contexts {context} --loops {loops}",
                "ticket criterion 3: the same workload on the llama.cpp build"))

    for build, config in (("ane", chosen), ("gpu", GPU_CONFIG)):
        if config is None:
            continue
        needs.append(Requirement(
            energy_key(build, config, GATE_CONTEXT, GATE_LOOPS),
            f"{build}: energy per generated token at {GATE_CONTEXT} tokens / "
            f"{GATE_LOOPS} loops",
            "python scripts/dory_sweep.py energy-plan  (the owner runs powermetrics; "
            "then energy --build <build> --log <file> --tokens <n> --seconds <s>)",
            "ticket criterion 4: powermetrics needs sudo, so the owner runs it"))
    return _dedupe(needs)


def _dedupe(needs: Sequence[Requirement]) -> List[Requirement]:
    """Drop repeats: the 24K / 2-loop gate cell is also in the shipping matrix."""
    seen, unique = set(), []
    for need in needs:
        if need.key in seen:
            continue
        seen.add(need.key)
        unique.append(need)
    return unique


def missing_cells(results: dict, *, energy_pending: bool = False,
                  shipping: Optional[str] = None) -> List[Requirement]:
    """The required cells that have no measurement yet."""
    missing: List[Requirement] = []
    for need in required_cells(results, shipping=shipping):
        if need.key == "shipping":
            # The sweep's state, not a measurement: the report turns it into
            # "no configuration meets every gate" once the deciding cells are in,
            # and refuses on those cells while they are not.
            continue
        if energy_pending and need.key.startswith("energy/"):
            continue
        if cell(results, need.key) is None:
            missing.append(need)
    return missing


# ---------------------------------------------------------------------------
# the decision
# ---------------------------------------------------------------------------

def config_verdict(results: dict, config: str) -> dict:
    """Every gate for one configuration, from the cells that exist."""
    quality_failures: List[str] = []
    speed_failures: List[str] = []
    worst_kld, worst_top1 = 0.0, 1.0
    measured_loops = 0
    for loops in LOOPS:
        measured = cell(results, kld_key(config, loops))
        if measured is None:
            quality_failures.append(f"{loops} loops: not measured")
            continue
        measured_loops += 1
        worst_kld = max(worst_kld, float(measured["mean_kld"]))
        worst_top1 = min(worst_top1, float(measured["top1"]))
        if float(measured["mean_kld"]) > GATE_MEAN_KLD:
            quality_failures.append(
                f"{loops} loops: mean KLD {float(measured['mean_kld']):.5f} > {GATE_MEAN_KLD}")
        if float(measured["top1"]) < GATE_TOP1:
            quality_failures.append(
                f"{loops} loops: top-1 {100 * float(measured['top1']):.2f} % < "
                f"{100 * GATE_TOP1:.0f} %")
    speed = cell(results, speed_key("ane", config, GATE_CONTEXT, GATE_LOOPS))
    decode = float(speed["decode_tok_s"]) if speed else None
    if speed is None:
        speed_failures.append(f"{GATE_CONTEXT} / {GATE_LOOPS} loops: not measured")
    else:
        if decode < GATE_DECODE_TOK_S:
            speed_failures.append(
                f"{GATE_CONTEXT} / {GATE_LOOPS} loops: {decode:.2f} tok/s < "
                f"{GATE_DECODE_TOK_S} tok/s")
        if decode < REASSESS_DECODE_TOK_S:
            speed_failures.append(
                f"{GATE_CONTEXT} / {GATE_LOOPS} loops: {decode:.2f} tok/s is under the "
                f"{REASSESS_DECODE_TOK_S} tok/s floor -- stop and reassess")
    return {
        "config": config,
        "quality_ok": not quality_failures and measured_loops == len(LOOPS),
        "speed_ok": not speed_failures,
        "failures": quality_failures + speed_failures,
        "worst_mean_kld": worst_kld,
        "worst_top1": worst_top1,
        "decode_tok_s": decode,
        "loops_measured": measured_loops,
    }


def choose_shipping(results: dict) -> Optional[str]:
    """The fastest configuration that meets every gate; None while undecided.

    Ties break towards quality (higher worst-case top-1), then towards the wider
    LUT, then by name, so the choice is reproducible from the stored cells alone.
    """
    verdicts = [config_verdict(results, name) for name in CONFIG_NAMES]
    passing = [verdict for verdict in verdicts
               if verdict["quality_ok"] and verdict["speed_ok"]]
    if not passing:
        return None
    passing.sort(key=lambda verdict: (
        -float(verdict["decode_tok_s"] or 0.0),
        -float(verdict["worst_top1"]),
        -int(verdict["config"].replace("lut", "")),
    ))
    return passing[0]["config"]


# ---------------------------------------------------------------------------
# measurements: sizes
# ---------------------------------------------------------------------------

def read_meta(model_dir) -> dict:
    """meta.yaml, without importing the converter (which wants coremltools)."""
    import yaml

    path = Path(model_dir) / "meta.yaml"
    if not path.is_file():
        raise SystemExit(f"no meta.yaml in {model_dir}")
    return yaml.safe_load(path.read_text())


def directory_bytes(path) -> int:
    return sum(file.stat().st_size for file in Path(path).rglob("*") if file.is_file())


def measure_sizes(model_dir, config: str) -> dict:
    """Package bytes for the files the meta names, checked against the meta."""
    model_dir = Path(model_dir)
    meta = read_meta(model_dir)
    params = meta["parameters"]
    expected = dict((name, blocks) for name, blocks, _ in CONFIGS)[config]
    lut_blocks = params.get("lut_blocks")
    if str(lut_blocks) != str(expected):
        raise SystemExit(
            f"this directory says lut_blocks={lut_blocks!r}, but --config {config} means "
            f"{expected} bits; the directory and the configuration do not match")
    if str(params.get("lut_lm_head")) != "6":
        raise SystemExit(
            f"this directory has lut_lm_head={params.get('lut_lm_head')!r}; the sweep "
            f"fixes the LM head at LUT6")

    by_component, missing = {}, []
    for name in meta["run_order"]:
        entry = meta["components"][name]
        total = 0
        for function in entry["functions"]:
            path = model_dir / entry["files"][function]
            if not path.exists():
                missing.append(path.name)
                continue
            total += directory_bytes(path) if path.is_dir() else path.stat().st_size
        by_component[name] = total
    if missing:
        raise SystemExit(f"missing packages in {model_dir}: {', '.join(sorted(missing))}")
    return {
        "total_bytes": sum(by_component.values()),
        "by_component": by_component,
        "context_length": int(params["context_length"]),
        "lut_blocks": int(lut_blocks),
        "lut_lm_head": int(params["lut_lm_head"]),
        "model_dir": str(model_dir),
        "files": sum(len(meta["components"][n]["functions"]) for n in meta["run_order"]),
    }


# ---------------------------------------------------------------------------
# measurements: quality (DANE-2's scoring, not reimplemented)
# ---------------------------------------------------------------------------

def dory_chat_score_command(model_dir, candidate: Path, *, loops: int,
                            chunk_size: int = 1024, max_windows: int = 0,
                            tokens_file: Optional[str] = None) -> List[str]:
    """The candidate command, kept pure so a test can assert its shape."""
    command = [sys.executable, "-m", "anemll.dory_chat", "score",
               "--model-dir", str(model_dir), "--out", str(candidate),
               "--loops", str(int(loops)), "--chunk-size", str(int(chunk_size))]
    if max_windows:
        command += ["--max-windows", str(int(max_windows))]
    if tokens_file:
        command += ["--tokens", str(tokens_file)]
    return command


def score_candidate(reference, candidate) -> dict:
    """DANE-2's score_files, reduced to the numbers the table shows."""
    result = score_files(reference, candidate)
    return {
        "mean_kld": float(result.mean_kld),
        "kld_stderr": float(result.kld_stderr),
        "kld_p90": float(result.kld_p90),
        "kld_99": float(result.kld_99),
        "kld_max": float(result.kld_max),
        "top1": float(result.top1_agreement),
        "n_rows": int(result.n_rows),
        "vocab_size": int(result.vocab_size),
    }


# ---------------------------------------------------------------------------
# measurements: speed (Neural Engine)
# ---------------------------------------------------------------------------

def workload_ids(count: int, tokenizer=None) -> np.ndarray:
    """A fixed prompt of exactly 'count' token ids.

    The evaluation tokens from DANE-2's fixture, tiled to length, so the Neural
    Engine and GPU runs see the same prompt.  The tokenizer is only a fallback
    when the fixture is absent.
    """
    fixture = REPO_ROOT / "tests" / "fixtures" / "dory_eval" / "eval_tokens.txt"
    if fixture.is_file():
        ids = np.loadtxt(fixture, dtype=np.int64).reshape(-1).astype(np.int32)
    elif tokenizer is not None:
        ids = np.asarray(tokenizer.encode("The quick brown fox jumps over the lazy dog. " * 64),
                         dtype=np.int32).reshape(-1)
    else:                                              # pragma: no cover - no fixture
        raise SystemExit(f"no {fixture} to build a prompt from")
    if ids.size == 0:                                  # pragma: no cover
        raise SystemExit("the eval fixture is empty")
    return np.tile(ids, int(np.ceil(count / ids.size)))[:count].astype(np.int32)


def _resident_memory() -> dict:
    """Peak RSS of this process (ru_maxrss is bytes on macOS)."""
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    machine = {}
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True)
        machine["swap"] = out.stdout.strip()
    except Exception:                                   # pragma: no cover
        pass
    return {"peak_rss_bytes": int(peak), "machine": machine}


def measure_speed(model_dir, config: str, contexts: Sequence[int], loops_list: Sequence[int],
                  *, tokens: int = 32, runtime_factory: Optional[Callable] = None,
                  warm_fraction: float = 0.25) -> Dict[str, dict]:
    """Prefill and decode tok/s plus peak RSS, with one load of the model.

    DoryRuntime is imported here, not at module scope: everything else in this
    script has to run where coremltools is absent (the tests do).  A caller may
    pass runtime_factory instead, which is how the tests drive this function
    without loading a model.
    """
    if runtime_factory is None:
        from anemll.dory_chat import ChatTokenizer, DoryRuntime

        runtime = DoryRuntime(model_dir, verbose=False)
        tokenizer = ChatTokenizer(Path(model_dir), read_meta(model_dir))
    else:
        runtime = runtime_factory(model_dir)
        tokenizer = None

    out: Dict[str, dict] = {}
    steps = int(tokens)
    for loops in loops_list:
        runtime.set_loops(int(loops))
        for context in contexts:
            key = speed_key("ane", config, int(context), int(loops))
            # One cell failing must not discard the cells that already worked: a
            # 24K context that overflows used to lose the 2K/8K/16K numbers with
            # it, because they only lived in this local.  The prompt leaves room
            # for the decode steps, because check_context reserves one position
            # and a full-context prompt raised before any measurement.
            try:
                prompt_tokens = max(int(context) - max(steps, 1), 0)
                if prompt_tokens < 1:
                    raise ValueError(
                        f"a {int(context)}-token context leaves no room for {steps} "
                        f"decode step(s)")
                ids = workload_ids(prompt_tokens, tokenizer)
                runtime.reset_state()
                started = time.perf_counter()
                logits = runtime.prefill(ids, 0)
                prefill_seconds = max(time.perf_counter() - started, 1e-9)
                window: List[float] = []
                position = int(len(ids))
                token = int(np.argmax(np.asarray(logits[-1]).reshape(-1)))
                for _ in range(steps):
                    started = time.perf_counter()
                    logits = runtime.decode(token, position)
                    window.append(time.perf_counter() - started)
                    token = int(np.argmax(np.asarray(logits).reshape(-1)))
                    position += 1
                warm = window[int(len(window) * warm_fraction):] or window
                mean = float(np.mean(warm)) if warm else float("nan")
                resident = _resident_memory()
                out[key] = {
                    "decode_tok_s": (1.0 / mean) if mean > 0 else float("nan"),
                    "decode_ms": mean * 1e3,
                    "prefill_tok_s": len(ids) / prefill_seconds,
                    "prefill_seconds": prefill_seconds,
                    "prompt_tokens": int(len(ids)),
                    "decode_headroom": int(context) - int(len(ids)),
                    "decoded_tokens": int(len(warm)),
                    "peak_rss_bytes": int(resident["peak_rss_bytes"]),
                    "machine": resident.get("machine", {}),
                }
            except Exception as exc:                  # one cell, not the whole sweep
                out[key] = {"error": f"{type(exc).__name__}: {exc}",
                            "context": int(context), "loops": int(loops)}
                out.setdefault("errors", []).append(key)
    if "errors" in out:
        print(f"[speed] {len(out['errors'])} cell(s) failed: {out['errors']}",
              file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# measurements: the GPU build (llama.cpp, Q8_0, last_loop)
# ---------------------------------------------------------------------------

def gpu_completion(port: int, prompt_ids: Sequence[int], n_predict: int) -> dict:
    """One non-streaming /completion, fed token ids, not text.

    The Neural Engine side feeds exactly workload_ids(context) token ids; a
    character slice of some text would tokenise to a different length and a
    different content, and the head-to-head table would compare two workloads.
    llama.cpp's server accepts a prompt as an array of token ids, which makes
    both sides literally the same input.
    """
    import urllib.request

    body = json.dumps({"prompt": [int(token) for token in prompt_ids],
                       "n_predict": int(n_predict),
                       "temperature": 0.0, "stream": False}).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{int(port)}/completion", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=7200) as response:
        return json.loads(response.read())


def wait_for_server(port: int, timeout_seconds: float = 1800.0) -> None:
    import urllib.error
    import urllib.request

    deadline = time.time() + float(timeout_seconds)
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{int(port)}/health", timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(2.0)
    raise SystemExit(f"llama-server did not answer on port {port} within {timeout_seconds:.0f}s")


def gpu_prompt_tokens(context: int) -> List[int]:
    """The token ids the Neural Engine side feeds at this context.

    Not "the same text": the same ids, produced by the same function.  Anything
    else (a character slice of some prompt text) tokenises to a different
    sequence, and the two columns of the head-to-head table would not be
    measuring the same workload.
    """
    return [int(token) for token in workload_ids(int(context))]


def check_prompt_count(timings: dict, expected: int, context: int) -> None:
    """Refuse a row whose server-side prompt length differs from what was sent."""
    reported = timings.get("prompt_n")
    if reported is None or int(reported) != int(expected):
        raise SystemExit(
            f"GPU row at {context} tokens: the server counted {reported} prompt tokens "
            f"but {expected} were sent.  Refusing to record a workload the Neural "
            f"Engine side did not measure.")


def measure_gpu(gguf, contexts: Sequence[int], loops_list: Sequence[int], *,
                port: int = 49562, tokens: int = 32,
                llama_server: Optional[str] = None) -> Dict[str, dict]:
    """The same cells from the GPU build: Q8_0 with last_loop (spec D4)."""
    import psutil

    binary = llama_server or os.path.expanduser(
        "~/.unsloth/llama.cpp-dory/build/bin/llama-server")
    prompts = {int(context): gpu_prompt_tokens(int(context)) for context in contexts}
    out: Dict[str, dict] = {}
    for loops in loops_list:
        process = subprocess.Popen(
            [binary, "-m", str(gguf), "-ngl", "999",
             "-c", str(max(int(value) for value in contexts) + int(tokens) + 64),
             "--override-kv",
             f"dory.num_loops=int:{int(loops)},dory.recurrent_kv_cache_mode=str:last_loop",
             "--port", str(int(port))],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            wait_for_server(int(port))
            for context in contexts:
                ids = prompts[int(context)]
                reply = gpu_completion(int(port), ids, int(tokens))
                timings = reply.get("timings") or {}
                check_prompt_count(timings, len(ids), int(context))
                try:
                    peak = int(psutil.Process(process.pid).memory_info().rss)
                except psutil.Error:                    # pragma: no cover
                    peak = 0
                out[speed_key("gpu", GPU_CONFIG, int(context), int(loops))] = {
                    "decode_tok_s": float(timings.get("predicted_per_second", "nan")),
                    "prefill_tok_s": float(timings.get("prompt_per_second", "nan")),
                    "prompt_tokens": len(ids),
                    "decoded_tokens": int(timings.get("predicted_n", 0)),
                    "peak_rss_bytes": peak,
                }
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:           # pragma: no cover
                process.kill()
    return out


# ---------------------------------------------------------------------------
# energy: the owner runs powermetrics, this records what comes back
# ---------------------------------------------------------------------------

POWER_SAMPLERS = {
    "ane": ("ANE Power",),
    "gpu": ("GPU Power",),
}


def energy_plan(config: str, gpu_config: str = GPU_CONFIG) -> str:
    """The exact commands and workload the owner runs with sudo."""
    lines = [
        "# Energy per generated token.  Run these with sudo in a quiet window;",
        "# -i 500 gives two samples a second, enough for a run that lasts minutes.",
        "#",
        "# 1. Neural Engine build (" + config + ", 2 loops, 24,576-token prompt, 64 tokens):",
        "sudo powermetrics --samplers ane_power,gpu_power,cpu_power -i 500 "
        "-o ane-24k-2loops.txt &",
        "PM=$!",
        "PYTHONPATH=<worktree> python -m anemll.dory_chat bench --model-dir <converted-dir> "
        "--context 24576 --loops 2 --tokens 64",
        "kill -INT $PM; wait $PM",
        "#    paste back: ane-24k-2loops.txt plus the JSON that bench prints (it carries",
        "#    decode_tok_s and the number of decoded tokens; pass that count as --tokens,",
        "#    and use the bench wall time for --seconds).",
        "#",
        "# 2. GPU build (Q8_0, last_loop), the same prompt and token count:",
        "sudo powermetrics --samplers ane_power,gpu_power,cpu_power -i 500 "
        "-o gpu-24k-2loops.txt &",
        "PM=$!",
        "PYTHONPATH=<worktree> python scripts/dory_sweep.py gpu --gguf <Q8_0.gguf> "
        "--contexts 24576 --loops 2 --tokens 64",
        "kill -INT $PM; wait $PM",
        "#",
        "# 3. Record both (power x time / tokens, from the samples above):",
        "python scripts/dory_sweep.py energy --build ane --log ane-24k-2loops.txt "
        "--tokens 64 --seconds <bench seconds>",
        "python scripts/dory_sweep.py energy --build gpu --log gpu-24k-2loops.txt "
        "--tokens 64 --seconds <gpu seconds>",
    ]
    return NEWLINE.join(lines) + NEWLINE


def parse_powermetrics(text: str, build: str) -> dict:
    """Mean sampled power (mW) for one build from a powermetrics log."""
    keys = POWER_SAMPLERS[build]
    samples: Dict[str, List[float]] = {}
    for line in text.splitlines():
        stripped = line.strip()
        for key in keys:
            if not stripped.startswith(key + ":"):
                continue
            value = stripped.split(":", 1)[1].strip().split()[0]
            try:
                samples.setdefault(key, []).append(float(value))
            except ValueError:
                pass
    if not samples:
        raise SystemExit("no " + " or ".join(keys) + " samples in the powermetrics log")
    values = [value for series in samples.values() for value in series]
    return {"mean_power_mw": float(np.mean(values)), "samples": len(values),
            "series": samples}


def energy_from_log(log_path, build: str, tokens: int, seconds: float) -> dict:
    """Joules per generated token = mean sampled power x run time / tokens."""
    if int(tokens) <= 0:
        raise SystemExit("--tokens must be the number of generated tokens in the run")
    if seconds is None or float(seconds) <= 0:
        raise SystemExit("--seconds must be the wall time of the timed workload")
    parsed = parse_powermetrics(Path(log_path).read_text(), build)
    joules = parsed["mean_power_mw"] / 1000.0 * float(seconds)
    return {
        "joules_per_token": joules / float(tokens),
        "mean_power_mw": parsed["mean_power_mw"],
        "samples": parsed["samples"],
        "tokens": int(tokens),
        "seconds": float(seconds),
    }


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

class SweepIncomplete(RuntimeError):
    """The report was asked to render while a required cell is unmeasured."""

    def __init__(self, missing: List[Requirement]):
        self.missing = missing
        super().__init__(f"{len(missing)} required cell(s) are unmeasured; "
                         f"refusing to emit a table")


def fmt_bytes(value) -> str:
    return f"{float(value) / 1024 ** 3:.2f} GiB"


def fmt_tok_s(value) -> str:
    return "--" if value is None or not np.isfinite(value) else f"{float(value):.1f}"


def fmt_pct(value) -> str:
    return "--" if value is None else f"{100.0 * float(value):.2f} %"


def fmt_gib(value) -> str:
    return "--" if not value else f"{float(value) / 1024 ** 3:.1f} GiB"


def quality_table(results: dict) -> List[str]:
    lines = ["| configuration | loops | mean KLD | top-1 | verdict |",
             "|---|---|---|---|---|"]
    for name in CONFIG_NAMES:
        verdict = config_verdict(results, name)
        for loops in LOOPS:
            measured = cell(results, kld_key(name, loops))
            if measured is None:
                lines.append(f"| {name} | {loops} | NOT MEASURED | NOT MEASURED | refused |")
                continue
            lines.append(
                f"| {name} | {loops} | {float(measured['mean_kld']):.5f} "
                f"+/- {float(measured['kld_stderr']):.5f} | "
                f"{fmt_pct(measured['top1'])} | "
                f"{'pass' if verdict['quality_ok'] else 'fail'} |")
    return lines


def size_table(results: dict) -> List[str]:
    lines = ["| configuration | on disk | components |", "|---|---|---|"]
    for name in CONFIG_NAMES:
        measured = cell(results, size_key(name))
        if measured is None:
            lines.append(f"| {name} | NOT MEASURED | NOT MEASURED |")
            continue
        parts = ", ".join(f"{component} {fmt_bytes(size)}"
                          for component, size in measured["by_component"].items())
        lines.append(f"| {name} | {fmt_bytes(measured['total_bytes'])} | {parts} |")
    return lines


def speed_table(results: dict, build: str, config: str) -> List[str]:
    header = ("| context (tokens) | " + " | ".join(
        f"decode {loops} loop(s) | prefill {loops} loop(s)" for loops in LOOPS)
        + " | peak RSS |")
    lines = [header, "|" + "---|" * (1 + 2 * len(LOOPS) + 1)]
    for context in CONTEXTS:
        values = [cell(results, speed_key(build, config, context, loops)) for loops in LOOPS]
        present = [value for value in values if value is not None]
        if not present:
            lines.append(f"| {context} |" + " NOT MEASURED |" * (2 * len(LOOPS))
                         + " NOT MEASURED |")
            continue
        parts = [str(context)]
        for value in values:
            if value is None:
                parts.extend(["NOT MEASURED", "NOT MEASURED"])
            else:
                parts.extend([fmt_tok_s(value["decode_tok_s"]),
                              fmt_tok_s(value["prefill_tok_s"])])
        parts.append(fmt_gib(max(int(value.get("peak_rss_bytes") or 0) for value in present)))
        lines.append("| " + " | ".join(parts) + " |")
    return lines


def comparison_table(results: dict, chosen: str) -> List[str]:
    lines = ["| metric | Neural Engine (" + chosen + ") | GPU (Q8_0, last_loop) |",
             "|---|---|---|"]
    ane = cell(results, speed_key("ane", chosen, GATE_CONTEXT, GATE_LOOPS)) or {}
    gpu = cell(results, speed_key("gpu", GPU_CONFIG, GATE_CONTEXT, GATE_LOOPS)) or {}
    ane_energy = cell(results, energy_key("ane", chosen, GATE_CONTEXT, GATE_LOOPS)) or {}
    gpu_energy = cell(results, energy_key("gpu", GPU_CONFIG, GATE_CONTEXT, GATE_LOOPS)) or {}
    rows = [
        ("decode tok/s", fmt_tok_s(ane.get("decode_tok_s")), fmt_tok_s(gpu.get("decode_tok_s"))),
        ("prefill tok/s", fmt_tok_s(ane.get("prefill_tok_s")),
         fmt_tok_s(gpu.get("prefill_tok_s"))),
        ("peak RSS", fmt_gib(ane.get("peak_rss_bytes")), fmt_gib(gpu.get("peak_rss_bytes"))),
        ("energy per generated token",
         (f"{float(ane_energy['joules_per_token']):.2f} J" if ane_energy else "not measured"),
         (f"{float(gpu_energy['joules_per_token']):.2f} J" if gpu_energy else "not measured")),
    ]
    for label, left, right in rows:
        lines.append(f"| {label} | {left} | {right} |")
    return lines


def verdict_block(results: dict, chosen: Optional[str]) -> List[str]:
    lines = ["| configuration | quality gates | speed gate (24K / 2 loops) | notes |",
             "|---|---|---|---|"]
    for name in CONFIG_NAMES:
        verdict = config_verdict(results, name)
        notes = "; ".join(verdict["failures"]) if verdict["failures"] else "meets every gate"
        lines.append(
            f"| {name} | {'pass' if verdict['quality_ok'] else 'fail'} | "
            f"{'pass' if verdict['speed_ok'] else 'fail'} | {notes} |")
    return lines


def provenance_block(results: dict, chosen: Optional[str]) -> List[str]:
    lines = ["| cell | command | load avg | when |", "|---|---|---|---|"]
    for key in sorted(results["cells"]):
        record_ = results["cells"][key].get("provenance") or {}
        command = str(record_.get("command") or "")[:110]
        lines.append(f"| {key} | {command} | {record_.get('loadavg')} | "
                     f"{record_.get('when', '')} |")
    return lines


def render_markdown(results: dict, *, energy_pending: bool = False) -> str:
    """The measurement section of the report, or SweepIncomplete."""
    chosen = choose_shipping(results)
    missing = missing_cells(results, energy_pending=energy_pending)
    if missing:
        raise SweepIncomplete(missing)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S%z")
    blocks: List[str] = []
    blocks.append(f"Generated by scripts/dory_sweep.py on {stamp} from "
                  f"{RESULTS_PATH}.")
    blocks.append("")
    if energy_pending:
        blocks.append("> **Energy is not measured yet.** The powermetrics runs need "
                      "sudo, so the owner runs them (the commands are at the end of "
                      "this page); every other cell here is measured.")
        blocks.append("")
    if chosen is None:
        blocks.append("> **NO CONFIGURATION MEETS EVERY GATE.** Nothing is shipped "
                      "from this sweep; the verdicts below say why.")
        blocks.append("")
    else:
        verdict = config_verdict(results, chosen)
        blocks.append(f"**Shipping configuration: {chosen}** -- "
                      f"{chosen.replace('lut', '')}-bit blocks with a LUT6 LM head, "
                      f"loop count chosen per request (default 2).")
        blocks.append("")
        blocks.append(
            f"It is the fastest configuration that clears every gate: worst-case mean "
            f"KLD {verdict['worst_mean_kld']:.5f} and worst-case top-1 "
            f"{fmt_pct(verdict['worst_top1'])} across loops 1, 2 and 4 (gates: KLD <= "
            f"{GATE_MEAN_KLD}, top-1 >= {fmt_pct(GATE_TOP1)}), and "
            f"{fmt_tok_s(verdict['decode_tok_s'])} tok/s at 24K / 2 loops "
            f"(gate >= {GATE_DECODE_TOK_S} tok/s).")
        blocks.append("")
    blocks.append("### The GPU baseline")
    blocks.append("")
    blocks.append("Q8_0, last_loop, the llama.cpp side build -- the same prompt text, "
                  "the same loop counts, the same measurement points.")
    blocks.append("")
    blocks.extend(speed_table(results, "gpu", GPU_CONFIG))
    blocks.append("")
    blocks.append("### Neural Engine vs GPU at 24K / 2 loops")
    blocks.append("")
    if chosen is not None:
        blocks.extend(comparison_table(results, chosen))
    blocks.append("")
    blocks.append("### Quality against the bf16 decode reference")
    blocks.append("")
    blocks.append("Decode mode (last_loop, the port's own batching): mean KLD and top-1 "
                  "per loop count.  Spec gates: KLD <= 0.05, top-1 >= 93 %.")
    blocks.append("")
    blocks.extend(quality_table(results))
    blocks.append("")
    blocks.append("### On disk")
    blocks.append("")
    blocks.extend(size_table(results))
    blocks.append("")
    blocks.append("### Speed on the Neural Engine")
    blocks.append("")
    if chosen is not None:
        blocks.extend(speed_table(results, "ane", chosen))
    blocks.append("")
    blocks.append("### Gates, configuration by configuration")
    blocks.append("")
    blocks.extend(verdict_block(results, chosen))
    blocks.append("")
    blocks.append("### Provenance")
    blocks.append("")
    blocks.append("Every cell above came from one of these runs; the command, the load "
                  "average it ran under and the artifact hashes are stored beside the "
                  "number in " + str(RESULTS_PATH) + ".")
    blocks.append("")
    blocks.extend(provenance_block(results, chosen))
    return NEWLINE.join(blocks) + NEWLINE


def write_report(results: dict, path: Path = REPORT_PATH, *,
                 energy_pending: bool = False) -> Path:
    """Replace the generated region of the report page."""
    table = render_markdown(results, energy_pending=energy_pending)
    path = Path(path)
    text = path.read_text() if path.exists() else (BEGIN + NEWLINE + END + NEWLINE)
    if BEGIN not in text or END not in text:
        raise SystemExit(f"{path} has no {BEGIN} / {END} markers to fill")
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    path.write_text(head + BEGIN + NEWLINE + NEWLINE + table + NEWLINE + END + tail)
    return path


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------

def _parse_ints(value: str) -> List[int]:
    return [int(part) for part in str(value).split(",") if part.strip()]


def _print_missing(missing: Sequence[Requirement]) -> None:
    print(f"REFUSING TO EMIT A TABLE: {len(missing)} required cell(s) are unmeasured.")
    print("")
    for need in missing:
        print(f"- {need.what}")
        print(f"    why : {need.why}")
        print(f"    fill: {need.fill}")
    print("")
    print("No value is interpolated, estimated or carried over from a neighbouring "
          "cell: measure the cells above and run this again.")


def _record_all(results: dict, results_path: Path, values: dict, *,
                artifacts: Sequence[str]) -> None:
    """Record every cell of a measuring command, then save the store."""
    command = " ".join(sys.argv[1:])
    for key, value in values.items():
        record(results, key, value, command=command, artifacts=artifacts)
    save_results(results, results_path)


def _cmd_plan(args, results, results_path) -> int:
    missing = missing_cells(results, energy_pending=False)
    print(f"sweep store: {results_path} ({len(results['cells'])} cells measured)")
    print(f"shipping configuration: {choose_shipping(results) or 'undecided'}")
    required = [need for need in required_cells(results) if need.key != "shipping"]
    print(f"required measurements: {len(required)}, missing: {len(missing)}")
    if missing:
        print("")
        _print_missing(missing)
    return 0


def _cmd_sizes(args, results, results_path) -> int:
    value = measure_sizes(args.model_dir, args.config)
    record(results, size_key(args.config), value,
           command=" ".join(sys.argv[1:]),
           artifacts=[Path(args.model_dir) / "meta.yaml"])
    results["configs"].setdefault(args.config, {}).update(
        {"lut_blocks": value["lut_blocks"], "lut_lm_head": value["lut_lm_head"],
         "model_dir": value["model_dir"], "context_length": value["context_length"]})
    save_results(results, results_path)
    print(json.dumps(value, indent=2))
    return 0


def _cmd_score(args, results, results_path) -> int:
    candidate = (Path(args.candidate) if args.candidate
                 else Path(args.results).parent / f"cand-{args.config}-loop{args.loops}.npz")
    if args.candidate is None:
        argv_ = dory_chat_score_command(
            args.model_dir, candidate, loops=args.loops, chunk_size=args.chunk_size,
            max_windows=args.max_windows, tokens_file=args.tokens_file)
        print("running: " + " ".join(argv_))
        completed = subprocess.run(argv_, capture_output=True, text=True)
        if completed.returncode != 0:
            print(completed.stdout[-2000:])
            print(completed.stderr[-2000:], file=sys.stderr)
            raise SystemExit(f"dory_chat score failed (rc {completed.returncode})")
    value = score_candidate(args.reference, candidate)
    value.update({"reference": str(args.reference), "candidate": str(candidate),
                  "config": args.config, "loops": int(args.loops)})
    record(results, kld_key(args.config, args.loops), value,
           command=" ".join(sys.argv[1:]),
           artifacts=[args.reference, candidate])
    save_results(results, results_path)
    print(json.dumps(value, indent=2))
    return 0


def _cmd_speed(args, results, results_path) -> int:
    values = measure_speed(args.model_dir, args.config, _parse_ints(args.contexts),
                           _parse_ints(args.loops), tokens=args.tokens)
    _record_all(results, results_path, values,
                artifacts=[Path(args.model_dir) / "meta.yaml"])
    print(json.dumps(values, indent=2))
    return 0


def _cmd_gpu(args, results, results_path) -> int:
    values = measure_gpu(args.gguf, _parse_ints(args.contexts), _parse_ints(args.loops),
                         tokens=args.tokens, port=args.port,
                         llama_server=args.llama_server)
    _record_all(results, results_path, values, artifacts=[args.gguf])
    print(json.dumps(values, indent=2))
    return 0


def _cmd_energy_plan(args, results, results_path) -> int:
    chosen = choose_shipping(results)
    if chosen is None:
        print("# The shipping configuration is not chosen yet: the quality and")
        print("# gate-speed cells decide it.  The command below works for whichever")
        print("# configuration wins.")
        print("")
        chosen = "<shipping config>"
    print(energy_plan(chosen))
    return 0


def _cmd_energy(args, results, results_path) -> int:
    config = args.config or (choose_shipping(results) if args.build == "ane"
                             else GPU_CONFIG) or CONFIG_NAMES[-1]
    value = energy_from_log(args.log, args.build, args.tokens, args.seconds)
    record(results, energy_key(args.build, config, args.context, args.loops), value,
           command=" ".join(sys.argv[1:]), artifacts=[args.log])
    save_results(results, results_path)
    print(json.dumps(value, indent=2))
    return 0


def _cmd_report(args, results, results_path) -> int:
    """Both 'report' and 'check': the second renders without writing."""
    try:
        table = render_markdown(results, energy_pending=args.energy_pending)
    except SweepIncomplete as incomplete:
        _print_missing(incomplete.missing)
        return 2
    if choose_shipping(results) is None:
        print(table)
        print("The sweep is complete but NO configuration meets every gate; "
              "nothing is shipped.", file=sys.stderr)
        return 3
    if args.command == "check":
        print(table)
        return 0
    written = write_report(results, Path(args.out), energy_pending=args.energy_pending)
    print(f"wrote {written}")
    return 0


#: One handler per subcommand, attached with set_defaults(func=...) so the
#: dispatch is the parser's job -- the same shape anemll/eval/kld.py and
#: anemll/dory_chat.py use, and no if-ladder to keep in step with the parser.
HANDLERS = {
    "plan": _cmd_plan,
    "sizes": _cmd_sizes,
    "score": _cmd_score,
    "speed": _cmd_speed,
    "gpu": _cmd_gpu,
    "energy-plan": _cmd_energy_plan,
    "energy": _cmd_energy,
    "report": _cmd_report,
    "check": _cmd_report,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", default=str(RESULTS_PATH),
                        help="where the measurements are stored")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    def add(name: str, help_text: str):
        command = sub.add_parser(name, help=help_text)
        command.set_defaults(func=HANDLERS[name])
        return command

    add("plan", "what the sweep needs, measured and missing")

    sizes = add("sizes", "package bytes for one converted configuration")
    sizes.add_argument("--config", required=True, choices=CONFIG_NAMES)
    sizes.add_argument("--model-dir", required=True)

    score = add("score", "mean KLD and top-1 against a reference")
    score.add_argument("--config", required=True, choices=CONFIG_NAMES)
    score.add_argument("--loops", type=int, required=True, choices=LOOPS)
    score.add_argument("--model-dir", required=True)
    score.add_argument("--reference", required=True)
    score.add_argument("--candidate", default=None,
                       help="score this .npz instead of building one with dory_chat score")
    score.add_argument("--chunk-size", type=int, default=1024)
    score.add_argument("--max-windows", type=int, default=0)
    score.add_argument("--tokens-file", default=None)

    speed = add("speed", "prefill and decode tok/s on the Neural Engine")
    speed.add_argument("--config", required=True, choices=CONFIG_NAMES)
    speed.add_argument("--model-dir", required=True)
    speed.add_argument("--contexts", default=",".join(str(v) for v in CONTEXTS))
    speed.add_argument("--loops", default=",".join(str(v) for v in LOOPS))
    speed.add_argument("--tokens", type=int, default=32,
                       help="tokens decoded per cell (the warm window drops the first quarter)")

    gpu = add("gpu", "the same cells from the llama.cpp GPU build")
    gpu.add_argument("--gguf", required=True)
    gpu.add_argument("--contexts", default=",".join(str(v) for v in CONTEXTS))
    gpu.add_argument("--loops", default=",".join(str(v) for v in LOOPS))
    gpu.add_argument("--tokens", type=int, default=32)
    gpu.add_argument("--port", type=int, default=49562)
    gpu.add_argument("--llama-server", default=None)

    add("energy-plan", "the powermetrics commands the owner runs")

    energy = add("energy", "record a powermetrics run")
    energy.add_argument("--build", required=True, choices=("ane", "gpu"))
    energy.add_argument("--log", required=True)
    energy.add_argument("--tokens", type=int, required=True)
    energy.add_argument("--seconds", type=float, required=True)
    energy.add_argument("--context", type=int, default=GATE_CONTEXT)
    energy.add_argument("--loops", type=int, default=GATE_LOOPS)
    energy.add_argument("--config", default=None)

    for name, help_text in (("report", "render the report page"),
                            ("check", "render, but do not write")):
        command = add(name, help_text)
        command.add_argument("--out", default=str(REPORT_PATH))
        command.add_argument("--energy-pending", action="store_true",
                             help="render with the energy row marked as the owner's "
                                  "pending powermetrics run (energy is not a gate)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    results_path = Path(args.results)
    results = load_results(results_path)
    return args.func(args, results, results_path)


if __name__ == "__main__":
    raise SystemExit(main())
