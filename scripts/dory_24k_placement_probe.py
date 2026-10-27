"""The 24K placement question, answered with the evidence this host can produce.

Spec D7 asks whether a 24,576-row full-attention cache stays on the Neural Engine
inside a block chunk; the answer decides one cache against the 16,384 + 8,192 tile
fallback.  Converting all five components at 24K to find out costs an hour of
palettization, so this probe converts a seven-block chunk (blocks 0-6: three
sliding, one full, three MLP) at the target context, unquantized -- quantization
does not move a graph between devices -- and answers with two kinds of evidence
instead of an op count, because an op count is not obtainable here:

  * Core ML's own compute plan, recorded VERBATIM.  On this host (macOS 27.2,
    coremltools 8.3) MLComputePlan attributes every operation to "other" and
    cannot build a plan for a stateful model at all, so this section says what the
    tool said and never turns it into a percentage.
  * LATENCY of the same converted chunk under ComputeUnit.CPU_ONLY and under
    ComputeUnit.CPU_AND_NE.  A graph that material work runs on the ANE is
    markedly faster with the ANE enabled; a graph that is not gets the same time.
    That is indirect evidence, and the report labels it as such.

The host's stateful path is ALSO intermittent: the same package loads and predicts
in one fresh process and returns error -14 in others, minutes apart.  So every
heavy step runs in a fresh process and is retried, and the artifact records how
many attempts it took, the error each failed attempt returned, and the load average
and swap at every attempt.

    python scripts/dory_24k_placement_probe.py --model <bf16 dir>
    python scripts/dory_24k_placement_probe.py --model <bf16 dir> --full-tile-rows 16384,8192

Nothing here writes a placement percentage, and nothing here is a gate that can
pass on an empty attribution: the verdict is the measured latency pair, and the
raw plan is attached so a reader can disagree with the tool.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

__all__ = [
    "BLOCKS",
    "build_report",
    "compute_plan_raw",
    "latency_verdict",
    "machine_facts",
    "probe_once",
    "run_attempts",
    "main",
]

BLOCKS = (0, 1, 2, 3, 4, 5, 6)          # 0,2,4 sliding; 6 full; 1,3,5 MLP
DEFAULT_ATTEMPTS = 10


# ---------------------------------------------------------------------------
# what the machine looked like at that moment
# ---------------------------------------------------------------------------

def machine_facts() -> dict:
    """Load average and swap, which every number below has to be read against."""
    facts = {}
    try:
        one, five, fifteen = os.getloadavg()
        facts.update(load_1m=round(one, 2), load_5m=round(five, 2),
                     load_15m=round(fifteen, 2))
    except (OSError, AttributeError):                 # pragma: no cover - not on posix
        pass
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"],
                             capture_output=True, text=True, timeout=20)
        facts["swap"] = out.stdout.strip() or None
    except Exception:                                 # pragma: no cover - host specific
        facts["swap"] = None
    return facts


# ---------------------------------------------------------------------------
# retries, each in a fresh process
# ---------------------------------------------------------------------------

def run_attempts(argv, attempts=DEFAULT_ATTEMPTS, timeout=7200, log=None) -> list:
    """Run argv in a fresh process until it succeeds; one record per attempt.

    A record is written even for the failures: their return code, their stderr
    tail (which is where -14 lands), the seconds it took, and the machine's load
    and swap at that moment.  The caller gets the whole list, so an artifact can
    show that it took three attempts and what the first two said.
    """
    records = []
    for index in range(1, max(int(attempts), 1) + 1):
        facts = machine_facts()
        started = time.time()
        try:
            process = subprocess.run(argv, capture_output=True, text=True,
                                     timeout=timeout)
            code, stderr = process.returncode, process.stderr
        except subprocess.TimeoutExpired:
            code, stderr = -1, f"timeout after {timeout}s"
        record = {"attempt": index, "ok": code == 0, "returncode": code,
                  "seconds": round(time.time() - started, 1), **facts}
        if code != 0:
            tail = [line for line in (stderr or "").strip().splitlines() if line.strip()]
            record["stderr_tail"] = tail[-6:]
            record["error_14"] = any("-14" in line or "error code: -14" in line
                                     for line in tail)
        records.append(record)
        if log is not None:
            state = "ok" if record["ok"] else f"failed ({code})"
            log(f"attempt {index}/{attempts}: {state} in {record['seconds']}s, "
                f"load {record.get('load_1m')}")
        if code == 0:
            break
    return records


# ---------------------------------------------------------------------------
# the tool's own word on placement
# ---------------------------------------------------------------------------

def compute_plan_raw(model_path) -> dict:
    """MLComputePlan's verdict, verbatim, plus whether it carries any device info.

    Never summarised into a percentage: the raw lines are the evidence, and
    "usable" is false when the plan attributed nothing but "other" (or failed
    outright, which is what happens for a stateful model on this host).
    """
    from coremltools.models.compute_plan import MLComputePlan

    path = Path(model_path)
    lines, devices = [], set()
    try:
        plan = MLComputePlan.load_from_path(str(path))
        program = plan.model_structure.program
        if program is None:
            return {"available": False, "usable": False, "devices": [],
                    "raw": [f"{path.name}: not an ML program"]}
        for function_name, function in program.functions.items():
            for operation in function.block.operations:
                usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
                device = str(getattr(usage, "preferred_compute_device", usage))
                device = device.rsplit(".", 1)[-1].lower()
                devices.add(device)
                outputs = getattr(operation, "outputs", []) or []
                label = outputs[0].name if outputs else "?"
                lines.append(f"{function_name}\t{getattr(operation, 'operator_name', '?')}"
                             f"\t{label}\t{device}")
    except Exception as error:                        # noqa: BLE001 - the point here
        return {"available": False, "usable": False, "devices": [],
                "raw": [f"{path.name}: {type(error).__name__}: {error}"]}
    usable = bool(devices - {"other", "none", "unknown"})
    return {"available": True, "usable": usable, "devices": sorted(devices),
            "raw": lines}


# ---------------------------------------------------------------------------
# the latency pair
# ---------------------------------------------------------------------------

def _time_predict(mlmodel, inputs, repeats: int, warmup: int) -> dict:
    """Median predict time over repeats, after warmup, on one state object.

    The same state is reused across the timed predictions: the rows written are
    the same on every one, so the timing is representative while the allocation of
    a 1 GB state per predict is not measured.
    """
    state = mlmodel.make_state()
    for _ in range(max(int(warmup), 0)):
        mlmodel.predict(inputs, state=state)
    samples = []
    for _ in range(max(int(repeats), 1)):
        started = time.perf_counter()
        mlmodel.predict(inputs, state=state)
        samples.append((time.perf_counter() - started) * 1e3)
    return {"median_ms": round(statistics.median(samples), 3),
            "min_ms": round(min(samples), 3),
            "max_ms": round(max(samples), 3),
            "repeats": len(samples)}


def latency_verdict(cpu_only_ms: float, cpu_and_ne_ms: float,
                    threshold: float = 1.5) -> dict:
    """What the latency pair says, stated as evidence and not as a placement.

    A graph whose work runs on the ANE is meaningfully faster with the ANE
    enabled; a graph that runs on the CPU alone gets the same number twice.  The
    threshold is a judgement call, so it is in the report next to the numbers.
    """
    ratio = float(cpu_only_ms) / max(float(cpu_and_ne_ms), 1e-9)
    accelerated = ratio >= float(threshold)
    return {
        "ratio": round(ratio, 2),
        "threshold": float(threshold),
        "verdict": "ane" if accelerated else "cpu",
        "method": "latency evidence only (CPU_ONLY vs CPU_AND_NE); op-level "
                  "placement is not obtainable on this host",
        "reading": ("the ANE version is materially faster, so the graph does run "
                    "at least partly on the Neural Engine" if accelerated else
                    "enabling the ANE changed nothing, so this graph is not running "
                    "on the Neural Engine"),
    }


# ---------------------------------------------------------------------------
# one heavy attempt, in its own process
# ---------------------------------------------------------------------------

def probe_once(model_path, context_length=24576, batch_size=64, tile_rows=None,
               repeats=8, warmup=2, verbose=True) -> dict:
    """Convert the seven-block chunk and measure the latency pair.  Heavy."""
    import coremltools as ct
    import numpy as np
    import torch

    from anemll.models import dory_chunks as dch
    from anemll.models.dory_ane_model import DoryANEConfig

    config = DoryANEConfig(context_length=int(context_length),
                           batch_size=int(batch_size),
                           full_cache_tiles=tile_rows)
    from anemll.ane_converter.dory_converter import _ComponentModel

    model = _ComponentModel(config, list(BLOCKS), load_lm_head=False)
    model.load_pretrained_weights(str(model_path), blocks=list(BLOCKS))
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.model.reset_caches()

    wrapper = dch.traced_block_chunk_class(bool(config.tiled), False)(
        model.model, 0, len(BLOCKS))
    wrapper.eval()

    seq = 1                                     # decode: the per-step question
    dtype = config.dtype
    args = [torch.zeros(1, seq, config.hidden_size, dtype=dtype),
            torch.zeros(1, 1, seq, config.mask_len, dtype=dtype),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, dtype=torch.int32)]
    declared = dch.block_input_names(bool(config.tiled), False, False)
    inputs = []
    for name in declared:
        if name == "causal_mask":
            inputs.append(ct.TensorType(name=name, shape=(1, 1, seq, config.mask_len),
                                        dtype=np.float16))
        elif name in ("current_pos", "cache_pos"):
            inputs.append(ct.TensorType(name=name, shape=(1,), dtype=np.int32))
        elif name.startswith("write_index_"):
            args.append(torch.zeros(1, dtype=torch.int32))
            inputs.append(ct.TensorType(name=name, shape=(1,), dtype=np.int32))
        elif name.startswith("write_gate_"):
            args.append(torch.zeros(1, dtype=dtype))
            inputs.append(ct.TensorType(name=name, shape=(1,), dtype=np.float16))
    states = []
    for attr in dch.state_attrs(config):
        buffer = getattr(model.model, attr)
        states.append(ct.StateType(name=f"model.{attr}",
                                   wrapped_type=ct.TensorType(
                                       shape=tuple(int(d) for d in buffer.shape),
                                       dtype=np.float16)))
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, tuple(args), strict=False)

    predict_inputs = {}
    for name in declared:
        if name == "hidden_states":
            predict_inputs[name] = np.zeros((1, seq, config.hidden_size), dtype=np.float16)
        elif name == "causal_mask":
            predict_inputs[name] = np.zeros((1, 1, seq, config.mask_len), dtype=np.float16)
        elif name in ("current_pos", "cache_pos") or name.startswith("write_index_"):
            predict_inputs[name] = np.array([0], dtype=np.int32)
        else:
            predict_inputs[name] = np.zeros((1,), dtype=np.float16)

    out_dir = Path(tempfile.mkdtemp(prefix="dory24k-probe-"))
    results = {"context_length": int(context_length), "batch_size": int(batch_size),
               "tile_rows": list(tile_rows) if tile_rows else [int(context_length)],
               "blocks": list(BLOCKS), "path": str(out_dir),
               "state_mib": round(sum(
                   int(np.prod(getattr(model.model, attr).shape, dtype=np.int64))
                   for attr in dch.state_attrs(config)) * 2 / 2 ** 20, 1)}

    timings = {}
    for label, unit in (("cpu_only", ct.ComputeUnit.CPU_ONLY),
                        ("cpu_and_ne", ct.ComputeUnit.CPU_AND_NE)):
        started = time.time()
        mlmodel = ct.convert(
            traced,
            inputs=inputs,
            outputs=[ct.TensorType(name="output_hidden_states", dtype=np.float16)],
            states=states,
            compute_precision=ct.precision.FLOAT16,
            compute_units=unit,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )
        converted = time.time() - started
        package = out_dir / f"probe_{label}.mlpackage"
        mlmodel.save(str(package))
        plan = compute_plan_raw(package)
        timing = _time_predict(mlmodel, predict_inputs, repeats, warmup)
        timings[label] = {**timing, "convert_seconds": round(converted, 1),
                          "compute_unit": str(unit).rsplit(".", 1)[-1]}
        results.setdefault("compute_plan", {})[label] = {
            "available": plan["available"], "usable": plan["usable"],
            "devices": plan["devices"], "raw": plan["raw"][:4000],
            "lines": len(plan["raw"])}
        if verbose:
            print(f"[{label}] convert {timings[label]['convert_seconds']}s, "
                  f"predict {timing['median_ms']} ms median "
                  f"({timing['min_ms']}-{timing['max_ms']}), "
                  f"plan devices {plan['devices'] or 'none'} "
                  f"usable={plan['usable']}", file=sys.stderr)

    results["latency"] = timings
    results["verdict"] = latency_verdict(timings["cpu_only"]["median_ms"],
                                        timings["cpu_and_ne"]["median_ms"])
    results["machine"] = machine_facts()
    return results


# ---------------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------------

def build_report(attempts, probe, machine=None) -> dict:
    """The artifact: what was measured, how many processes it took, and the caveats."""
    used = len(attempts)
    ok = bool(probe)
    report = {
        "question": "does a 24,576-row full-attention cache stay on the Neural Engine "
                    "inside a block chunk (spec D7)?",
        "method": ["latency evidence: the same chunk converted with CPU_ONLY and with "
                   "CPU_AND_NE, timed in a fresh process",
                   "the compute plan's raw output, recorded verbatim and never "
                   "summarised into a percentage"],
        "attempts_used": used,
        "attempts": attempts,
        "machine": machine or machine_facts(),
        "probe": probe if ok else None,
        "caveats": [
            "MLComputePlan attributes every operation to 'other' on this host (macOS "
            "27.2, coremltools 8.3) and cannot build a plan for a stateful model, so an "
            "op-level placement count is not obtainable and is not reported",
            "the stateful path is intermittent here: the same package loads and predicts "
            "in one fresh process and returns error -14 in others, which is why every "
            "attempt is a fresh process and the attempt list is part of the result",
        ],
    }
    if ok:
        report["verdict"] = probe["verdict"]
        report["latency"] = probe["latency"]
        report["compute_plan"] = probe.get("compute_plan", {})
        report["state_mib"] = probe.get("state_mib")
        report["tile_rows"] = probe.get("tile_rows")
    else:
        report["verdict"] = {"verdict": "inconclusive",
                             "method": "no attempt completed: see the attempt list"}
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="24K placement probe (latency evidence)")
    parser.add_argument("--model", required=True, help="BF16 safetensors directory")
    parser.add_argument("--context-length", type=int, default=24576)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--full-tile-rows", default=None,
                        help="e.g. 16384,8192 to probe the tile fallback")
    parser.add_argument("--attempts", type=int, default=DEFAULT_ATTEMPTS)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--out", default=None, help="write the report JSON here")
    parser.add_argument("--attempt", action="store_true",
                        help="internal: run exactly one attempt in this process")
    parser.add_argument("--attempt-out", default=None,
                        help="internal: where one attempt writes its JSON")
    args = parser.parse_args(argv)

    tiles = None
    if args.full_tile_rows:
        tiles = tuple(int(part) for part in
                      args.full_tile_rows.replace(" ", "").split(",") if part)

    if args.attempt:
        probe = probe_once(args.model, args.context_length, args.batch_size, tiles,
                           repeats=args.repeats, warmup=args.warmup)
        probe["machine"] = machine_facts()
        Path(args.attempt_out).write_text(json.dumps(probe, default=str) + chr(10))
        return 0

    with tempfile.TemporaryDirectory(prefix="dory24k-attempt-") as scratch:
        attempt_out = Path(scratch) / "attempt.json"
        command = [sys.executable, str(Path(__file__).resolve()), "--attempt",
                   "--model", str(args.model),
                   "--context-length", str(args.context_length),
                   "--batch-size", str(args.batch_size),
                   "--repeats", str(args.repeats), "--warmup", str(args.warmup),
                   "--attempt-out", str(attempt_out)]
        if args.full_tile_rows:
            command += ["--full-tile-rows", str(args.full_tile_rows)]
        attempts = run_attempts(command, attempts=args.attempts,
                               log=lambda line: print(line, file=sys.stderr))
        probe = None
        if attempts and attempts[-1]["ok"] and attempt_out.is_file():
            probe = json.loads(attempt_out.read_text())

    report = build_report(attempts, probe)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str) + chr(10))
    print(json.dumps({"verdict": report["verdict"],
                      "attempts_used": report["attempts_used"],
                      "latency": report.get("latency"),
                      "state_mib": report.get("state_mib"),
                      "tile_rows": report.get("tile_rows")}, indent=2, default=str))
    return 0 if report["verdict"]["verdict"] in ("ane", "cpu") else 1


if __name__ == "__main__":
    raise SystemExit(main())
