"""Run a converted Dory 24K program set -- one Core ML file per stage -- on the ANE.

The plan this drives
--------------------
At 24,576 tokens a Core ML program may hold one (layer, head) K/V cache (24.0 MiB)
and nothing much wider, so 'anemll.ane_converter.dory_per_head' emits 102 programs
instead of five components and 'meta.yaml' carries the wiring:

* 'components' maps a program to its files and its declared surface;
* 'per_head_programs' adds what a dispatcher needs and a Core ML file cannot say:
  the blocks a program advances, its KV head, whether it sits in the loop zone,
  and which gate it emits or consumes.

    embeddings            -> hidden_states, gate_0
    entry_gate            -> h_entry, gate_16             once per loop pass
    per period k = 0..8
      sliding_attn_L<b>   -> contribution_h0..h3          one (2,4,1280,256) carrier
      sepilogue_L<b>      -> h_next, gate_<b+2>
      attn_full_<k>_h<h>  -> head_contribution            one (2,1,24576,256) cache
      epilogue_<k>        -> output_hidden_states, gate_<...>
    lm_head               -> logits

56 dispatches per loop pass, 136 per decoded token at loops=2.  The gate chain is
not re-derived here: 'gate_out' names the attention block whose gate a program
emits, 'gate_in' plus 'blocks[0]' names the one it consumes, and a consumer that
runs before its producer raises with both program names.

The three things that are easy to get wrong, and where they live
----------------------------------------------------------------
* **State, once.** Every stated program gets ONE MLState, created from its infer
  package and kept for the whole run (63 states, 999 MiB at 24K).  Nothing here
  creates a state per token, and no package is loaded twice.
* **Positions and the mask.** 'current_pos' is the position of the first row,
  'cache_pos' the ring offset the sliding carriers are written at, and the mask
  is one (1, 1, seq, mask_len) tensor whose first 'context_length' columns are
  the full-attention causal mask and whose last 'sliding_rows' columns are the
  sliding ring in ring-slot order.  Both widths come from the meta.  A decode
  step writes one column of it and re-uses the buffer.
* **The loop.** The recurrent zone is re-entered 'loops' times; the loop-entry
  program runs first on every pass and takes 'h_in' -- the hidden state that
  *entered* the zone -- so its gate is recomputed per pass instead of arriving
  from the program before the zone, which does not exist.

The seam
--------
Model loading sits behind 'ProgramFactory.load(path) -> ProgramHandle', whose
contract is 'make_state()' and 'predict(feed, state=...)'.  The tests drive the
whole chain through a stub factory, so the dispatch order, the gate threading,
the loop and the position bookkeeping are tested without Core ML.

Command line
------------
    python -m anemll.runtime.dory_ane_runtime plan --model-dir DIR
    python -m anemll.runtime.dory_ane_runtime run  --model-dir DIR --prompt "..." --max-tokens 32
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

__all__ = [
    "ContextOverflowError",
    "GateError",
    "PlanError",
    "ProgramError",
    "ProgramHandle",
    "ProgramFactory",
    "CoreMLProgramFactory",
    "Program",
    "Plan",
    "DoryProgramRuntime",
    "build_causal_mask",
    "resolve_model_path",
    "peak_rss_mib",
    "build_parser",
    "main",
]

#: the value meta.yaml uses for "called once per loop pass"
LOOPS = "loops"

#: the leaf a program takes a layer's gate on
GATE_INPUT = "gate"

#: the RoPE row tensors a 24K program may declare
ROPE_INPUTS = ("cos_sliding", "sin_sliding", "cos_full", "sin_full")

#: which output of a program is the residual stream it advances.  A program of a
#: kind that is absent here (a head program, a sliding attention program, the LM
#: head) produces no hidden state: it hands contributions or logits to the next.
HIDDEN_OUTPUTS = {
    "embeddings": "hidden_states",
    "entry_gate": "h_entry",
    "sliding_layer": "h_next",
    "sliding_epilogue": "h_next",
    "full_epilogue": "output_hidden_states",
}

#: kinds that emit per-KV-head attention contributions instead of a hidden state
CONTRIBUTION_KINDS = ("attn_head", "sliding_attn")

#: what the split LM head names its parts
LOGITS_PREFIX = "logits_"

NEG_INF = float("-inf")


# ---------------------------------------------------------------------------
# Errors: every one of them names the program that caused it
# ---------------------------------------------------------------------------

class DoryRuntimeError(RuntimeError):
    """Base class for a plan, a dispatch or a package that does not add up."""


class PlanError(DoryRuntimeError):
    """The meta does not describe a plan this runtime can drive."""


class GateError(DoryRuntimeError):
    """The gate chain is broken: consumed before produced, stale, or twice."""


class ProgramError(DoryRuntimeError):
    """One program's declared surface and the feed disagree."""

    def __init__(self, program: str, message: str):
        super().__init__(f"program {program}: {message}")
        self.program = program


class ContextOverflowError(ValueError):
    """A call would run past the context the caches hold."""


# ---------------------------------------------------------------------------
# The seam: what the runtime needs from a loaded package
# ---------------------------------------------------------------------------

@runtime_checkable
class ProgramHandle(Protocol):
    """One loaded Core ML package: make_state() and predict().

    'ct.models.MLModel' satisfies this as it is.  A test stub satisfies it with
    two methods, which is the whole point of the seam: the dispatch order, the
    gate threading and the positions can be checked without Core ML.
    """

    def make_state(self) -> object:  # pragma: no cover - protocol
        """A fresh MLState for this model, or raise if it declares none."""

    def predict(self, inputs: dict, state: object = None) -> dict:  # pragma: no cover
        """Run one dispatch."""


@runtime_checkable
class ProgramFactory(Protocol):
    """Loads one package (.mlpackage or compiled .mlmodelc)."""

    def load(self, path: Path) -> ProgramHandle:  # pragma: no cover - protocol
        """Return a handle for the package at 'path'."""


def _coreml():
    # imported where it is used: coremltools takes a second to load, and this
    # module has to stay importable (and testable) without it.
    import coremltools as ct

    return ct


def resolve_model_path(model_dir, file_name) -> Path:
    """The file the meta names, or a sibling with the other package suffix."""
    path = Path(model_dir) / str(file_name)
    if path.exists():
        return path
    for suffix in (".mlpackage", ".mlmodelc"):
        alternative = path.with_suffix(suffix)
        if alternative.exists():
            return alternative
    raise FileNotFoundError(f"neither {path.name} nor its .mlpackage/.mlmodelc sibling "
                            f"exists in {model_dir}")


def _check_coreml_cache() -> None:
    """Fail with the real cause when Core ML cannot write its model cache.

    Loading a .mlpackage makes Core ML compile it and cache the result under
    ~/Library/Caches/com.apple.CoreML.  When that directory is not writable, the
    failure surfaces as "Failed to build the model execution plan ... error code
    -14", which looks like a broken package but is a permission problem.  The
    same guard exists in anemll/dory_chat.py; it is repeated here so that the
    program runtime does not import that module (and with it torch) just to load
    a model.
    """
    cache = Path.home() / "Library" / "Caches" / "com.apple.CoreML"
    try:
        cache.mkdir(parents=True, exist_ok=True)
        probe = cache / ".dory_program_cache_probe"
        probe.write_text("ok")
        probe.unlink()
    except Exception as exc:  # noqa: BLE001 - any failure means "not writable"
        raise RuntimeError(
            f"Core ML cache not writable at {cache}: {exc}. Core ML must write its "
            "compiled-model cache there to load a model; when the write is denied it "
            "reports error -14, which looks like a broken package but is a permission "
            "problem.") from exc


class CoreMLProgramFactory:
    """ct.models.MLModel(path, compute_units=CPU_AND_NE), and nothing else."""

    def __init__(self, compute_units=None):
        self.compute_units = compute_units

    def load(self, path):
        ct = _coreml()
        _check_coreml_cache()
        units = (self.compute_units if self.compute_units is not None
                 else ct.ComputeUnit.CPU_AND_NE)
        path = Path(path)
        if path.suffix == ".mlmodelc":
            return ct.models.CompiledMLModel(str(path), units)
        return ct.models.MLModel(str(path), compute_units=units)


# ---------------------------------------------------------------------------
# The mask: one tensor, two layouts, built the way the model slices it
# ---------------------------------------------------------------------------

def _fill_full_rows(full, start: int, seq: int) -> None:
    """Columns <= query position are 0, the rest -inf, in place.

    'full' is the (seq, context_length) view of the mask's full-attention part.
    """
    full.fill(NEG_INF)
    for row in range(int(seq)):
        full[row, : int(start) + row + 1] = 0.0


def _fill_sliding_rows(slide, start: int, seq: int, sliding_rows: int,
                       sliding_window: int) -> None:
    """The sliding ring mask in ring-slot order, in place.

    Slot j holds the newest position q_j <= last with q_j % rows == j
    (last = start + seq - 1); a query at pq may see it iff
    0 <= pq - q_j <= sliding_window - 1 and q_j >= 0.  The same arithmetic as
    anemll.utils.dory_caches.sliding_ring_mask, which the model's eager path uses.
    """
    rows = int(sliding_rows)
    last = int(start) + int(seq) - 1
    slots = np.arange(rows, dtype=np.int64)
    tok_pos = last - np.mod(last - slots, rows)      # position currently in slot j
    allowed = np.empty(rows, dtype=bool)
    written = tok_pos >= 0
    for row in range(int(seq)):
        # age is negative for a slot that holds a position this batch has not
        # reached yet: those are masked, and so are slots never written at all
        age = (int(start) + row) - tok_pos
        np.greater_equal(age, 0, out=allowed)
        allowed &= age <= int(sliding_window) - 1
        allowed &= written
        view = slide[row]
        view.fill(NEG_INF)
        view[allowed] = 0.0


def build_causal_mask(start: int, seq: int, context_length: int, sliding_rows: int,
                      sliding_window: int, dtype=np.float16) -> np.ndarray:
    """The single mask input: full-attention part then sliding part, concatenated.

    Shape (1, 1, seq, context_length + sliding_rows), float16, 0 / -inf.
    Built from the meta's widths, never from a constant here.
    """
    context_length = int(context_length)
    sliding_rows = int(sliding_rows)
    mask = np.empty((1, 1, int(seq), context_length + sliding_rows), dtype=dtype)
    _fill_full_rows(mask[0, 0, :, :context_length], start, seq)
    _fill_sliding_rows(mask[0, 0, :, context_length:], start, seq, sliding_rows,
                       sliding_window)
    return mask


class _DecodeMask:
    """The one-row mask a decode step uses, advanced in place.

    A decode step differs from the one before it by a single column of the
    full-attention part -- column pos becomes visible -- so the buffer is
    allocated once per run and never rebuilt, unless the position goes backwards
    (a resumed session), in which case it is refilled from scratch.
    """

    __slots__ = ("mask", "primed", "context_length", "sliding_rows", "sliding_window")

    def __init__(self, context_length: int, sliding_rows: int, sliding_window: int):
        self.context_length = int(context_length)
        self.sliding_rows = int(sliding_rows)
        self.sliding_window = int(sliding_window)
        self.mask = np.full((1, 1, 1, self.context_length + self.sliding_rows), NEG_INF,
                            dtype=np.float16)
        self.primed = -1

    def advance(self, pos: int) -> np.ndarray:
        pos = int(pos)
        if pos < 0:
            raise ContextOverflowError(f"position {pos} is negative")
        if pos >= self.context_length:
            raise ContextOverflowError(
                f"position {pos} is outside the {self.context_length}-token context")
        if pos <= self.primed:                      # backwards: start again
            self.mask.fill(NEG_INF)
            self.primed = -1
        while self.primed < pos:
            self.primed += 1
            self.mask[0, 0, 0, self.primed] = 0.0
        _fill_sliding_rows(self.mask[0, 0, :, self.context_length:], pos, 1,
                           self.sliding_rows, self.sliding_window)
        return self.mask


def peak_rss_mib() -> float:
    """Peak resident set size of this process, in MiB."""
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes, Linux KiB
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


# ---------------------------------------------------------------------------
# The plan, read out of meta.yaml
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Program:
    """One Core ML file pair and everything the dispatcher needs about it."""

    name: str
    kind: str
    blocks: Tuple[int, ...]
    kv_head: Optional[int]
    calls_per_token: object
    loop_entry: bool
    loop_exit: bool
    gate_in: bool
    gate_out: Optional[int]
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]
    states: Tuple[dict, ...]
    files: dict
    functions: Tuple[str, ...]

    # -- the surface -------------------------------------------------------
    @property
    def is_recurrent(self) -> bool:
        return self.calls_per_token == LOOPS

    @property
    def hidden_output(self) -> Optional[str]:
        """The output carrying the residual stream, or None if it carries none."""
        return HIDDEN_OUTPUTS.get(self.kind)

    @property
    def advances_hidden(self) -> bool:
        return self.hidden_output is not None

    @property
    def gate_name(self) -> Optional[str]:
        """gate_<block> this program emits, i.e. the consumer's handle."""
        return None if self.gate_out is None else f"gate_{int(self.gate_out)}"

    @property
    def gate_input(self) -> Optional[str]:
        """gate_<block> this program consumes, from blocks[0]."""
        if not self.gate_in or not self.blocks:
            return None
        return f"gate_{int(self.blocks[0])}"

    @property
    def state_mib(self) -> float:
        return sum(float(item.get("bytes", 0)) for item in self.states) / 1024 ** 2

    @property
    def state_names(self) -> Tuple[str, ...]:
        return tuple(str(item.get("name")) for item in self.states)

    def __str__(self) -> str:  # pragma: no cover - reporting only
        gate = ""
        if self.gate_in:
            gate += f" <- {self.gate_input}"
        if self.gate_out is not None:
            gate += f" -> {self.gate_name}"
        zone = "loop" if self.is_recurrent else "once"
        return (f"{self.name:<18} {self.kind:<17} blocks={list(self.blocks)!s:<9} "
                f"{zone:<5}{gate}")


@dataclass
class Plan:
    """The whole program set: what runs once, what runs per loop pass, and why."""

    model_dir: Path
    meta: dict
    programs: Dict[str, Program]
    order: Tuple[str, ...]
    pre_loop: Tuple[str, ...]
    loop: Tuple[str, ...]
    post_loop: Tuple[str, ...]
    context_length: int
    sliding_rows: int
    sliding_window: int
    mask_len: int
    head_dim: int
    hidden_size: int
    batch_size: int
    loops_min: int
    loops_max: int
    loops_default: int
    zones: dict
    rope_tables: Optional[dict]
    producers_of_block: Dict[int, str] = field(default_factory=dict)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_meta(cls, meta: dict, model_dir) -> "Plan":
        params = meta.get("parameters") or {}
        components = meta.get("components") or {}
        per_program = meta.get("per_head_programs") or {}
        declared = per_program.get("programs") or {}
        if not declared:
            raise PlanError(
                "meta.yaml has no per_head_programs.programs section: this runtime "
                "drives the per-(layer, head) plan (one file per program, each with "
                "its own state), not the five-component layout - use "
                "anemll.dory_chat.DoryRuntime for that one")
        order = tuple(meta.get("run_order") or ())
        if not order:
            raise PlanError("meta.yaml has no run_order")
        if set(order) != set(components):
            missing = sorted(set(order) - set(components))
            extra = sorted(set(components) - set(order))
            raise PlanError(f"run_order and components disagree: {missing} are in the "
                            f"order but not in components, {extra} the other way round")

        programs: Dict[str, Program] = {}
        for name in order:
            entry = components[name] or {}
            wire = declared.get(name)
            if wire is None:
                raise PlanError(f"component {name} is in run_order but has no entry in "
                                f"per_head_programs.programs")
            files = entry.get("files") or {}
            functions = tuple(entry.get("functions") or ())
            if "infer" not in functions:
                raise PlanError(f"{name} declares no infer function "
                                f"(functions={list(functions)})")
            for function in functions:
                if not files.get(function):
                    raise PlanError(f"{name} has no file for its {function} function")
            gate_out = wire.get("gate_out", entry.get("gate_out"))
            program = Program(
                name=name,
                kind=str(wire.get("kind") or entry.get("kind") or ""),
                blocks=tuple(int(b) for b in (wire.get("blocks") or ())),
                kv_head=None if wire.get("kv_head") is None else int(wire["kv_head"]),
                calls_per_token=wire.get("calls_per_token",
                                         entry.get("calls_per_token", 1)),
                loop_entry=bool(wire.get("loop_entry", False)),
                loop_exit=bool(wire.get("loop_exit", False)),
                gate_in=bool(wire.get("gate_in", entry.get("gate_in", False))),
                gate_out=None if gate_out is None else int(gate_out),
                inputs=tuple(str(item) for item in
                             (wire.get("inputs") or entry.get("inputs") or ())),
                outputs=tuple(str(item) for item in
                              (wire.get("outputs") or entry.get("outputs") or ())),
                states=tuple(wire.get("states") or ()),
                files={str(k): str(v) for k, v in files.items()},
                functions=functions,
            )
            _cross_check(program, entry)
            programs[name] = program

        context = int(params.get("context_length") or 0)
        sliding_rows = int(params.get("sliding_rows") or 0)
        mask_len = int(params.get("mask_len") or 0)
        if context <= 0 or sliding_rows <= 0:
            raise PlanError("parameters.context_length and parameters.sliding_rows must "
                            "be positive: they are the two mask widths")
        if mask_len != context + sliding_rows:
            raise PlanError(f"parameters.mask_len ({mask_len}) is not context_length + "
                            f"sliding_rows ({context} + {sliding_rows})")
        zones = params.get("zones") or {}
        for key in ("input", "recurrent", "output"):
            if not isinstance(zones.get(key), (list, tuple)) or len(zones[key]) != 2:
                raise PlanError(f"parameters.zones.{key} must be a [start, end) pair")

        if params.get("argmax_in_model"):
            raise PlanError("this conversion put the argmax inside the LM head "
                            "(parameters.argmax_in_model); this runtime reads logits, so "
                            "convert with --no-argmax to drive it")
        loops = meta.get("loops") or {}
        plan = cls(
            model_dir=Path(model_dir),
            meta=meta,
            programs=programs,
            order=order,
            pre_loop=(),
            loop=(),
            post_loop=(),
            context_length=context,
            sliding_rows=sliding_rows,
            sliding_window=int(params.get("sliding_window") or sliding_rows),
            mask_len=mask_len,
            head_dim=int(params.get("head_dim") or 0),
            hidden_size=int(params.get("hidden_size") or 0),
            batch_size=int(params.get("batch_size") or 1),
            loops_min=int(loops.get("min", 1)),
            loops_max=int(loops.get("max", 1)),
            loops_default=int(loops.get("default", 1)),
            zones=zones,
            rope_tables=(params.get("rope_tables")
                         if params.get("rope_tables_as_inputs") else None),
        )
        plan._split_zones()
        plan._check_loop_zone()
        plan.producers_of_block = plan._map_contributions()
        return plan

    # -- zones -------------------------------------------------------------
    def _split_zones(self) -> None:
        """Place every program in the pre-loop, loop or post-loop group.

        The rule is the meta's own: a program is in the loop zone when
        calls_per_token is "loops"; a program that runs once belongs to the side
        of the recurrent zone its blocks sit on.  A program that straddles the
        boundary, or a once-program with no blocks that is neither the embeddings
        nor the LM head, is refused rather than placed by guesswork.
        """
        rec_start, rec_end = (int(v) for v in self.zones["recurrent"])
        buckets: Dict[str, List[str]] = {"pre": [], "loop": [], "post": []}
        rank = {"pre": 0, "loop": 1, "post": 2}
        last = -1
        for name in self.order:
            program = self.programs[name]
            if program.is_recurrent:
                if program.loop_entry:
                    # The entry program is listed before the input zone -- it has
                    # to be, since nothing in the zone exists yet -- and the
                    # runtime hoists it to the head of every pass, so its place in
                    # run_order says nothing about the zone order.
                    buckets["loop"].append(name)
                    continue
                where = "loop"
            elif program.kind == "embeddings":
                where = "pre"
            elif program.kind == "lm_head":
                where = "post"
            elif not program.blocks:
                raise PlanError(
                    f"{name} runs once, advances no block and is neither the embeddings "
                    f"nor the LM head: the runtime cannot place it around the loop")
            elif program.blocks[-1] < rec_start:
                where = "pre"
            elif program.blocks[0] >= rec_end:
                where = "post"
            else:
                raise PlanError(
                    f"{name} advances blocks {list(program.blocks)}, which straddle the "
                    f"recurrent zone {rec_start}..{rec_end - 1}; a program has to sit "
                    f"wholly on one side of the loop")
            if rank[where] < last:
                raise PlanError(
                    f"run_order puts {name} ({where}-loop) after a later group: the loop "
                    f"zone is not one contiguous run of programs")
            last = rank[where]
            buckets[where].append(name)
        self.pre_loop = tuple(buckets["pre"])
        self.loop = tuple(buckets["loop"])
        self.post_loop = tuple(buckets["post"])
        if not self.pre_loop or not self.post_loop:
            raise PlanError("the plan needs programs before and after the loop zone")
        if self.programs[self.pre_loop[0]].kind != "embeddings":
            raise PlanError(f"{self.pre_loop[0]} runs first, and it is not the embeddings "
                            f"program")
        if self.programs[self.post_loop[-1]].kind != "lm_head":
            raise PlanError(f"{self.post_loop[-1]} runs last, and it is not the LM head")
        if not self.loop:
            raise PlanError("the plan has no recurrent programs: nothing to loop")

    def _check_loop_zone(self) -> None:
        """The loop-entry program opens every pass, and every gate has one owner."""
        entries = [name for name in self.loop if self.programs[name].loop_entry]
        if len(entries) != 1:
            raise PlanError(f"the loop zone needs exactly one loop-entry program, found "
                            f"{entries or 'none'}")
        entry = self.programs[entries[0]]
        if self.loop[0] != entry.name:
            raise PlanError(
                f"the loop-entry program {entry.name} must run first on every pass, but "
                f"run_order puts {self.loop[0]} first")
        if not entry.advances_hidden:
            raise PlanError(f"the loop-entry program {entry.name} produces no hidden state "
                            f"(kind {entry.kind!r})")
        if entry.gate_out is None:
            raise PlanError(f"the loop-entry program {entry.name} emits no gate, so the "
                            f"first attention layer of the zone would have none per pass")
        if not [name for name in self.loop
                if self.programs[name].gate_input == entry.gate_name]:
            raise PlanError(f"no program in the loop consumes {entry.gate_name}, which "
                            f"{entry.name} emits on every pass")
        producers = self._gate_producers()
        chain = (self.meta.get("per_head_programs") or {}).get("gate_chain") or {}
        for gate, producer in chain.items():
            if producers.get(gate) != producer:
                raise PlanError(f"per_head_programs.gate_chain says {gate} comes from "
                                f"{producer}; the programs say {producers.get(gate)}")
        seen = set()
        for name in self.order:
            program = self.programs[name]
            if not program.gate_in:
                continue
            if program.gate_input is None:
                raise PlanError(f"{name} takes a gate but advances no block, so its gate "
                                f"cannot be named")
            if program.gate_input not in producers:
                raise PlanError(f"{name} consumes {program.gate_input}, which no program "
                                f"emits")
            if program.gate_input in seen:
                raise PlanError(f"{program.gate_input} is consumed by two programs, the "
                                f"second being {name}")
            seen.add(program.gate_input)

    def _gate_producers(self) -> Dict[str, str]:
        producers: Dict[str, str] = {}
        for name in self.order:
            program = self.programs[name]
            if program.gate_out is None:
                continue
            if program.gate_name in producers:
                raise PlanError(f"{program.gate_name} is emitted by both "
                                f"{producers[program.gate_name]} and {name}")
            producers[program.gate_name] = name
        return producers

    def _map_contributions(self) -> Dict[int, str]:
        """Which program emits the per-KV-head contributions of each block."""
        out: Dict[int, str] = {}
        for name in self.order:
            program = self.programs[name]
            if program.kind in CONTRIBUTION_KINDS and program.blocks:
                out[int(program.blocks[0])] = name
        return out

    # -- reporting ---------------------------------------------------------
    def dispatches_per_token(self, loops: int) -> int:
        return len(self.pre_loop) + int(loops) * len(self.loop) + len(self.post_loop)

    def state_mib(self) -> float:
        return sum(p.state_mib for p in self.programs.values())

    def stated_programs(self) -> List[str]:
        return [name for name in self.order if self.programs[name].states]

    def summary(self) -> str:
        stated = self.stated_programs()
        return (f"{len(self.order)} programs, {len(stated)} with state "
                f"({self.state_mib():.1f} MiB), {len(self.loop)} dispatches per loop pass, "
                f"{self.dispatches_per_token(self.loops_default)} per token at "
                f"loops={self.loops_default}; context {self.context_length}, mask "
                f"{self.mask_len} = {self.context_length} full + {self.sliding_rows} sliding")


def _cross_check(program: Program, entry: dict) -> None:
    """The two meta sections have to describe the same program."""
    for field_name in ("calls_per_token", "gate_in", "gate_out"):
        if field_name in entry and entry[field_name] != getattr(program, field_name):
            raise PlanError(
                f"{program.name}: components says {field_name}={entry[field_name]!r} while "
                f"per_head_programs says {getattr(program, field_name)!r}")
    for field_name in ("inputs", "outputs"):
        declared = tuple(str(item) for item in (entry.get(field_name) or ()))
        mine = tuple(getattr(program, field_name))
        if declared and declared != mine:
            raise PlanError(
                f"{program.name}: components declares {field_name}={list(declared)} while "
                f"per_head_programs says {list(mine)}")


# ---------------------------------------------------------------------------
# Timings
# ---------------------------------------------------------------------------

@dataclass
class _Timing:
    calls: int = 0
    seconds: float = 0.0


# ---------------------------------------------------------------------------
# One forward call: the feed every program of a batch shares
# ---------------------------------------------------------------------------

class _Batch:
    """The inputs, buffers and intermediate tensors of one forward call.

    Everything a program can be handed is here, and a program only gets what it
    declares -- a declared input this class cannot supply raises with the
    program's name rather than being silently dropped.
    """

    def __init__(self, runtime: "DoryProgramRuntime", ids: np.ndarray, start: int,
                 function: str):
        self.rt = runtime
        self.plan = runtime.plan
        self.ids = ids
        self.seq = int(ids.shape[1])
        self.start = int(start)
        self.function = function
        self.mask = runtime._mask_for(self.start, self.seq)
        self.current_pos = runtime._pos_buffer(start)
        self.cache_pos = runtime._pos_buffer(runtime._cache_pos(self.start, self.seq))
        self.rope = runtime._rope_rows(self.start, self.seq)
        self.hidden: Optional[np.ndarray] = None
        self.hidden_owner: Optional[str] = None
        self.h_in: Optional[np.ndarray] = None
        self.gates: Dict[str, Tuple[str, np.ndarray, int]] = {}
        self.consumed = set()
        self.contributions: Dict[Tuple[int, int], np.ndarray] = {}
        self.pass_id = -1
        self.logits: Optional[np.ndarray] = None

    # -- pass scoping ------------------------------------------------------
    def enter_pass(self, pass_id: int) -> None:
        """A new loop pass: gates from the pass before are no longer admissible."""
        self.pass_id = int(pass_id)
        self.consumed = set()

    # -- one dispatch ------------------------------------------------------
    def run(self, name: str) -> None:
        program = self.plan.programs[name]
        feed = self._feed(program)
        out = self.rt._predict(program, feed, self.function)
        self._absorb(program, out)

    def _feed(self, program: Program) -> dict:
        feed: Dict[str, np.ndarray] = {}
        for name in program.inputs:
            if name == "hidden_states":
                if self.hidden is None:
                    raise ProgramError(program.name, "needs hidden_states, but no program "
                                                     "before it produced any")
                feed[name] = self.hidden
            elif name == "input_ids":
                feed[name] = self.ids
            elif name == "causal_mask":
                feed[name] = self.mask
            elif name == "current_pos":
                feed[name] = self.current_pos
            elif name == "cache_pos":
                feed[name] = self.cache_pos
            elif name == "h_in":
                if self.h_in is None:
                    raise ProgramError(program.name, "needs h_in, but the loop entry has "
                                                     "not been reached")
                feed[name] = self.h_in
            elif name == GATE_INPUT:
                feed[name] = self._take_gate(program)
            elif name in ROPE_INPUTS:
                if name not in self.rope:
                    raise ProgramError(program.name, f"declares the RoPE rows {name}, and "
                                                     f"this package carries none")
                feed[name] = self.rope[name]
            elif name.startswith("head_") or name.startswith("contribution_h"):
                feed[name] = self._take_contribution(program, name)
            elif name.startswith("write_index_") or name.startswith("write_gate_"):
                raise ProgramError(
                    program.name, f"declares the tiled full-cache write input {name}; the "
                                  f"per-(layer, head) plan is untiled (one cache per "
                                  f"program), so this runtime does not drive it")
            else:
                raise ProgramError(program.name, f"declares the input {name!r}, which this "
                                                 f"runtime does not know how to supply")
        missing = [name for name in program.inputs if name not in feed]
        if missing:
            raise ProgramError(program.name, f"declared input(s) {missing} were not supplied")
        return feed

    def _take_gate(self, program: Program) -> np.ndarray:
        name = program.gate_input
        if name is None:
            raise ProgramError(program.name, "declares a gate input, but the plan gives it "
                                             "no block to name the gate with")
        entry = self.gates.get(name)
        if entry is None:
            raise GateError(f"program {program.name} consumes {name} before it is produced")
        producer, array, pass_id = entry
        if pass_id != self.pass_id:
            raise GateError(
                f"program {program.name} consumes {name} produced by {producer} in loop "
                f"pass {pass_id + 1}, but it is running in pass {self.pass_id + 1}: the "
                f"gate is stale")
        if name in self.consumed:
            raise GateError(f"program {program.name} consumes {name}, which this pass "
                            f"already consumed")
        self.consumed.add(name)
        return array

    def _join_split_outputs(self, program: Program, out: dict) -> dict:
        """Rejoin a program whose output the converter split along the last axis.

        The LM head cannot be a single 131,072-wide convolution -- that exceeds the
        ANE's 65,536-element axis limit -- so the converter divides the vocabulary
        into N slices and the package returns logits_1 .. logits_N.  They are
        consecutive vocabulary blocks in index order, so concatenating them on the
        last axis rebuilds exactly the declared tensor.  A declared name that is
        absent is rebuilt only when every slice for it is present.
        """
        if all(name in out for name in program.outputs):
            return out
        joined = dict(out)
        for name in program.outputs:
            if name in joined:
                continue
            slices = [k for k in out if k.startswith(name + "_")
                      and k[len(name) + 1:].isdigit()]
            if len(slices) < 2:
                continue
            slices.sort(key=lambda k: int(k[len(name) + 1:]))
            joined[name] = np.concatenate([np.asarray(out[k]) for k in slices], axis=-1)
        return joined

    def _take_contribution(self, program: Program, name: str) -> np.ndarray:
        if name.startswith("head_"):
            head = int(name[len("head_"):])
        else:
            head = int(name[len("contribution_h"):])
        if not program.blocks:
            raise ProgramError(program.name, f"takes {name}, but advances no block to key "
                                             f"the contribution with")
        key = (int(program.blocks[0]), head)
        if key not in self.contributions:
            producer = self.plan.producers_of_block.get(key[0], "nothing")
            raise ProgramError(
                program.name, f"takes {name}, the contribution of KV head {head} for block "
                              f"{key[0]}, and it has not been produced ({producer} covers "
                              f"that block)")
        return self.contributions[key]

    def _absorb(self, program: Program, out: dict) -> None:
        out = self._join_split_outputs(program, out)
        missing = [name for name in program.outputs if name not in out]
        if missing:
            raise ProgramError(program.name, f"did not return its declared output(s) "
                                             f"{missing}; it returned {sorted(out)}")
        if program.kind == "attn_head":
            if program.kv_head is None:
                raise ProgramError(program.name, "is a head program without a kv_head")
            self.contributions[(int(program.blocks[0]), int(program.kv_head))] = \
                out["head_contribution"]
        elif program.kind == "sliding_attn":
            for name in program.outputs:
                if name.startswith("contribution_h"):
                    self.contributions[(int(program.blocks[0]),
                                        int(name[len("contribution_h"):]))] = out[name]
        if program.advances_hidden:
            self.hidden = out[program.hidden_output]
            self.hidden_owner = program.name
        if program.gate_out is not None:
            name = program.gate_name
            if name not in out:
                raise ProgramError(program.name, f"declares gate_out {program.gate_out} but "
                                                 f"returned no {name}")
            previous = self.gates.get(name)
            if (previous is not None and previous[2] == self.pass_id
                    and name not in self.consumed):
                raise GateError(f"{name} is produced twice in one pass, by "
                                f"{previous[0]} and {program.name}")
            self.gates[name] = (program.name, out[name], self.pass_id)
        if program.kind == "lm_head":
            parts = [out[name] for name in program.outputs if name.startswith(LOGITS_PREFIX)]
            if not parts:
                parts = [out[name] for name in program.outputs]
            self.logits = np.concatenate(parts, axis=-1)


# ---------------------------------------------------------------------------
# The runtime
# ---------------------------------------------------------------------------

class DoryProgramRuntime:
    """The 24K program set, loaded once and driven per call.

    Construction reads meta.yaml and builds the plan; nothing is loaded until
    load() (or the first forward call), so 'plan', 'report' and the tests cost
    nothing.  load() instantiates every package once and creates one MLState per
    stated program, which stays alive for the life of the runtime: the caches are
    the model's whole memory of the conversation.
    """

    def __init__(self, model_dir, *, factory: Optional[ProgramFactory] = None,
                 loops: Optional[int] = None, verbose: bool = False,
                 meta: Optional[dict] = None):
        self.model_dir = Path(model_dir)
        self.meta = meta if meta is not None else self._read_meta(self.model_dir / "meta.yaml")
        self.plan = Plan.from_meta(self.meta, self.model_dir)
        self.factory = factory if factory is not None else CoreMLProgramFactory()
        self.verbose = bool(verbose)
        self._loops = self._check_loops(self.plan.loops_default if loops is None else loops)
        self._handles: Dict[str, Dict[str, ProgramHandle]] = {}
        # prefill packages are loaded for their single dispatch and released at
        # once: 102 infer models + 999.0 MiB of state is already the resident set
        # this host can hold, and a second copy of every package pushes Core ML
        # into silently not loading some of them.
        self._prefill_paths: Dict[str, Path] = {}
        self._paths: Dict[str, Dict[str, Path]] = {}
        self._phase_active: str = ""
        self.first_token_seconds: float = 0.0
        self._states: Dict[str, object] = {}
        self.load_seconds = 0.0
        self._kind_time: Dict[str, _Timing] = {}
        self._program_time: Dict[str, _Timing] = {}
        self._phase = {"prefill": _Timing(), "decode": _Timing()}
        self._wall: Dict[str, dict] = {}
        self._phase_tokens = {"prefill": 0, "decode": 0}
        self._phase_dispatches = {"prefill": 0, "decode": 0}
        self._phase_name = "infer"
        self._rope_tables: Optional[dict] = None
        self._decode_mask: Optional[_DecodeMask] = None
        self._prefill_masks: Dict[int, np.ndarray] = {}
        self._ids_one = np.zeros((1, 1), dtype=np.int32)
        self._pos_buffers = (np.zeros((1,), dtype=np.int32), np.zeros((1,), dtype=np.int32))
        self._pos_index = 0

    # -- meta --------------------------------------------------------------
    @staticmethod
    def _read_meta(path: Path) -> dict:
        import yaml

        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist: point --model-dir at the "
                                    f"conversion output that holds meta.yaml")
        with open(path, "r", encoding="utf-8") as handle:
            meta = yaml.safe_load(handle)
        if not isinstance(meta, dict):
            raise PlanError(f"{path} does not hold a mapping")
        return meta

    def _check_loops(self, loops) -> int:
        loops = int(loops)
        lo, hi = self.plan.loops_min, self.plan.loops_max
        if not lo <= loops <= hi:
            raise ValueError(f"loops must be {lo}..{hi}, got {loops}")
        return loops

    @property
    def loops(self) -> int:
        return self._loops

    def set_loops(self, loops: int) -> bool:
        """Select the loop count; a change resets the caches.  True if reset."""
        loops = self._check_loops(loops)
        if loops == self._loops:
            return False
        self._loops = loops
        self.reset()
        return True

    # -- loading -----------------------------------------------------------
    @property
    def loaded(self) -> bool:
        return bool(self._handles)

    def load(self) -> float:
        """Load every package once, create one state per stated program, keep both."""
        if self.loaded:
            return self.load_seconds
        started = time.perf_counter()
        try:
            self._load_all()
        except Exception:
            # a half-loaded runtime is worse than none: drop what was opened so a
            # caller can fix the cause and try again
            self._handles.clear()
            self._states.clear()
            raise
        if self.plan.rope_tables:
            self._rope_tables = self._load_rope_tables(self.plan.rope_tables)
        self.load_seconds = time.perf_counter() - started
        return self.load_seconds

    def _load_all(self) -> None:
        """Create one state per stated program and hold no model open yet.

        Two measured facts decide this shape.  A state OUTLIVES the model that
        made it: a state from an infer package keeps working after that model is
        released, and another package accepts it (verified on attn_full_0_h0).
        And ~102 concurrently loaded packages is the ceiling here: at 204, Core
        ML silently leaves models unloaded and make_state raises 'not loaded with
        the Core ML Framework', while 102 is reliable.

        So models are loaded per PHASE -- prefill models while the cache fills,
        infer models while decoding, never both -- and the states are created up
        front from the infer packages, which are then released.
        """
        for name in self.plan.order:
            program = self.plan.programs[name]
            handles: Dict[str, ProgramHandle] = {}
            for function in program.functions:
                path = resolve_model_path(self.model_dir, program.files[function])
                self._paths.setdefault(name, {})[function] = path
            if not program.states:
                continue
            source_path = (self._paths[name].get("infer")
                           or next(iter(self._paths[name].values())))
            source = self.factory.load(source_path)
            make_state = getattr(source, "make_state", None)
            if make_state is None:
                raise ProgramError(name, "declares state(s) "
                                         f"{list(program.state_names)} but its package "
                                         f"has no make_state()")
            # ONE state per program, created here and never again: the caches have
            # to survive every later dispatch.  It outlives the model that made it,
            # so the model is released immediately and the state is handed to
            # whatever phase is running.
            self._states[name] = make_state()
            del source

    def _enter_phase(self, function: str) -> None:
        """Make one phase's packages resident, releasing the other phase's."""
        if self._phase_active == function:
            return
        self._handles.clear()
        for name in self.plan.order:
            path = self._paths.get(name, {}).get(function)
            if path is None:
                continue
            try:
                self._handles[name] = {function: self.factory.load(path)}
            except Exception as exc:  # noqa: BLE001 - re-raised with the program name
                raise RuntimeError(f"loading {name}/{function} from {path.name} "
                                   f"failed: {exc}") from exc
            if self.verbose:
                print(f"loaded {name}/{function}: {path.name}", file=sys.stderr)
        self._phase_active = function

    def _load_rope_tables(self, declared: dict) -> dict:
        name = declared.get("file")
        if not name:
            raise PlanError("parameters.rope_tables_as_inputs is on but "
                            "parameters.rope_tables.file is missing")
        path = self.model_dir / str(name)
        if not path.is_file():
            raise FileNotFoundError(f"{path.name} is missing: the packages were converted "
                                    f"with the RoPE rows as inputs")
        with np.load(path) as data:
            tables = {key: np.asarray(data[key]) for key in declared.get("keys") or ()}
        shape = (int(declared["rows"]), int(declared["head_dim"]))
        for key, array in tables.items():
            if tuple(array.shape) != shape:
                raise PlanError(f"{path.name}: {key} has shape {array.shape}, "
                                f"expected {shape}")
            if array.dtype != np.float16:
                tables[key] = array.astype(np.float16)
        return tables

    def reset(self) -> None:
        """Throw the caches away: new MLState per stated program, empty mask."""
        self._states = self.new_states()
        self._decode_mask = None
        self._prefill_masks.clear()

    # -- state handover, for a session cache -------------------------------
    def new_states(self) -> Dict[str, object]:
        """A fresh, empty cache for every stated program, without installing it.

        Creating a state needs a loaded model, but it does not need to *stay*
        loaded: a state outlives the model that made it.  That is what lets an
        engine hold several conversations' caches (one dict each) while only ever
        keeping ~102 packages resident.
        """
        made: Dict[str, object] = {}
        for name in self.plan.stated_programs():
            handles = self._handles.get(name) or {}
            source = handles.get("infer") or (next(iter(handles.values())) if handles else None)
            temporary = source is None
            if temporary:
                path = (self._paths.get(name, {}).get("infer")
                        or next(iter(self._paths.get(name, {}).values()), None))
                if path is None:
                    # new_states() can be called before load() has recorded paths
                    program = self.plan.programs[name]
                    function = "infer" if "infer" in program.files else next(iter(program.files))
                    path = resolve_model_path(self.model_dir, program.files[function])
                    self._paths.setdefault(name, {})[function] = path
                source = self.factory.load(path)
            make_state = getattr(source, "make_state", None)
            if make_state is None:
                raise ProgramError(name, "declares state(s) but its package has no make_state()")
            made[name] = make_state()
            if temporary:
                del source
        return made

    def take_states(self) -> Dict[str, object]:
        """Hand the current caches to the caller and leave this runtime without any."""
        states, self._states = self._states, {}
        self._decode_mask = None
        self._prefill_masks.clear()
        return states

    def install_states(self, states: Dict[str, object]) -> None:
        """Adopt a cache created earlier (by this runtime or another instance)."""
        missing = set(self.plan.stated_programs()) - set(states)
        if missing:
            raise ProgramError(sorted(missing)[0],
                               f"the state cache is missing {len(missing)} program(s)")
        self._states = dict(states)
        self._decode_mask = None
        self._prefill_masks.clear()

    # -- buffers -----------------------------------------------------------
    def _pos_buffer(self, value: int) -> np.ndarray:
        """A reusable (1,) int32 position tensor (two, so a feed can hold both)."""
        buffer = self._pos_buffers[self._pos_index % 2]
        self._pos_index += 1
        buffer[0] = int(value)
        return buffer

    def _cache_pos(self, start: int, seq: int) -> int:
        rows = int(self.plan.sliding_rows)
        first = int(start) % rows
        if int(seq) > 1 and first + int(seq) > rows:
            raise ValueError(
                f"a {seq}-token batch at position {start} would straddle the {rows}-row "
                f"sliding ring ({first} + {seq} > {rows}); feed whole batches or start "
                f"at a multiple of the batch size")
        return first

    def _mask_for(self, start: int, seq: int) -> np.ndarray:
        if int(seq) == 1:
            if self._decode_mask is None:
                self._decode_mask = _DecodeMask(self.plan.context_length,
                                                self.plan.sliding_rows,
                                                self.plan.sliding_window)
            return self._decode_mask.advance(start)
        key = int(seq)
        mask = self._prefill_masks.get(key)
        if mask is None:
            mask = np.empty((1, 1, key, self.plan.mask_len), dtype=np.float16)
            self._prefill_masks[key] = mask
        _fill_full_rows(mask[0, 0, :, :self.plan.context_length], start, seq)
        _fill_sliding_rows(mask[0, 0, :, self.plan.context_length:], start, seq,
                           self.plan.sliding_rows, self.plan.sliding_window)
        return mask

    def _rope_rows(self, start: int, seq: int) -> dict:
        if self._rope_tables is None:
            return {}
        end = int(start) + int(seq)
        out = {}
        for key, table in self._rope_tables.items():
            if end > int(table.shape[0]):
                raise ContextOverflowError(
                    f"a {seq}-token call at {start} needs RoPE rows up to {end} and the "
                    f"exported table has {int(table.shape[0])}")
            out[key] = table[int(start):end].reshape(1, 1, int(seq), int(table.shape[1]))
        return out

    # -- one dispatch ------------------------------------------------------
    def _predict(self, program: Program, feed: dict, function: str) -> dict:
        handles = self._handles.get(program.name) or {}
        handle = handles.get(function)
        if handle is None and function == "prefill":
            path = self._prefill_paths.get(program.name)
            if path is not None:
                try:
                    handle = self.factory.load(path)
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(f"loading {program.name}/prefill from {path.name} "
                                       f"failed: {exc}") from exc
                handles[function] = handle          # kept until release_prefill()
                if self.verbose:
                    print(f"loaded {program.name}/prefill: {path.name}", file=sys.stderr)
        if handle is None:
            raise PlanError(f"{program.name} has no {function} package (the meta lists "
                            f"{sorted(handles) or 'none'}); a prefill needs one per program")
        state = self._states.get(program.name)
        started = time.perf_counter()
        try:
            if state is None:
                out = handle.predict(feed)
            else:
                out = handle.predict(feed, state=state)
        except Exception as exc:  # noqa: BLE001 - re-raised with the program name
            raise ProgramError(program.name, f"{function} predict failed: {exc}") from exc
        self._record(program, time.perf_counter() - started)
        return out

    def _record(self, program: Program, seconds: float) -> None:
        kind = self._kind_time.setdefault(program.kind, _Timing())
        kind.calls += 1
        kind.seconds += seconds
        one = self._program_time.setdefault(program.name, _Timing())
        one.calls += 1
        one.seconds += seconds
        phase = "prefill" if self._phase_name == "prefill" else "decode"
        self._phase[phase].calls += 1
        self._phase[phase].seconds += seconds
        self._phase_dispatches[phase] += 1

    # -- the forward call --------------------------------------------------
    def _forward(self, ids, start: int, function: str) -> np.ndarray:
        if not self.loaded:
            self.load()
        self._enter_phase(function)
        self._phase_name = "prefill" if function == "prefill" else "infer"
        plan = self.plan
        seq = int(ids.shape[1])
        if function == "prefill" and seq != plan.batch_size:
            raise ValueError(
                f"the prefill packages are traced at {plan.batch_size} rows and this call "
                f"has {seq}: feed whole batches through prefill and the tail through "
                f"infer (feed() does that)")
        if function == "infer" and seq != 1:
            raise ValueError(f"the infer packages are traced at one row and this call has "
                             f"{seq}")
        batch = _Batch(self, ids, start, function)
        batch.enter_pass(-1)
        for name in plan.pre_loop:
            batch.run(name)
        if batch.hidden is None:
            raise PlanError("no program before the loop produced a hidden state")
        batch.h_in = batch.hidden
        for index in range(self._loops):
            batch.enter_pass(index)
            for name in plan.loop:
                batch.run(name)
        batch.enter_pass(self._loops - 1)
        for name in plan.post_loop:
            batch.run(name)
        if batch.logits is None:
            raise PlanError("the plan produced no logits: no lm_head program ran")
        return batch.logits

    # -- public forward paths ---------------------------------------------
    def prefill(self, tokens: Sequence[int], start: int = 0) -> np.ndarray:
        """One full prefill batch (parameters.batch_size rows) through _prefill.

        Returns the logits for every row, (batch_size, vocab): the last row
        predicts the first generated token.
        """
        ids = np.asarray([int(t) for t in tokens], dtype=np.int32).reshape(1, -1)
        if ids.shape[1] != self.plan.batch_size:
            raise ValueError(f"prefill takes exactly {self.plan.batch_size} tokens (the "
                             f"batch the packages are traced at), got {ids.shape[1]}")
        self._check_context(start, ids.shape[1])
        self.load()                       # not part of the rate: load once, up front
        started = time.perf_counter()
        logits = self._forward(ids, int(start), "prefill")
        self._phase_tokens["prefill"] += int(ids.shape[1])
        self._count_wall("prefill", time.perf_counter() - started)
        return logits[0]

    def prepare_decode(self) -> float:
        """Load the infer packages now so the first decode is not charged for it.

        Models are loaded per phase, and switching from the prefill set to the infer
        set means opening 102 packages -- measured at 83 to 200 s on this machine.
        That cost belongs to the run, not to a token, so generate() pays it here,
        once, instead of letting the first decode absorb it.
        """
        self.load()
        started = time.perf_counter()
        self._enter_phase("infer")
        return time.perf_counter() - started

    def decode(self, token_id: int, pos: int) -> np.ndarray:
        """One token at pos through the _infer packages; logits, (vocab,)."""
        self._check_context(pos, 1)
        self.load()
        self._ids_one[0, 0] = int(token_id)
        started = time.perf_counter()
        logits = self._forward(self._ids_one, int(pos), "infer")
        self._phase_tokens["decode"] += 1
        self._count_wall("decode", time.perf_counter() - started)
        return logits[0, 0]

    def feed(self, tokens: Sequence[int], start: int = 0, *,
             prefill_threshold: int = 256) -> np.ndarray:
        """Feed any number of tokens and return their logits.

        Whole 64-token batches go through the _prefill packages at ~130 tok/s, but
        packing a batch switches the resident model set: the prefill packages are
        opened and the infer ones dropped, which costs 80-175 s here.  Below
        `prefill_threshold` tokens that switch is far more expensive than feeding
        one token at a time through the infer path (~10 tok/s), so short prompts --
        every turn of a chat or an agent loop -- stay on the decode side.  A 200
        token prompt costs ~20 s instead of a 100 s switch plus a second.
        """
        tokens = [int(t) for t in tokens]
        if not tokens:
            raise ValueError("feed needs at least one token")
        self._check_context(start, len(tokens))
        rows = []
        done = 0
        batch = int(self.plan.batch_size)
        if batch > 1 and len(tokens) < max(batch, int(prefill_threshold)):
            batch = 1
        while batch > 1 and done + batch <= len(tokens):
            rows.append(self.prefill(tokens[done:done + batch], start + done))
            done += batch
        while done < len(tokens):
            rows.append(self.decode(tokens[done], start + done)[None, :])
            done += 1
        return np.concatenate(rows, axis=0)

    def generate(self, tokens: Sequence[int], max_new_tokens: int, *,
                 eos_ids: Sequence[int] = ()) -> Tuple[List[int], List[int]]:
        """Greedy decode from tokens; returns (prompt, generated).

        The prompt is fed at position 0, so this always starts from empty caches:
        if this runtime has dispatched before, they are reset first, and the
        returned ids are the whole answer to this prompt.
        """
        if self._dispatched():
            self.reset()
        prompt = [int(t) for t in tokens]
        logits = self.feed(prompt, 0)
        self.prepare_decode()      # not part of the rate: the switch is a one-off
        produced: List[int] = []
        stop = {int(t) for t in eos_ids}
        for _ in range(int(max_new_tokens)):
            next_id = int(np.argmax(logits[-1]))
            produced.append(next_id)
            if next_id in stop or len(produced) >= int(max_new_tokens):
                break
            logits = np.stack([self.decode(next_id, len(prompt) + len(produced) - 1)])
        return prompt, produced

    # -- guards ------------------------------------------------------------
    def _dispatched(self) -> bool:
        return bool(sum(self._phase_dispatches.values()))

    def _check_context(self, start: int, seq: int) -> None:
        context = int(self.plan.context_length)
        end = int(start) + int(seq)
        if end > context:
            raise ContextOverflowError(
                f"{seq} token(s) at position {start} would end at {end}, past the "
                f"{context}-token context these caches hold")

    def _count_wall(self, phase: str, seconds: float) -> None:
        entry = self._wall.setdefault(phase, {"seconds": 0.0, "calls": 0})
        # The first decode after the phase switch also builds the execution plan for
        # 102 freshly opened packages; on this machine that is 80-100 s against a
        # steady 108 ms.  Charging it to the token rate would report 0.38 tok/s for a
        # loop that actually runs at 9.26, so it is kept as its own number -- the
        # time to first token -- and left out of the steady rate.
        if phase == "decode" and entry["calls"] == 0:
            self.first_token_seconds = float(seconds)
            entry["calls"] = 1
            entry["seconds"] = 0.0
            return
        entry["seconds"] += float(seconds)
        entry["calls"] += 1

    # -- reporting ---------------------------------------------------------
    def stats(self) -> dict:
        def rate(phase: str) -> dict:
            tokens = self._phase_tokens[phase]
            wall = (self._wall.get(phase) or {}).get("seconds", 0.0)
            dispatches = self._phase_dispatches[phase]
            return {
                "tokens": tokens,
                "seconds": round(wall, 3),
                "tokens_per_second": round(tokens / wall, 3) if wall > 0 and tokens else None,
                "first_token_seconds": (round(self.first_token_seconds, 3)
                                        if phase == "decode" else None),
                "dispatches": dispatches,
                "dispatches_per_token": round(dispatches / tokens, 3) if tokens else None,
                "model_seconds": round(self._phase[phase].seconds, 3),
            }

        kinds = {}
        dispatch_total = sum(item.calls for item in self._kind_time.values()) or 1
        for kind, item in sorted(self._kind_time.items(), key=lambda kv: -kv[1].seconds):
            kinds[kind] = {
                "calls": item.calls,
                "seconds": round(item.seconds, 3),
                "ms_per_call": (round(1000.0 * item.seconds / item.calls, 3)
                                if item.calls else None),
                "share": round(item.calls / dispatch_total, 4),
            }
        programs = {}
        for name, item in self._program_time.items():
            programs[name] = {
                "calls": item.calls,
                "seconds": round(item.seconds, 3),
                "ms_per_call": (round(1000.0 * item.seconds / item.calls, 3)
                                if item.calls else None),
            }
        return {
            "model_dir": str(self.model_dir),
            "loops": self._loops,
            "programs": len(self.plan.order),
            "stated_programs": len(self.plan.stated_programs()),
            "state_mib": round(self.plan.state_mib(), 1),
            "dispatches_per_token": self.plan.dispatches_per_token(self._loops),
            "load_seconds": round(self.load_seconds, 3),
            "prefill": rate("prefill"),
            "decode": rate("decode"),
            "per_kind": kinds,
            "per_program": programs,
            "peak_rss_mib": round(peak_rss_mib(), 1),
        }

    def report(self) -> str:
        stats = self.stats()
        lines = [f"model      {stats['model_dir']}",
                 f"plan       {stats['programs']} programs "
                 f"({stats['stated_programs']} with state, {stats['state_mib']:.1f} MiB), "
                 f"{stats['dispatches_per_token']} dispatches per token at "
                 f"loops={stats['loops']}",
                 f"load       {stats['load_seconds']:.2f} s"]
        if stats["decode"]["first_token_seconds"]:
            lines.append(f"{'first tok':<10} {stats['decode']['first_token_seconds']:.2f} s "
                         f"(phase switch + execution plan)")
        for phase in ("prefill", "decode"):
            item = stats[phase]
            if item["tokens"] and item["tokens_per_second"]:
                lines.append(
                    f"{phase:<10} {item['tokens']} tokens in {item['seconds']:.2f} s = "
                    f"{item['tokens_per_second']:.2f} tok/s, {item['dispatches']} "
                    f"dispatches ({item['dispatches_per_token']:.1f}/token)")
            elif item["tokens"]:
                lines.append(f"{phase:<10} {item['tokens']} tokens (first token only)")
        lines.append("per kind")
        for kind, item in stats["per_kind"].items():
            lines.append(f"  {kind:<18} {item['calls']:>6} calls  {item['seconds']:8.3f} s "
                         f" {item['ms_per_call']:7.3f} ms/call  {100 * item['share']:5.1f}%")
        lines.append(f"peak RSS   {stats['peak_rss_mib']:.0f} MiB")
        return "\n".join(lines)

    # -- one self-check the real packages need -----------------------------
    def verify_state_handoff(self, tokens: Sequence[int]) -> dict:
        """Prove the prefill packages' state is the one the decoder reads.

        Core ML cannot copy buffers between two models, so this runtime hands the
        MLState created from a program's infer package to that program's prefill
        package as well, and the caches carry across because both declare the same
        state.  That is an assumption about Core ML, not a fact the meta states --
        so it is checked: the same tokens are fed (a) as one prefill batch
        followed by one decode step and (b) one token at a time, and the
        next-token logits are compared.  They can only agree if the prefill wrote
        the buffers the decoder reads.

        Costs one prefill batch and two decodes, plus one decode per token for the
        one-token path (batch_size * dispatches_per_token), so it is opt-in.  It
        resets the caches at the start of each path, so call it on a fresh runtime
        (or call reset() afterwards) and not in the middle of a conversation.
        """
        tokens = [int(t) for t in tokens]
        if not tokens:
            raise ValueError("verify_state_handoff needs at least one token")
        if len(tokens) > self.plan.batch_size:
            raise ValueError(f"verify_state_handoff feeds one prefill batch of "
                             f"{self.plan.batch_size} tokens, got {len(tokens)}")
        if len(tokens) < self.plan.batch_size:
            # one prefill batch is what the packages take; repeating the last token
            # keeps both paths on the same ids, which is all the check needs
            tokens = tokens + [tokens[-1]] * (self.plan.batch_size - len(tokens))
        self.reset()
        prefilled = self.prefill(tokens, 0)
        first = int(np.argmax(prefilled[-1]))
        after_prefill = self.decode(first, len(tokens))

        self.reset()
        one_by_one = None
        for pos, token in enumerate(tokens):
            one_by_one = self.decode(token, pos)
        after_decode = self.decode(first, len(tokens))

        a = np.asarray(after_prefill, dtype=np.float32)
        b = np.asarray(after_decode, dtype=np.float32)
        norm = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
        self.reset()          # leave nothing of the check behind
        return {
            "tokens": len(tokens),
            "first_token_prefill": first,
            "first_token_one_by_one": int(np.argmax(one_by_one)),
            "max_abs_diff": float(np.max(np.abs(a - b))),
            "cosine": float(np.dot(a, b) / norm),
            "same_argmax": bool(int(np.argmax(a)) == int(np.argmax(b))),
        }


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _describe_plan(runtime: DoryProgramRuntime) -> str:
    plan = runtime.plan
    lines = [f"model dir   {plan.model_dir}",
             f"plan        {plan.summary()}",
             "",
             f"input zone  ({len(plan.pre_loop)} programs)"]
    for name in plan.pre_loop:
        lines.append("  " + str(plan.programs[name]))
    lines.append(f"loop zone   ({len(plan.loop)} programs per pass, "
                 f"{plan.loops_default} pass(es) by default)")
    for name in plan.loop:
        lines.append("  " + str(plan.programs[name]))
    lines.append(f"output zone ({len(plan.post_loop)} programs)")
    for name in plan.post_loop:
        lines.append("  " + str(plan.programs[name]))
    lines.append("")
    lines.append(f"state       {plan.state_mib():.1f} MiB across "
                 f"{len(plan.stated_programs())} programs")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m anemll.runtime.dory_ane_runtime",
        description="Drive a converted Dory per-(layer, head) program set on the ANE.")
    sub = parser.add_subparsers(dest="command", required=True)

    plan_cmd = sub.add_parser("plan", help="print the dispatch plan from meta.yaml")
    plan_cmd.add_argument("--model-dir", required=True)
    plan_cmd.add_argument("--loops", type=int, default=None)

    run_cmd = sub.add_parser("run", help="prefill a prompt and decode tokens")
    run_cmd.add_argument("--model-dir", required=True)
    run_cmd.add_argument("--prompt", default=None,
                         help="chat prompt (needs tokenizer.json beside the meta)")
    run_cmd.add_argument("--tokens", default=None,
                         help="raw token ids, comma separated, instead of --prompt")
    run_cmd.add_argument("--max-tokens", type=int, default=32)
    run_cmd.add_argument("--loops", type=int, default=None)
    run_cmd.add_argument("--json", action="store_true", help="print the stats as JSON")
    run_cmd.add_argument("--verify-state-handoff", action="store_true",
                         help="check that prefill fills the state decode reads")
    run_cmd.add_argument("--verbose", action="store_true")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    runtime = DoryProgramRuntime(args.model_dir, loops=args.loops,
                                 verbose=getattr(args, "verbose", False))
    if args.command == "plan":
        print(_describe_plan(runtime))
        return 0

    runtime.load()
    tokens = _resolve_tokens(runtime, args)
    if args.verify_state_handoff:
        result = runtime.verify_state_handoff(tokens[:runtime.plan.batch_size])
        if args.json:
            print(json.dumps({"verify_state_handoff": result}, indent=2))
        else:
            print("state handoff: " + ", ".join(f"{k}={v}" for k, v in result.items()))
        return 0

    eos = ((runtime.meta.get("tokenizer") or {}).get("eos_token_ids") or ())
    prompt, produced = runtime.generate(tokens, args.max_tokens, eos_ids=eos)
    stats = runtime.stats()
    if args.json:
        print(json.dumps(stats, indent=2))
        return 0
    print(runtime.report())
    print()
    print(f"prompt  {len(prompt)} tokens")
    print(f"reply   {produced}")
    text = _decode_tokens(runtime, produced)
    if text is not None:
        print(f"text    {text}")
    return 0


def _decode_tokens(runtime: DoryProgramRuntime, produced: List[int]):
    """The generated ids as text, or None when this directory carries no tokenizer.

    A missing tokenizer must not cost the caller the run's numbers, which is why
    this is best-effort and the ids are printed either way.
    """
    if not produced:
        return None
    try:
        from anemll.dory_chat import ChatTokenizer

        return ChatTokenizer(runtime.model_dir, runtime.meta).decode(produced)
    except Exception:  # noqa: BLE001 - metrics matter more than the text
        return None


def _resolve_tokens(runtime: DoryProgramRuntime, args) -> List[int]:
    if args.tokens:
        return [int(item) for item in str(args.tokens).split(",") if item.strip()]
    if not args.prompt:
        raise SystemExit("pass --prompt or --tokens")
    from anemll.dory_chat import ChatTokenizer

    tokenizer = ChatTokenizer(runtime.model_dir, runtime.meta)
    text = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}])
    return [int(t) for t in tokenizer.encode(text)]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
