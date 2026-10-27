"""Convert Dory1.0-4B into the five multifunction Core ML chunks of spec D6.

One command: scripts/convert_dory.py.  This module holds the parts:

1. the component plan -- embeddings, input (blocks 0-15), recurrent (16-55),
   output (56-71), LM head split into even parts -- with explicit, structurally
   chosen layer ranges instead of ANEMLL's balanced split (spec D6; the
   balanced formula in llama_converter.py:761-768 does not apply to a model
   whose blocks are not interchangeable);
2. the meta-file schema.  ANEMLL 0.3.5 records context length and a chunk count
   and nothing else: no KV mode, no loop range, no component manifest, no
   tokenizer or chat template.  Dory needs all of them at run time, so the
   keys are invented here and validated by validate_meta();
3. the ~2 GB per-file guard, checked twice: from a parameter count before any
   conversion starts (so a doomed run costs seconds, not an hour), and again
   on the bytes the finished .mlpackage actually occupies;
4. the per-function ANE placement check, which the converter runs itself and
   which fails the run when any op lands on the GPU or the CPU.

Tracing discipline follows ANEMLL 0.3.5 (llama_converter.py:544-550, :838-847):
reset the cache buffers, torch.jit.trace, reset them again on both the eager
module and the traced copy, then ct.convert with the state list, FLOAT16
precision, CPU_AND_NE, iOS 18 and the mlprogram format.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import coremltools as ct
from coremltools.optimize.coreml import OptimizationConfig, OpPalettizerConfig

from anemll.ane_converter.base_converter import BaseConverter
from anemll.ane_converter.metadata import AddMetadata
from anemll.models import dory_ane_model as dam
from anemll.models import dory_chunks as dch
from anemll.utils import dory_caches as dc

__all__ = [
    "META_FORMAT",
    "MAX_MODEL_BYTES",
    "ModelTooLargeError",
    "PlacementError",
    "MetaError",
    "ComponentSpec",
    "component_specs",
    "estimate_component_params",
    "estimate_component_bytes",
    "check_size",
    "check_disk",
    "disk_needed",
    "build_meta",
    "validate_meta",
    "write_meta",
    "read_meta",
    "placement_report",
    "assert_ane_only",
    "DoryConverter",
]

#: Version of the meta schema this module writes (see build_meta).
META_FORMAT = 1

#: What the runtime uses when the weights ship no chat template.  The BF16
#: checkpoint and the community GGUF both carry the special tokens but no
#: template (checked: no chat_template in tokenizer_config.json, none in the
#: GGUF metadata), while NVIDIA's README launches llama.cpp with "--jinja
#: --reasoning on".  ChatML is what the <|im_start|>/<|im_end|> pair in the
#: vocabulary means, and an open  thinking block is reasoning on by default.
FALLBACK_TEMPLATE = "chatml-think"

#: The ANE file limit that matters on this Mac: ~2 GB per Core ML file.
MAX_MODEL_BYTES = 2 * 1024 ** 3

#: Refuse earlier than the hard limit, so a conversion never ends just over it.
SIZE_WARN_FRACTION = 0.90

#: Components whose blocks are in the input zone / recurrent zone / output zone.
MASK_WIDTH = "mask_len"


class ModelTooLargeError(RuntimeError):
    """A Core ML file would exceed the ANE per-file limit."""


class PlacementError(RuntimeError):
    """A converted op did not land on the Neural Engine."""


class MetaError(ValueError):
    """The meta file is missing or contradicts itself."""


# ---------------------------------------------------------------------------
# The component plan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ComponentSpec:
    """One Core ML file: what it computes, how often, and which LUT applies.

    calls_per_token is an int or the string 'loops' (the host calls the
    component once per loop pass).  blocks is (start, end) for block chunks and
    None for the embedding and LM-head components.
    """

    name: str
    kind: str                      # "embeddings" | "blocks" | "lm_head"
    blocks: Optional[tuple] = None
    loop_entry: bool = False
    loop_exit: bool = False
    calls_per_token: object = 1
    lut_role: str = "blocks"
    tiled: bool = False
    rope_tables: bool = False

    @property
    def inputs(self) -> list:
        if self.kind == "embeddings":
            return ["input_ids"]
        if self.kind == "lm_head":
            return ["hidden_states"]
        # one source of truth: the wrapper classes pick their signature from the
        # same function, so a converted input name cannot drift from the trace
        return dch.block_input_names(self.tiled, self.loop_entry, self.rope_tables)

    @property
    def is_recurrent(self) -> bool:
        return self.loop_entry or self.loop_exit


def input_spec(name: str, seq: int, config):
    """(traced shape, torch dtype, Core ML dtype) of one declared input.

    One table feeds both the traced example tuple and the ct.TensorType list, so
    an input cannot appear in one and not the other.  That is how the per-tile
    write index and gate, and the four RoPE row tensors, stay in step with the
    wrapper signature that block_input_names() describes.
    """
    if name == "input_ids":
        return (1, int(seq)), torch.int32, np.int32
    if name == "causal_mask":
        return (1, 1, int(seq), int(config.mask_len)), config.dtype, np.float16
    if name in ("current_pos", "cache_pos") or name.startswith("write_index_"):
        return (1,), torch.int32, np.int32
    if name.startswith("write_gate_"):
        return (1,), config.dtype, np.float16
    if name.startswith("cos_") or name.startswith("sin_"):
        # the rows the graph would have sliced from its own table (24K option)
        return (1, 1, int(seq), int(config.head_dim)), config.dtype, np.float16
    if name in ("gate", "contribution"):
        # a layer's gate travels as a leaf between programs (the program that
        # produced this layer's input computed it, because a program that owns a
        # cache cannot carry its own gate convolution), and it is one value per
        # attention channel, num_attention_heads * head_dim wide.  The plain
        # "contribution" is the older single wide sliding output, kept so that
        # shape stays convertible for measurement; the default surface is the
        # four per-KV-head contribution_h* slices below.
        return ((1, int(seq), int(config.num_attention_heads) * int(config.head_dim)),
                config.dtype, np.float16)
    if name.startswith("head_") or name.startswith("contribution_h"):
        # one per-KV-head attention contribution: the rep query heads of that KV
        # head, head_dim channels each.  A full-attention head program's output
        # and one slice of a sliding attention program's are the same width by
        # construction, which is why the epilogue after each of them takes the
        # same interface.
        rep = int(config.num_attention_heads) // int(config.num_key_value_heads)
        return ((1, int(seq), rep * int(config.head_dim)), config.dtype, np.float16)
    # hidden_states, h_in: one hidden vector per row
    return (1, int(seq), int(config.hidden_size)), config.dtype, np.float16


#: The RoPE row tables travel beside the package when rope_tables_as_inputs is on.
ROPE_TABLE_FILE = "rope_tables.npz"
ROPE_TABLE_KEYS = ("cos_sliding", "sin_sliding", "cos_full", "sin_full")


def rope_tables(config) -> dict:
    """The four cos/sin row tables, built by the model's own rotary module.

    Exported once at conversion time so the runtime slices rows instead of
    computing them: the formula lives in one place, DoryANERotaryEmbedding, which
    is also where the graph's constant table came from.  Two implementations of
    the same table would be exactly the drift this version of the converter is
    meant to remove.
    """
    dtype = getattr(config, "dtype", torch.float16)
    if dtype not in (torch.float16, torch.float32):
        raise SystemExit(f"cannot export RoPE tables as {dtype}: float16 or float32 only")
    rows = int(config.context_length) + 1
    tables = {}
    for kind, base in (("sliding", float(config.rope_theta_sliding)),
                       ("full", float(config.rope_theta_full))):
        module = dam.DoryANERotaryEmbedding(int(config.head_dim), base, rows, dtype)
        tables[f"cos_{kind}"] = module.cos_cached.detach().cpu().numpy()
        tables[f"sin_{kind}"] = module.sin_cached.detach().cpu().numpy()
    return {name: tables[name] for name in ROPE_TABLE_KEYS}


def rope_table_descriptor(config):
    """What the meta records about the sidecar, or None when it is not written."""
    if not tiling_flags(config)[1]:
        return None
    return {
        "file": ROPE_TABLE_FILE,
        "rows": int(config.context_length) + 1,
        "head_dim": int(config.head_dim),
        "dtype": "float16" if getattr(config, "dtype", torch.float16) is torch.float16
                 else "float32",
        "keys": list(ROPE_TABLE_KEYS),
    }


def write_rope_tables(config, output_dir) -> Path:
    """Write the sidecar next to the package; returns its path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ROPE_TABLE_FILE
    np.savez_compressed(path, **rope_tables(config))
    return path


def tiling_flags(config) -> tuple:
    """(tiled, rope_tables) as the components of this config must be built."""
    tiles = tuple(getattr(config, "full_tiles", (1,)) or (1,))
    return len(tiles) > 1, bool(getattr(config, "rope_tables_as_inputs", False))


def component_specs(config, recurrent_splits: int = 1) -> list:
    """The five components of spec D6, in run order.

    With recurrent_splits > 1 the recurrent block (blocks 16-55) becomes two or
    more files, each called once per loop pass; the loop entry norm stays on the
    first and the loop exit norm on the last, so the math is unchanged.
    """
    recurrent_splits = int(recurrent_splits)
    if recurrent_splits < 1:
        raise ValueError("recurrent_splits must be >= 1")
    input_end = int(config.input_layer_count)
    rec_start = input_end
    rec_end = rec_start + int(config.recurrent_layer_count)
    num_blocks = int(config.num_hidden_layers)
    if rec_end > num_blocks:
        raise ValueError(f"input+recurrent blocks ({rec_end}) exceed the model's "
                         f"{num_blocks} blocks")

    tiled, rope_tables = tiling_flags(config)
    flags = dict(tiled=tiled, rope_tables=rope_tables)
    specs = [ComponentSpec("embeddings", "embeddings", None, calls_per_token=1,
                           lut_role="embeddings")]
    specs.append(ComponentSpec("input", "blocks", (0, input_end), calls_per_token=1,
                               **flags))

    base, rem = divmod(rec_end - rec_start, recurrent_splits)
    start = rec_start
    for i in range(recurrent_splits):
        end = start + base + (1 if i < rem else 0)
        name = "recurrent" if recurrent_splits == 1 else f"recurrent_{i + 1}of{recurrent_splits}"
        specs.append(ComponentSpec(
            name, "blocks", (start, end),
            loop_entry=(i == 0),
            loop_exit=(i == recurrent_splits - 1),
            calls_per_token="loops",
            **flags))
        start = end

    specs.append(ComponentSpec("output", "blocks", (rec_end, num_blocks), calls_per_token=1,
                               **flags))
    specs.append(ComponentSpec("lm_head", "lm_head", None, calls_per_token=1,
                               lut_role="lm_head"))
    return specs


def block_params(config, block: int) -> int:
    """Parameters of one physical block (attention or MLP)."""
    h = int(config.hidden_size)
    inter = int(config.intermediate_size)
    nh = int(config.num_attention_heads)
    nkv = int(config.num_key_value_heads)
    hd = int(config.head_dim)
    if dc.block_is_mlp(block):
        return 3 * h * inter
    return h * nh * hd + 2 * h * nkv * hd + nh * hd * h + h * nh * hd


def estimate_component_params(spec: ComponentSpec, config) -> int:
    """Parameters one component holds, used by the size guard before converting.

    A per-program spec (dory_per_head.ProgramSpec) carries its own exact count,
    because one (layer, head) program holds a head's rows of the projections and
    not a whole block; a ComponentSpec has no such attribute and is estimated
    from its block range exactly as before.
    """
    hint = getattr(spec, "params", None)
    if hint is not None:
        return int(hint)
    if spec.kind == "embeddings" or spec.kind == "lm_head":
        return int(config.vocab_size) * int(config.hidden_size)
    if spec.blocks is None:
        raise ValueError(f"component {spec.name} has no block range")
    return sum(block_params(config, b) for b in range(spec.blocks[0], spec.blocks[1]))


def estimate_component_bytes(spec: ComponentSpec, config, lut_bits) -> int:
    """Rough on-disk size of a converted component, in bytes.

    A palettized weight costs lut_bits, plus one fp16 scale per group of
    per_grouped_channel weights (16 bits / 8 weights = 2 bits), plus the small
    LUT tables and the fp16 non-weight tensors; 5 % is added for those.  An
    unpalettized (None) component is fp16.
    """
    params = estimate_component_params(spec, config)
    if lut_bits is None:
        return int(params * 2 * 1.02)
    # coremltools palettizes per grouped channel over a group of 16 weights, so
    # the fp16 scale costs one bit per 16 weights, not one scale per 8.  Measured
    # on the lm_head package: 335.5M params -> 242 MiB = 0.759 B/param (6.07 bits).
    return int(params * (int(lut_bits) + 1 / 16.0) / 8.0 * 1.02)


def disk_needed(estimates) -> int:
    """Disk a conversion needs on top of the model it writes.

    Components convert one at a time, so the peak is the largest component
    tripled (unquantized function package + quantized function package + the
    welded multifunction package) plus a gigabyte of slack, not the sum of all
    five.
    """
    sizes = [int(size) for _, size in estimates]
    if not sizes:
        return 0
    return int(max(sizes) * 3.2) + 1024 ** 3


def check_disk(path, estimates, reserve_bytes: int = 2 * 1024 ** 3) -> dict:
    """Refuse to start a conversion that would leave the volume under 2 GB.

    There is one volume and it is shared with the rest of the port (references,
    tests, other tickets' artifacts), so the guard is on the *free* space left
    after the conversion, not on the conversion's own size.
    """
    needed = disk_needed(estimates)
    usage = shutil.disk_usage(str(path))
    free = int(usage.free)
    if free - needed < reserve_bytes:
        raise RuntimeError(
            f"not enough disk: {free / 1024 ** 3:.1f} GB free, this conversion "
            f"needs about {needed / 1024 ** 3:.1f} GB and must leave "
            f"{reserve_bytes / 1024 ** 3:.0f} GB for the rest of the port. Free "
            f"space (delete a superseded package or a reference) and run again.")
    return {"free": free, "needed": needed}


def nearest_existing(path) -> Path:
    """The closest ancestor of path that exists -- disk_usage needs one."""
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def check_size(label: str, size_bytes: int, limit_bytes: int = MAX_MODEL_BYTES) -> int:
    """Raise ModelTooLargeError when a component would not fit in one file.

    The message has to be actionable: it names the limit, the estimate, and the
    flag that fixes it (a wider recurrence split, a narrower LUT, or both).
    """
    if size_bytes <= limit_bytes:
        return size_bytes
    gb = size_bytes / 1024 ** 3
    limit_gb = limit_bytes / 1024 ** 3
    raise ModelTooLargeError(
        f"{label} would be about {gb:.2f} GB, over the {limit_gb:.1f} GB Core ML "
        f"per-file limit that the Neural Engine requires. Fix it with either "
        f"--recurrent-splits 2 (splits the recurrent block into two files, each "
        f"called once per loop) or a narrower LUT (--lut-blocks 4), then run "
        f"again. Nothing was written."
    )


# ---------------------------------------------------------------------------
# The meta file: what ANEMLL 0.3.5 does not record and the runtime needs
# ---------------------------------------------------------------------------

def _lut_value(bits):
    """Normalise a LUT argument to an int or None (unquantized)."""
    if bits in (None, "", "none", "None", 0, "0"):
        return None
    return int(bits)


def _state_descriptions(config) -> list:
    """One Core ML state per cache buffer, with its shape and byte cost.

    The tiled fallback (spec D7) replaces the single full cache with one buffer
    per tile, so the list comes from dch.state_attrs -- the same function the
    model registers its buffers from and the runtime builds its MLState from.  A
    hard-coded pair here was exactly what made a tiled spec unconvertible.
    """
    tiles = tuple(getattr(config, "full_tiles", (1,)) or (1,))
    names = dch.state_names(config)
    out = []
    tiles_seen = 0
    for (attr, shape), name in zip(dch.state_layout(config), names):
        kind = "full" if attr.startswith("kv_cache_full") else "sliding"
        elements = int(np.prod(shape))
        entry = {
            "name": name,
            "buffer": attr,
            "kind": kind,
            # K and V of every layer-slot in this buffer, interleaved along dim 0
            "slots": int(shape[0]) // 2,
            "shape": [int(d) for d in shape],
            "rows": int(shape[2]),
            "dtype": "float16",
            "bytes": elements * 2,
        }
        if kind == "full" and len(tiles) > 1:
            entry["tile"] = tiles_seen + 1
            entry["tile_offset"] = int(sum(int(r) for r in tiles[:tiles_seen]))
            tiles_seen += 1
        out.append(entry)
    return out


def build_meta(*, config, specs, files, lut_bits, lm_head_chunk_sizes, prefix="dory",
               argmax=False, tokenizer="tokenizer.json",
               chat_template="chat_template.jinja", bos_token_id=1, eos_token_id=11,
               eos_token_ids=(2, 11), created=None, per_program=False) -> dict:
    """The meta file: context, chunk layout, LUT widths, KV mode, loop range,
    tokenizer and chat template.

    keys ANEMLL 0.3.5 has: model_info, parameters.context_length,
    parameters.batch_size, the lut_* triple, split_lm_head, argmax_in_model.
    keys invented here (none of them exist upstream): dory_meta (schema
    version), parameters.state_length / sliding_rows / mask_len / zones /
    num_full_caches / num_sliding_caches, the whole kv section, the loops
    section, the tokenizer section, and the components manifest that maps a
    component name to its file, its functions, its block range and how many
    times the host calls it per token.
    """
    context = int(config.context_length)
    batch = int(config.batch_size)
    states = _state_descriptions(config)
    zone_input = [0, int(config.input_layer_count)]
    zone_recurrent = [zone_input[1], zone_input[1] + int(config.recurrent_layer_count)]
    zone_output = [zone_recurrent[1], int(config.num_hidden_layers)]

    components = {}
    for spec in specs:
        bits = lut_bits.get(spec.lut_role)
        entry = {
            "kind": spec.kind,
            "files": dict(files[spec.name]),
            "functions": ["infer", "prefill"] if batch > 1 else ["infer"],
            "calls_per_token": spec.calls_per_token,
            "lut": _lut_value(bits),
            "inputs": list(spec.inputs),
        }
        if spec.kind == "blocks":
            entry["blocks"] = [int(spec.blocks[0]), int(spec.blocks[1])]
            entry["loop_entry"] = bool(spec.loop_entry)
            entry["loop_exit"] = bool(spec.loop_exit)
            entry["tiled"] = bool(spec.tiled)
            entry["rope_tables"] = bool(spec.rope_tables)
        if spec.kind == "lm_head":
            entry["outputs"] = ([f"argmax_idx_{i}" for i in range(1, len(lm_head_chunk_sizes) + 1)]
                                if argmax else
                                [f"logits_{i}" for i in range(1, len(lm_head_chunk_sizes) + 1)])
            entry["splits"] = len(lm_head_chunk_sizes)
        else:
            entry["outputs"] = (["output_hidden_states"] if spec.kind == "blocks"
                                else ["hidden_states"])
        # a per-program spec (dory_per_head.ProgramSpec) names its own outputs,
        # and a program in the gate chain has two: its hidden state and the next
        # attention layer's gate leaf
        program_outputs = getattr(spec, "outputs", None)
        if isinstance(program_outputs, (list, tuple)):
            entry["outputs"] = [str(name) for name in program_outputs]
            entry["gate_in"] = bool(getattr(spec, "gate_in", False))
            entry["gate_out"] = getattr(spec, "gate_out", None)
        components[spec.name] = entry

    meta = {
        "dory_meta": META_FORMAT,
        "model_info": {
            "name": f"anemll-{prefix}-ctx{context}",
            "description": (
                f"Dory1.0-4B on the Apple Neural Engine: {len(specs)} components, "
                f"context {context}, prefill batch {batch}, loops 1-4 "
                f"({config.sliding_window}-token sliding window)."),
            "architecture": "dory",
            "framework": "Core ML",
            "format": "mlprogram",
            "created": created or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        "parameters": {
            "model_prefix": prefix,
            "context_length": context,
            "state_length": int(config.state_length),
            "batch_size": batch,
            "vocab_size": int(config.vocab_size),
            "hidden_size": int(config.hidden_size),
            "intermediate_size": int(config.intermediate_size),
            "num_hidden_layers": int(config.num_hidden_layers),
            "num_attention_heads": int(config.num_attention_heads),
            "num_key_value_heads": int(config.num_key_value_heads),
            "head_dim": int(config.head_dim),
            "sliding_window": int(config.sliding_window),
            "sliding_rows": int(config.sliding_rows),
            "mask_len": int(config.mask_len),
            "zones": {"input": zone_input, "recurrent": zone_recurrent,
                      "output": zone_output},
            "full_attention_blocks": list(dc.full_attention_blocks()),
            "sliding_attention_blocks": list(dc.sliding_attention_blocks()),
            "num_full_caches": int(config.num_full_caches),
            "num_sliding_caches": int(config.num_sliding_caches),
            "num_chunks": len(specs),
            "split_lm_head": len(lm_head_chunk_sizes),
            "lm_head_chunk_sizes": [int(s) for s in lm_head_chunk_sizes],
            "argmax_in_model": bool(argmax),
            "full_cache_tiles": [int(rows) for rows in
                                 (getattr(config, "full_tiles", (1,)) or (1,))],
            "full_cache_tiled": bool(tiling_flags(config)[0]),
            "rope_tables_as_inputs": bool(tiling_flags(config)[1]),
            "rope_tables": rope_table_descriptor(config),
            "lut_embeddings": _lut_value(lut_bits.get("embeddings")),
            "lut_blocks": _lut_value(lut_bits.get("blocks")),
            "lut_lm_head": _lut_value(lut_bits.get("lm_head")),
            "lut_per_channel": 8,
        },
        "kv": {
            "mode": "last_loop",
            "dtype": "float16",
            "write_scheme": "ring",
            "invalidate_on_loop_change": True,
            "tiled": bool(tiling_flags(config)[0]),
            # DANE-5: how the caches are carved up, so a reader of the artifact
            # can tell 45 states of 24.0 MiB from the aggregated pair without
            # rebuilding the shapes.  state_layout describes both.
            "heads_per_state": getattr(config, "heads_per_state", None),
            "slots_per_sliding_carrier": int(
                getattr(config, "slots_per_sliding_carrier", 3) or 3),
            "state_layout": (
                "per_head" if getattr(config, "heads_per_state", None) is not None
                else "aggregated"),
            "state_summary": dch.state_summary(config, per_program=per_program),
            "write_frame": ("one full-attention tile per call: the host passes "
                            "write_index_i / write_gate_i for every tile"
                            if tiling_flags(config)[0] else
                            "one contiguous write at cache_pos"),
            "states": states,
        },
        "loops": {"min": 1, "max": 4, "default": 2},
        "tokenizer": {
            "file": tokenizer,
            "chat_template": chat_template,
            "chat_template_fallback": None if chat_template else FALLBACK_TEMPLATE,
            "reasoning_default": True,
            "reasoning_marker": "think",
            "bos_token_id": int(bos_token_id),
            "eos_token_id": int(eos_token_id),
            "eos_token_ids": [int(t) for t in eos_token_ids],
        },
        "components": components,
        "run_order": [spec.name for spec in specs],
    }
    return meta


def validate_meta(meta) -> list:
    """Every way the meta file can be wrong, as a list of human sentences.

    The runtime calls this on load, so a broken conversion is a message and not
    a garbage answer; the converter calls it on write, so a broken schema never
    reaches disk.
    """
    problems = []

    def need(cond, message):
        if not cond:
            problems.append(message)

    if not isinstance(meta, dict):
        return ["the meta file is not a mapping"]
    need(meta.get("dory_meta") == META_FORMAT,
         f"dory_meta must be {META_FORMAT}, got {meta.get('dory_meta')!r}")
    params = meta.get("parameters") or {}
    context = params.get("context_length")
    need(isinstance(context, int) and context > 0, "parameters.context_length must be a positive int")
    batch = params.get("batch_size")
    need(isinstance(batch, int) and batch >= 1, "parameters.batch_size must be >= 1")
    sliding_rows = params.get("sliding_rows")
    mask_len = params.get("mask_len")
    if isinstance(context, int) and isinstance(sliding_rows, int) and isinstance(mask_len, int):
        need(mask_len == context + sliding_rows,
             f"parameters.mask_len ({mask_len}) must be context_length + sliding_rows "
             f"({context} + {sliding_rows})")
        need(sliding_rows >= min(context, params.get("sliding_window") or sliding_rows),
             "parameters.sliding_rows is smaller than the sliding window it must hold")

    loops = meta.get("loops") or {}
    lo, hi, default = loops.get("min"), loops.get("max"), loops.get("default")
    need(isinstance(lo, int) and isinstance(hi, int) and isinstance(default, int),
         "loops.min / loops.max / loops.default must be ints")
    if isinstance(lo, int) and isinstance(hi, int) and isinstance(default, int):
        need(1 <= lo <= default <= hi, "loops must satisfy 1 <= min <= default <= max")

    kv = meta.get("kv") or {}
    need(kv.get("mode") == "last_loop",
         f"kv.mode must be 'last_loop', got {kv.get('mode')!r}")
    states = kv.get("states") or []
    tiles = params.get("full_cache_tiles") or ([context] if isinstance(context, int) else [1])
    tiled = bool(params.get("full_cache_tiled", len(tiles) > 1))
    heads_per_state = kv.get("heads_per_state")
    split = heads_per_state is not None
    need(bool(kv.get("tiled", tiled)) == tiled,
         f"kv.tiled ({kv.get('tiled')}) disagrees with parameters.full_cache_tiled ({tiled})")
    need(not (tiled and split),
         "kv.heads_per_state and the tiled fallback cannot be combined: the layout refuses "
         "it, and a meta claiming both would describe a model nobody built")
    full_states = [state for state in states if state.get("kind") == "full"]
    sliding_states = [state for state in states if state.get("kind") == "sliding"]
    need(bool(full_states), "kv.states declares no full-attention buffers")
    need(bool(sliding_states), "kv.states declares no sliding buffers")
    if split:
        # one state per (layer-slot, head group), and the windowed caches grouped
        # by layer-slot -- the DANE-5 layout that stays under the per-state
        # ceiling at 24,576 rows
        num_kv = params.get("num_key_value_heads")
        n_full_slots = params.get("num_full_caches")
        n_sliding_slots = params.get("num_sliding_caches")
        group = kv.get("slots_per_sliding_carrier")
        usable = (isinstance(heads_per_state, int) and heads_per_state >= 1
                  and isinstance(num_kv, int) and num_kv % heads_per_state == 0)
        need(usable,
             f"kv.heads_per_state ({heads_per_state!r}) must divide "
             f"parameters.num_key_value_heads ({num_kv!r})")
        if usable and isinstance(n_full_slots, int):
            expected = int(n_full_slots) * (int(num_kv) // int(heads_per_state))
            need(len(full_states) == expected,
                 f"kv.states declares {len(full_states)} full-attention buffers, expected "
                 f"{expected}: one per (layer-slot, head group) with {heads_per_state} "
                 f"head(s) of {num_kv}")
        need(isinstance(group, int) and group >= 1,
             f"kv.slots_per_sliding_carrier must be a positive int, got {group!r}")
        if isinstance(group, int) and group >= 1 and isinstance(n_sliding_slots, int):
            expected = -(-int(n_sliding_slots) // int(group))
            need(len(sliding_states) == expected,
                 f"kv.states declares {len(sliding_states)} sliding carriers, expected "
                 f"{expected} holding {group} layer-slot(s) of {n_sliding_slots}")
    else:
        n_full = len(tiles) if tiled else 1
        need(len(states) == n_full + 1,
             f"kv.states must declare one buffer per full-attention tile plus the sliding "
             f"cache ({n_full + 1} with {n_full} tile(s)), got {len(states)}")
        need(len(full_states) == n_full,
             f"kv.states declares {len(full_states)} full-attention buffers, expected {n_full}")
        need(len(sliding_states) == 1,
             f"kv.states declares {len(sliding_states)} sliding buffers, expected 1")
    for state in states:
        need(bool(state.get("name")), "every kv state needs a name")
        need(isinstance(state.get("shape"), list) and state.get("shape"),
             f"kv state {state.get('name')} needs a shape")
    if isinstance(context, int) and full_states:
        if split:
            # every split state holds the whole context for one head group; the
            # tiles, which are the case that sums to the context, cannot be on
            wrong = [state.get("name") for state in full_states
                     if len(state.get("shape") or []) != 4
                     or int((state.get("shape") or [0, 0, 0, 0])[2]) != context]
            need(not wrong,
                 f"every split full-attention state must hold context_length {context} rows, "
                 f"these do not: {wrong[:3]}")
        else:
            rows = 0
            for state in full_states:
                shape = state.get("shape") or []
                if len(shape) == 4:
                    rows += int(shape[2])
            need(rows == context,
                 f"the full-attention tiles hold {rows} rows in total, expected context_length "
                 f"{context}")
    if tiled:
        need(len(tiles) > 1, "parameters.full_cache_tiled is true but full_cache_tiles "
                             "lists a single tile")
        need(sum(int(r) for r in tiles) == context,
             f"full_cache_tiles {list(tiles)} sum to {sum(int(r) for r in tiles)}, "
             f"not context_length {context}")

    rope_inputs = bool(params.get("rope_tables_as_inputs", False))
    declared = params.get("rope_tables")
    if rope_inputs:
        need(isinstance(declared, dict),
             "parameters.rope_tables must describe the sidecar when "
             "rope_tables_as_inputs is on")
        if isinstance(declared, dict):
            rows = declared.get("rows")
            need(isinstance(rows, int) and isinstance(context, int) and rows >= context + 1,
                 f"parameters.rope_tables.rows ({rows}) must cover context_length + 1 "
                 f"({context + 1 if isinstance(context, int) else '?'})")
            need(declared.get("head_dim") == params.get("head_dim"),
                 f"parameters.rope_tables.head_dim ({declared.get('head_dim')}) must equal "
                 f"parameters.head_dim ({params.get('head_dim')})")
            need(bool(declared.get("file")), "parameters.rope_tables.file is required")
            need(list(declared.get("keys") or []) == list(ROPE_TABLE_KEYS),
                 f"parameters.rope_tables.keys must be {list(ROPE_TABLE_KEYS)}, "
                 f"got {declared.get('keys')}")
    else:
        need(declared in (None, {}),
             f"parameters.rope_tables is set ({declared!r}) but rope_tables_as_inputs is off")

    tokenizer = meta.get("tokenizer") or {}
    need(bool(tokenizer.get("file")), "tokenizer.file is required")
    need("chat_template" in tokenizer,
         "tokenizer.chat_template must be present (a file name, or null with a fallback)")
    need(bool(tokenizer.get("chat_template")) or bool(tokenizer.get("chat_template_fallback")),
         "either tokenizer.chat_template or tokenizer.chat_template_fallback is required")

    components = meta.get("components") or {}
    order = meta.get("run_order") or []
    need(isinstance(components, dict), "components must be a mapping")
    need(list(order) == list(components.keys()),
         "run_order must list every component, in order")
    if order:
        need(order[0] == "embeddings", "embeddings must run first")
        need(order[-1] == "lm_head", "lm_head must run last")
    covered = set()
    for name, comp in components.items():
        comp_files = comp.get("files") or {}
        need(isinstance(comp_files, dict) and bool(comp_files),
             f"component {name} needs a files mapping")
        funcs = comp.get("functions") or []
        for function in funcs:
            need(bool(comp_files.get(function)),
                 f"component {name} needs a file for its {function} function")
        need("infer" in funcs, f"component {name} must provide the infer function")
        if batch and batch > 1:
            need("prefill" in funcs, f"component {name} must provide the prefill function")
        calls = comp.get("calls_per_token")
        need(isinstance(calls, int) or calls == "loops",
             f"component {name} calls_per_token must be an int or 'loops'")
        if comp.get("kind") == "blocks":
            blocks = comp.get("blocks") or []
            need(len(blocks) == 2 and blocks[1] > blocks[0],
                 f"component {name} needs a block range")
            if len(blocks) == 2:
                covered.update(range(blocks[0], blocks[1]))
                if comp.get("loop_entry") or comp.get("loop_exit"):
                    need(calls == "loops",
                         f"recurrent component {name} must be called once per loop")
                else:
                    need(calls == 1, f"component {name} must be called once per token")
    n_blocks = params.get("num_hidden_layers")
    if isinstance(n_blocks, int) and covered:
        need(sorted(covered) == list(range(n_blocks)),
             f"the block components must cover blocks 0..{n_blocks - 1}, "
             f"covered {len(covered)}")

    head = components.get("lm_head") or {}
    sizes = params.get("lm_head_chunk_sizes") or []
    vocab = params.get("vocab_size")
    if sizes and isinstance(vocab, int):
        need(len(sizes) == params.get("split_lm_head"),
             "lm_head_chunk_sizes must match split_lm_head")
        need(sum(sizes) == vocab,
             f"lm_head_chunk_sizes must sum to vocab_size ({sum(sizes)} != {vocab})")
        need(head.get("splits") == len(sizes),
             "the lm_head component must declare its splits")
    return problems


def write_meta(path, meta) -> Path:
    """Write meta.yaml (and refuse to write an invalid one)."""
    import yaml

    problems = validate_meta(meta)
    if problems:
        raise MetaError("refusing to write an invalid meta file: " + "; ".join(problems))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(meta, handle, sort_keys=False, width=100)
    return path


def read_meta(path):
    """Read and validate meta.yaml."""
    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        meta = yaml.safe_load(handle)
    problems = validate_meta(meta)
    if problems:
        raise MetaError("invalid meta file " + str(path) + ": " + "; ".join(problems))
    return meta


# ---------------------------------------------------------------------------
# ANE placement: the converter checks itself
# ---------------------------------------------------------------------------

def _device_of(usage) -> str:
    """Classify an MLComputePlan device usage as 'ane', 'gpu', 'cpu' or 'none'."""
    if usage is None:
        return "none"
    text = repr(usage).lower()
    if "ane" in text or "neural" in text:
        return "ane"
    if "gpu" in text or "mps" in text:
        return "gpu"
    if "cpu" in text:
        return "cpu"
    return "none"


def placement_report(model_path) -> dict:
    """Per-function op placement from Core ML's own compute plan.

    Compiles the package with xcrun coremlcompiler into a scratch directory and
    walks every operation of every function, asking the compute plan which
    device it will run on.  Returns {function: {ane, gpu, cpu, none, ops}}.
    """
    from coremltools.models.compute_plan import MLComputePlan

    model_path = Path(model_path)
    tmp = Path(tempfile.mkdtemp(prefix="dory_plan_"))
    try:
        if model_path.suffix == ".mlpackage":
            result = subprocess.run(
                ["xcrun", "coremlcompiler", "compile", str(model_path), str(tmp)],
                capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"coremlcompiler failed for {model_path}: "
                                   f"{result.stderr.strip()[:400]}")
            compiled = sorted(tmp.glob("*.mlmodelc"))
            if not compiled:
                raise RuntimeError(f"no .mlmodelc produced for {model_path}")
            compiled_path = compiled[0]
        else:
            compiled_path = model_path
        plan = MLComputePlan.load_from_path(str(compiled_path))
        program = plan.model_structure.program
        if program is None:
            raise RuntimeError(f"{model_path} is not an ML program")
        report = {}
        for name, function in program.functions.items():
            counts = {"ane": 0, "gpu": 0, "cpu": 0, "none": 0}
            fallbacks = []
            for op in function.block.operations:
                device = _device_of(plan.get_compute_device_usage_for_mlprogram_operation(op))
                counts[device] += 1
                if device in ("gpu", "cpu"):
                    outputs = getattr(op, "outputs", []) or []
                    label = outputs[0].name if outputs else getattr(op, "operator_name", "?")
                    fallbacks.append(f"{getattr(op, 'operator_name', '?')}:{label}")
            counts["ops"] = sum(v for k, v in counts.items() if k != "ops")
            counts["fallbacks"] = fallbacks
            report[name] = counts
        return report
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def assert_ane_only(report: dict, label: str) -> dict:
    """Fail unless every function runs its ops on the Neural Engine.

    Two ways to fail, both of them hard: an op attributed to the GPU or the CPU,
    and a function the compute plan attributed no device to (ane + gpu + cpu ==
    0).  An empty attribution is not evidence of placement, so it must never
    read as a pass.
    """
    def offending(counts) -> bool:
        return bool(counts["gpu"] or counts["cpu"]) or not counts["ane"]

    bad = {fn: counts for fn, counts in report.items() if offending(counts)}
    if not bad:
        return report
    lines = []
    for fn, counts in bad.items():
        if counts["ane"] or counts["gpu"] or counts["cpu"]:
            lines.append(f"{fn}: {counts['ane']} ANE, {counts['gpu']} GPU, "
                         f"{counts['cpu']} CPU")
        else:
            lines.append(f"{fn}: the compute plan attributed no device to any of "
                         f"{counts['ops']} operations")
        for op in counts["fallbacks"][:20]:
            lines.append(f"  - {op}")
    raise PlacementError(
        f"{label} has operations off the Neural Engine (spec gate: every op on "
        f"the ANE): " + "; ".join(lines))


# ---------------------------------------------------------------------------
# The converter
# ---------------------------------------------------------------------------

class _ComponentModel(dam.DoryANEForCausalLM):
    """DoryANEForCausalLM restricted to the blocks one component converts.

    Inherits ticket DANE-4's weight loader unchanged -- the folding of the four
    static scales (spec D5) happens exactly once, in that loader -- and changes
    only which blocks exist, so converting the output block does not allocate
    the recurrent block's 1.8 B parameters.  The cache buffers are built for the
    full 36 slots regardless, which is what keeps the state union identical
    across components.
    """

    def __init__(self, config, block_indices, load_lm_head: bool = False,
                 head_map=None, state_subset=None, embed_vocab=None,
                 gate_in_blocks=(), gate_only_blocks=(), tail_only_blocks=(),
                 attn_only_blocks=(), contrib_per_head_blocks=()):
        """head_map / state_subset belong to the per-program path (dory_per_head):
        they build one KV-head group of a block and restrict the declared states
        to the ones that program owns.  Both default to the whole five-component
        behaviour.

        embed_vocab sizes the (unused) token embedding; a program that does not
        convert embeddings passes 1 so it does not allocate 131,072 x 2,560
        weights it will never trace, 56 times over.

        contrib_per_head_blocks refines attn_only_blocks: that block's attention
        scores one KV head at a time and emits one tensor per KV head, which is
        the only shape a stateful sliding attention program may have (a 4096-wide
        tensor anywhere in it builds no plan, measured).
        """
        torch.nn.Module.__init__(self)
        self.config = config
        self.embed_tokens = torch.nn.Embedding(
            int(config.vocab_size if embed_vocab is None else embed_vocab),
            config.hidden_size, dtype=config.dtype)
        self.model = dam.DoryANEModel(config, block_indices=list(block_indices),
                                      head_map=head_map, state_subset=state_subset,
                                      gate_in_blocks=tuple(gate_in_blocks),
                                      gate_only_blocks=tuple(gate_only_blocks),
                                      tail_only_blocks=tuple(tail_only_blocks),
                                      attn_only_blocks=tuple(attn_only_blocks),
                                      contrib_per_head_blocks=tuple(
                                          contrib_per_head_blocks))
        self.load_lm_head = bool(load_lm_head)
        if load_lm_head:
            self.lm_head = torch.nn.Conv2d(config.hidden_size, config.vocab_size,
                                           kernel_size=1, bias=False, dtype=config.dtype)
            self.register_buffer("logit_scale", torch.empty(0), persistent=False)
        else:
            self.lm_head = None
        self._active_loops = None


class DoryConverter(BaseConverter):
    """Convert Dory's five components to multifunction Core ML packages."""

    def __init__(self, model_path, output_dir, config=None, prefix="dory",
                 lut_blocks=6, lut_lm_head=6, lut_embeddings=None,
                 lut_per_channel=8, lm_head_splits=8, recurrent_splits=1,
                 argmax=False, check_placement=True, verbose=True,
                 resume=False):
        super().__init__(model=None)
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        if config is None:
            raise ValueError("a DoryANEConfig is required (DoryANEConfig() or "
                             "DoryANEConfig.from_json(model_dir/config.json))")
        self.config = config
        self.prefix = prefix
        self.lut_bits = {
            "embeddings": _lut_value(lut_embeddings),
            "blocks": _lut_value(lut_blocks),
            "lm_head": _lut_value(lut_lm_head),
        }
        self.lut_per_channel = int(lut_per_channel)
        self.lm_head_splits = int(lm_head_splits)
        self.recurrent_splits = int(recurrent_splits)
        self.argmax = bool(argmax)
        self.check_placement_gate = bool(check_placement)
        self.verbose = bool(verbose)
        #: skip a function whose package already exists and verifies, so an
        #: interrupted run resumes at the pass it was killed in
        self.resume = bool(resume)
        from anemll.models.dory_chunks import lm_head_split_sizes
        self.lm_head_chunk_sizes = lm_head_split_sizes(self.config.vocab_size,
                                                       self.lm_head_splits)
        self.placement: dict = {}
        self.sizes: dict = {}
        self.seconds: dict = {}

    # -- plan --------------------------------------------------------------
    @property
    def specs(self) -> list:
        return component_specs(self.config, self.recurrent_splits)

    def _finished_package(self, spec: ComponentSpec, function: str):
        """Bytes of an existing package that still verifies, or None.

        Component conversion is two ~25-minute palettize passes; when a run is
        interrupted between them, redoing the finished one wastes the window.
        A package counts as finished only when it loads, matches the size
        prediction within 15 %, holds a single function and records the
        function its name claims.
        """
        from coremltools.models.utils import load_spec

        package = self.output_dir / self.file_name(spec, function)
        if not package.is_dir():
            return None
        try:
            spec_proto = load_spec(str(package))
        except Exception:
            return None
        bits = self.lut_bits.get(spec.lut_role)
        predicted = estimate_component_bytes(spec, self.config, bits)
        size = self._dir_size(package)
        if predicted and abs(size - predicted) > 0.15 * predicted:
            return None
        if len(list(spec_proto.description.functions)) > 1:
            return None
        recorded = str(spec_proto.description.metadata.userDefined.get(
            "com.anemll.functions", ""))
        return size if recorded == function else None

    def spec(self, name: str) -> ComponentSpec:
        for spec in self.specs:
            if spec.name == name:
                return spec
        raise KeyError(f"no component named {name}; known: {[s.name for s in self.specs]}")

    def file_name(self, spec: ComponentSpec, function: str = None) -> str:
        """Package name. One package per function: multifunction files are
        rejected by this macOS build (see docs/dory.md), so infer and prefill
        are separate files that each hold one function."""
        lut = _lut_value(self.lut_bits.get(spec.lut_role))
        suffix = f"_lut{lut}" if lut else ""
        tail = f"_{function}" if function else ""
        return f"{self.prefix}_{spec.name}{suffix}{tail}.mlpackage"

    def estimates(self) -> list:
        """(component, estimated bytes) for the whole conversion."""
        out = []
        for spec in self.specs:
            bits = self.lut_bits.get(spec.lut_role)
            out.append((spec, estimate_component_bytes(spec, self.config, bits)))
        return out

    def guard_sizes(self) -> dict:
        """Refuse the whole run before converting anything that cannot fit."""
        sizes = {}
        for spec, size in self.estimates():
            check_size(spec.name, size)
            sizes[spec.name] = size
        return sizes

    # -- model construction ------------------------------------------------
    def build_component(self, spec: ComponentSpec):
        """The component's modules plus the traceable wrapper around them."""
        indices = list(range(*spec.blocks)) if spec.kind == "blocks" else []
        model = _ComponentModel(self.config, indices,
                                load_lm_head=(spec.kind == "lm_head"))
        model.load_pretrained_weights(str(self.model_path), blocks=indices,
                                      load_lm_head=(spec.kind == "lm_head"))
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        model.model.reset_caches()

        if spec.kind == "embeddings":
            wrapper = dch.EmbeddingChunk(self.config, model.embed_tokens, model.model)
        elif spec.kind == "lm_head":
            wrapper = dch.LMHeadChunk(self.config, model.lm_head.weight.data,
                                      self.lm_head_splits, model.model,
                                      argmax=self.argmax)
            model.lm_head = None  # the weights are in the wrapper now
        else:
            # the signature has to be exactly spec.inputs / block_input_names order
            wrapper = dch.traced_block_chunk_class(spec.tiled, spec.rope_tables)(
                model.model, spec.blocks[0], spec.blocks[1],
                loop_entry=spec.loop_entry, loop_exit=spec.loop_exit)
        wrapper.eval()
        for param in wrapper.parameters():
            param.requires_grad = False
        return model, wrapper

    # -- tracing / conversion ---------------------------------------------
    def _example_inputs(self, spec: ComponentSpec, seq: int):
        """One zero tensor per declared input, in the declared order.

        int32 end to end for input_ids: torch's embedding accepts it (checked)
        and the Core ML input is declared int32, so there is no dtype cast to get
        wrong between the trace and the converted graph.
        """
        args = []
        for name in spec.inputs:
            shape, torch_dtype, _ = input_spec(name, seq, self.config)
            args.append(torch.zeros(*shape, dtype=torch_dtype))
        return tuple(args)

    def _coreml_inputs(self, spec: ComponentSpec, seq: int) -> list:
        types = []
        for name in spec.inputs:
            shape, _, ml_dtype = input_spec(name, seq, self.config)
            types.append(ct.TensorType(name=name, shape=shape, dtype=ml_dtype))
        return types

    def _coreml_outputs(self, spec: ComponentSpec) -> list:
        names = (self._lm_head_output_names() if spec.kind == "lm_head"
                 else ["output_hidden_states" if spec.kind == "blocks" else "hidden_states"])
        types = []
        for name in names:
            if name.startswith("argmax_idx"):
                types.append(ct.TensorType(name=name, dtype=np.int32))
            else:
                types.append(ct.TensorType(name=name, dtype=np.float16))
        return types

    def _lm_head_output_names(self) -> list:
        if self.argmax:
            names = []
            for i in range(1, self.lm_head_splits + 1):
                names.extend([f"argmax_idx_{i}", f"argmax_val_{i}"])
            return names
        return [f"logits_{i}" for i in range(1, self.lm_head_splits + 1)]

    def states(self, wrapper) -> list:
        """The state list every component declares: one buffer per cache, in order.

        dch.state_attrs derives the union from the config, so the untiled model
        declares the same two buffers an untiled conversion always did, and a
        tiled model declares one per tile plus the sliding cache.
        """
        states = []
        for attr in dch.state_attrs(self.config):
            buffer = getattr(wrapper.model, attr)
            states.append(ct.StateType(
                name=f"model.{attr}",
                wrapped_type=ct.TensorType(shape=tuple(int(d) for d in buffer.shape),
                                           dtype=np.float16)))
        return states

    @staticmethod
    def _reset_caches(module):
        with torch.no_grad():
            for name, buffer in module.named_buffers():
                if "kv_cache_" in name:
                    buffer.zero_()

    def trace(self, spec: ComponentSpec, wrapper, seq: int):
        """Trace one function of one component (ANEMLL's discipline)."""
        self._reset_caches(wrapper)
        example = self._example_inputs(spec, seq)
        with torch.no_grad():
            traced = torch.jit.trace(wrapper, example, strict=False)
        self._reset_caches(wrapper)
        self._reset_caches(traced)
        return traced

    def convert_function(self, spec: ComponentSpec, wrapper, seq: int):
        """Trace and convert one function; returns the unquantized MLModel."""
        traced = self.trace(spec, wrapper, seq)
        return ct.convert(
            traced,
            inputs=self._coreml_inputs(spec, seq),
            outputs=self._coreml_outputs(spec),
            states=self.states(wrapper),
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS18,
            convert_to="mlprogram",
        )

    def palettize(self, model, bits, label: str):
        """LUT-palettize weights after conversion; never silently skip it."""
        if bits is None:
            return model
        from coremltools.optimize.coreml import palettize_weights

        config = OptimizationConfig(global_config=OpPalettizerConfig(
            mode="kmeans", nbits=int(bits), granularity="per_grouped_channel",
            group_size=self.lut_per_channel, num_kmeans_workers=1))
        try:
            return palettize_weights(model, config)
        except ValueError as exc:
            if "Pool not running" not in str(exc):
                raise
            config.global_config.num_kmeans_workers = 1
            return palettize_weights(model, config)

    # -- one component end to end -----------------------------------------
    def convert_component(self, name: str) -> dict:
        """Convert one component: one package per function, sized and placement-gated."""
        spec = self.spec(name)
        bits = self.lut_bits.get(spec.lut_role)
        check_size(spec.name, estimate_component_bytes(spec, self.config, bits))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        model, wrapper = self.build_component(spec)
        emitted = {}
        try:
            for function, seq in self._functions(spec):
                if self.resume:
                    done = self._finished_package(spec, function)
                    if done is not None:
                        print(f"  {spec.name}/{function}: already built "
                              f"({done / 1024 ** 2:.0f} MB), skipped")
                        self.sizes[f"{spec.name}:{function}"] = done
                        emitted[function] = self.output_dir / self.file_name(spec, function)
                        continue
                converted = self.convert_function(spec, wrapper, seq)
                converted = self.palettize(converted, bits, f"{spec.name}/{function}")
                out_path = self.output_dir / self.file_name(spec, function)
                self._finalize_package(spec, function, converted, out_path)
                size = self._dir_size(out_path)
                check_size(f"{spec.name}/{function}", size)
                emitted[function] = out_path
                self.sizes[f"{spec.name}:{function}"] = size
                if self.verbose:
                    print(f"  {spec.name}/{function}: {out_path.name} "
                          f"{size / 1024 ** 2:.0f} MB")
        finally:
            del wrapper
            del model
        for function, out_path in emitted.items():
            self._gate_placement(spec, out_path, function)
        self.seconds[spec.name] = time.time() - started
        if self.verbose:
            total = sum(self.sizes[f"{spec.name}:{fn}"] for fn in emitted) / 1024 ** 2
            print(f"{spec.name}: {len(emitted)} packages, {total:.0f} MB total, "
                  f"{self.seconds[spec.name]:.0f}s")
        return emitted

    def _package_metadata(self, spec: ComponentSpec, function: str) -> dict:
        """The com.anemll.* metadata recorded inside one package."""
        return {
            "context_length": self.config.context_length,
            "state_length": self.config.state_length,
            "batch_size": self.config.batch_size,
            "num_chunks": len(self.specs),
            "lut_bits": _lut_value(self.lut_bits.get(spec.lut_role)),
            "function_names": [function],
            "vocab_size": self.config.vocab_size if spec.kind == "lm_head" else None,
            "lm_head_chunk_sizes": (self.lm_head_chunk_sizes
                                    if spec.kind == "lm_head" else None),
            "argmax_in_model": bool(self.argmax) if spec.kind == "lm_head" else False,
        }

    def _finalize_package(self, spec: ComponentSpec, function: str, model,
                          out_path: Path) -> None:
        """Attach metadata to the in-memory model and save it.

        No reload: the model that just came out of ct.convert + palettize is the
        one to write, so the save never depends on a package being loadable.
        """
        AddMetadata(model, self._package_metadata(spec, function))
        shutil.rmtree(out_path, ignore_errors=True)
        model.save(str(out_path))

    def _gate_placement(self, spec: ComponentSpec, out_path: Path,
                        function: str = None) -> None:
        """The spec's placement gate: refuse a package that is not all on the ANE."""
        label = f"{spec.name}/{function}" if function else spec.name
        if not self.check_placement_gate:
            print(f"WARNING: placement gate skipped for {label}; the spec "
                  f"requires every op on the Neural Engine", file=sys.stderr)
            return
        report = placement_report(out_path)
        self.placement.setdefault(spec.name, {})[function or "main"] = report
        assert_ane_only(report, f"{label} ({out_path.name})")

    def _functions(self, spec: ComponentSpec) -> list:
        """(function_name, sequence_length) pairs this component provides."""
        pairs = [("infer", 1)]
        if int(self.config.batch_size) > 1:
            pairs.append(("prefill", int(self.config.batch_size)))
        return pairs

    def _unused_combine(self, spec: ComponentSpec, saved: list, out_path: Path) -> None:
        """Retired: multifunction packages do not load on this OS (docs/dory.md)."""
        sources = [(str(path), "main", function) for function, path in saved]
        metadata = {
            "context_length": self.config.context_length,
            "state_length": self.config.state_length,
            "batch_size": self.config.batch_size,
            "num_chunks": len(self.specs),
            "lut_bits": _lut_value(self.lut_bits.get(spec.lut_role)),
            "function_names": [fn for fn, _ in saved],
            "vocab_size": self.config.vocab_size if spec.kind == "lm_head" else None,
            "lm_head_chunk_sizes": (self.lm_head_chunk_sizes
                                    if spec.kind == "lm_head" else None),
            "argmax_in_model": bool(self.argmax) if spec.kind == "lm_head" else False,
        }
        # Weld into a temp package, then load THAT and save to the real name.
        # coremltools 8.3's save_multifunction() returns None (it only writes
        # the package), and MLModel.save() deletes its destination before
        # copying, so saving a package onto itself would delete the source.
        temp_package = out_path.parent / (".combine_" + out_path.name)
        shutil.rmtree(temp_package, ignore_errors=True)
        shutil.rmtree(out_path, ignore_errors=True)
        if len(sources) > 1:
            with prepare_dedup_sources(sources) as deduped:
                descriptor = ct.utils.MultiFunctionDescriptor()
                for path, src, tgt in deduped:
                    descriptor.add_function(path, src, tgt)
                descriptor.default_function_name = sources[0][2]
                ct.utils.save_multifunction(descriptor, str(temp_package))
        else:
            shutil.copytree(sources[0][0], temp_package)
        combined = ct.models.MLModel(str(temp_package))
        if combined is None:
            raise RuntimeError(f"could not load the combined package {temp_package}")
        AddMetadata(combined, metadata)
        combined.save(str(out_path))
        shutil.rmtree(temp_package, ignore_errors=True)

    @staticmethod
    def _dir_size(path: Path) -> int:
        path = Path(path)
        if path.is_file():
            return path.stat().st_size
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())

    # -- deliverables ------------------------------------------------------
    def compiled_name(self, spec: ComponentSpec, function: str = None) -> str:
        return self.file_name(spec, function)[: -len(".mlpackage")] + ".mlmodelc"

    def compile_packages(self) -> dict:
        """Compile every package to .mlmodelc (the cached form the runtime loads)."""
        done = {}
        for spec in self.specs:
            for function, _ in self._functions(spec):
                package = self.output_dir / self.file_name(spec, function)
                if not package.exists():
                    continue
                started = time.time()
                result = subprocess.run(
                    ["xcrun", "coremlcompiler", "compile", str(package),
                     str(self.output_dir)],
                    capture_output=True, text=True)
                key = f"{spec.name}:{function}"
                if result.returncode != 0:
                    raise RuntimeError(f"coremlcompiler failed for {package.name}: "
                                       f"{result.stderr.strip()[:400]}")
                done[key] = {"seconds": time.time() - started,
                             "path": str(self.output_dir / self.compiled_name(spec, function))}
                if self.verbose:
                    print(f"compiled {self.compiled_name(spec, function)} "
                          f"in {done[key]['seconds']:.1f}s")
        return done

    TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                       "chat_template.jinja", "chat_template.json", "added_tokens.json",
                       "special_tokens_map.json", "vocab.json", "merges.txt",
                       "generation_config.json")

    def copy_tokenizer(self) -> list:
        """Copy the tokenizer and chat template next to the models (meta points at them).

        A chat_template.jinja in the weights directory is copied; failing that,
        a chat_template embedded in tokenizer_config.json is written out as one,
        so the artifact always says how prompts were formatted (or the meta
        records that it had to fall back -- see FALLBACK_TEMPLATE).
        """
        copied = []
        for name in self.TOKENIZER_FILES:
            source = self.model_path / name
            if not source.is_file():
                continue
            target = self.output_dir / name
            if source.resolve() == target.resolve():
                copied.append(name)
                continue
            shutil.copy2(source, target)
            copied.append(name)
        template_path = self.output_dir / "chat_template.jinja"
        if not template_path.is_file():
            config_path = self.model_path / "tokenizer_config.json"
            if config_path.is_file():
                try:
                    with open(config_path, "r", encoding="utf-8") as handle:
                        raw = json.load(handle)
                except (OSError, ValueError):
                    raw = {}
                template = raw.get("chat_template")
                if isinstance(template, list) and template:
                    template = template[0].get("template") if isinstance(template[0], dict) \
                        else template[0]
                if isinstance(template, str) and template.strip():
                    template_path.write_text(template, encoding="utf-8")
                    copied.append("chat_template.jinja")
        return copied

    def manifest_files(self) -> dict:
        """Component -> {function: the file the runtime should load}.

        The compiled form is preferred when it exists (it is the cached compile
        the runtime benefits from); otherwise the package, which Core ML
        compiles on first load.
        """
        files = {}
        for spec in self.specs:
            entry = {}
            for function, _ in self._functions(spec):
                compiled = self.output_dir / self.compiled_name(spec, function)
                entry[function] = (self.compiled_name(spec, function) if compiled.exists()
                                   else self.file_name(spec, function))
            files[spec.name] = entry
        return files

    def write_artifacts(self) -> dict:
        """Everything the runtime directory needs: tokenizer, config, meta, report."""
        copied = self.copy_tokenizer()
        source_config = self.model_path / "config.json"
        if source_config.is_file():
            shutil.copy2(source_config, self.output_dir / "config.json")
        files = self.manifest_files()
        if tiling_flags(self.config)[1]:
            table_path = write_rope_tables(self.config, self.output_dir)
            if self.verbose:
                print(f"wrote {table_path.name}: "
                      f"{table_path.stat().st_size / 1024 ** 2:.1f} MB", file=sys.stderr)
        meta = build_meta(
            config=self.config, specs=self.specs, files=files,
            lut_bits=self.lut_bits, lm_head_chunk_sizes=self.lm_head_chunk_sizes,
            prefix=self.prefix, argmax=self.argmax,
            tokenizer="tokenizer.json",
            chat_template=("chat_template.jinja" if "chat_template.jinja" in copied
                           else None))
        yaml_path = write_meta(self.output_dir / "meta.yaml", meta)
        report_path = self.write_report(meta, copied)
        return {"meta": str(yaml_path), "report": str(report_path)}

    def write_report(self, meta: dict, tokenizer_files: list) -> Path:
        """conversion_report.json: sizes, timings and per-function placement."""
        report = {
            "components": {name: {
                "bytes": {fn: self.sizes.get(f"{name}:{fn}") for fn in meta["components"][name]["files"]},
                "seconds": self.seconds.get(name),
                "placement": self.placement.get(name)} for name in meta["run_order"]},
            "estimates": {spec.name: estimate_component_bytes(
                spec, self.config, self.lut_bits.get(spec.lut_role))
                for spec in self.specs},
            "tokenizer_files": list(tokenizer_files),
            "created": meta["model_info"]["created"],
            "config": {
                "context_length": self.config.context_length,
                "batch_size": self.config.batch_size,
                "lut": {role: _lut_value(bits) for role, bits in self.lut_bits.items()},
                "lm_head_splits": self.lm_head_splits,
                "recurrent_splits": self.recurrent_splits,
            },
        }
        path = self.output_dir / "conversion_report.json"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        return path

    def cache_summary_line(self) -> str:
        """One line about the declared states, for the conversion plan.

        The five-component path declares the whole union in every file, which is
        what dch.state_summary describes; the per-program path (dory_per_head)
        overrides this because there each program declares a slice of it.
        """
        return dch.state_summary(self.config)

    # -- the whole run -----------------------------------------------------
    def convert(self, component="all"):
        """BaseConverter entry point: 'all' or a component name."""
        if component in ("all", None):
            return self.run()
        return self.convert_component(component)

    def run(self, only=None, dry_run=False, do_compile=True) -> dict:
        """Convert every component, compile, and write meta.yaml.

        A dry run writes nothing at all: it prints the plan, the per-file size
        guard's verdict and the disk check, and returns.
        """
        wanted = None if not only else {name.strip() for name in str(only).split(",")}
        plan = [(spec, estimate_component_bytes(spec, self.config,
                                                self.lut_bits.get(spec.lut_role)))
                for spec in self.specs if wanted is None or spec.name in wanted]
        if self.verbose:
            print(f"Dory conversion plan ({len(plan)} components, context "
                  f"{self.config.context_length}, batch {self.config.batch_size}):")
            for spec, size in plan:
                print(f"  {spec.name:14s} {size / 1024 ** 2:8.0f} MB estimated"
                      f"  blocks={spec.blocks}  lut={self.lut_bits.get(spec.lut_role)}"
                      f"  calls/token={spec.calls_per_token}")
            tiled, rope = tiling_flags(self.config)
            print(f"  caches: {self.cache_summary_line()}")
            print(f"  mask_len={int(self.config.mask_len)}")
            print(f"  tiled={tiled}  rope_tables_as_inputs={rope}")
        for spec, size in plan:
            check_size(spec.name, size)
        disk = check_disk(nearest_existing(self.output_dir), plan)
        if self.verbose:
            print(f"disk: {disk['free'] / 1024 ** 3:.1f} GB free, "
                  f"{disk['needed'] / 1024 ** 3:.1f} GB needed")
        if dry_run:
            return {"dry_run": True, "estimates": {s.name: n for s, n in plan},
                    "disk": disk, "components": {s.name: n for s, n in plan}}

        self.output_dir.mkdir(parents=True, exist_ok=True)
        started = time.time()
        for spec, _ in plan:
            if self.verbose:
                print(f"converting {spec.name} ...")
            self.convert_component(spec.name)
        compiled = self.compile_packages() if do_compile else {}
        artifacts = self.write_artifacts()
        total = time.time() - started
        if self.verbose:
            print(f"conversion finished in {total:.0f}s; meta at {artifacts['meta']}")
        return {"components": {spec.name: self.sizes.get(spec.name) for spec, _ in plan},
                "compiled": compiled, "seconds": total, **artifacts}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_tile_rows(value):
    """'16384,8192' -> (16384, 8192), or None for the untiled layout."""
    if value in (None, "", "none", "None"):
        return None
    if isinstance(value, (tuple, list)):
        return tuple(int(v) for v in value)
    try:
        rows = tuple(int(part) for part in str(value).replace(" ", "").split(",") if part)
    except ValueError as exc:
        raise SystemExit(f"not a tile layout: {value!r} (want rows like 16384,8192)") from exc
    if not rows or any(row <= 0 for row in rows):
        raise SystemExit(f"not a tile layout: {value!r} (every tile needs positive rows)")
    return rows


def build_config(model_path, context_length=None, batch_size=None, config=None,
                 full_tile_rows=None, rope_tables_as_inputs=False,
                 heads_per_state=None, slots_per_sliding_carrier=None):
    """DoryANEConfig from the model directory's config.json, with overrides."""
    model_path = Path(model_path)
    config_json = model_path / "config.json"
    overrides = {}
    if context_length is not None:
        overrides["context_length"] = int(context_length)
    if batch_size is not None:
        overrides["batch_size"] = int(batch_size)
    tiles = parse_tile_rows(full_tile_rows)
    if tiles is not None:
        overrides["full_cache_tiles"] = tiles
    if rope_tables_as_inputs:
        overrides["rope_tables_as_inputs"] = True
    if heads_per_state is not None:
        overrides["heads_per_state"] = int(heads_per_state)
    if slots_per_sliding_carrier is not None:
        overrides["slots_per_sliding_carrier"] = int(slots_per_sliding_carrier)
    if config is not None:
        return config
    if config_json.is_file():
        return dam.DoryANEConfig.from_json(str(config_json), **overrides)
    return dam.DoryANEConfig(**overrides)


def parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert Dory1.0-4B into the five multifunction Core ML chunks "
                    "of spec D6 (embeddings, input, recurrent, output, split LM head).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", required=True, help="directory with the BF16 safetensors")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--prefix", default="dory", help="file-name prefix")
    parser.add_argument("--context-length", type=int, default=1024,
                        help="rows of a full-attention cache (tracer bullet: 1024)")
    parser.add_argument("--batch-size", type=int, default=64, help="prefill batch size")
    parser.add_argument("--lut-blocks", default="6", help="LUT bits for the block chunks (none to skip)")
    parser.add_argument("--lut-lm-head", default="6", help="LUT bits for the LM head")
    parser.add_argument("--lut-embeddings", default="none",
                        help="LUT bits for the embeddings (spec D8: unquantized or 8)")
    parser.add_argument("--lut-per-channel", type=int, default=8,
                        help="per_grouped_channel group size")
    parser.add_argument("--lm-head-splits", type=int, default=8,
                        help="even LM-head parts (vocab 131072: 8 x 16384)")
    parser.add_argument("--recurrent-splits", type=int, default=1,
                        help="split the recurrent block into N files if it exceeds 2 GB")
    parser.add_argument("--argmax", action="store_true",
                        help="return local argmax indices from the LM head instead of logits")
    parser.add_argument("--no-placement-check", action="store_true",
                        help="skip the per-function ANE placement gate (never do this for a gate)")
    parser.add_argument("--no-compile", action="store_true", help="skip the .mlmodelc step")
    parser.add_argument("--only", default=None,
                        help="comma-separated component names (default: all)")
    parser.add_argument("--full-tile-rows", default=None,
                        help="split the full-attention caches into tiles (D7 fallback), "
                             "e.g. '16384,8192'; unset keeps one cache per attention block")
    parser.add_argument("--rope-tables-as-inputs", action="store_true",
                        help="pass the RoPE cos/sin rows as graph inputs instead of tracing "
                             "them as constants (24K: the tables are ~100 MB of constants)")
    parser.add_argument("--heads-per-state", type=int, default=None,
                        help="split every KV cache into one state per group of this many "
                             "key/value heads (DANE-5: 1 gives 36 full states of 24.0 MiB "
                             "at 24,576 rows, under the measured 24 MiB per-state ceiling); "
                             "unset keeps the aggregated caches")
    parser.add_argument("--slots-per-sliding-carrier", type=int, default=None,
                        help="how many sliding layer-slots share one state (default 3: "
                             "9 carriers of 15.0 MiB at 24,576 rows, because the state "
                             "COUNT has its own ceiling; --per-head-programs needs 1, "
                             "because one program per sliding layer owns its carrier)")
    parser.add_argument("--per-head-programs", action="store_true",
                        help="emit the per-(layer, head) plan instead of the five "
                             "components: one program per full-attention (layer, KV "
                             "head) holding exactly one 24.0 MiB state, one sliding "
                             "attention program per sliding layer holding its 5.0 MiB "
                             "carrier (attention only, no gate and no tail), the "
                             "unstated slide epilogue that finishes each of those layers "
                             "(gate, out_norm, o_proj, the MLP, the next gate out), and "
                             "unstated epilogue/MLP programs for the full-attention "
                             "layers (needed above ~16K context; --heads-per-state 1 and "
                             "--slots-per-sliding-carrier 1 are implied)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and the size estimates, convert nothing")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    heads_per_state = args.heads_per_state
    slots_per_carrier = args.slots_per_sliding_carrier
    if args.per_head_programs:
        if heads_per_state not in (None, 1):
            print(f"stopped: --per-head-programs declares one (layer, head) cache per "
                  f"state, so it needs --heads-per-state 1, not {heads_per_state}",
                  file=sys.stderr)
            return 2
        heads_per_state = 1
        # one program per sliding layer owns that layer's carrier, so a carrier
        # shared by three layers would be declared by three programs
        if slots_per_carrier is None:
            slots_per_carrier = 1
        elif slots_per_carrier != 1:
            print(f"stopped: --per-head-programs gives every sliding layer its own "
                  f"program, which declares that layer's carrier, so it needs "
                  f"--slots-per-sliding-carrier 1, not {slots_per_carrier}",
                  file=sys.stderr)
            return 2
    config = build_config(args.model, args.context_length, args.batch_size,
                          full_tile_rows=args.full_tile_rows,
                          rope_tables_as_inputs=args.rope_tables_as_inputs,
                          heads_per_state=heads_per_state,
                          slots_per_sliding_carrier=slots_per_carrier)
    converter_class = DoryConverter
    if args.per_head_programs:
        from anemll.ane_converter.dory_per_head import PerHeadConverter
        converter_class = PerHeadConverter
    converter = converter_class(
        model_path=args.model, output_dir=args.output, config=config, prefix=args.prefix,
        lut_blocks=args.lut_blocks, lut_lm_head=args.lut_lm_head,
        lut_embeddings=args.lut_embeddings, lut_per_channel=args.lut_per_channel,
        lm_head_splits=args.lm_head_splits, recurrent_splits=args.recurrent_splits,
        argmax=args.argmax, check_placement=not args.no_placement_check)
    try:
        result = converter.run(only=args.only, dry_run=args.dry_run,
                               do_compile=not args.no_compile)
    except ModelTooLargeError as exc:
        print(f"stopped: {exc}", file=sys.stderr)
        return 2
    except PlacementError as exc:
        print(f"placement gate failed: {exc}", file=sys.stderr)
        return 3
    if not args.dry_run:
        print(json.dumps({"components": {k: v for k, v in result.get("components", {}).items()}},
                         indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
