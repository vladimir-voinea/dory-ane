"""The traceable pieces a Dory Core ML chunk is made of.

Spec D6 (five components), D5 (ANE-safe form), D7 (caches).

The math lives in :mod:anemll.models.dory_ane_model (ticket DANE-4) and is not
repeated here.  This module only *wraps* that math into the shapes Core ML
traces: fixed input/output names, a batch dimension of 1 (decode) or
batch_size (prefill), and the union-of-states rule.

Union of states
---------------
The runtime creates **one** MLState from one component and passes it to all
five (ANEMLL tests/chat.py create_unified_state).  Core ML silently falls back
to per-chunk states when the state declarations differ, which for Dory would
stop the input block's cache writes from reaching the recurrent block.  Every
wrapper here therefore carries the *same* two buffers -- the 9 full-attention
caches and the 27 sliding ones -- and adds '0 * state[0,0,0,0]' into its output
so Core ML does not prune an unused one.  That is the Gemma 3 trick
(gemma3_converter.py:341-371, :1474-1481), extended to two buffers.

Component I/O
-------------
============ ================================ =============================
component    inputs                            output
============ ================================ =============================
embeddings   input_ids (1,seq) int32           hidden_states (1,seq,2560)
input        hidden_states, causal_mask,       output_hidden_states
             current_pos, cache_pos
recurrent    ... plus h_in                     output_hidden_states
output       same as input                     output_hidden_states
lm_head      hidden_states                     logits_1..N
============ ================================ =============================

seq is 1 for the infer function and batch_size for prefill.  current_pos is the
position of the first row (it selects the RoPE rows), cache_pos is where those
rows are written in the ring caches.  Both are int32 (1,) inputs, exactly like
ANEMLL's current_pos.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from anemll.models.dory_ane_model import (
    DoryANEChunk,
    DoryANEModel,
    _own_caches,
    q_len_of,
)

__all__ = [
    "STATE_ATTRS",
    "state_attrs",
    "state_names",
    "block_input_names",
    "BlockChunk",
    "TiledBlockChunk",
    "RopeBlockChunk",
    "TiledRopeBlockChunk",
    "traced_block_chunk_class",
    "program_state_names",
    "program_state_bytes",
    "HeadAttentionChunk",
    "RopeHeadAttentionChunk",
    "CONTRIBUTIONS",
    "contribution_names",
    "SlidingAttentionChunk",
    "RopeSlidingAttentionChunk",
    "WideSlidingAttentionChunk",
    "RopeWideSlidingAttentionChunk",
    "SlidingGroupChunk",
    "RopeSlidingGroupChunk",
    "SlidingLayerChunk",
    "RopeSlidingLayerChunk",
    "EntryGateChunk",
    "sliding_layer_input_names",
    "FullEpilogueChunk",
    "SlidingEpilogueChunk",
    "WideSlidingEpilogueChunk",
    "EmbeddingChunk",
    "LMHeadChunk",
    "empty_state_carrier",
    "lm_head_split_sizes",
]

#: The two state buffers of an untiled model, in declaration order.
STATE_ATTRS = ("kv_cache_full", "kv_cache_sliding")


def _caches():
    """dory_caches, imported lazily.

    This module deliberately imports nothing at module scope (the extracted
    build loads it before its siblings exist), so the shape helpers are fetched
    on first use instead of bound at import time.
    """
    from anemll.utils import dory_caches

    return dory_caches


def state_layout(config=None) -> tuple:
    """Every state buffer as (name, shape), in declaration order.

    Three axes decide the list, and only the first exists by default:

      tiles  the tiled full-attention fallback (spec D7) splits the full cache
             by context rows, one buffer per tile (DANE-6's path).
      heads  the measured single-state ceiling on this host -- 24 MiB loads,
             32 MiB fails, 3/3 attempts each -- splits each cache by key/value
             head, so one buffer holds (2 slots, heads_per_state, rows, head_dim).
      slots  the aggregate buffer interleaves K and V along dim 0: K at slot,
             V at slot + num_slots, as dory_caches.write_cache_rows expects.

    Default: no tiles, no head split -> exactly the two buffers this module
    declared before, byte for byte, so nothing that exists today changes shape.
    Tiles and head-splitting are mutually exclusive until the tiled path learns
    the head axis: both at once would produce a list neither caller understands,
    so asking for it raises instead of guessing.
    """
    tiles = getattr(config, "full_tiles", (1,)) if config is not None else (1,)
    heads_per_state = getattr(config, "heads_per_state", None) if config is not None else None
    if heads_per_state is None and config is not None:
        heads_per_state = getattr(config, "cache_heads_per_state", None)
    if heads_per_state is None:
        if len(tiles) == 1:
            full = [("kv_cache_full", _caches().cache_shapes(
                config.context_length, config.num_key_value_heads, config.head_dim,
                config.sliding_window, config.batch_size)[0])]
        else:
            shapes = _caches().full_cache_shapes(
                config.context_length, config.num_key_value_heads, config.head_dim,
                tiles, num_slots=len(_caches().cache_slots()[0]))
            full = [(f"kv_cache_full_{index}", shape)
                    for index, shape in enumerate(shapes, start=1)]
        _, sliding_shape = _caches().cache_shapes(
            config.context_length, config.num_key_value_heads, config.head_dim,
            config.sliding_window, config.batch_size)
        return tuple(full + [("kv_cache_sliding", sliding_shape)])

    if len(tiles) > 1:
        raise ValueError(
            "the tiled full cache and the per-head cache split cannot be combined yet: "
            "tiles split the full cache by context rows, heads split every cache by key "
            "head, and a model needs one of them at a time until the tiled path carries "
            "the head axis too.")

    heads = int(heads_per_state)
    num_heads = int(config.num_key_value_heads)
    if heads < 1 or num_heads % heads:
        raise ValueError(f"cache_heads_per_state must divide {num_heads} heads, got {heads}")
    full_slots, sliding_slots = _caches().cache_slots()
    full_rows = _caches().full_cache_rows(config.context_length)
    sliding_rows = _caches().sliding_cache_rows(config.context_length, config.sliding_window,
                                         config.batch_size)
    head_dim = int(config.head_dim)
    layout = []
    for slot in range(len(full_slots)):
        for head in range(0, num_heads, heads):
            layout.append((f"kv_cache_full_{slot}_{head}",
                           (2, heads, full_rows, head_dim)))
    # The sliding cache is windowed, so its carriers are grouped by SLOT, not by
    # head: one carrier holds a whole layer's K and V for every head (4 heads x
    # 1,280 rows = 5.0 MiB per slot at 24K), and the group size is the largest
    # divisor of the slot count whose carrier still fits under the measured
    # per-state ceiling -- 27 slots become 9 carriers of 15.0 MiB rather than 27
    # carriers of 5.0 MiB, because the state COUNT has its own ceiling (56 loads,
    # 63 fails) and 36 full + 27 sliding = 63 would sit exactly on the failure.
    slots_total = len(sliding_slots)
    group = int(getattr(config, "slots_per_sliding_carrier", 3))
    if group < 1 or slots_total % group:
        raise ValueError(
            f"slots_per_sliding_carrier must be a positive divisor of {slots_total} "
            f"(the number of sliding layer-slots), got {group}.  Each carrier holds "
            f"K and V of {group} layer(s); the per-state ceiling (24 MiB loads, 32 MiB "
            f"fails) caps a carrier at "
            f"{int(24.0 // (2 * sliding_rows * head_dim * num_heads * 2 / 1024 ** 2))} "
            f"layer(s) at this context length.")
    for start in range(0, slots_total, group):
        count = min(group, slots_total - start)
        layout.append((f"kv_cache_sliding_{start}_{start + count - 1}",
                       (2 * count, num_heads, sliding_rows, head_dim)))
    return tuple(layout)


def program_state_names(config, *, blocks=(), kv_head=None) -> tuple:
    """The states ONE program declares, selected out of state_layout.

    The per-(layer, head) path gives every program its own buffers instead of the
    union every component declares today: a full-attention head program holds
    exactly its (layer-slot, head) buffer -- 24.0 MiB at 24,576 rows, the one
    shape that fits under the measured per-state ceiling -- and a sliding group
    holds the carriers covering the sliding slots of the blocks it computes.

    Selection is by NAME out of state_layout(config), not by re-deriving shapes
    here: a program can only declare a buffer the model, the converter and the
    runtime all agree on, and a layout knob that renames or regroups the carriers
    (heads_per_state, slots_per_sliding_carrier) changes this list with it.

    blocks=() with kv_head=None is an unstated program (the epilogue, the
    embeddings, the LM head): it declares nothing and carries no cache.
    """
    layout = state_layout(config)
    known = {name for name, _ in layout}
    slots = _caches().cache_slots()
    full_slots, sliding_slots = slots
    is_full = _caches().block_is_full_attention
    is_sliding = _caches().block_is_sliding_attention
    wanted = []
    for block in tuple(blocks):
        if is_full(block):
            if kv_head is None:
                continue          # the epilogue computes the tail, not a cache
            wanted.append(f"kv_cache_full_{full_slots[block]}_{int(kv_head)}")
        elif is_sliding(block):
            slot = sliding_slots[block]
            for name, _shape in layout:
                if not name.startswith("kv_cache_sliding_"):
                    continue
                first, last = name.rsplit("_", 2)[-2:]
                if int(first) <= slot <= int(last) and name not in wanted:
                    wanted.append(name)
    missing = [name for name in wanted if name not in known]
    if missing:
        raise ValueError(
            f"program states {missing[:4]} are not in this conversion's state layout "
            f"(heads_per_state={getattr(config, 'heads_per_state', None)!r}, "
            f"slots_per_sliding_carrier="
            f"{getattr(config, 'slots_per_sliding_carrier', 3)!r}); the per-head path "
            f"needs the split layout, e.g. --heads-per-state 1")
    return tuple(wanted)


def program_state_bytes(config, names) -> list:
    """[(name, shape, MiB)] for the states a program declares, for the report."""
    shapes = dict(state_layout(config))
    out = []
    for name in names:
        shape = shapes[name]
        elements = 1
        for dim in shape:
            elements *= int(dim)
        out.append((name, tuple(int(d) for d in shape), elements * 2 / 1024 ** 2))
    return out


def state_attrs(config=None) -> tuple:
    """Every state buffer a component declares, in declaration order.

    The union is derived from the config rather than hard-coded: the converter,
    the runtime's single MLState and every component must agree on the list or
    Core ML refuses the shared state.  See state_layout for the axes.
    """
    return tuple(name for name, _ in state_layout(config))


def state_summary(config=None, per_program: bool = False) -> str:
    """One line for the conversion report and the meta: sizes and the ceiling.

    Anyone who later sees a -14 on this model should find these numbers next to
    the artifact rather than re-deriving them from shapes.

    per_program is for the one-program-per-(layer, head|layer) path, where no
    program ever declares the union: the COUNT of buffers stops being a limit
    (each program declares one) and only the per-state size is still a ceiling.
    """
    layout = state_layout(config)
    sizes = []
    for _, shape in layout:
        elements = 1
        for dim in shape:
            elements *= int(dim)
        sizes.append(elements * 2 / 1024 ** 2)
    total = sum(sizes)
    largest = max(sizes) if sizes else 0.0
    smallest = min(sizes) if sizes else 0.0
    if per_program:
        verdict = "OK" if largest <= 24.0 else "OVER THE MEASURED LIMIT"
        return ("%d states, %.1f-%.1f MiB each, %.1f MiB total, max single state "
                "%.1f MiB (per-program path: each program declares its own state, so "
                "the 24 MiB per-state ceiling is the only limit): %s"
                % (len(layout), smallest, largest, total, largest, verdict))
    verdict = "OK" if largest <= 24.0 and len(layout) <= 56 else "OVER THE MEASURED LIMIT"
    return ("%d states, %.1f-%.1f MiB each, %.1f MiB total, max single state "
            "%.1f MiB / %d states (limits measured at 24 MiB / 56 states OK, "
            "32 MiB / 63 states FAIL): %s"
            % (len(layout), smallest, largest, total, largest, len(layout), verdict))


def state_names(config=None, prefix: str = "model") -> tuple:
    """Qualified names of the states, as Core ML sees them.

    coremltools 8.3 renames dots inside a state name (model.kv_cache_full ->
    model_kv_cache_full) but consistently, so the union works as long as every
    component is built by this module.
    """
    return tuple(f"{prefix}.{attr}" for attr in state_attrs(config))


def block_input_names(tiled: bool, loop_entry: bool, rope_tables: bool = False,
                      rope_kinds=("sliding", "full")) -> list:
    """The traced input names of a block component, in signature order.

    The converter, the traced wrapper and the runtime's call must agree on this
    order exactly: Core ML maps the declared inputs positionally onto the traced
    graph.  A tiled full cache needs the per-tile write index and gate, because
    a traced graph cannot branch on the position (see BlockChunk).

    rope_kinds narrows the row tables to the ones a program actually uses -- a
    sliding group takes the sliding table, a per-(layer, head) full-attention
    program the full one.  A declared-but-unused input would be pruned by the
    converter, so the declaration has to match the graph.
    """
    names = ["hidden_states", "causal_mask", "current_pos", "cache_pos"]
    if tiled:
        names += ["write_index_1", "write_gate_1", "write_index_2", "write_gate_2"]
    if rope_tables:
        # the cos/sin tensors the host slices from its own tables (24K)
        for kind in rope_kinds:
            names += [f"cos_{kind}", f"sin_{kind}"]
    if loop_entry:
        names.append("h_in")
    return names


def sliding_layer_input_names(loop_entry: bool, rope_tables: bool) -> list:
    """The traced inputs of a one-sliding-layer program, in signature order.

    The gate sits fifth, before the optional ones, because it is the one input
    every such program has and the one the wrapper must receive positionally:
    torch.jit.trace maps the example tuple onto the signature by position, so
    the declaration here and SlidingLayerChunk.forward have to agree exactly.
    A declared-but-unused input is pruned by the converter, so the RoPE rows and
    h_in appear only when the program actually uses them.
    """
    names = ["hidden_states", "causal_mask", "current_pos", "cache_pos", "gate"]
    if rope_tables:
        names += ["cos_sliding", "sin_sliding"]
    if loop_entry:
        names.append("h_in")
    return names


class _StateTouching(nn.Module):
    """Base class for wrappers that keep the full state union alive.

    Subclasses set self.model to the DoryANEModel that owns the cache buffers
    (an empty one for the embedding/LM-head components).  The buffers must be
    the *module's own* registered buffers, not copies: Core ML matches states by
    the attribute path from the traced root.
    """

    model: DoryANEModel

    def touch_states(self, out):
        """Add 0 * every state, so none is pruned as unused.

        The list comes from the model's own buffers rather than a hard-coded
        pair: the split layout declares 45 of them, and one left out is a state
        Core ML would drop from the union the runtime shares between components.

        A program that emits several tensors (the sliding attention program's one
        contribution per KV head) touches each of them: every output has to depend
        on the state, or the state is pruned from that output's branch of the
        graph.  Adding zero does not move a number, so the slices stay the
        aggregated tensor's channel blocks.
        """
        buffers = self.model.state_buffers
        if not buffers:
            # an unstated program (epilogue, MLP, embeddings in the per-program
            # path) declares nothing, so there is no state to keep alive
            return out
        dummy = buffers[0][0, 0, 0, 0] * 0.0
        for cache in buffers[1:]:
            dummy = dummy + cache[0, 0, 0, 0] * 0.0
        dummy = dummy.view(1, 1, 1)
        if isinstance(out, tuple):
            return tuple(one + dummy for one in out)
        return out + dummy


def empty_state_carrier(config) -> DoryANEModel:
    """A DoryANEModel with no blocks -- states only.

    The embedding and LM-head components compute nothing on a cache, but they
    must still declare both to stay in the union.
    """
    return DoryANEModel(config, block_indices=[])


class BlockChunk(_StateTouching):
    """A contiguous range of physical blocks (input / recurrent / output).

    loop_entry / loop_exit are the recurrent block's entry and exit norms
    (h <- L2(h + h_in) and the trailing L2).  They are compile-time flags, so
    splitting the recurrent block across two files keeps exactly one entry and
    one exit.
    """

    def __init__(self, model, start_block: int, end_block: int,
                 loop_entry: bool = False, loop_exit: bool = False):
        super().__init__()
        self.model = model
        self.chunk = DoryANEChunk(model, start_block, end_block,
                                  loop_entry=loop_entry, loop_exit=loop_exit)
        self.start_block = int(start_block)
        self.end_block = int(end_block)
        self.loop_entry = bool(loop_entry)
        self.loop_exit = bool(loop_exit)

    def run(self, hidden_states, causal_mask, current_pos, cache_pos, h_in=None,
            write_index_1=None, write_gate_1=None, write_index_2=None, write_gate_2=None,
            cos_sliding=None, sin_sliding=None, cos_full=None, sin_full=None):
        out = self.chunk(hidden_states, causal_mask, current_pos, cache_pos, h_in,
                         write_index_1, write_gate_1, write_index_2, write_gate_2,
                         cos_sliding=cos_sliding, sin_sliding=sin_sliding,
                         cos_full=cos_full, sin_full=sin_full)
        return self.touch_states(out)

    def forward(self, hidden_states, causal_mask, current_pos, cache_pos, h_in=None):
        return self.run(hidden_states, causal_mask, current_pos, cache_pos, h_in)


class TiledBlockChunk(BlockChunk):
    """The same chunk, traced with the per-tile write index and gate.

    A separate class rather than an optional argument: the traced signature has
    to place every tensor input first (torch.jit.trace cannot be handed a None
    placeholder), while the eager callers keep h_in as the fifth positional
    argument.  block_input_names() is the single source of truth for the order.
    """

    def forward(self, hidden_states, causal_mask, current_pos, cache_pos,
                write_index_1, write_gate_1, write_index_2, write_gate_2, h_in=None):
        return self.run(hidden_states, causal_mask, current_pos, cache_pos, h_in,
                        write_index_1, write_gate_1, write_index_2, write_gate_2)


class RopeBlockChunk(BlockChunk):
    """Traced with the RoPE cos/sin rows as inputs (the 24K shape); h_in last.

    A traced signature has to put every tensor input first, because
    torch.jit.trace cannot be handed a None placeholder, so each present group of
    inputs gets a class whose parameter order is exactly block_input_names().
    h_in stays the fifth positional argument of the plain BlockChunk, which is
    where the eager callers and the block tests pass it.
    """

    def forward(self, hidden_states, causal_mask, current_pos, cache_pos,
                cos_sliding, sin_sliding, cos_full, sin_full, h_in=None):
        return self.run(hidden_states, causal_mask, current_pos, cache_pos, h_in,
                        cos_sliding=cos_sliding, sin_sliding=sin_sliding,
                        cos_full=cos_full, sin_full=sin_full)


class TiledRopeBlockChunk(BlockChunk):
    """Both: the tile write inputs and the RoPE rows, h_in last."""

    def forward(self, hidden_states, causal_mask, current_pos, cache_pos,
                write_index_1, write_gate_1, write_index_2, write_gate_2,
                cos_sliding, sin_sliding, cos_full, sin_full, h_in=None):
        return self.run(hidden_states, causal_mask, current_pos, cache_pos, h_in,
                        write_index_1, write_gate_1, write_index_2, write_gate_2,
                        cos_sliding=cos_sliding, sin_sliding=sin_sliding,
                        cos_full=cos_full, sin_full=sin_full)


def traced_block_chunk_class(tiled: bool, rope_tables: bool):
    """The wrapper whose signature is exactly block_input_names(tiled, _, rope)."""
    if tiled and rope_tables:
        return TiledRopeBlockChunk
    if tiled:
        return TiledBlockChunk
    if rope_tables:
        return RopeBlockChunk
    return BlockChunk


# ---------------------------------------------------------------------------
# The per-(layer, head) program path (DANE-5, 24,576 tokens)
#
# One program per (layer, head) full-attention cache is what the measured
# 24 MiB-per-state ceiling forces: 24,576 rows x 256 dims of K and V is 24.0 MiB,
# and a program may hold ~34 MiB.  The wrappers below are that decomposition --
# a head attention program, a sliding group, and the unstated epilogue that
# finishes the layer the per-head programs cannot (the L2 norm and o_proj mix all
# 16 heads, so they cannot live in a one-head program).
# ---------------------------------------------------------------------------

class HeadAttentionChunk(_StateTouching):
    """One (layer, KV head) attention program, holding exactly one cache.

    Weights are this head's rows of the layer's projections (q/g: rep query
    heads, k/v: one KV head), never the whole layer, and the module is the same
    DoryANEAttention whose group loop the aggregated layout runs -- the program
    is that loop's single pass, not a second implementation of attention.

    Output is the head's gated contribution to the layer's 4096-wide attention
    output: (1, seq, rep * head_dim), exactly the slice the aggregated path
    produces for this head, in the same channel order.
    """

    def __init__(self, model, block: int, kv_head: int):
        super().__init__()
        self.model = model
        self.block = int(block)
        self.kv_head = int(kv_head)
        matches = [i for i, b in enumerate(model.block_indices) if b == self.block]
        if len(matches) != 1:
            raise ValueError(f"block {block} is not in this program's model "
                             f"{tuple(model.block_indices)}")
        self.index = matches[0]
        layer = model.layers[self.index]
        if layer.is_mlp:
            raise ValueError(f"block {block} is an MLP block, not an attention block")
        if not getattr(layer.mixer, "per_head", False):
            raise ValueError(f"block {block} was built whole; a head program needs the "
                             f"per-head attention module (kv_head={kv_head})")

    def run(self, hidden_states, causal_mask, current_pos, cache_pos,
            cos_full=None, sin_full=None):
        model = self.model
        masks_full, _mask_sliding = model.split_mask(causal_mask)
        full_table, _sliding_table = model.cache_tables()
        layer = model.layers[self.index]
        rope = None
        if model.config.rope_tables_as_inputs:
            if cos_full is None:
                # eager use: the rows the graph would have been handed
                cos_full, sin_full = layer.mixer.rope.table_rows(
                    current_pos, q_len_of(hidden_states))
            rope = (cos_full, sin_full)
        out = layer.mixer(hidden_states, masks_full, full_table[layer.slot],
                          layer.slot, layer.num_slots, current_pos, cache_pos,
                          None, None, rope)
        return self.touch_states(out)

    def forward(self, hidden_states, causal_mask, current_pos, cache_pos):
        return self.run(hidden_states, causal_mask, current_pos, cache_pos)


class RopeHeadAttentionChunk(HeadAttentionChunk):
    """The same program with the full-attention RoPE rows as graph inputs (24K)."""

    def forward(self, hidden_states, causal_mask, current_pos, cache_pos,
                cos_full, sin_full):
        return self.run(hidden_states, causal_mask, current_pos, cache_pos,
                        cos_full, sin_full)


#: How many raw contributions a sliding attention program emits: one per KV head,
#: in the aggregated layer's channel order.
#:
#: This is the interface the full-attention path already has between
#: attn_full_<k>_h<h> and epilogue_<k> -- one (1, seq, rep * head_dim)
#: contribution per KV head, concatenated by the epilogue -- reused for the
#: sliding layers because it is the surface the plan builder accepts: a stateful
#: program that owns a cache and emits ONE (1, seq, 4096) tensor does not build a
#: plan, while the same compute emitted as four 1024-wide tensors does (measured).
#: Fixed at four because a traced graph's signature is a fixed-arity function;
#: num_key_value_heads is four for this model, and SlidingAttentionChunk refuses
#: a model where it is not, rather than emitting a surface the converter would
#: mislabel.
CONTRIBUTIONS = 4


def contribution_names(prefix: str = "contribution") -> list:
    """The output names of one sliding attention program, in channel order.

    One function owns the count and the names, because the converter's declared
    outputs, the traced wrapper and the epilogue's declared inputs all have to
    agree on them: Core ML maps declared outputs onto the traced graph by
    position, so the four slices have to be named in the order they are produced.
    """
    return [f"{prefix}_h{index}" for index in range(CONTRIBUTIONS)]


class SlidingAttentionChunk(_StateTouching):
    """One sliding layer's attention over its own carrier, and nothing else.

    The sliding twin of HeadAttentionChunk: a single state -- this layer's
    (2, num_kv_heads, sliding_rows, head_dim) carrier, (2, 4, 1280, 256) = 5.0
    MiB at 24,576 rows -- read and written by the layer's own module, which holds
    the q/k/v rows of every head and neither a gate nor a tail.

    The output is the RAW contribution, as CONTRIBUTIONS tensors of
    (1, seq, rep * head_dim) each: exactly the slices the aggregated sliding layer
    computes before it multiplies by its gate and runs out_norm / o_proj, in the
    aggregated channel order (KV head 0's query heads first).  The layer's own
    module produces them one KV head at a time (mixer.contrib_per_head) and never
    builds the (1, seq, 4096) tensor at all: the plan builder refuses a stateful
    program that materialises a 4096-wide tensor ANYWHERE (measured -- four
    outputs that are torch.chunk() of that tensor still build no plan), so
    chunking a wide result after the fact is not enough.  The epilogue after this
    program concatenates the four, which is the interface epilogue_<k> already
    takes from the four attn_full_<k>_h<h> programs.  out * gate commutes with the
    concatenation, so the numbers do not move.

    A program that owns a cache must not carry the gate convolution (measured), so
    out * gate, the tail and the MLP after the layer all live in the unstated
    SlidingEpilogueChunk that follows this one.
    tests/test_per_head_programs.py asserts that the pair reproduces the
    aggregated layer, arithmetic for arithmetic.
    """

    #: what this program needs from the layer's module: the contribution emitted
    #: one KV head at a time.  False only for the wide shape kept for
    #: re-measurement, which accepts the layer either way (its job is to be 4096
    #: wide and fail to load).
    needs_contrib_per_head = True

    def __init__(self, model: DoryANEModel, block: int):
        super().__init__()
        self.model = model
        self.block = int(block)
        self.index = _index_of(model, self.block)
        layer = model.layers[self.index]
        if layer.is_mlp or layer.is_full:
            raise ValueError(f"block {block} is not a sliding-attention block: this "
                             f"program computes one sliding layer's attention")
        if not getattr(layer.mixer, "attn_only", False):
            raise ValueError(
                f"block {block} was built with a gate or a tail: a sliding attention "
                f"program is the attention only, so build the block with "
                f"attn_only_blocks=(...,) and let the epilogue after it do the rest")
        if (self.needs_contrib_per_head
                and not getattr(layer.mixer, "contrib_per_head", False)):
            raise ValueError(
                f"block {block} emits its contribution as one 4096-wide tensor: a "
                f"sliding attention program builds no such tensor anywhere, so the "
                f"program would not load.  Build the block with "
                f"contrib_per_head_blocks=(...,) as well (one KV head scored per "
                f"pass, one {CONTRIBUTIONS}-tensor output)")
        if not model.split_cache:
            raise ValueError("a sliding attention program declares that layer's own "
                             "carrier, so the model needs the split layout "
                             "(heads_per_state=1)")
        if int(layer.mixer.kv_heads) != CONTRIBUTIONS:
            raise ValueError(
                f"block {block} has {int(layer.mixer.kv_heads)} KV heads, but a sliding "
                f"attention program emits {CONTRIBUTIONS} contributions: the epilogue "
                f"after it takes a fixed number of inputs, exactly like the four head "
                f"programs a full-attention layer has")

    def attend(self, hidden_states, causal_mask, current_pos, cache_pos,
               cos_sliding=None, sin_sliding=None):
        """The layer's attention, in the shape the layer emits it.

        The aggregated layer's arithmetic, one KV head per pass
        (mixer.contrib_per_head): CONTRIBUTIONS tensors of
        (1, seq, rep * head_dim), in the aggregated channel order.  Nothing here
        slices a wide result -- the wide result is never built.
        """
        model = self.model
        _masks_full, mask_sliding = model.split_mask(causal_mask)
        _full_table, sliding_table = model.cache_tables()
        layer = model.layers[self.index]
        rope = None
        if model.config.rope_tables_as_inputs:
            if cos_sliding is None:
                # eager use: the rows the graph would have been handed
                cos_sliding, sin_sliding = layer.mixer.rope.table_rows(
                    current_pos, q_len_of(hidden_states))
            rope = (cos_sliding, sin_sliding)
        out = layer.mixer(hidden_states, [mask_sliding],
                          [sliding_table[layer.slot]], layer.slot, layer.num_slots,
                          current_pos, cache_pos, None, None, rope)
        return self.touch_states(out)

    def run(self, hidden_states, causal_mask, current_pos, cache_pos,
            cos_sliding=None, sin_sliding=None):
        """The raw contribution as CONTRIBUTIONS tensors, in channel order."""
        out = self.attend(hidden_states, causal_mask, current_pos, cache_pos,
                          cos_sliding, sin_sliding)
        if not isinstance(out, tuple) or len(out) != CONTRIBUTIONS:
            raise RuntimeError(
                f"the attention of block {self.block} emitted one tensor, not "
                f"{CONTRIBUTIONS} contributions: it was built without "
                f"contrib_per_head, so it would build the 4096-wide tensor the plan "
                f"builder refuses")
        return out

    def forward(self, hidden_states, causal_mask, current_pos, cache_pos):
        return self.run(hidden_states, causal_mask, current_pos, cache_pos)


class RopeSlidingAttentionChunk(SlidingAttentionChunk):
    """The same program with the sliding RoPE rows as graph inputs (24K)."""

    def forward(self, hidden_states, causal_mask, current_pos, cache_pos,
                cos_sliding, sin_sliding):
        return self.run(hidden_states, causal_mask, current_pos, cache_pos,
                        cos_sliding, sin_sliding)


class WideSlidingAttentionChunk(SlidingAttentionChunk):
    """The previous shape, kept for re-measurement: ONE (1, seq, 4096) output.

    A stateful program whose single output is 4096 channels wide is one of the
    surfaces the plan builder refuses (measured) -- so is the same compute with
    four outputs that are torch.chunk() of that tensor, which is why the default
    program emits its slices from four single-head passes instead.  This class
    exists so the refusal can be re-measured without reverting the split, and so
    the shape the sliding_attn_L<b> program had before this change stays
    constructible.  Nothing in the conversion builds it: the plan's sliding_attn
    kind is the per-KV-head form.
    """

    needs_contrib_per_head = False

    def run(self, hidden_states, causal_mask, current_pos, cache_pos,
            cos_sliding=None, sin_sliding=None):
        """ONE (1, seq, 4096) tensor -- the surface that was measured to fail.

        A layer built contrib_per_head emits the four slices; joining them here
        restores the wide graph (same numbers, one 4096-wide tensor), which is
        exactly what a re-measurement of the refusal needs.  A layer built without
        it returns the wide tensor directly, as it did before this change.
        """
        out = self.attend(hidden_states, causal_mask, current_pos, cache_pos,
                          cos_sliding, sin_sliding)
        if isinstance(out, tuple):
            return torch.cat(list(out), dim=-1)
        return out

    def forward(self, hidden_states, causal_mask, current_pos, cache_pos):
        return self.run(hidden_states, causal_mask, current_pos, cache_pos)


class RopeWideSlidingAttentionChunk(WideSlidingAttentionChunk):
    """The wide shape with the sliding RoPE rows as graph inputs (24K)."""

    def forward(self, hidden_states, causal_mask, current_pos, cache_pos,
                cos_sliding, sin_sliding):
        return self.run(hidden_states, causal_mask, current_pos, cache_pos,
                        cos_sliding, sin_sliding)


class SlidingGroupChunk(BlockChunk):
    """A run of sliding-only blocks: its attention caches and its MLPs.

    The three sliding layers of one 8-block period are separated by MLPs, so the
    only run that can be computed inside one program is the run *including* those
    MLPs: a program that held three caches but only did attention would have to
    be called three times, one per layer, and would save no dispatch at all.
    Three caches of 5.0 MiB is 15.0 MiB, under the ceiling, and one call advances
    the residual stream through three layers.
    """

    def __init__(self, model, start_block: int, end_block: int,
                 loop_entry: bool = False, loop_exit: bool = False):
        super().__init__(model, start_block, end_block,
                         loop_entry=loop_entry, loop_exit=loop_exit)
        for index in self.chunk._indices:
            layer = model.layers[index]
            if layer.is_full:
                raise ValueError(
                    f"block {layer.block} is a full-attention block: a sliding group "
                    f"cannot hold it (its per-head caches are 24.0 MiB each), so the "
                    f"group must end before it")


class RopeSlidingGroupChunk(SlidingGroupChunk):
    """A sliding group with only the sliding RoPE rows as inputs."""

    def forward(self, hidden_states, causal_mask, current_pos, cache_pos,
                cos_sliding, sin_sliding, h_in=None):
        return self.run(hidden_states, causal_mask, current_pos, cache_pos, h_in,
                        cos_sliding=cos_sliding, sin_sliding=sin_sliding)


# ---------------------------------------------------------------------------
# One program per sliding layer: the gate arrives as an input leaf
#
# The measured rule this section exists for: a Core ML program that owns a
# cache must not carry a second convolution chain that joins its output.  A
# stateful attention program that computes its own gate never builds a plan;
# the same program with the gate as an *input leaf* does, and it still does when
# it finishes the layer and emits the NEXT layer's gate as a second output
# (measured one variant at a time on the real per-head program).
#
# Dory's sliding layers compute their gate from their own input, so the shipped
# `sliding_k` group -- three sliding layers plus the MLPs between them -- cannot
# be converted at all.  One program per sliding layer, with the gate handed in
# and the next layer's gate handed out, is the decomposition that both loads and
# keeps the arithmetic identical.
# ---------------------------------------------------------------------------


def _index_of(model: DoryANEModel, block: int) -> int:
    """The position of a physical block in this program's model."""
    matches = [i for i, b in enumerate(model.block_indices) if b == int(block)]
    if len(matches) != 1:
        raise ValueError(f"block {block} is not in this program's model "
                         f"{tuple(model.block_indices)}")
    return matches[0]


class SlidingLayerChunk(_StateTouching):
    """One sliding layer, the MLP after it, and the NEXT layer's gate.

    This wrapper is the whole fix for the sliding layers: the gate is an
    input, and the gate of the layer this program hands its hidden state to is
    emitted as a second output, so the chain stays continuous

        embeddings -> sliding(0) -> sliding(2) -> sliding(4) -> epilogue -> ...

    The MLP after the layer is inside the program (blocks L and L+1 are
    advanced by one call): an attention-only program would have to be called
    once per layer anyway, and the MLP is what makes the dispatch count fall.

    The arithmetic is the aggregated path's, op for op: the same attention over
    the same cache, the same `out * gate` with the same gate values (the
    previous program computed them from exactly this layer's input), the same
    out_norm / o_proj, the same nGPT blend, the same MLP.  tests/
    test_per_head_programs.py asserts the whole chain against the aggregated
    model at 4,096 and at 24,576.
    """

    def __init__(self, model: DoryANEModel, block: int,
                 mlp_block: Optional[int] = None, gate_block: Optional[int] = None,
                 loop_entry: bool = False, loop_exit: bool = False):
        super().__init__()
        self.model = model
        self.block = int(block)
        self.index = _index_of(model, self.block)
        self.mlp_block = None if mlp_block is None else int(mlp_block)
        self.mlp_index = (None if mlp_block is None
                          else _index_of(model, int(mlp_block)))
        self.gate_block = None if gate_block is None else int(gate_block)
        self.loop_entry = bool(loop_entry)
        self.loop_exit = bool(loop_exit)
        layer = model.layers[self.index]
        if layer.is_mlp or layer.is_full:
            raise ValueError(f"block {block} is not a sliding-attention block: one "
                             f"sliding program holds exactly one sliding layer")
        if not layer.mixer.gate_in:
            raise ValueError(
                f"block {block} was built with its own g_proj: a sliding program "
                f"receives the gate as an input leaf, because a second conv chain "
                f"joining a state-derived output does not build a plan; build the "
                f"block with gate_in_blocks=(...,)")
        if self.mlp_index is not None and not model.layers[self.mlp_index].is_mlp:
            raise ValueError(f"block {self.mlp_block} is not an MLP block")
        if self.gate_block is not None:
            source = model.layer_for(self.gate_block)
            if source.is_mlp or source.mixer.g_proj is None:
                raise ValueError(
                    f"the gate for block {self.gate_block} is computed in this "
                    f"program, so it has to hold that block as a gate-only module")

    def gate_next(self, hidden_states):
        """sigmoid(g_proj(h) * s_gate^2) of the layer this program feeds."""
        return self.model.layer_for(self.gate_block).mixer.head_gate(hidden_states)

    def run(self, hidden_states, causal_mask, current_pos, cache_pos, gate,
            h_in=None, cos_sliding=None, sin_sliding=None):
        model = self.model
        masks_full, mask_sliding = model.split_mask(causal_mask)
        full_table, sliding_table = model.cache_tables()
        layer = model.layers[self.index]
        rope = None
        if model.config.rope_tables_as_inputs:
            if cos_sliding is None:
                # eager use: the rows the graph would have been handed
                cos_sliding, sin_sliding = layer.mixer.rope.table_rows(
                    current_pos, q_len_of(hidden_states))
            rope = (cos_sliding, sin_sliding)
        h = hidden_states
        if self.loop_entry:
            if h_in is None:
                raise ValueError("a loop-entry sliding program needs h_in")
            h = model.l2(h + h_in)
        h = layer(h, masks_full, mask_sliding,
                  *_own_caches(layer, full_table, sliding_table),
                  current_pos, cache_pos, None, None, rope, None, gate)
        if self.mlp_index is not None:
            mlp = model.layers[self.mlp_index]
            h = mlp.blend(h, mlp.mixer(h))
        if self.loop_exit:
            h = model.l2(h)
        h = self.touch_states(h)
        if self.gate_block is None:
            return h
        return h, self.gate_next(h)

    def forward(self, hidden_states, causal_mask, current_pos, cache_pos, gate,
                h_in=None):
        """The declared signature of a sliding program without RoPE inputs."""
        return self.run(hidden_states, causal_mask, current_pos, cache_pos, gate,
                        h_in)


class RopeSlidingLayerChunk(SlidingLayerChunk):
    """The same program with the sliding RoPE rows as graph inputs (24K)."""

    def forward(self, hidden_states, causal_mask, current_pos, cache_pos, gate,
                cos_sliding, sin_sliding, h_in=None):
        return self.run(hidden_states, causal_mask, current_pos, cache_pos, gate,
                        h_in, cos_sliding, sin_sliding)


class EntryGateChunk(_StateTouching):
    """Unstated: the loop entry's L2, and the gate of the layer it feeds.

    The recurrent zone is re-entered on every loop pass with a *different*
    hidden state (the previous pass's output plus h_in), so the first layer
    inside the zone has a gate that changes per pass and cannot be emitted by
    the program before the zone.  This program computes the entry's
    l2(h + h_in) and that layer's gate in one call.

    It is unstated, which is exactly why it may hold a gate convolution: the
    rule the sliding layers obey is about a second conv chain joining the output
    of a program that owns a cache, and the epilogue (also unstated) has always
    computed a gate this way.
    """

    def __init__(self, model: DoryANEModel, gate_block: int):
        super().__init__()
        self.model = model
        self.gate_block = int(gate_block)
        layer = model.layer_for(self.gate_block)
        if layer.is_mlp or layer.mixer.g_proj is None:
            raise ValueError(f"block {self.gate_block} cannot supply a gate: this "
                             f"program needs it built as a gate-only module")

    def forward(self, hidden_states, h_in):
        h = self.model.l2(hidden_states + h_in)
        gate = self.model.layer_for(self.gate_block).mixer.head_gate(h)
        return self.touch_states(h), gate

class FullEpilogueChunk(_StateTouching):
    """Unstated: finishes a full-attention block, then computes the next MLP block.

    The per-(layer, head) programs cannot finish their layer -- out_norm is an L2
    over all 4,096 channels and o_proj mixes all 16 heads -- so that tail, the
    nGPT residual blend, and the MLP block that follows it (blocks 8k+7 in Dory's
    layout) are one unstated program.  It carries no cache, which is why it can
    hold whole o_proj / gate_proj / down_proj matrices and still be small on the
    ANE.
    """

    def __init__(self, model, full_block: int, mlp_block: Optional[int] = None,
                 loop_exit: bool = False, gate_block: Optional[int] = None,
                 gate_in: bool = False):
        """gate_in: the gate arrives as an input leaf; gate_block: also emit the
        NEXT layer's gate as a second output.

        With the defaults (no gate leaf, no gate output) this is exactly the
        program the 24K epilogue has always been: it computes its own gate from
        hidden_states, which is safe here because the epilogue declares no
        cache.  In the per-sliding-layer chain the gate comes in from the
        program that produced this layer's input, and the gate of the next
        sliding layer goes out, so one program per attention layer emits every
        gate exactly once.
        """
        super().__init__()
        self.model = model
        self.full_block = int(full_block)
        self.mlp_block = None if mlp_block is None else int(mlp_block)
        self.loop_exit = bool(loop_exit)
        self.full_index = self._index(self.full_block)
        self.mlp_index = (None if self.mlp_block is None else self._index(self.mlp_block))
        self.gate_block = None if gate_block is None else int(gate_block)
        self.gate_in = bool(gate_in)
        mixer = model.layers[self.full_index].mixer
        if self.gate_in and mixer.g_proj is not None:
            raise ValueError(f"block {self.full_block} holds a g_proj this program "
                             f"would never use; build it with gate_in")
        if not self.gate_in and mixer.g_proj is None:
            raise ValueError(f"block {self.full_block} was built gate_in but this "
                             f"epilogue computes its own gate; drop gate_in or pass "
                             f"the gate leaf")
        if self.gate_block is not None:
            source = model.layer_for(self.gate_block)
            if source.is_mlp or source.mixer.g_proj is None:
                raise ValueError(
                    f"the gate for block {self.gate_block} is computed here, so this "
                    f"program has to hold that block as a gate-only module")

    def _index(self, block: int) -> int:
        matches = [i for i, b in enumerate(self.model.block_indices) if b == block]
        if len(matches) != 1:
            raise ValueError(f"block {block} is not in this program's model "
                             f"{tuple(self.model.block_indices)}")
        return matches[0]

    def run(self, hidden_states, head_0, head_1, head_2, head_3, gate=None):
        return self.finish_layer(
            hidden_states,
            torch.cat([head_0, head_1, head_2, head_3], dim=-1), gate)

    def finish_layer(self, hidden_states, contributions, gate):
        """The layer tail, given the whole 4096-wide contribution tensor.

        run() concatenates the four per-KV-head slices before calling this, so a
        full-attention epilogue (four head programs) and a sliding one (four
        contributions from one attention program) reach the same arithmetic.  The
        gate multiply is elementwise, so concatenating first is exactly the
        aggregated layer's out * gate.
        """
        model = self.model
        layer = model.layers[self.full_index]
        # The gate lives here, not in the head programs: it is an independent
        # convolution chain, and a program that joins it to the state-derived
        # attention output at the end fails to build a plan (measured - the same
        # graph without the gate loads, with it fails, and an add instead of a
        # multiply fails identically).  out * gate commutes with the concatenation,
        # so applying it after the cat is exactly the aggregated arithmetic.
        if gate is None:
            # the gate lives here, not in the head programs: it is an
            # independent convolution chain, and a program that joins it to the
            # state-derived attention output fails to build a plan (measured).
            # The epilogue declares no cache, so computing it here is safe; in
            # the per-sliding-layer chain the previous program sends it in.
            gate = layer.mixer.head_gate(hidden_states)
        contributions = contributions * gate
        # the layer's own tail, the same two operations the aggregated graph runs
        h = layer.blend(hidden_states, layer.mixer.finish(contributions))
        if self.mlp_index is not None:
            mlp = model.layers[self.mlp_index]
            h = mlp.blend(h, mlp.mixer(h))
        if self.loop_exit:
            h = model.l2(h)
        h = self.touch_states(h)
        if self.gate_block is None:
            return h
        return h, model.layer_for(self.gate_block).mixer.head_gate(h)

    def forward(self, hidden_states, head_0, head_1, head_2, head_3, gate=None):
        return self.run(hidden_states, head_0, head_1, head_2, head_3, gate)


class SlidingEpilogueChunk(FullEpilogueChunk):
    """Unstated: finishes a sliding layer, then computes the MLP after it.

    The second half of the split sliding path (sepilogue_L<b>).  Its inputs are
    hidden_states, the four raw contributions the sliding attention program
    emitted -- contribution_h0 ... contribution_h3, in the same channel order the
    aggregated layer uses -- and the gate of THIS layer, computed by the program
    that produced hidden_states.  The inherited run() concatenates the four and
    calls finish_layer: out * gate, out_norm, o_proj, the nGPT residual blend, the
    MLP block after the layer, and the NEXT attention layer's gate as a second
    output.

    Four contribution inputs rather than one 4096-wide tensor is the interface
    the full-attention path already has between attn_full_<k>_h<h> and
    epilogue_<k>, and it is the surface the attention program can emit
    (measured).  The gate multiply is elementwise, so concatenating them first is
    exactly the aggregated layer's arithmetic.

    It declares no state, which is why it may hold the gate convolution of the
    next layer: the rule the sliding layers obey is about a second conv chain
    joining the output of a program that owns a cache, and this program owns
    none.  FullEpilogueChunk's full_block is simply the block this program
    finishes.
    """

    def forward(self, hidden_states, contribution_h0, contribution_h1,
                contribution_h2, contribution_h3, gate):
        return self.run(hidden_states, contribution_h0, contribution_h1,
                        contribution_h2, contribution_h3, gate)


class WideSlidingEpilogueChunk(FullEpilogueChunk):
    """The previous shape: ONE (1, seq, 4096) contribution input.

    The counterpart of WideSlidingAttentionChunk -- it takes the single wide
    tensor that program emits, so the pair whose surface the plan builder refuses
    stays constructible for a re-measurement.  Nothing in the conversion builds
    it: the plan's sliding_epilogue kind is the four-contribution form.
    """

    def run(self, hidden_states, contribution, gate):
        return self.finish_layer(hidden_states, contribution, gate)

    def forward(self, hidden_states, contribution, gate):
        return self.run(hidden_states, contribution, gate)


class EmbeddingChunk(_StateTouching):
    """Token ids -> hidden states, and optionally the first layer's gate.

    The gate of block 0 depends on the embedding output and on nothing else, so
    the program that produces that output is the one that can emit it, and one
    dispatch is saved by doing so: the per-sliding-layer chain then starts at
    `gate_0` with no extra program.  gate_block is built as a gate-only module
    beside the embedding table (it holds g_proj and nothing else).
    """

    def __init__(self, config, embed_tokens: nn.Embedding, model: DoryANEModel,
                 gate_block: Optional[int] = None):
        super().__init__()
        self.model = model
        self.embed_tokens = embed_tokens
        self.gate_block = None if gate_block is None else int(gate_block)
        if self.gate_block is not None:
            layer = model.layer_for(self.gate_block)
            if layer.is_mlp or layer.mixer.g_proj is None:
                raise ValueError(f"block {self.gate_block} cannot supply the first "
                                 f"layer's gate: it has to be built gate-only")

    def forward(self, input_ids):
        h = self.embed_tokens(input_ids)
        if self.gate_block is None:
            return self.touch_states(h)
        gate = self.model.layer_for(self.gate_block).mixer.head_gate(h)
        return self.touch_states(h), gate


def lm_head_split_sizes(vocab_size: int, splits: int) -> list:
    """Even vocab_size // splits row slices, or a clear error.

    ANEMLL's LM-head split is even by construction (llama_model.py:1212-1221)
    and Dory's 131,072 is divisible by 8 and 16.  An uneven split is refused
    rather than silently padded: the runtime rebuilds the vocabulary from
    lm_head_chunk_sizes in this order.
    """
    vocab_size = int(vocab_size)
    splits = int(splits)
    if splits < 2:
        raise ValueError(
            f"the LM head must be split into at least 2 parts (got {splits}): "
            "a 131,072-wide conv output exceeds the 65,536 ANE axis limit")
    if vocab_size % splits:
        raise ValueError(f"vocab_size {vocab_size} is not divisible by {splits} "
                         "LM-head splits; pick a divisor of the vocabulary")
    size = vocab_size // splits
    return [size] * splits


class LMHeadChunk(_StateTouching):
    """The LM head, split into N independent 1x1 convolutions.

    Each part is vocab//N output channels (16,384 for N=8), which keeps a single
    ANE axis inside its documented range.  The parts are separate graph outputs,
    so one file holds the whole head and the runtime concatenates logits_1..N in
    order.

    argmax replaces the logits with argmax_idx_i (int32, a **local** index
    inside part i) plus argmax_val_i (fp16); the global token id is the local
    index plus that part's offset, computed on the host (ANEMLL
    llama_converter.py:661-671).
    """

    def __init__(self, config, weight: torch.Tensor, splits: int, model: DoryANEModel,
                 argmax: bool = False):
        super().__init__()
        self.model = model
        self.splits = int(splits)
        self.argmax = bool(argmax)
        sizes = lm_head_split_sizes(config.vocab_size, self.splits)
        self.chunk_sizes = sizes
        weight = weight.detach()
        if weight.shape[0] != config.vocab_size:
            raise ValueError(f"lm_head weight has {weight.shape[0]} rows, "
                             f"config says vocab_size {config.vocab_size}")
        parts = []
        start = 0
        for size in sizes:
            conv = nn.Conv2d(config.hidden_size, size, kernel_size=1, bias=False,
                             dtype=weight.dtype)
            conv.weight.data.copy_(
                weight[start:start + size].view(size, config.hidden_size, 1, 1))
            parts.append(conv)
            start += size
        self.parts = nn.ModuleList(parts)

    def forward(self, hidden_states):
        x = hidden_states.permute(0, 2, 1).unsqueeze(2)
        outs = []
        for conv in self.parts:
            logits = conv(x).squeeze(2).permute(0, 2, 1)
            if self.argmax:
                val, idx = torch.max(logits, dim=-1)
                outs.append(idx.to(torch.int32))
                outs.append(val)
            else:
                outs.append(logits)
        # keep both states alive on the first output only
        outs[0] = self.touch_states(outs[0])
        return tuple(outs)

    @property
    def output_names(self) -> list:
        if self.argmax:
            names = []
            for i in range(1, self.splits + 1):
                names.extend([f"argmax_idx_{i}", f"argmax_val_{i}"])
            return names
        return [f"logits_{i}" for i in range(1, self.splits + 1)]
