"""The per-(layer, head) program path: 24,576-token context on the ANE (DANE-5).

Why this exists
---------------
A Core ML program that does real attention may keep about 34 MiB of state
resident (measured: 33.8 MiB loads, 45.0 MiB fails; one 24.0 MiB state loads, two
of 48.0 MiB do not).  Weights do not compete with that budget and the number of
carriers is not the limit; size per state is.

At 24,576 tokens one (layer, head) K+V cache is exactly 24.0 MiB, so **exactly
one such cache fits in one program**.  The five-component design cannot convert
at that context at all: every component declares the whole state union, which is
864 MiB of full-attention cache plus 135 MiB of sliding cache in one program.
This module therefore emits a different plan:

* **36 attention programs**, one per (full-attention layer, KV head), each
  declaring exactly one state of shape (2, 1, 24576, 256) fp16 = 24.0 MiB;
* **27 sliding attention programs**, one per sliding layer, each declaring that
  layer's 5.0 MiB carrier (all four KV heads at once) and computing the layer's
  attention and nothing else -- one KV head scored per pass, one (1, seq, 1024)
  tensor out per head, never the wide contribution;
* **27 unstated sliding epilogue programs** that apply the gate, finish the
  sliding layer they follow (out_norm, o_proj, the nGPT blend) and compute the
  MLP after it;
* **9 unstated epilogue programs** that finish a full-attention layer (out_norm
  over all 4,096 channels, o_proj, the nGPT residual blend) and then compute the
  MLP block that follows it -- the part of the residual stream the per-head
  programs cannot compute, because an L2 norm and o_proj mix all 16 heads;
* the **embeddings**, the **loop-entry gate** and the **LM head**, unstated.

Why one program per sliding layer, and why the gate has to travel
-----------------------------------------------------------------
Measured on the real per-head program, one variant at a time: a program that
owns a cache converts while it is a passthrough, while it writes and reads its
cache, while it scores, and while it projects queries -- and stops converting the
moment it carries a *second* convolution chain that joins the attention output.
Computing its own gate (`g_proj` -> sigmoid -> `out * gate`) is exactly that, and
an `out + gate` variant fails identically, so it is not the elementwise op.  The
same graph with the gate arriving as an **input leaf** converts, and so does one
that takes the gate, applies it, finishes the layer, and emits the NEXT layer's
gate as a second output.  Dory's sliding layers all compute their own gate, which
is why the older `sliding_{k}` program -- three sliding layers plus the MLPs
between them, 15.0 MiB of state -- could never load.

So the plan hands the gate around instead of computing it inside a stateful
program.  Every attention layer's gate is computed by the program that produces
the hidden state it depends on, and consumed as a leaf by the one that needs it:

    embeddings      -> h0, gate_0      (the embedding output's own gate)
    sliding_attn_L0 -> c0_h0..c0_h3    (block 0's attention, raw, one tensor
                                       per KV head -- the same four-slice
                                       interface epilogue_<k> takes)
    sepilogue_L0    -> h2, gate_2      (gate * cat(c0), block 0's tail,
                                       block 1's MLP)
    sliding_attn_L2 -> c2_h0..c2_h3
    sepilogue_L2    -> h4, gate_4
    epilogue_0      -> h8, gate_8      (block 6's tail + block 7's MLP)
    ...
    epilogue_1      -> h16, (no gate)  the loop entry is next, and re-entering
                                       the zone changes its input, so
    entry_gate      -> h16', gate_16   an unstated program emits that one
                                       per loop pass

Splitting a sliding layer in two is what makes it convert.  The measured rule is
that a program owning a cache may not carry a second convolution chain joining
its output: one program holding the layer, its gate and its MLP (the old
sliding_L0) never built a plan at 24,576, while the same layer cut into
"attention only" and "everything after it, unstated" is the shape the
full-attention path already ships.  The rule is about the program that owns the
cache, so the gate convolution is free to live in the unstated half.

There is a second measured rule on the same program, one level lower: a stateful
program may not materialise a 4096-wide tensor *anywhere*.  With a single
(1, seq, 4096) contribution output the split sliding attention program built no
plan (the same graph without the gate, without the mask, without RoPE and with a
512-row carrier loads, so neither the compute nor the state was the problem), and
so does the same program emitting four 1024-wide outputs that are torch.chunk()
of that tensor: chunking a wide result after the fact cannot help.  So the
attention program scores ONE KV head per pass -- one (1, seq, 1024) tensor per
head, in the aggregated channel order -- and never builds the wide tensor at all
(mixer.contrib_per_head; the eager test watches the matmuls and the converted
graph has eight of them, two per head, and no concat of the four).  The epilogue
concatenates the slices, which is the interface the full-attention path already
uses between attn_full_<k>_h<h> and epilogue_<k>; the gate multiply is
elementwise, so cat-then-multiply is the aggregated layer's arithmetic exactly.

One consequence worth stating: a gate convolution now lives in exactly one
program per attention layer across the whole conversion (the program that emits
it), so the per-layer plan holds no duplicated gate weights.

Nothing here changes the arithmetic
-----------------------------------
Every program is built from the same modules the five-component path uses:
a head program is one pass of DoryANEAttention's group loop (same projections,
same RoPE, same norms, same cache write and read, same matmul), the epilogue runs
the same out_norm / o_proj and the same residual blend, and a sliding group is
the same DoryANEChunk over a shorter block range.  tests/test_per_head_programs.py
asserts the head contributions equal the aggregated path's, on a fixed seed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from anemll.ane_converter.dory_converter import (
    DoryConverter,
    _ComponentModel,
    build_meta,
    check_size,
    estimate_component_bytes,
    write_meta,
    write_rope_tables,
    tiling_flags,
)
from anemll.models import dory_chunks as dch
from anemll.utils import dory_caches as dc

__all__ = [
    "ProgramSpec",
    "program_specs",
    "program_by_name",
    "validate_program_specs",
    "program_manifest",
    "declared_states",
    "PerHeadConverter",
]


@dataclass(frozen=True)
class ProgramSpec:
    """One Core ML file of the per-head plan.

    kind is one of embeddings | entry_gate | sliding_attn | sliding_epilogue |
    sliding_layer | attn_head | sliding_group | full_epilogue | lm_head.  states
    are the buffers this program alone declares (empty for an unstated program);
    blocks are the physical blocks whose residual stream it advances -- for a
    head program the attention block it computes one head of, for a sliding
    attention program the sliding layer it computes the attention of (consumed,
    not advanced: the epilogue after it finishes the block), for an epilogue the
    attention block it finishes plus the MLP after it, and for the older
    one-program sliding layer that layer and the MLP after it.

    gate_in / gate_out describe the gate chain, which is what makes a stateful
    attention program convertible at all:

      gate_in   the program receives its layer's gate as an input leaf, instead
                of computing it from hidden_states inside the program, which
                does not build a plan (measured).
      gate_out  the attention block whose gate this program emits as a second
                output: sigmoid(g_proj(h_next) * s_gate^2) of the NEXT layer,
                which is the layer this program's hidden state feeds.

    Every attention block's gate is therefore produced exactly once, by the
    program that computes the hidden state it depends on, and consumed exactly
    once; validate_program_specs checks both halves of that.
    """

    name: str
    kind: str
    blocks: tuple = ()
    kv_head: Optional[int] = None
    states: tuple = ()
    calls_per_token: object = 1
    loop_entry: bool = False
    loop_exit: bool = False
    rope_kinds: tuple = ()
    params: Optional[int] = None
    #: the LUT role this program's weights are palettized with (DoryConverter
    #: keys its lut_bits table by role; every program is a block-like file)
    lut_role: str = "blocks"
    #: a per-head program never tiles its cache -- one state is one head
    tiled: bool = False
    #: the gate arrives as an input leaf (no g_proj in this program)
    gate_in: bool = False
    #: the attention block whose gate this program emits as a second output
    gate_out: Optional[int] = None

    @property
    def rope_tables(self) -> bool:
        return bool(self.rope_kinds)

    @property
    def inputs(self) -> list:
        if self.kind == "embeddings":
            return ["input_ids"]
        if self.kind == "lm_head":
            return ["hidden_states"]
        if self.kind == "entry_gate":
            # the loop entry's l2(h + h_in), then the gate of the layer it feeds
            return ["hidden_states", "h_in"]
        if self.kind == "full_epilogue":
            # one contribution per KV head, in head order, then the gate leaf
            names = ["hidden_states"] + [f"head_{h}" for h in range(self.heads)]
            if self.gate_in:
                names.append("gate")
            return names
        if self.kind == "sliding_epilogue":
            # the raw contributions the attention program before it emitted --
            # one per KV head, in channel order, exactly the interface
            # epilogue_<k> takes from the four attn_full_<k>_h<h> programs --
            # and the gate of the layer this program finishes.  The gate is
            # unconditional: the wrapper's signature has no default for it, since
            # a sliding layer's epilogue always receives one (validate_program_specs
            # refuses a sliding_epilogue that would compute its own).
            return ["hidden_states"] + dch.contribution_names() + ["gate"]
        if self.kind == "sliding_layer":
            # one function owns the order: the wrapper's signature is built from it
            return dch.sliding_layer_input_names(self.loop_entry, self.rope_tables)
        # a head program and a sliding attention program: hidden_states, mask,
        # the two positions, and the RoPE rows of their own kind
        return dch.block_input_names(False, self.loop_entry, self.rope_tables,
                                     self.rope_kinds)

    @property
    def heads(self) -> int:
        return 4

    @property
    def gate_output(self) -> Optional[str]:
        """Name of the second output, or None: `gate_<block>` names its consumer."""
        return None if self.gate_out is None else f"gate_{int(self.gate_out)}"

    @property
    def outputs(self) -> list:
        gate = [] if self.gate_output is None else [self.gate_output]
        if self.kind == "attn_head":
            return ["head_contribution"]
        if self.kind == "sliding_attn":
            # the raw, un-gated contribution, one tensor per KV head: the program
            # holds no gate, and a stateful program whose single output is 4096
            # channels wide does not build a plan (measured)
            return dch.contribution_names()
        if self.kind == "embeddings":
            return ["hidden_states"] + gate
        if self.kind == "entry_gate":
            return ["h_entry"] + gate
        if self.kind in ("sliding_layer", "sliding_epilogue"):
            return ["h_next"] + gate
        if self.kind == "lm_head":
            return ["logits"]
        return ["output_hidden_states"] + gate

    @property
    def is_recurrent(self) -> bool:
        return self.calls_per_token == "loops"


def _recurrent(blocks, config) -> bool:
    """True when a program's blocks sit inside the recurrent (looped) zone."""
    start = int(config.input_layer_count)
    end = start + int(config.recurrent_layer_count)
    return bool(blocks) and blocks[0] >= start and blocks[-1] < end


def _head_params(config) -> int:
    """Parameters of one (layer, KV head) attention program.

    q and g keep rep query heads' rows, k and v one KV head's; o_proj and the
    layer-wide out_norm belong to the epilogue, so they are not here.
    """
    h = int(config.hidden_size)
    hd = int(config.head_dim)
    rep = int(config.num_attention_heads) // int(config.num_key_value_heads)
    return h * hd * (2 * rep + 2)


def _gate_params(config) -> int:
    """One layer's gate convolution: every attention channel from hidden_size."""
    return (int(config.hidden_size) * int(config.num_attention_heads)
            * int(config.head_dim))


def _mlp_params(config) -> int:
    """gate_proj + up_proj + down_proj of one MLP block."""
    return 3 * int(config.hidden_size) * int(config.intermediate_size)


def _sliding_attn_params(config) -> int:
    """One sliding attention program: q/k/v of the whole layer, and nothing else.

    Every query head's q rows and every KV head's k/v rows, which is the one
    convolution chain such a program may hold: it receives no gate and keeps no
    tail, so there is no g_proj and no o_proj here.  o_proj, out_norm and the MLP
    are the epilogue's (_epilogue_params).
    """
    h = int(config.hidden_size)
    hd = int(config.head_dim)
    return (h * int(config.num_attention_heads) * hd
            + 2 * h * int(config.num_key_value_heads) * hd)


def _sliding_layer_params(config) -> int:
    """One sliding program: the layer's q/k/v/o projections plus the MLP after it.

    The gate is not here: this program receives its own gate as an input leaf,
    and only the NEXT layer's gate convolution, which it emits for its
    successor, is counted -- by the caller, because the last program before a
    loop entry emits none.
    """
    h = int(config.hidden_size)
    hd = int(config.head_dim)
    nh = int(config.num_attention_heads)
    nkv = int(config.num_key_value_heads)
    return h * nh * hd + 2 * h * nkv * hd + nh * hd * h + _mlp_params(config)


def _epilogue_params(config, gate_in: bool = False, gate_out: bool = False) -> int:
    """Parameters of one epilogue: o_proj, the MLP, and the gates it holds.

    With the gate arriving as an input leaf the layer's own g_proj is gone, and
    the one it emits for the next sliding layer is added instead -- so a gate
    convolution lives in exactly one program per attention layer across the
    whole conversion.
    """
    h = int(config.hidden_size)
    hd = int(config.head_dim)
    nh = int(config.num_attention_heads)
    params = (nh * hd) * h + _mlp_params(config)
    if not gate_in:
        params += _gate_params(config)
    if gate_out:
        params += _gate_params(config)
    return params


def _gate_out_after(config, end: int) -> Optional[int]:
    """The attention block whose gate a program whose last block is `end` emits.

    It is the next attention block, which is exactly the layer the program's
    hidden state feeds: gate_L = sigmoid(g_proj_L(h_in_L) * s_gate_L^2), and
    h_in_L is this program's output.  Two places end the chain instead:

      * the start of the recurrent zone -- the zone is re-entered on every loop
        pass with a different hidden state, so an unstated `entry_gate` program
        computes that layer's gate per pass;
      * the last block of the model, which has no successor.
    """
    if int(end) + 1 == int(config.input_layer_count):
        return None
    for block in range(int(end) + 1, int(config.num_hidden_layers)):
        if dc.block_is_attention(block):
            return block
    return None


def _block_params(config, block: int) -> int:
    """Parameters of one whole physical block (attention or MLP)."""
    h = int(config.hidden_size)
    inter = int(config.intermediate_size)
    nh = int(config.num_attention_heads)
    nkv = int(config.num_key_value_heads)
    hd = int(config.head_dim)
    if dc.block_is_mlp(block):
        return 3 * h * inter
    return h * nh * hd + 2 * h * nkv * hd + nh * hd * h + h * nh * hd


def program_specs(config, sliding: str = "split") -> list:
    """The per-head plan, in run order.

    heads_per_state must be 1: the whole point is that a 24.0 MiB state holds
    exactly one (layer, head) cache at this context length.

    `sliding` decides what a sliding layer becomes:

      "split"  two programs per sliding layer (the default and the only shape
               that builds a plan at 24,576): sliding_attn_L<b> declares that
               layer's 5.0 MiB carrier and computes its attention only -- all
               four KV heads, no gate, no tail -- emitting one 1024-wide
               contribution per KV head, and sepilogue_L<b> concatenates them,
               applies the gate, finishes the layer (out_norm, o_proj, the nGPT
               blend) and computes the MLP after it, unstated.  Splitting is what
               the full-attention layers already do (per-head programs plus an
               unstated epilogue), and it is what makes the sliding path load:
               one program holding the layer, its gate chain and its MLP does
               not build a plan (measured, 175 MB, 8 convolutions), and neither
               does a stateful program emitting one 4096-wide tensor (measured;
               four 1024-wide outputs load).
      "layer"  one program per sliding layer: sliding_L<b> does attention, the
               gate, the MLP and the next layer's gate in one stateful program.
               This is the shape measured NOT to build a plan at 24,576, kept so
               the failure can be re-measured against the split.
      "group"  the older program holding three sliding layers and the three
               MLPs between them: each of its three layers computes its own
               gate, so it is the program that never loaded.

    The gate chain of the "split" plan is

        embeddings --gate_0--> sliding_attn_L0 -> sepilogue_L0 --gate_2-->
        sliding_attn_L2 -> sepilogue_L2 --gate_4--> sliding_attn_L4 ->
        sepilogue_L4 --gate_6--> attn_full_0_h* -> epilogue_0 --gate_8-->
        sliding_attn_L8 -> ... -> epilogue_1 --(none: loop entry)-->
        [entry_gate] -> sliding_attn_L16

    where entry_gate emits gate_16 on every loop pass, because the recurrent
    zone is re-entered with a different hidden state each pass.
    """
    heads = getattr(config, "heads_per_state", None)
    if heads != 1:
        raise ValueError(
            "the per-(layer, head) program path needs heads_per_state=1 (one KV head "
            f"per state), got {heads!r}. Pass --heads-per-state 1.")
    num_kv = int(config.num_key_value_heads)
    if num_kv != 4:
        raise ValueError(f"the epilogue program takes one contribution per KV head and "
                         f"is written for Dory's 4, got {num_kv}")
    periods, remainder = divmod(int(config.num_hidden_layers), dc.PERIOD)
    if remainder:
        raise ValueError(f"{config.num_hidden_layers} blocks is not a whole number of "
                         f"{dc.PERIOD}-block periods")
    if sliding not in ("split", "layer", "group"):
        raise ValueError(f"sliding is 'split', 'layer' or 'group', got {sliding!r}")
    # every plan but the old group hands the gate along instead of computing it
    # inside a stateful program, so the chain wiring is the same either way
    chain = sliding in ("split", "layer")
    split = sliding == "split"
    if chain:
        group = int(getattr(config, "slots_per_sliding_carrier", 3) or 3)
        if group != 1:
            raise ValueError(
                "one program per sliding layer declares that layer's own carrier, so "
                f"slots_per_sliding_carrier has to be 1 here (got {group}): a carrier "
                "shared by three layer-slots would be declared by three programs. "
                "Pass --slots-per-sliding-carrier 1.")
    entry_block = int(config.input_layer_count)

    specs = [ProgramSpec(
        "embeddings", "embeddings", calls_per_token=1,
        gate_out=(0 if chain else None),
        params=(int(config.vocab_size) * int(config.hidden_size)
                + (_gate_params(config) if chain else 0)))]
    if chain:
        # the loop entry: l2(h + h_in) and the gate of the layer it feeds.  No
        # program before the zone can emit this one, because every pass adds a
        # different h to h_in.
        specs.append(ProgramSpec(
            "entry_gate", "entry_gate", blocks=(), calls_per_token="loops",
            loop_entry=True, gate_out=entry_block, params=_gate_params(config)))
    for k in range(periods):
        base = k * dc.PERIOD
        full_block = base + dc.FULL_ATTENTION_OFFSET
        if chain:
            for block in range(base, full_block, 2):
                if split:
                    # the layer's attention on its own: one state (that layer's
                    # carrier, 5.0 MiB at 24,576 rows), one convolution chain
                    # (q/k/v), and an output that joins nothing
                    specs.append(ProgramSpec(
                        f"sliding_attn_L{block}", "sliding_attn", blocks=(block,),
                        states=dch.program_state_names(config, blocks=(block,)),
                        calls_per_token=("loops" if _recurrent((block,), config)
                                         else 1),
                        rope_kinds=("sliding",),
                        params=_sliding_attn_params(config)))
                run = (block, block + 1)          # the layer and its MLP
                gate_out = _gate_out_after(config, run[-1])
                if split:
                    # ... and the unstated program that applies the gate,
                    # finishes the layer, computes the MLP after it and emits
                    # the next layer's gate
                    specs.append(ProgramSpec(
                        f"sepilogue_L{block}", "sliding_epilogue", blocks=run,
                        states=(),
                        calls_per_token=("loops" if _recurrent(run, config) else 1),
                        gate_in=True, gate_out=gate_out,
                        params=_epilogue_params(config, gate_in=True,
                                                gate_out=gate_out is not None)))
                else:
                    specs.append(ProgramSpec(
                        f"sliding_L{block}", "sliding_layer", blocks=run,
                        states=dch.program_state_names(config, blocks=(block,)),
                        calls_per_token=("loops" if _recurrent(run, config) else 1),
                        rope_kinds=("sliding",), gate_in=True, gate_out=gate_out,
                        params=_sliding_layer_params(config)
                        + (_gate_params(config) if gate_out is not None else 0)))
        else:
            run = tuple(range(base, full_block))          # 3 sliding layers + 3 MLPs
            specs.append(ProgramSpec(
                f"sliding_{k}", "sliding_group", blocks=run,
                states=dch.program_state_names(config, blocks=run),
                calls_per_token="loops" if _recurrent(run, config) else 1,
                loop_entry=(run[0] == entry_block),
                rope_kinds=("sliding",),
                params=sum(_block_params(config, b) for b in run)))
        for head in range(num_kv):
            specs.append(ProgramSpec(
                f"attn_full_{k}_h{head}", "attn_head", blocks=(full_block,),
                kv_head=head,
                states=dch.program_state_names(config, blocks=(full_block,),
                                               kv_head=head),
                calls_per_token="loops" if _recurrent((full_block,), config) else 1,
                rope_kinds=("full",),
                params=_head_params(config)))
        tail = (full_block, full_block + 1)
        gate_out = _gate_out_after(config, tail[-1]) if chain else None
        specs.append(ProgramSpec(
            f"epilogue_{k}", "full_epilogue", blocks=tail, states=(),
            calls_per_token="loops" if _recurrent(tail, config) else 1,
            loop_exit=(tail[-1] == entry_block
                       + int(config.recurrent_layer_count) - 1),
            gate_in=chain, gate_out=gate_out,
            params=_epilogue_params(config, gate_in=chain,
                                    gate_out=gate_out is not None)))
    specs.append(ProgramSpec("lm_head", "lm_head", calls_per_token=1,
                             params=int(config.vocab_size) * int(config.hidden_size)))
    validate_program_specs(config, specs)
    return specs


def program_by_name(config, name: str) -> ProgramSpec:
    """The one program with this name, for --only on a single package."""
    specs = program_specs(config)
    for spec in specs:
        if spec.name == name:
            return spec
    raise KeyError(f"no program named {name}; the plan has {len(specs)}: "
                   f"{[s.name for s in specs][:3]} ... "
                   f"{[s.name for s in specs][-1]}")


def validate_program_specs(config, specs) -> None:
    """Refuse a plan that leaves a block out, doubles one, or over-declares state.

    Three structural claims, each of which a wrong plan would break silently:

    * every physical block is advanced by exactly one program (its residual
      stream is computed once);
    * every full-attention block has exactly num_key_value_heads head programs,
      and every sliding block -- in the split plan -- has exactly one attention
      program of its own;
    * every state of the model's layout is declared by exactly one program, and
      each program's declared state stays under the per-program budget.
    """
    problems = []
    residual = {}
    for spec in specs:
        if spec.kind in ("attn_head", "sliding_attn"):
            # a head program computes one head's K/V and attention and a sliding
            # attention program one layer's, but neither advances the residual
            # stream: its block is finished by the epilogue that follows it,
            # which is what the next block starts from
            continue
        for block in spec.blocks:
            if block in residual:
                problems.append(f"block {block} is advanced by both {residual[block]} "
                                f"and {spec.name}")
            residual[block] = spec.name
    missing = [b for b in range(int(config.num_hidden_layers)) if b not in residual]
    if missing:
        problems.append(f"no program advances blocks {missing[:6]}")

    head_map = {}
    for spec in specs:
        if spec.kind != "attn_head":
            continue
        block = spec.blocks[0]
        if not dc.block_is_full_attention(block):
            problems.append(f"{spec.name} is a head program for block {block}, which is "
                            f"not a full-attention block")
        head_map.setdefault(block, []).append(spec.kv_head)
    for block in range(int(config.num_hidden_layers)):
        if not dc.block_is_full_attention(block):
            continue
        heads = sorted(head_map.get(block, []))
        if heads != list(range(int(config.num_key_value_heads))):
            problems.append(f"full-attention block {block} has head programs "
                            f"{heads}, expected one per head 0.."
                            f"{int(config.num_key_value_heads) - 1}")

    sliding_map = {}
    for spec in specs:
        if spec.kind != "sliding_attn" or not spec.blocks:
            continue
        block = spec.blocks[0]
        if not dc.block_is_sliding_attention(block):
            problems.append(f"{spec.name} is a sliding attention program for block "
                            f"{block}, which is not a sliding-attention block")
        if block in sliding_map:
            problems.append(f"sliding block {block} has two attention programs, "
                            f"{sliding_map[block]} and {spec.name}")
        sliding_map[block] = spec.name
    if sliding_map:
        for block in range(int(config.num_hidden_layers)):
            if dc.block_is_sliding_attention(block) and block not in sliding_map:
                problems.append(f"sliding block {block} has no attention program: in "
                                f"the split plan a sliding layer's attention is its "
                                f"own program")

    layout = dch.state_layout(config)
    known = {name for name, _ in layout}
    declared = {}
    for spec in specs:
        for name in spec.states:
            if name not in known:
                problems.append(f"{spec.name} declares {name}, which the layout does "
                                f"not define")
            if name in declared:
                problems.append(f"state {name} is declared by both {declared[name]} and "
                                f"{spec.name}")
            declared[name] = spec.name
    undeclared = sorted(known - set(declared))
    if undeclared:
        problems.append(f"no program declares {len(undeclared)} state(s): "
                        f"{undeclared[:4]}")

    shapes = dict(layout)
    for spec in specs:
        total = 0
        for name in spec.states:
            shape = shapes.get(name)
            if shape is None:
                continue
            elements = 1
            for dim in shape:
                elements *= int(dim)
            total += elements * 2
        if total > 30 * 1024 ** 2:
            problems.append(f"{spec.name} declares {total / 1024 ** 2:.1f} MiB of state, "
                            f"over the 30 MiB per-program budget")

    # -- the gate chain ---------------------------------------------------
    # A stateful attention program that computes its own gate does not build a
    # plan, so the plan is only consistent if every attention layer's gate is
    # emitted by exactly one program and taken as an input leaf by exactly one.
    produced, consumed = {}, {}
    for spec in specs:
        if spec.gate_out is not None:
            block = int(spec.gate_out)
            if not dc.block_is_attention(block):
                problems.append(f"{spec.name} emits gate_{block}, and block {block} is "
                                f"not an attention block")
            if block in produced:
                problems.append(f"gate_{block} is emitted by both {produced[block]} and "
                                f"{spec.name}")
            produced[block] = spec.name
        if not spec.gate_in:
            continue
        if not spec.blocks:
            problems.append(f"{spec.name} takes a gate but advances no block")
            continue
        block = int(spec.blocks[0])
        if block in consumed:
            problems.append(f"gate_{block} is consumed by both {consumed[block]} and "
                            f"{spec.name}")
        consumed[block] = spec.name
    if produced or consumed:
        for block in range(int(config.num_hidden_layers)):
            if not dc.block_is_attention(block):
                continue
            if block not in produced:
                problems.append(f"no program emits gate_{block}, the gate of block {block}")
            if block not in consumed:
                problems.append(f"no program takes gate_{block}: block {block} would have "
                                f"to compute its own gate, which does not build a plan")
    # a sliding program is a sliding layer plus the MLP right after it, never a
    # bare attention block: the MLP is what makes one call advance two blocks
    for spec in specs:
        if spec.kind not in ("sliding_layer", "sliding_epilogue"):
            continue
        if (len(spec.blocks) != 2 or not dc.block_is_sliding_attention(spec.blocks[0])
                or not dc.block_is_mlp(spec.blocks[1])):
            problems.append(f"{spec.name} covers blocks {list(spec.blocks)}, which is not "
                            f"(sliding layer, the MLP after it)")
    if problems:
        raise ValueError("the per-head plan is inconsistent: " + "; ".join(problems))


def declared_states(config, specs=None) -> list:
    """Every state of a 24K conversion with its size, and which program holds it."""
    specs = program_specs(config) if specs is None else specs
    owner = {}
    for spec in specs:
        for name in spec.states:
            owner[name] = spec
    rows = []
    for name, shape in dch.state_layout(config):
        elements = 1
        for dim in shape:
            elements *= int(dim)
        size = elements * 2
        spec = owner.get(name)
        rows.append({
            "name": name,
            "shape": [int(dim) for dim in shape],
            "bytes": size,
            "mib": size / 1024 ** 2,
            "program": None if spec is None else spec.name,
            "calls_per_token": None if spec is None else spec.calls_per_token,
        })
    return rows


def program_manifest(config, specs=None) -> dict:
    """What the meta records about this plan: per program, its states and sizes."""
    specs = program_specs(config) if specs is None else specs
    shapes = dict(dch.state_layout(config))
    programs = {}
    for spec in specs:
        states = []
        for name in spec.states:
            shape = shapes[name]
            elements = 1
            for dim in shape:
                elements *= int(dim)
            states.append({"name": name, "shape": [int(d) for d in shape],
                           "bytes": elements * 2, "mib": elements * 2 / 1024 ** 2})
        programs[spec.name] = {
            "kind": spec.kind,
            "blocks": [int(b) for b in spec.blocks],
            "kv_head": spec.kv_head,
            "calls_per_token": spec.calls_per_token,
            "loop_entry": bool(spec.loop_entry),
            "loop_exit": bool(spec.loop_exit),
            "rope_tables": list(spec.rope_kinds),
            "gate_in": bool(spec.gate_in),
            "gate_out": spec.gate_out,
            "inputs": list(spec.inputs),
            "outputs": list(spec.outputs),
            "states": states,
            "state_bytes": sum(item["bytes"] for item in states),
        }
    return {
        "layout": "per_head_programs",
        "num_programs": len(specs),
        # which program produces each attention layer's gate leaf, so a runtime
        # can hand it to the one that consumes it without re-deriving the chain
        "gate_chain": {f"gate_{spec.gate_out}": spec.name
                        for spec in specs if spec.gate_out is not None},
        "heads_per_state": getattr(config, "heads_per_state", None),
        "slots_per_sliding_carrier": int(
            getattr(config, "slots_per_sliding_carrier", 3) or 3),
        "recurrent_dispatches_per_loop_pass": sum(
            1 for spec in specs if spec.is_recurrent),
        "programs": programs,
    }


class PerHeadConverter(DoryConverter):
    """Convert the per-(layer, head) plan instead of the five components.

    Every other part of the conversion -- tracing discipline, palettization, the
    per-file size guard, the ANE placement gate -- is DoryConverter's, unchanged.
    What changes is the plan and which buffers a package declares: a head program
    declares the one state it owns, and the unstated programs declare none.
    """

    @property
    def specs(self) -> list:
        return program_specs(self.config)

    def spec(self, name: str) -> ProgramSpec:
        for spec in self.specs:
            if spec.name == name:
                return spec
        raise KeyError(f"no program named {name}; the plan has {len(self.specs)}")

    def states(self, wrapper) -> list:
        """Only the buffers this program declared (never the 45-buffer union)."""
        import coremltools as ct

        states = []
        for attr in wrapper.model.state_buffer_names:
            buffer = getattr(wrapper.model, attr)
            states.append(ct.StateType(
                name=f"model.{attr}",
                wrapped_type=ct.TensorType(
                    shape=tuple(int(d) for d in buffer.shape), dtype=np.float16)))
        return states

    def _coreml_outputs(self, spec: ProgramSpec) -> list:
        """The graph output of a program, which for a block-like program is a
        hidden-state tensor under its own name.

        It cannot be called "hidden_states": that is the program's first *input*,
        and Core ML refuses a function whose variable is both ("Var names
        ['hidden_states'] is used both as function's input and output").
        """
        import coremltools as ct

        if spec.kind == "lm_head":
            return super()._coreml_outputs(spec)
        # the spec names its outputs, and a program in the gate chain has two:
        # its hidden state and the NEXT layer's gate leaf
        return [ct.TensorType(name=name, dtype=np.float16) for name in spec.outputs]

    def build_component(self, spec: ProgramSpec):
        """The program's modules and the wrapper that traces its signature."""
        config = self.config
        blocks = [int(b) for b in spec.blocks]
        head_map = ({blocks[0]: spec.kv_head} if spec.kind == "attn_head" else None)
        # a program holds the blocks it advances plus, when it emits the next
        # layer's gate, that layer as a gate-only module (g_proj and nothing
        # else) -- so the gate convolution lives in exactly one program per
        # attention layer across the whole conversion
        gate_only = [int(spec.gate_out)] if spec.gate_out is not None else []
        gate_in = [int(blocks[0])] if (spec.gate_in and blocks) else []
        # an epilogue finishes a layer the attention programs computed, so it
        # needs out_norm and o_proj only -- not the q/k/v it would never call
        tail_only = ([int(blocks[0])]
                     if (spec.kind in ("full_epilogue", "sliding_epilogue")
                         and spec.gate_in and blocks)
                     else [])
        # a sliding attention program is the attention and nothing else: no
        # gate convolution and no tail, which is the shape that builds a plan
        attn_only = ([int(blocks[0])]
                     if (spec.kind == "sliding_attn" and blocks) else [])
        # ... and it scores one KV head at a time, emitting one tensor per KV
        # head: a stateful program that materialises a 4096-wide tensor anywhere
        # -- the output, or the intermediate a chunked output would be cut from --
        # builds no plan (measured)
        contrib_per_head = ([int(blocks[0])]
                            if (spec.kind == "sliding_attn" and blocks) else [])
        held = list(blocks) + [b for b in gate_only if b not in blocks]
        model = _ComponentModel(
            config, held, load_lm_head=(spec.kind == "lm_head"),
            head_map=head_map, state_subset=spec.states,
            embed_vocab=(config.vocab_size if spec.kind == "embeddings" else 1),
            gate_in_blocks=gate_in, gate_only_blocks=gate_only,
            tail_only_blocks=tail_only, attn_only_blocks=attn_only,
            contrib_per_head_blocks=contrib_per_head)
        model.load_pretrained_weights(
            str(self.model_path), blocks=held,
            load_lm_head=(spec.kind == "lm_head"),
            load_embeddings=(spec.kind == "embeddings"))
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        model.model.reset_caches()

        if spec.kind == "embeddings":
            wrapper = dch.EmbeddingChunk(config, model.embed_tokens, model.model,
                                         gate_block=spec.gate_out)
        elif spec.kind == "lm_head":
            wrapper = dch.LMHeadChunk(config, model.lm_head.weight.data,
                                      self.lm_head_splits, model.model,
                                      argmax=self.argmax)
            model.lm_head = None
        elif spec.kind == "attn_head":
            cls = (dch.RopeHeadAttentionChunk if spec.rope_tables
                   else dch.HeadAttentionChunk)
            wrapper = cls(model.model, blocks[0], spec.kv_head)
        elif spec.kind == "sliding_group":
            cls = (dch.RopeSlidingGroupChunk if spec.rope_tables
                   else dch.SlidingGroupChunk)
            wrapper = cls(model.model, blocks[0], blocks[-1] + 1,
                          loop_entry=spec.loop_entry, loop_exit=spec.loop_exit)
        elif spec.kind == "sliding_layer":
            cls = (dch.RopeSlidingLayerChunk if spec.rope_tables
                   else dch.SlidingLayerChunk)
            wrapper = cls(model.model, blocks[0],
                          blocks[1] if len(blocks) > 1 else None,
                          gate_block=spec.gate_out,
                          loop_entry=spec.loop_entry, loop_exit=spec.loop_exit)
        elif spec.kind == "sliding_attn":
            cls = (dch.RopeSlidingAttentionChunk if spec.rope_tables
                   else dch.SlidingAttentionChunk)
            wrapper = cls(model.model, blocks[0])
        elif spec.kind == "entry_gate":
            wrapper = dch.EntryGateChunk(model.model, spec.gate_out)
        elif spec.kind in ("full_epilogue", "sliding_epilogue"):
            cls = (dch.SlidingEpilogueChunk if spec.kind == "sliding_epilogue"
                   else dch.FullEpilogueChunk)
            wrapper = cls(
                model.model, blocks[0], blocks[1] if len(blocks) > 1 else None,
                loop_exit=spec.loop_exit, gate_block=spec.gate_out,
                gate_in=spec.gate_in)
        else:
            raise ValueError(f"unknown program kind {spec.kind!r} for {spec.name}")
        wrapper.eval()
        for param in wrapper.parameters():
            param.requires_grad = False
        return model, wrapper

    def _package_metadata(self, spec: ProgramSpec, function: str) -> dict:
        meta = super()._package_metadata(spec, function)
        meta.update({
            "component_kind": spec.kind,
            "program_blocks": [int(b) for b in spec.blocks],
            "program_kv_head": spec.kv_head,
            "program_calls_per_token": spec.calls_per_token,
            "program_gate_in": bool(spec.gate_in),
            "program_gate_out": spec.gate_out,
            "declared_states": list(spec.states),
            "state_bytes": sum(
                int(np.prod(shape)) * 2
                for name, shape in dch.state_layout(self.config)
                if name in set(spec.states)),
        })
        return meta

    def cache_summary_line(self) -> str:
        """The per-program state budget: what a program may hold, not the union."""
        shapes = dict(dch.state_layout(self.config))
        sizes, largest_name, largest = [], None, 0.0
        for spec in self.specs:
            total = 0
            for name in spec.states:
                shape = shapes[name]
                elements = 1
                for dim in shape:
                    elements *= int(dim)
                total += elements * 2
            sizes.append(total)
            if total > largest:
                largest, largest_name = total, spec.name
        with_state = [size for size in sizes if size]
        dispatches = sum(1 for spec in self.specs if spec.is_recurrent)
        return ("%d programs, %d of them with state: %.1f-%.1f MiB each "
                "(largest: %s), %d dispatches per loop pass; measured ceiling "
                "~34 MiB per program"
                % (len(self.specs), len(with_state),
                   min(with_state) / 1024 ** 2 if with_state else 0.0,
                   largest / 1024 ** 2, largest_name, dispatches))

    def write_artifacts(self) -> dict:
        """The runtime directory: tokenizer, config, the plan in meta, and a report."""
        import json
        import shutil

        copied = self.copy_tokenizer()
        source_config = self.model_path / "config.json"
        if source_config.is_file():
            shutil.copy2(source_config, self.output_dir / "config.json")
        files = self.manifest_files()
        if tiling_flags(self.config)[1]:
            write_rope_tables(self.config, self.output_dir)
        meta = build_meta(
            config=self.config, specs=self.specs, files=files,
            lut_bits=self.lut_bits, lm_head_chunk_sizes=self.lm_head_chunk_sizes,
            prefix=self.prefix, argmax=self.argmax, per_program=True,
            tokenizer="tokenizer.json",
            chat_template=("chat_template.jinja" if "chat_template.jinja" in copied
                           else None))
        meta["per_head_programs"] = program_manifest(self.config, self.specs)
        meta["kv"]["program_layout"] = "per_head_programs"
        meta["kv"]["declared_states_by_program"] = {
            spec.name: list(spec.states) for spec in self.specs if spec.states}
        yaml_path = write_meta(self.output_dir / "meta.yaml", meta)
        report_path = self.output_dir / "conversion_report.json"
        report = {
            "layout": "per_head_programs",
            "programs": {name: {
                "bytes": {fn: self.sizes.get(f"{name}:{fn}")
                          for fn in meta["components"][name]["files"]},
                "seconds": self.seconds.get(name),
                "states": meta["per_head_programs"]["programs"][name]["states"],
            } for name in meta["run_order"]},
            "declared_states": declared_states(self.config, self.specs),
            "tokenizer_files": list(copied),
            "created": meta["model_info"]["created"],
        }
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        return {"meta": str(yaml_path), "report": str(report_path)}
