"""Per-(layer, head) programs must reproduce the aggregated path, exactly.

The claim this file gates is the one the whole DANE-5 conversion rests on: a
program that holds ONE (layer, head) cache and one head's rows of the layer's
projections computes exactly what that head computes inside the aggregated
DoryANEAttention group loop.  If that is not true, 36 separate programs cannot
be the same model as one program holding all four heads -- and the 24 MiB
per-state ceiling then has no compliant design at all.

What is compared, and against what
----------------------------------
The reference is the AGGREGATED layout, the shipped five-component path: one
model whose attention module holds the whole layer (all 16 query heads, all 4 KV
heads) and whose caches are the two aggregated buffers.  The per-head side is
four single-head programs, each with its own one-head state, plus the unstated
epilogue that finishes the layer (out_norm over 4096, o_proj, residual blend) and
computes the MLP block after it.  Both sides are driven with the same hidden
states, the same mask and the same positions, and their caches are carried from
step to step.

The tolerance is the one tests/test_cache_layout_numerics.py justifies: the
arithmetic is identical (same matmul, same softmax, same dtypes, same order
within a head), so the difference should be zero; a rel-1e-3 bound plus an exact
argmax-of-the-readout check is what the KLD gate actually consumes.
"""
from __future__ import annotations

import gc

import pytest
import torch
import torch.nn as nn

from anemll.ane_converter import dory_per_head as dph
from anemll.models import dory_ane_model as dam
from anemll.models import dory_chunks as dch
from anemll.utils import dory_caches as dc

#: one full-attention block and the MLP after it: the unit an epilogue covers
FULL_BLOCK = 6
MLP_BLOCK = 7
READOUT_VOCAB = 512


def _config(context_length: int, **overrides):
    return dam.DoryANEConfig(hidden_size=128, intermediate_size=96,
                             num_attention_heads=16, num_key_value_heads=4,
                             head_dim=256, vocab_size=1024,
                             context_length=context_length, batch_size=8,
                             compute_mode="matmul", **overrides)


def _seed_weights(model) -> None:
    """Fixed seed, and the magnitudes that make the softmax genuinely soft.

    The sq scale is ~256 in production (compute_sq), and the streaming test below
    relies on it: with a tiny sq every logit lands near zero, the softmax is
    uniform, and a head reading another head's cache would still come out equal.
    """
    torch.manual_seed(7)
    with torch.no_grad():
        for _, parameter in model.named_parameters():
            parameter.normal_(0.0, 0.05)
        for name, buffer in model.named_buffers():
            if name.startswith("kv_cache") or "cos_cached" in name or "sin_cached" in name:
                continue
            if not buffer.numel():
                continue
            if name.endswith(".sq"):
                buffer.copy_(torch.rand(buffer.shape, dtype=torch.float32) * 100.0 + 200.0)
                continue
            buffer.normal_(0.0, 0.05).abs_().add_(0.1)


def _copy_head_weights(agg, head) -> None:
    """The rows the loader slices (weight_row_slice / head_slice).

    Same rows, same values: the per-head module is the aggregated layer sliced,
    never re-initialised, so a difference below cannot come from the weights.
    """
    with torch.no_grad():
        for name in ("q_proj", "k_proj", "v_proj", "g_proj"):
            getattr(head, name).weight.copy_(
                getattr(agg, name).weight[head.weight_row_slice(name)])
        head.sq.copy_(head.head_slice(agg.sq))
        if head.sg.numel():
            head.sg.copy_(head.head_slice(agg.sg))


def _hidden(hidden_size: int, seq: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, seq, hidden_size, generator=generator, dtype=torch.float16)


class _Readout(nn.Module):
    """A seeded 1x1 readout, so the comparison lands where the KLD gate looks."""

    def __init__(self, hidden_size: int, vocab: int = READOUT_VOCAB, seed: int = 31):
        super().__init__()
        torch.manual_seed(seed)
        self.conv = nn.Conv2d(hidden_size, vocab, kernel_size=1, bias=False,
                              dtype=torch.float16)
        with torch.no_grad():
            self.conv.weight.normal_(0.0, 0.02)

    def forward(self, hidden_states):
        x = hidden_states.permute(0, 2, 1).unsqueeze(2)
        return self.conv(x).squeeze(2).permute(0, 2, 1).float()


def _steps(context_length: int):
    """(start, q_len) per call; the caches carry between them.

    The late steps are the interesting ones: near position 0 the softmax is
    one-hot and a head reading another head's cache would pass unnoticed, while
    deep in the context every head's K and V have to come from its own buffer.
    """
    if context_length >= 24576:
        return ((0, 8), (20000, 1))
    return ((0, 8), (8, 1), (1500, 1))


def _mask(config, start, q_len):
    return dc.build_causal_mask(start, q_len, config.context_length,
                                config.sliding_rows, config.sliding_window,
                                dtype=config.dtype)


def _logits(model, readout, config, chunk=None, steps=None):
    out = []
    steps = _steps(config.context_length) if steps is None else steps
    with torch.no_grad():
        for start, q_len in steps:
            hidden = _hidden(config.hidden_size, q_len, 100 + start)
            cache_pos = start % config.sliding_rows
            mask = _mask(config, start, q_len)
            h = (model.process_blocks(hidden, mask, start, cache_pos)
                 if chunk is None else chunk(hidden, mask, start, cache_pos))
            out.append(readout(h))
    return out


def _per_head_models(config, aggregated_attention, num_kv: int):
    """One single-head model per KV head, holding one state each."""
    chunks, models = [], []
    for head in range(num_kv):
        name = f"kv_cache_full_0_{head}"
        model = dam.DoryANEModel(config, block_indices=[FULL_BLOCK],
                                 head_map={FULL_BLOCK: head},
                                 state_subset=(name,)).eval()
        assert model.state_buffer_names == (name,), "one state per head program"
        assert tuple(model.state_buffers[0].shape) == (
            2, 1, config.context_length, config.head_dim)
        _copy_head_weights(aggregated_attention, model.layers[0].mixer)
        chunks.append(dch.HeadAttentionChunk(model, FULL_BLOCK, head).eval())
        models.append(model)
    return chunks, models


def _epilogue_model(config, aggregated_model):
    model = dam.DoryANEModel(config, block_indices=[FULL_BLOCK, MLP_BLOCK],
                             state_subset=()).eval()
    assert model.state_buffer_names == ()
    with torch.no_grad():
        model.layers[0].load_state_dict(aggregated_model.layers[0].state_dict())
        model.layers[1].load_state_dict(aggregated_model.layers[1].state_dict())
    return model, dch.FullEpilogueChunk(model, FULL_BLOCK, MLP_BLOCK).eval()


@pytest.mark.parametrize("context_length", [4096, 24576])
def test_head_programs_plus_epilogue_match_the_aggregated_layer(context_length):
    """Four head programs + the epilogue == the aggregated layer, per step."""
    config = _config(context_length)
    aggregated = dam.DoryANEModel(config, block_indices=[FULL_BLOCK, MLP_BLOCK]).eval()
    _seed_weights(aggregated)
    readout = _Readout(config.hidden_size).eval()
    expected = _logits(aggregated, readout, config)
    attention = aggregated.layers[0].mixer

    # the per-head side is a conversion config: one head per state, so the
    # single-head models own the buffer the plan names
    config_split = _config(context_length, heads_per_state=1,
                           slots_per_sliding_carrier=3)
    head_chunks, head_models = _per_head_models(
        config_split, attention, int(config.num_key_value_heads))
    _epilogue_m, epilogue = _epilogue_model(config_split, aggregated)
    for model in head_models:
        model.reset_caches()

    actual = []
    with torch.no_grad():
        for start, q_len in _steps(context_length):
            hidden = _hidden(config.hidden_size, q_len, 100 + start)
            mask = _mask(config, start, q_len)
            cache_pos = start % config.sliding_rows
            contributions = [chunk(hidden, mask, start, cache_pos)
                             for chunk in head_chunks]
            actual.append(readout(epilogue(hidden, *contributions)))

    assert len(actual) == len(expected) == len(_steps(context_length))
    for index, (got, want) in enumerate(zip(actual, expected)):
        assert got.shape == want.shape
        max_abs = (got - want).abs().max().item()
        scale = want.abs().max().item() or 1.0
        assert max_abs <= 1e-3 * scale, (
            f"step {index} at context {context_length}: the per-head decomposition "
            f"diverged from the aggregated path, max abs {max_abs:.3e} against scale "
            f"{scale:.3e}")
        assert torch.equal(got.argmax(-1), want.argmax(-1)), (
            f"step {index} at context {context_length}: argmax differs, so the "
            f"decomposition changes which token the readout picks")


def test_sliding_group_matches_the_aggregated_chunk():
    """A sliding group over blocks 0-5 == the same blocks in one aggregated chunk."""
    context_length = 4096
    config_ref = _config(context_length)
    reference = dam.DoryANEModel(config_ref, block_indices=list(range(0, 6))).eval()
    _seed_weights(reference)
    readout = _Readout(config_ref.hidden_size).eval()
    expected = _logits(reference, readout, config_ref)

    weights = {name: tensor.detach().clone()
               for name, tensor in reference.state_dict().items()}
    del reference
    gc.collect()

    config = _config(context_length, heads_per_state=1, slots_per_sliding_carrier=3)
    carrier = "kv_cache_sliding_0_2"
    group = dam.DoryANEModel(config, block_indices=list(range(0, 6)),
                             state_subset=(carrier,)).eval()
    group.load_state_dict(weights)
    assert group.state_buffer_names == (carrier,)
    chunk = dch.SlidingGroupChunk(group, 0, 6).eval()
    actual = _logits(group, readout, config, chunk=chunk)

    for index, (got, want) in enumerate(zip(actual, expected)):
        max_abs = (got - want).abs().max().item()
        scale = want.abs().max().item() or 1.0
        assert max_abs <= 1e-3 * scale, (
            f"step {index}: the sliding group diverged from the aggregated chunk, "
            f"max abs {max_abs:.3e} against scale {scale:.3e}")


# -- the plan and its declared states -------------------------------------

def test_24k_plan_declares_one_24_mib_state_per_head_program():
    """The declared state list of a 24K conversion, with sizes."""
    config = _config(24576, heads_per_state=1, slots_per_sliding_carrier=1,
                     rope_tables_as_inputs=True)
    specs = dph.program_specs(config)
    kinds = {}
    for spec in specs:
        kinds[spec.kind] = kinds.get(spec.kind, 0) + 1
    assert kinds == {"embeddings": 1, "entry_gate": 1, "sliding_attn": 27,
                     "sliding_epilogue": 27, "attn_head": 36,
                     "full_epilogue": 9, "lm_head": 1}

    rows = dph.declared_states(config, specs)
    full = [row for row in rows if row["name"].startswith("kv_cache_full")]
    sliding = [row for row in rows if row["name"].startswith("kv_cache_sliding")]
    assert len(full) == 36 and len(sliding) == 27
    assert all(row["shape"] == [2, 1, 24576, 256] for row in full)
    assert all(row["mib"] == pytest.approx(24.0) for row in full)
    # one sliding layer's own carrier: K and V of 4 KV heads over 1,280 rows
    assert all(row["shape"] == [2, 4, 1280, 256] for row in sliding)
    assert all(row["mib"] == pytest.approx(5.0) for row in sliding)
    assert sum(row["mib"] for row in full) == pytest.approx(864.0)
    assert sum(row["mib"] for row in rows) == pytest.approx(999.0)
    assert all(row["program"] for row in rows), "a state no program declares"
    manifest = dph.program_manifest(config, specs)
    for name, program in manifest["programs"].items():
        assert program["state_bytes"] <= 30 * 1024 ** 2, (
            f"{name} declares {program['state_bytes'] / 1024 ** 2:.1f} MiB")


def test_24k_head_program_declares_exactly_one_state():
    """A head program is built with one buffer, and its traced graph reads it."""
    config = _config(24576, heads_per_state=1, slots_per_sliding_carrier=1)
    name = "kv_cache_full_2_1"
    model = dam.DoryANEModel(config, block_indices=[22], head_map={22: 1},
                             state_subset=(name,)).eval()
    assert model.state_buffer_names == (name,)
    assert model.state_buffers[0].numel() * 2 == 24576 * 256 * 2 * 2
    chunk = dch.HeadAttentionChunk(model, 22, 1).eval()
    hidden = torch.zeros(1, 1, config.hidden_size, dtype=torch.float16)
    mask = torch.zeros(1, 1, 1, config.mask_len, dtype=torch.float16)
    pos = torch.tensor([0], dtype=torch.int32)
    with torch.no_grad():
        traced = torch.jit.trace(chunk, (hidden, mask, pos, pos), strict=False)
    graph = str(traced.inlined_graph)
    assert name in graph, "the traced graph never reads its own state"
    assert "kv_cache_sliding" not in graph, "a head program touched a sliding buffer"


def test_plan_covers_every_block_once_and_rejects_a_broken_one():
    """The plan validator actually rejects plans; it is not a rubber stamp."""
    config = _config(24576, heads_per_state=1, slots_per_sliding_carrier=1)
    specs = dph.program_specs(config)
    advanced = []
    for spec in specs:
        if spec.kind not in ("attn_head", "sliding_attn"):
            advanced.extend(spec.blocks)
    assert sorted(advanced) == list(range(int(config.num_hidden_layers)))

    broken = [spec for spec in specs if spec.name != "epilogue_3"]
    with pytest.raises(ValueError, match="inconsistent"):
        dph.validate_program_specs(config, broken)

    # losing the unstated half of a sliding layer leaves its attention output
    # with nobody to finish the block, and the plan must refuse it
    for name in ("sepilogue_L0", "sliding_attn_L0"):
        broken = [spec for spec in specs if spec.name != name]
        with pytest.raises(ValueError, match="inconsistent"):
            dph.validate_program_specs(config, broken)


def test_the_shipped_4096_plan_is_untouched():
    """The existing conversion path must not move: five components, two states."""
    from anemll.ane_converter import dory_converter as dconv

    config = _config(4096)
    specs = dconv.component_specs(config)
    assert [spec.name for spec in specs] == ["embeddings", "input", "recurrent",
                                             "output", "lm_head"]
    assert [spec.calls_per_token for spec in specs] == [1, 1, "loops", 1, 1]
    assert len(dch.state_layout(config)) == 2
    assert dch.state_attrs(config) == ("kv_cache_full", "kv_cache_sliding")
    with pytest.raises(ValueError, match="heads_per_state=1"):
        dph.program_specs(config)


# ---------------------------------------------------------------------------
# The plan as a whole: one program per sliding layer, the gate travelling
#
# The measured rule this section tests the wiring of: a program that owns a
# cache must not carry a second convolution chain that joins its output, so no
# sliding layer computes its own gate.  Every attention layer's gate is emitted
# by the program that produces its input and taken as a leaf by the program
# that consumes it, and the numbers must not move because of it.
# ---------------------------------------------------------------------------

#: one whole 8-block period: sliding, mlp, sliding, mlp, sliding, mlp, full, mlp
PERIOD = tuple(range(0, 8))
SLIDING = (0, 2, 4)
FULL_BLOCK = 6
MLP_AFTER_FULL = 7

#: what a layer of each mode never builds, and therefore never loads
GONE_GATE_IN = ("mixer.g_proj.weight",)
GONE_GATE_ONLY = ("mixer.q_proj.weight", "mixer.k_proj.weight",
                  "mixer.v_proj.weight", "mixer.o_proj.weight")
GONE_PER_HEAD = ("mixer.o_proj.weight",)
#: the epilogue finishes a layer the head programs computed: no attention, no gate
GONE_TAIL = ("mixer.q_proj.weight", "mixer.k_proj.weight", "mixer.v_proj.weight")
GONE_GATE_IN_TAIL = GONE_GATE_IN + GONE_TAIL
#: a sliding attention program is the attention only: no gate, no tail
GONE_ATTN_ONLY = ("mixer.g_proj.weight", "mixer.o_proj.weight")


def _tokens(config, start, q_len):
    """The same ids for the same step on both sides (a per-step seed)."""
    generator = torch.Generator().manual_seed(9000 + int(start))
    return torch.randint(0, config.vocab_size, (1, int(q_len)), generator=generator)


def _embedding(config):
    module = nn.Embedding(config.vocab_size, config.hidden_size, dtype=torch.float16)
    generator = torch.Generator().manual_seed(4242)
    with torch.no_grad():
        module.weight.normal_(0.0, 0.02, generator=generator)
    return module


def _copy_layer(dst, src, gone=()):
    """Give a program's layer the aggregated layer's weights.

    A gate-in layer has no g_proj and a gate-only layer has no attention, so
    the tensors a mode removes are the only ones allowed to be missing: that
    assertion is what stops a program quietly loading half a layer.
    """
    # a mode REMOVES tensors, so they show up as keys the destination has no
    # room for; nothing the destination expects may be left unfilled
    missing, unexpected = dst.load_state_dict(src.state_dict(), strict=False)
    assert not missing, missing
    assert set(unexpected) <= set(gone), (sorted(unexpected), sorted(gone))


def _layer(model, block):
    return model.layers[model.block_indices.index(block)]


def _sliding_carrier(config, block):
    return dch.program_state_names(config, blocks=(block,))[0]


def _build_chain(config, reference, embed):
    """The plan's programs for one period, wired exactly as the plan wires them.

    Returns the embeddings program, the three sliding attention programs, the
    three slide epilogues, the epilogue and the four head programs -- one object
    per Core ML file the conversion would write for blocks 0-7.
    """
    gate_model = dam.DoryANEModel(config, block_indices=[SLIDING[0]],
                                  state_subset=(), gate_only_blocks=[SLIDING[0]]).eval()
    _copy_layer(_layer(gate_model, SLIDING[0]), _layer(reference, SLIDING[0]),
                gone=GONE_GATE_ONLY)
    embeddings = dch.EmbeddingChunk(config, embed, gate_model,
                                    gate_block=SLIDING[0]).eval()

    attention, epilogues = {}, {}
    for block in SLIDING:
        # the stateful half: this layer's carrier, its attention, nothing else
        attn_model = dam.DoryANEModel(
            config, block_indices=[block],
            state_subset=(_sliding_carrier(config, block),),
            attn_only_blocks=[block],
            contrib_per_head_blocks=[block]).eval()
        _copy_layer(_layer(attn_model, block), _layer(reference, block),
                    gone=GONE_ATTN_ONLY)
        attention[block] = dch.SlidingAttentionChunk(attn_model, block).eval()

        # the unstated half: gate in, layer tail, the MLP, the next gate out
        model = dam.DoryANEModel(
            config, block_indices=[block, block + 1, block + 2],
            state_subset=(), gate_in_blocks=[block], tail_only_blocks=[block],
            gate_only_blocks=[block + 2]).eval()
        _copy_layer(_layer(model, block), _layer(reference, block),
                    gone=GONE_GATE_IN_TAIL)
        _copy_layer(_layer(model, block + 1), _layer(reference, block + 1))
        _copy_layer(_layer(model, block + 2), _layer(reference, block + 2),
                    gone=GONE_GATE_ONLY)
        epilogues[block] = dch.SlidingEpilogueChunk(
            model, block, block + 1, gate_block=block + 2, gate_in=True).eval()

    epilogue_model = dam.DoryANEModel(
        config, block_indices=[FULL_BLOCK, MLP_AFTER_FULL], state_subset=(),
        gate_in_blocks=[FULL_BLOCK], tail_only_blocks=[FULL_BLOCK]).eval()
    _copy_layer(_layer(epilogue_model, FULL_BLOCK), _layer(reference, FULL_BLOCK),
                gone=GONE_GATE_IN_TAIL)
    _copy_layer(_layer(epilogue_model, MLP_AFTER_FULL),
                _layer(reference, MLP_AFTER_FULL))
    epilogue = dch.FullEpilogueChunk(epilogue_model, FULL_BLOCK, MLP_AFTER_FULL,
                                     gate_in=True).eval()

    heads = {}
    for head in range(int(config.num_key_value_heads)):
        name = f"kv_cache_full_0_{head}"
        model = dam.DoryANEModel(config, block_indices=[FULL_BLOCK],
                                 head_map={FULL_BLOCK: head},
                                 state_subset=(name,)).eval()
        assert model.state_buffer_names == (name,), "one state per head program"
        mixer = _layer(model, FULL_BLOCK).mixer
        _copy_head_weights(_layer(reference, FULL_BLOCK).mixer, mixer)
        heads[head] = dch.HeadAttentionChunk(model, FULL_BLOCK, head).eval()
    return embeddings, attention, epilogues, epilogue, heads


@pytest.mark.parametrize("context_length", [4096, 24576])
def test_the_per_layer_chain_reproduces_the_aggregated_period(context_length):
    """The whole plan -- embeddings, 3 sliding attention programs, 3 slide
    epilogues, 4 head programs, epilogue -- equals the aggregated model over one
    8-block period."""
    config = _config(context_length)
    reference = dam.DoryANEModel(config, block_indices=list(PERIOD)).eval()
    _seed_weights(reference)
    embed = _embedding(config)
    readout = _Readout(config.hidden_size).eval()
    steps = _steps(context_length)

    expected = []
    with torch.no_grad():
        for start, q_len in steps:
            mask = _mask(config, start, q_len)
            hidden = embed(_tokens(config, start, q_len))
            expected.append(readout(reference.process_blocks(
                hidden, mask, start, start % config.sliding_rows)))

    # one head per state and one carrier per sliding layer: the conversion
    # config, not the aggregated one
    config_split = _config(context_length, heads_per_state=1,
                           slots_per_sliding_carrier=1)
    embeddings, attention, epilogues, epilogue, heads = _build_chain(
        config_split, reference, embed)

    gate_width = int(config.num_attention_heads) * int(config.head_dim)
    actual = []
    with torch.no_grad():
        for start, q_len in steps:
            mask = _mask(config, start, q_len)
            cache_pos = start % config.sliding_rows
            ids = _tokens(config, start, q_len)
            hidden, gate = embeddings(ids)
            assert torch.equal(hidden, embed(ids)), (
                "the embeddings program must emit the hidden state the aggregated "
                "path starts from")
            # the first link of the chain: the embeddings program's gate_0 is
            # bit for bit the gate the aggregated block 0 computes for itself
            assert torch.equal(gate, _layer(reference, SLIDING[0]).mixer.head_gate(hidden)), (
                "gate_0 is not the gate block 0 would have computed")
            h = hidden
            for block in SLIDING:
                # the layer's attention on its own, then the unstated program
                # that consumes the gate, finishes the layer and runs its MLP
                contribution = attention[block](h, mask, start, cache_pos)
                assert len(contribution) == dch.CONTRIBUTIONS
                for slice_ in contribution:
                    assert slice_.shape == (1, q_len, gate_width // dch.CONTRIBUTIONS), (
                        "a sliding attention program emits the layer's raw "
                        "contribution one KV head at a time: the rep query heads of "
                        "that head, un-gated")
                h, gate = epilogues[block](h, *contribution, gate)
            contributions = [heads[head](h, mask, start, cache_pos)
                             for head in sorted(heads)]
            actual.append(readout(epilogue(h, *contributions, gate)))

    assert len(actual) == len(expected) == len(steps)
    for index, (got, want) in enumerate(zip(actual, expected)):
        assert got.shape == want.shape
        max_abs = (got - want).abs().max().item()
        scale = want.abs().max().item() or 1.0
        assert max_abs <= 1e-3 * scale, (
            f"step {index} at context {context_length}: the per-layer chain diverged "
            f"from the aggregated path, max abs {max_abs:.3e} against scale {scale:.3e}")
        assert torch.equal(got.argmax(-1), want.argmax(-1)), (
            f"step {index} at context {context_length}: argmax differs, so the "
            f"decomposition changes which token the readout picks")


@pytest.mark.parametrize("context_length", [4096, 24576])
def test_the_sliding_attention_program_emits_the_aggregated_raw_contribution(
        context_length):
    """sliding_attn_L0 is block 0's attention, bit for bit, and nothing else.

    The aggregated module ends with finish(out * gate); with a gate of ones and
    finish() bypassed it returns exactly out -- the raw contribution the sliding
    attention program emits.  Same projections, same RoPE, same ring write and
    read, same matmul, so the two tensors have to be equal, not merely close.

    The program now scores ONE KV head per pass instead of four at once, so this
    also pins that the four single-head passes reproduce the batched matmul's
    channel blocks exactly -- at 4,096 and at the 24,576 context the conversion
    ships, where a scoring difference would hide in the ring.
    """
    config = _config(context_length)
    aggregated = dam.DoryANEModel(config, block_indices=[0]).eval()
    _seed_weights(aggregated)
    hidden = _hidden(config.hidden_size, 8, 1234)
    mask = _mask(config, 0, 8)

    config_split = _config(context_length, heads_per_state=1,
                           slots_per_sliding_carrier=1)
    carrier = _sliding_carrier(config_split, 0)
    model = dam.DoryANEModel(config_split, block_indices=[0], attn_only_blocks=[0],
                             contrib_per_head_blocks=[0],
                             state_subset=(carrier,)).eval()
    assert model.state_buffer_names == (carrier,), (
        "a sliding attention program declares its own layer's carrier")
    _copy_layer(_layer(model, 0), _layer(aggregated, 0), gone=GONE_ATTN_ONLY)
    program = dch.SlidingAttentionChunk(model, 0).eval()

    with torch.no_grad():
        got = program(hidden, mask, 0, 0)
        layer = _layer(aggregated, 0)
        _full, sliding = aggregated.cache_tables()
        _masks, mask_sliding = aggregated.split_mask(mask)
        ones = torch.ones(1, 8, layer.mixer.q_channels, dtype=torch.float16)
        finish = layer.mixer.finish
        layer.mixer.finish = lambda contributions: contributions
        try:
            want = layer.mixer(hidden, [mask_sliding], sliding[layer.slot],
                               layer.slot, layer.num_slots, 0, 0, None, None,
                               None, ones)
        finally:
            layer.mixer.finish = finish

    wide = int(config.num_attention_heads) * int(config.head_dim)
    width = wide // dch.CONTRIBUTIONS
    assert want.shape == (1, 8, wide)
    assert len(got) == dch.CONTRIBUTIONS, (
        "a sliding attention program emits one contribution per KV head")
    joined = torch.cat(list(got), dim=-1)
    assert torch.equal(joined, want), (
        "the sliding attention program is not the aggregated layer's attention, "
        f"max abs {(joined - want).abs().max().item():.3e}")
    for index, contribution in enumerate(got):
        assert contribution.shape == (1, 8, width)
        assert torch.equal(
            contribution, want[..., index * width:(index + 1) * width]), (
            f"contribution_h{index} is not KV head {index}'s channels of the "
            f"aggregated attention output: the slice order is the channel order "
            f"the epilogue concatenates in")

    # ... and the module really is the attention only
    mixer = _layer(model, 0).mixer
    assert mixer.g_proj is None and mixer.o_proj is None and mixer.out_norm is None


def test_the_sliding_attention_program_never_builds_the_wide_tensor(monkeypatch):
    """One KV head per pass, so the 4096-wide contribution never exists.

    Chunking a wide result is not enough: a stateful program that materialises the
    whole contribution anywhere builds no plan (measured).  This watches the
    matmuls the run makes.  The aggregated loop scores a KV-head GROUP at a time --
    a four-batched pair of matmuls -- while the program must score one head at a
    time, twice per KV head (scores, then the value product), and no call may carry
    the four-head axis or produce a 4096-wide row.
    """
    config = _config(4096, heads_per_state=1, slots_per_sliding_carrier=1)
    model = dam.DoryANEModel(
        config, block_indices=[0], state_subset=(_sliding_carrier(config, 0),),
        attn_only_blocks=[0], contrib_per_head_blocks=[0]).eval()
    _seed_weights(model)
    program = dch.SlidingAttentionChunk(model, 0).eval()
    hidden = _hidden(config.hidden_size, 8, 8080)
    mask = _mask(config, 0, 8)

    kv = int(config.num_key_value_heads)
    d = int(config.head_dim)
    calls = []
    real_matmul = torch.matmul

    def recording(first, second, *args, **kwargs):
        calls.append(tuple(int(dim) for dim in first.shape))
        return real_matmul(first, second, *args, **kwargs)

    monkeypatch.setattr(torch, "matmul", recording)
    try:
        with torch.no_grad():
            got = program(hidden, mask, 0, 0)
    finally:
        monkeypatch.undo()

    assert len(got) == kv
    for contribution in got:
        assert contribution.shape == (1, 8,
                                      int(config.num_attention_heads) * d // kv)
    batched = [shape for shape in calls
               if len(shape) == 4 and shape[-1] == d and shape[1] != 1]
    assert not batched, (
        f"the program scored {batched} -- a multi-head batch, which is the "
        f"aggregated loop's shape and builds the wide intermediate")
    per_head = [shape for shape in calls if len(shape) == 4 and shape[1] == 1]
    assert len(per_head) == 2 * kv, (
        f"expected one scores matmul and one value matmul per KV head "
        f"({2 * kv} calls), saw {len(per_head)}: {per_head}")


def test_the_wide_single_contribution_pair_is_the_same_numbers():
    """The 4096-wide pair the plan builder refuses is still constructible and equal.

    WideSlidingAttentionChunk emits ONE (1, seq, 4096) tensor and
    WideSlidingEpilogueChunk takes it back; the default pair emits four 1024-wide
    slices that the epilogue concatenates first.  The shape kept for re-measuring
    the refusal must not be a different model, so the two are compared on the
    same weights: writing the carrier twice at the same cache_pos is idempotent,
    so the wide call sees exactly the rows the split call did.
    """
    context_length = 4096
    reference = dam.DoryANEModel(
        _config(context_length), block_indices=[0, 1]).eval()
    _seed_weights(reference)

    config = _config(context_length, heads_per_state=1, slots_per_sliding_carrier=1)
    attn_model = dam.DoryANEModel(
        config, block_indices=[0], state_subset=(_sliding_carrier(config, 0),),
        attn_only_blocks=[0], contrib_per_head_blocks=[0]).eval()
    _copy_layer(_layer(attn_model, 0), _layer(reference, 0), gone=GONE_ATTN_ONLY)
    epi_model = dam.DoryANEModel(
        config, block_indices=[0, 1], state_subset=(),
        gate_in_blocks=[0], tail_only_blocks=[0]).eval()
    _copy_layer(_layer(epi_model, 0), _layer(reference, 0), gone=GONE_GATE_IN_TAIL)
    _copy_layer(_layer(epi_model, 1), _layer(reference, 1))

    split = dch.SlidingAttentionChunk(attn_model, 0).eval()
    wide = dch.WideSlidingAttentionChunk(attn_model, 0).eval()
    epilogue = dch.SlidingEpilogueChunk(epi_model, 0, 1, gate_in=True).eval()
    wide_epilogue = dch.WideSlidingEpilogueChunk(epi_model, 0, 1, gate_in=True).eval()

    hidden = _hidden(config.hidden_size, 8, 4321)
    mask = _mask(config, 0, 8)
    gate = _layer(reference, 0).mixer.head_gate(hidden)
    whole = int(config.num_attention_heads) * int(config.head_dim)

    with torch.no_grad():
        contributions = split(hidden, mask, 0, 0)
        one = wide(hidden, mask, 0, 0)
        h_split = epilogue(hidden, *contributions, gate)
        h_wide = wide_epilogue(hidden, one, gate)

    assert len(contributions) == dch.CONTRIBUTIONS
    assert one.shape == (1, 8, whole)
    assert torch.equal(one, torch.cat(list(contributions), dim=-1)), (
        "the wide output is not the four slices joined")
    assert torch.equal(h_wide, h_split), (
        "the wide pair and the split pair finish the layer differently")

    # The wide chunk also still accepts a layer built WITHOUT contrib_per_head, and
    # then it is the aggregated single pass itself: joining the slices above and
    # returning the wide tensor below have to be the same tensor.
    plain = dam.DoryANEModel(
        config, block_indices=[0], state_subset=(_sliding_carrier(config, 0),),
        attn_only_blocks=[0]).eval()
    _copy_layer(_layer(plain, 0), _layer(reference, 0), gone=GONE_ATTN_ONLY)
    wide_plain = dch.WideSlidingAttentionChunk(plain, 0).eval()
    with torch.no_grad():
        direct = wide_plain(hidden, mask, 0, 0)
    assert direct.shape == (1, 8, whole), (
        "the wide chunk must be ONE 4096-wide tensor on either layer")
    assert torch.equal(direct, one), (
        "the per-head passes and the aggregated pass are not the same tensor")


def test_the_gate_chain_covers_every_attention_layer_exactly_once():
    """No attention layer's gate is computed inside its own stateful program."""
    config = _config(24576, heads_per_state=1, slots_per_sliding_carrier=1,
                     rope_tables_as_inputs=True)
    specs = dph.program_specs(config)
    chain = dph.program_manifest(config, specs)["gate_chain"]
    attention = [b for b in range(int(config.num_hidden_layers))
                 if dc.block_is_attention(b)]
    assert sorted(int(name.split("_")[1]) for name in chain) == attention
    assert chain["gate_0"] == "embeddings"
    assert chain[f"gate_{int(config.input_layer_count)}"] == "entry_gate"

    for spec in specs:
        if spec.kind in ("sliding_epilogue", "full_epilogue"):
            assert spec.gate_in, f"{spec.name} would compute its own gate"
            assert "gate" in spec.inputs, f"{spec.name} has no gate leaf"
        if spec.kind == "sliding_attn":
            # the stateful half of a split sliding layer takes no gate and
            # emits none: it computes the attention and stops
            assert not spec.gate_in and spec.gate_out is None
            assert "gate" not in spec.inputs, (
                f"{spec.name} takes a gate, but a program that owns a cache must "
                f"not join one to its output")
            assert spec.outputs == dch.contribution_names(), (
                f"{spec.name} must emit one contribution per KV head, in channel "
                f"order, not one 4096-wide tensor")
        if spec.kind == "sliding_epilogue":
            assert spec.inputs == (["hidden_states"] + dch.contribution_names()
                                   + ["gate"]), (
                f"{spec.name} must take the same four contributions the full "
                f"path's epilogue takes, plus its gate leaf")
        if spec.kind == "attn_head":
            assert not spec.gate_in and spec.gate_out is None
        if spec.gate_out is not None:
            assert spec.outputs[-1] == f"gate_{spec.gate_out}", (
                f"{spec.name} emits {spec.outputs} but the chain says "
                f"gate_{spec.gate_out}")

    entry = dph.program_by_name(config, "entry_gate")
    assert entry.calls_per_token == "loops", (
        "the loop entry is re-entered with a different hidden state on every pass, "
        "so its gate has to be recomputed per pass")
    assert entry.inputs == ["hidden_states", "h_in"]
    assert dph.program_by_name(config, "embeddings").outputs == ["hidden_states",
                                                                 "gate_0"]


def test_a_sliding_attention_program_holds_one_convolution_chain():
    """sliding_attn_L0 is q/k/v and nothing else, and the plan says so.

    The gate must not be in this program: it owns a cache, and a second
    convolution chain joining its output is exactly what fails to build a plan.
    The parameter count is what the program actually builds, so the plan cannot
    quietly gain a conv back.
    """
    config = _config(2048, heads_per_state=1, slots_per_sliding_carrier=1)
    spec = dph.program_by_name(config, "sliding_attn_L0")
    model = dam.DoryANEModel(config, block_indices=list(spec.blocks),
                             attn_only_blocks=[spec.blocks[0]],
                             state_subset=spec.states)
    mixer = _layer(model, spec.blocks[0]).mixer
    assert mixer.g_proj is None, "a sliding attention program must not hold a gate"
    assert mixer.o_proj is None and mixer.out_norm is None, (
        "the layer tail belongs to the epilogue after it")
    assert sum(p.numel() for p in model.parameters()) == spec.params
    h = int(config.hidden_size)
    hd = int(config.head_dim)
    assert spec.params == (h * int(config.num_attention_heads) * hd
                           + 2 * h * int(config.num_key_value_heads) * hd)


def test_a_slide_epilogue_holds_the_gate_it_applies_and_the_next_one():
    """sepilogue_L0: the gate in as a leaf, the tail, the MLP, the next gate out.

    One gate convolution in the program -- the next layer's -- and the layer it
    finishes keeps o_proj and out_norm but no q/k/v and no g_proj.
    """
    config = _config(2048, heads_per_state=1, slots_per_sliding_carrier=1)
    spec = dph.program_by_name(config, "sepilogue_L0")
    assert spec.inputs == ["hidden_states", "contribution_h0", "contribution_h1",
                           "contribution_h2", "contribution_h3", "gate"]
    assert spec.outputs == ["h_next", f"gate_{spec.gate_out}"]
    assert spec.states == ()
    model = dam.DoryANEModel(
        config, block_indices=list(spec.blocks) + [spec.gate_out],
        gate_in_blocks=[spec.blocks[0]], tail_only_blocks=[spec.blocks[0]],
        gate_only_blocks=[spec.gate_out], state_subset=())
    mixer = _layer(model, spec.blocks[0]).mixer
    assert mixer.g_proj is None, "the gate arrives as a leaf"
    assert mixer.q_proj is None and mixer.k_proj is None and mixer.v_proj is None
    assert mixer.o_proj is not None and mixer.out_norm is not None, (
        "the epilogue is what finishes the layer")
    assert _layer(model, spec.gate_out).mixer.q_proj is None, (
        "the next layer is held for its gate only")
    assert sum(p.numel() for p in model.parameters()) == spec.params

    epilogue_spec = dph.program_by_name(config, "epilogue_0")
    epilogue = dam.DoryANEModel(
        config, block_indices=list(epilogue_spec.blocks) + [epilogue_spec.gate_out],
        gate_in_blocks=[epilogue_spec.blocks[0]],
        tail_only_blocks=[epilogue_spec.blocks[0]],
        gate_only_blocks=[epilogue_spec.gate_out], state_subset=())
    assert sum(p.numel() for p in epilogue.parameters()) == epilogue_spec.params


def test_the_unsplit_shapes_stay_selectable_for_measurement():
    """Both older sliding shapes are kept, because both are the measured failures.

    "group" is the three-layer sliding group; "layer" is the one-program-per-
    sliding-layer shape (attention, gate, MLP and the next gate in one stateful
    program) that the split replaces.  Neither may be what the default plan
    builds -- they are here so a load can be re-measured against them.
    """
    config = _config(4096, heads_per_state=1, slots_per_sliding_carrier=3)
    groups = [spec for spec in dph.program_specs(config, sliding="group")
              if spec.kind == "sliding_group"]
    assert len(groups) == 9
    assert all(not spec.gate_in and spec.gate_out is None for spec in groups)
    with pytest.raises(ValueError, match="slots_per_sliding_carrier has to be 1"):
        dph.program_specs(config)

    unsplit = dph.program_specs(
        _config(4096, heads_per_state=1, slots_per_sliding_carrier=1), sliding="layer")
    layers = [spec for spec in unsplit if spec.kind == "sliding_layer"]
    assert len(layers) == 27
    assert all(spec.gate_in and spec.gate_out is not None for spec in layers), (
        "the unsplit shape still hands the gate in and out; it is the MLP and the "
        "layer tail living in the same stateful program that fails")
    assert not [spec for spec in unsplit
                if spec.kind in ("sliding_attn", "sliding_epilogue")]
