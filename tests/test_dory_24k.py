"""24,576-token behaviour: the tile fallback, the sliding wrap and the config.

Spec D7 (caches and the tile fallback), D5 (ANE-safe cache writes), and the
DANE-6 gates.  The heavy numbers (ANE placement, recall, tok/s at 24K) come from
the gated runs recorded in docs/dory-24k.md; these tests pin the arithmetic that
those runs stand on: the tile split, the log-sum-exp merge, the ring at the wrap
boundary, and the cache budget.
"""

from __future__ import annotations

import math

import numpy as np

import pytest
import torch

from anemll.models import dory_ane_model as dm
from anemll.utils import dory_caches as dc

from tests.test_dory_ane_model import TINY, _damped, _write_safetensors, tiny_weights  # noqa: F401


CONTEXT_24K = 24576
TILE_24K = (16384, 8192)


# ---------------------------------------------------------------------------
# The shipped configuration and its budget
# ---------------------------------------------------------------------------

def test_24k_full_caches_are_nine_of_96_mib():
    """Spec D7: 9 full-attention caches x 24,576 rows x 4 KV heads x 256 = 864 MiB."""
    shapes = dc.full_cache_shapes(CONTEXT_24K, 4, 256)
    assert shapes == [(18, 4, CONTEXT_24K, 256)]
    per_cache = 2 * dc.full_cache_rows(CONTEXT_24K) * 4 * 256 * 2
    assert per_cache == 96 * 1024 * 1024
    assert len(shapes[0]) and len(dc.full_attention_blocks()) == 9
    assert per_cache * 9 == 864 * 1024 * 1024


def test_24k_config_budget_including_sliding_caches():
    config = dm.DoryANEConfig(context_length=CONTEXT_24K)
    model = dm.DoryANEModel(config, block_indices=[])
    assert config.full_rows == CONTEXT_24K
    assert config.sliding_rows == dc.sliding_cache_rows(CONTEXT_24K, 1024, 64) == 1280
    assert config.mask_len == CONTEXT_24K + 1280
    assert not config.tiled
    assert model.cache_bytes() == (864 + 135) * 1024 * 1024  # 999 MiB measured


def test_tile_fallback_is_16384_plus_8192_and_loses_nothing():
    tiles = dc.full_cache_tiles(CONTEXT_24K, TILE_24K)
    assert tiles == TILE_24K
    assert sum(tiles) == CONTEXT_24K
    assert dc.full_tile_offsets(tiles) == (0, 16384)
    config = dm.DoryANEConfig(context_length=CONTEXT_24K, full_cache_tiles=TILE_24K)
    assert config.tiled
    model = dm.DoryANEModel(config, block_indices=[])
    assert [tuple(b.shape) for b in model.full_caches] == [(18, 4, 16384, 256),
                                                           (18, 4, 8192, 256)]
    # the same bytes in two buffers: a tile split changes the op width, not the bill
    assert model.cache_bytes() == dm.DoryANEModel(dm.DoryANEConfig(
        context_length=CONTEXT_24K), block_indices=[]).cache_bytes()


@pytest.mark.parametrize("tiles", [(16384, 16384), (24576, 8192),
                                   (16384, 8192, 4096), (0, 24576)])
def test_a_tile_layout_that_does_not_add_up_is_refused(tiles):
    with pytest.raises(ValueError):
        dc.full_cache_tiles(CONTEXT_24K, tiles)


def test_24k_masks_are_the_context_plus_the_ring():
    masks = dc.build_tile_masks(CONTEXT_24K - 70, 64, TILE_24K, 1280, 1024)
    assert [tuple(m.shape) for m in masks] == [(1, 1, 64, 16384), (1, 1, 64, 8192),
                                               (1, 1, 64, 1280)]
    # a query in the second tile sees nothing in the first tile that is its future
    whole = dc.build_causal_mask(CONTEXT_24K - 70, 64, CONTEXT_24K, 1280, 1024)
    assert torch.equal(masks[0], whole.narrow(-1, 0, 16384))
    assert torch.equal(masks[1], whole.narrow(-1, 16384, 8192))
    assert torch.equal(masks[2], whole.narrow(-1, CONTEXT_24K, 1280))


def test_a_tile_the_query_cannot_see_is_entirely_masked():
    """Positions in the first tile: the second tile is all -inf, not NaN."""
    masks = dc.build_tile_masks(4, 2, TILE_24K, 1280, 1024)
    assert torch.isinf(masks[1]).all()
    assert (masks[1] < 0).all()
    assert not torch.isinf(masks[0][0, 0, :, :6]).all()


# ---------------------------------------------------------------------------
# The log-sum-exp merge (spec D7's combine)
# ---------------------------------------------------------------------------

def test_merge_of_two_tiles_equals_one_softmax():
    torch.manual_seed(0)
    rows_a, rows_b, d = 16, 8, 4
    q = torch.randn(1, 2, 5, d)
    k_a, k_b = torch.randn(1, 2, rows_a, d), torch.randn(1, 2, rows_b, d)
    v_a, v_b = torch.randn(1, 2, rows_a, d), torch.randn(1, 2, rows_b, d)
    scale = 1.0 / math.sqrt(d)

    # reference: one concatenated cache, one softmax
    keys = torch.cat([k_a, k_b], dim=2)
    values = torch.cat([v_a, v_b], dim=2)
    want = torch.softmax(torch.matmul(q, keys.transpose(-1, -2)) * scale, dim=-1)
    want = torch.matmul(want, values)

    logits_a = torch.matmul(q, k_a.transpose(-1, -2)) * scale
    logits_b = torch.matmul(q, k_b.transpose(-1, -2)) * scale
    got, _ = dc.merge_attention_partials([(logits_a, v_a), (logits_b, v_b)])
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-5)


def test_merge_ignores_a_tile_that_is_entirely_masked():
    torch.manual_seed(1)
    d = 4
    q = torch.randn(1, 1, 3, d)
    k_a, v_a = torch.randn(1, 1, 6, d), torch.randn(1, 1, 6, d)
    k_b, v_b = torch.randn(1, 1, 4, d), torch.randn(1, 1, 4, d)
    logits_a = torch.matmul(q, k_a.transpose(-1, -2))
    logits_b = torch.full((1, 1, 3, 4), float("-inf"))
    got, total = dc.merge_attention_partials([(logits_a, v_a), (logits_b, v_b)])
    want = torch.softmax(logits_a, dim=-1) @ v_a
    assert torch.allclose(got, want, rtol=1e-5, atol=1e-5)
    # total is the softmax denominator (not the normalised weights): one tile
    denominator = torch.exp(logits_a - logits_a.amax(dim=-1, keepdim=True)).sum(-1, keepdim=True)
    assert torch.allclose(total, denominator, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# The sliding cache across the wrap boundary (fill then rotate)
# ---------------------------------------------------------------------------

def test_fill_then_rotate_holds_exactly_the_last_rows():
    """The ring and a physical shift hold the same rows, in different orders.

    rows > window on purpose: a rotate can only drop rows that are already
    outside the window, which is why the cache is sized window + batch.
    """
    rows, window, dim = 12, 8, 3
    ring = torch.zeros(2, 1, rows, dim)
    shifted = torch.zeros(2, 1, rows, dim)
    history = []
    generator = torch.Generator().manual_seed(7)
    for pos in range(4 * rows):
        key = torch.randn(1, 1, 1, dim, generator=generator)
        value = torch.randn(1, 1, 1, dim, generator=generator)
        history.append((key.clone(), value.clone()))
        if dc.sliding_phase(pos, window) == "fill":
            dc.write_cache_rows(ring, 0, key, value, pos, 1)
            dc.write_cache_rows(shifted, 0, key, value, pos, 1)
        else:
            dc.write_cache_rows(ring, 0, key, value, pos % rows, 1)
            dc.rotate_cache_rows(shifted, 0, key, value, rows, 1)

    last = len(history) - 1
    # the shifted cache is chronological: row r holds position last - rows + 1 + r
    for row in range(rows):
        key, value = history[last - rows + 1 + row]
        assert torch.equal(shifted[0, 0, row], key[0, 0, 0])
        assert torch.equal(shifted[1, 0, row], value[0, 0, 0])
        # the ring holds the same position in slot pos % rows
        slot = (last - rows + 1 + row) % rows
        assert torch.equal(ring[0, 0, slot], key[0, 0, 0])
        assert torch.equal(ring[1, 0, slot], value[0, 0, 0])


def test_the_ring_mask_at_the_wrap_is_exactly_the_window():
    """One mask column per slot: after the wrap each query sees 1,024 rows."""
    rows, window = 1088, 1024
    newest = 4999                                     # far past the first wrap
    mask = dc.sliding_ring_mask(newest, 1, rows, window)
    visible = (mask[0, 0, 0] == 0).sum().item()
    assert visible == window
    # and the visible slots are the newest window positions, in ring order
    slots = (mask[0, 0, 0] == 0).nonzero().flatten().tolist()
    assert len(slots) == window
    positions = sorted(newest - ((newest - slot) % rows) for slot in slots)
    assert positions == list(range(newest - window + 1, newest + 1))


def test_a_batch_that_straddles_the_ring_seam_is_refused():
    rows = dc.sliding_cache_rows(CONTEXT_24K, 1024, 64)      # 1280
    assert rows % 64 == 0
    with pytest.raises(ValueError):
        dc.cache_positions(rows - 32, 64, rows)


# ---------------------------------------------------------------------------
# The tiled model against the single-cache model
# ---------------------------------------------------------------------------

TILE_TINY = dict(TINY, context_length=16, sliding_window=8, batch_size=4)


def _build(path, module, **overrides):
    config = module.DoryANEConfig(**{**TILE_TINY, "fold_scales": True, **overrides})
    model = module.DoryANEForCausalLM(config)
    model.load_pretrained_weights(str(path))
    for block in model.model.layers:
        block.alpha.data.mul_(0.3)         # contractive: rounding cannot hide a bug
    model.eval()
    return model


@pytest.fixture(scope="module")
def weights_path(tmp_path_factory, tiny_weights):  # noqa: F811
    path = tmp_path_factory.mktemp("tiles") / "tiny.safetensors"
    _write_safetensors(str(path), _damped(tiny_weights))
    return path


@pytest.fixture(scope="module")
def tiled_and_single(weights_path):
    tiled = _build(weights_path, dm, full_cache_tiles=(8, 8))
    single = _build(weights_path, dm)
    assert tiled.config.tiled and not single.config.tiled
    return tiled, single


def test_the_tiled_model_matches_the_single_cache(tiled_and_single):
    """Same weights, same tokens: an 8+8 split must not move the answer.

    The two paths sum the same softmax in a different order (one 16-row
    denominator against two partials merged by log-sum-exp), so they agree to
    float32 rounding rather than bit for bit -- the gate that matters at 24K is
    the KLD against the reference, not bit equality.

    Prefill covers the tile boundary inside one batch (positions 6..13), which
    the eager path writes per tile; decode then walks past the boundary one
    token at a time.
    """
    tiled, single = tiled_and_single
    ids = [3, 127, 64, 9, 12, 88, 5, 41]
    got = tiled.prefill(ids, 0, loops=2).detach()
    want = single.prefill(ids, 0, loops=2).detach()
    assert torch.allclose(got, want, rtol=1e-4, atol=5e-4), (got - want).abs().max()

    tiled.reset_caches()
    single.reset_caches()
    straddling = [3, 127, 64, 9, 12, 88, 5, 41]
    got = tiled.prefill(straddling, 0, loops=2).detach()
    want = single.prefill(straddling, 0, loops=2).detach()
    assert torch.allclose(got, want, rtol=1e-4, atol=5e-4)

    # decode across positions 8 (the boundary) and past the sliding window
    tiled.reset_caches()
    single.reset_caches()
    tiled.prefill(ids, 0, loops=1)
    single.prefill(ids, 0, loops=1)
    for pos, token in enumerate([4, 8, 15, 16, 23, 42, 7], start=len(ids)):
        got = tiled.decode_step(token, pos, loops=1).detach()
        want = single.decode_step(token, pos, loops=1).detach()
        assert torch.allclose(got, want, rtol=1e-4, atol=5e-4), pos


def test_tile_write_inputs_are_one_hot_in_position(tiled_and_single):
    """The traced path's (index, gate) pair is right on both sides of the seam."""
    tiled, _ = tiled_and_single
    writes = tiled.tile_write_inputs(4, 4)
    assert set(writes) == {"write_index_1", "write_gate_1",
                           "write_index_2", "write_gate_2"}
    assert writes["write_index_1"].item() == 4 and writes["write_gate_1"].item() == 1.0
    assert writes["write_gate_2"].item() == 0.0          # this batch is all tile 1
    writes = tiled.tile_write_inputs(12, 4)
    assert writes["write_index_2"].item() == 4 and writes["write_gate_2"].item() == 1.0
    assert writes["write_gate_1"].item() == 0.0
    assert tiled.tile_write_inputs(0, 1)["write_index_1"].item() == 0
    # a batch across the seam cannot be expressed by one write per tile
    with pytest.raises(ValueError):
        tiled.tile_write_inputs(6, 4)


def test_the_gated_write_reproduces_the_eager_write(weights_path):
    """A traced chunk (index + gate) writes what the eager chunk writes.

    This is the contract the converted tiled package relies on: the graph only
    ever lerps at a host-computed index, so the two must agree exactly.
    """
    tiled = _build(weights_path, dm, full_cache_tiles=(8, 8))
    ids = [3, 127, 64, 9, 12, 88, 5, 41]
    mask = tiled.build_mask(0, len(ids))
    hidden = tiled.embed_tokens(torch.tensor([ids]))

    chunk = tiled.chunk_input
    eager = chunk(hidden, mask, 0, 0).detach()
    tiled.reset_caches()
    writes = tiled.tile_write_inputs(0, len(ids))
    gated = chunk(hidden, mask, torch.tensor([0], dtype=torch.int32),
                  torch.tensor([0], dtype=torch.int32), **writes).detach()
    assert torch.allclose(eager, gated, rtol=1e-6, atol=1e-6)

    # and past the seam: a single-token write gated to the second tile
    tiled.reset_caches()
    one = hidden[:, :1]
    mask_one = tiled.build_mask(12, 1)
    eager = chunk(one, mask_one, 12, 12).detach()
    tiled.reset_caches()
    writes = tiled.tile_write_inputs(12, 1)
    gated = chunk(one, mask_one, torch.tensor([12], dtype=torch.int32),
                  torch.tensor([12], dtype=torch.int32), **writes).detach()
    assert torch.allclose(eager, gated, rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# The runtime at 24K: progress, the over-context error, the tiled inputs
# ---------------------------------------------------------------------------

def test_prefill_reports_progress_and_a_rate():
    """A 24K prompt must never look hung: the callback fires, and it ends at 100%."""
    from tests.test_dory_chat import make_runtime

    runtime = make_runtime()
    seen = []
    rows = runtime.prefill(np.arange(200, dtype=np.int32), 0,
                           progress=lambda done, total, elapsed: seen.append((done, total)))
    assert rows.shape == (200, runtime.params["vocab_size"])
    assert seen[-1] == (200, 200)
    assert [done for done, _ in seen] == sorted(done for done, _ in seen)
    assert len(seen) < 200            # batched, not one line per token


def test_a_prompt_over_the_context_is_one_clear_error():
    """Spec user story 13: a clear error, not a crash, not silent truncation."""
    from tests.test_dory_chat import make_runtime
    from anemll import dory_chat as chat

    runtime = make_runtime()
    context = int(runtime.params["context_length"])
    with pytest.raises(chat.ContextOverflowError) as raised:
        runtime.prefill(np.arange(context + 5, dtype=np.int32), 0)
    message = str(raised.value)
    assert f"{context + 5:,}" in message and f"{context:,}" in message
    assert "too long" in message and "not truncated silently" in message
    assert raised.value.used == context + 5 and raised.value.context == context


def test_the_cli_turns_the_overflow_into_a_line_and_exit_code_2(capsys, monkeypatch):
    from anemll import dory_chat as chat

    def explode(args):
        raise chat.ContextOverflowError("the prompt is 30,000 tokens and this model's "
                                        "context is 24,576 tokens", used=30000, context=24576)

    monkeypatch.setattr(chat, "run_session", explode)
    code = chat.main(["chat", "--model-dir", "nowhere"])
    assert code == 2
    captured = capsys.readouterr()
    assert "error: the prompt is 30,000 tokens" in captured.err
    assert "Traceback" not in captured.err


def test_decode_past_the_context_is_refused():
    from tests.test_dory_chat import make_runtime
    from anemll import dory_chat as chat

    runtime = make_runtime()
    context = int(runtime.params["context_length"])
    with pytest.raises(chat.ContextOverflowError):
        runtime.decode(1, context)


def test_rope_tables_as_inputs_is_the_same_math(weights_path):
    """The 24K option: cos/sin rows come in as inputs instead of constants.

    At 24,576 rows the tables are ~100 MB of graph constants per package (600 MB
    in the recurrent chunk, against the 2 GB file limit) and slicing a constant
    with a runtime position is a placement risk.  The host hands the same rows in
    instead; the numbers must not move, so the two configurations are compared
    on the same weights, prefill and decode.
    """
    from anemll.models import dory_ane_model as dm

    ids = [3, 127, 64, 9, 12, 88, 5, 41]
    with_tables = _build(weights_path, dm)
    as_inputs = _build(weights_path, dm, rope_tables_as_inputs=True)
    got = as_inputs.prefill(ids, 0, loops=2).detach()
    want = with_tables.prefill(ids, 0, loops=2).detach()
    assert torch.equal(got, want)
    with_tables.reset_caches()
    as_inputs.reset_caches()
    with_tables.prefill(ids, 0, loops=1)
    as_inputs.prefill(ids, 0, loops=1)
    got = as_inputs.decode_step(23, len(ids), loops=1).detach()
    want = with_tables.decode_step(23, len(ids), loops=1).detach()
    assert torch.equal(got, want)

