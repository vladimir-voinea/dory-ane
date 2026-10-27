"""Tests for the Dory ANE-safe form and its `last_loop` caches (DANE-4).

The cache-layout half of this file needs no weights.  The model half runs NVIDIA's
real BF16 safetensors -- there are no mocked numbers anywhere.
"""

from __future__ import annotations

import torch

from anemll.utils import dory_caches as dc


# ---------------------------------------------------------------------------
# Cache inventory and rows (math-dory.md G2 / G3)
# ---------------------------------------------------------------------------

def test_cache_inventory_is_27_sliding_and_9_full():
    assert len(dc.sliding_attention_blocks()) == 27
    assert len(dc.full_attention_blocks()) == 9
    assert len([b for b in range(72) if dc.block_is_mlp(b)]) == 36
    assert dc.full_attention_blocks() == (6, 14, 22, 30, 38, 46, 54, 62, 70)


def test_cache_inventory_per_zone():
    def counts(start, end):
        s = sum(dc.block_is_sliding_attention(b) for b in range(start, end))
        f = sum(dc.block_is_full_attention(b) for b in range(start, end))
        m = sum(dc.block_is_mlp(b) for b in range(start, end))
        return s, f, m

    assert counts(0, 16) == (6, 2, 8)      # input zone
    assert counts(16, 56) == (15, 5, 20)   # recurrent zone (shared across loops)
    assert counts(56, 72) == (6, 2, 8)     # output zone


def test_slots_are_dense_and_unique():
    full, sliding = dc.cache_slots()
    assert sorted(full.values()) == list(range(9))
    assert sorted(sliding.values()) == list(range(27))
    assert set(full) | set(sliding) == {b for b in range(72) if dc.block_is_attention(b)}


def test_sliding_cache_rows_follows_llama_cpp_sizing():
    # PAD(min(ctx, window + ubatch), 256) -- 1280 at the shipped 24K context.
    assert dc.sliding_cache_rows(24576, 1024, 64) == 1280
    assert dc.sliding_cache_rows(24576, 1024, 512) == 1536
    assert dc.sliding_cache_rows(512, 1024, 64) == 512
    assert dc.sliding_cache_rows(1000, 1024, 64) == 1024 == dc._pad_to(min(1000, 1088), 256)
    assert dc.sliding_cache_rows(24576, 1024, 64) == 1280 > 1024  # the G3 correction


def test_cache_shapes():
    full, sliding = dc.cache_shapes(512, 4, 256, 1024, 64)
    assert full == (18, 4, 512, 256)     # 2 * 9 slots, K and V interleaved
    assert sliding == (54, 4, 512, 256)  # 2 * 27 slots


# ---------------------------------------------------------------------------
# Cache writes: single-row / n-row slices only (spec D5)
# ---------------------------------------------------------------------------

def _empty_cache(rows=16, heads=2, dim=4, slots=3):
    return torch.zeros(2 * slots, heads, rows, dim)


def test_decode_write_touches_only_one_row():
    cache = _empty_cache()
    k = torch.full((1, 2, 1, 4), 7.0)
    v = torch.full((1, 2, 1, 4), 9.0)
    before = cache.clone()
    dc.write_cache_rows(cache, slot=1, key=k, value=v, cache_pos=5, num_slots=3)
    changed = (cache != before).any(dim=(1, 3))          # (2*slots, rows)
    untouched = [0, 2, 3, 5]                            # slot 1 = K, slot 4 = V
    assert not changed[untouched].any()
    assert changed[1, 5] and changed[1].sum() == 1       # K row 5 only
    assert changed[4, 5] and changed[4].sum() == 1       # V row 5 only (slot + num_slots)
    assert torch.equal(cache[1, :, 5, :], k[0, :, 0, :])
    assert torch.equal(cache[4, :, 5, :], v[0, :, 0, :])


def test_prefill_write_is_one_contiguous_slice():
    cache = _empty_cache(rows=32, slots=3)
    k = torch.arange(2 * 8 * 4, dtype=torch.float32).reshape(1, 2, 8, 4)
    v = k + 100
    dc.write_cache_rows(cache, slot=2, key=k, value=v, cache_pos=8, num_slots=3)
    assert torch.equal(cache[2, :, 8:16, :], k[0])
    assert torch.equal(cache[5, :, 8:16, :], v[0])
    # nothing outside the 8 written rows moved
    assert cache[2, :, :8, :].abs().sum() == 0
    assert cache[2, :, 16:, :].abs().sum() == 0


def test_write_accepts_a_tensor_cache_pos():
    """The traced form reads cache_pos from an int32 graph input."""
    cache = _empty_cache()
    k = torch.ones(1, 2, 1, 4)
    dc.write_cache_rows(cache, slot=0, key=k, value=k, cache_pos=torch.tensor([3]), num_slots=3)
    assert cache[0, :, 3, :].sum() == 8.0


def test_ring_positions_never_straddle_the_seam():
    rows = dc.sliding_cache_rows(24576, 1024, 64)        # 1280
    assert rows == 1280
    for start in range(0, 24576, 64):
        first = dc.cache_positions(start, 64, rows)      # raises if it straddles
        assert 0 <= first < rows and first % 64 == 0
    # the last aligned batch of a ring: 1216 + 64 == 1280, exactly at the seam
    assert dc.cache_positions(1216, 64, 1280) == 1216
    assert dc.cache_positions(1280, 64, 1280) == 0


def test_ring_write_that_would_wrap_is_refused():
    try:
        dc.cache_positions(1248, 64, 1280)
    except ValueError as exc:
        assert "straddle" in str(exc)
    else:
        raise AssertionError("a prefill write crossing the ring seam must raise")


def test_reset_caches_zeroes_them():
    cache = torch.ones(4, 2, 8, 4)
    dc.reset_caches(cache)
    assert cache.abs().sum() == 0


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------

def test_full_causal_mask():
    m = dc.full_causal_mask(start=3, q_len=2, rows=8)
    assert m.shape == (1, 1, 2, 8)
    allowed = (m[0, 0] == 0)
    assert allowed[0].tolist() == [True, True, True, True, False, False, False, False]
    assert allowed[1].tolist() == [True, True, True, True, True, False, False, False]


def test_sliding_mask_during_fill_sees_only_the_past():
    rows, window = 512, 1024
    m = dc.sliding_ring_mask(start=0, q_len=3, rows=rows, sliding_window=window)
    allowed = (m[0, 0] == 0)
    # query 0 sees slot 0 only, query 2 sees slots 0..2, and the batch's own
    # newest written slot (2) is visible to earlier queries? No: causal.
    assert allowed[0, :4].tolist() == [True, False, False, False]
    assert allowed[2, :4].tolist() == [True, True, True, False]
    assert allowed.sum(dim=1).tolist() == [1, 2, 3]


def test_sliding_mask_in_steady_state_is_exactly_the_window():
    rows, window = 1280, 1024
    for pos in (1279, 1280, 1400, 5000, 24575):
        m = dc.sliding_ring_mask(start=pos, q_len=1, rows=rows, sliding_window=window)
        allowed = (m[0, 0, 0] == 0)
        assert int(allowed.sum()) == window, pos
        # the allowed slots are exactly the tokens at positions pos-1023..pos
        last = pos
        for j in range(rows):
            tok_pos = last - ((last - j) % rows)
            expected = (tok_pos >= 0) and (0 <= pos - tok_pos <= window - 1)
            assert bool(allowed[j]) == expected
        # the newest token is visible, one token past the window is not
        assert allowed[pos % rows]
        assert not allowed[(pos - window) % rows]


def test_mask_input_is_the_concatenation():
    ctx, rows = 32, 512
    m = dc.build_causal_mask(start=4, q_len=2, context_length=ctx, sliding_rows=rows)
    assert m.shape == (1, 1, 2, ctx + rows)
    assert torch.equal(m[..., :ctx], dc.full_causal_mask(4, 2, ctx))
    assert torch.equal(m[..., ctx:], dc.sliding_ring_mask(4, 2, rows))

# ===========================================================================
# Part 2 -- the model against an independent reference
#
# The oracle below is written from math-dory.md 4.0-4.6 in plain torch: no
# convolutions, no layer-norm trick, no cache, its own RoPE by complex
# rotation.  Nothing is shared with the module under test.
# ===========================================================================

import math
import os

import pytest

from anemll.models import dory_ane_model as dm

TINY = dict(hidden_size=32, intermediate_size=64, num_attention_heads=4,
            num_key_value_heads=1, head_dim=8, vocab_size=128,
            context_length=64, batch_size=4, dtype=torch.float32)


@pytest.fixture(scope="module")
def tiny_weights():
    """A complete, HF-named random Dory checkpoint (tiny dims)."""
    cfg = TINY
    g = torch.Generator().manual_seed(1234)
    H, I, V = cfg["hidden_size"], cfg["intermediate_size"], cfg["vocab_size"]
    nh, nkv, hd = 4, 1, 8
    w = {}
    w["backbone.embeddings.weight"] = torch.randn(V, H, generator=g)
    w["lm_head.weight"] = torch.randn(V, H, generator=g)
    w["lm_head.logit_scale"] = torch.rand(V, generator=g) + 0.5
    for p in range(72):
        if p % 8 == 6 or p % 8 in (0, 2, 4):
            w[f"backbone.layers.{p}.mixer.q_proj.weight"] = torch.randn(nh * hd, H, generator=g) * 0.2
            w[f"backbone.layers.{p}.mixer.k_proj.weight"] = torch.randn(nkv * hd, H, generator=g) * 0.2
            w[f"backbone.layers.{p}.mixer.v_proj.weight"] = torch.randn(nkv * hd, H, generator=g) * 0.2
            w[f"backbone.layers.{p}.mixer.o_proj.weight"] = torch.randn(H, nh * hd, generator=g) * 0.2
            w[f"backbone.layers.{p}.mixer.g_proj.weight"] = torch.randn(nh * hd, H, generator=g) * 0.2
            w[f"backbone.layers.{p}.mixer.sqk"] = torch.randn(nh * hd, generator=g) * 0.02
            w[f"backbone.layers.{p}.mixer.s_gate"] = torch.rand(nh * hd, generator=g) * 0.4
            w[f"backbone.layers.{p}.attn_alpha"] = torch.randn(H, generator=g) * 0.3
        else:
            w[f"backbone.layers.{p}.mixer.gate_proj.weight"] = torch.randn(I, H, generator=g) * 0.2
            w[f"backbone.layers.{p}.mixer.up_proj.weight"] = torch.randn(I, H, generator=g) * 0.2
            w[f"backbone.layers.{p}.mixer.down_proj.weight"] = torch.randn(H, I, generator=g) * 0.2
            w[f"backbone.layers.{p}.mixer.suv_gate"] = torch.rand(I, generator=g) + 0.2
            w[f"backbone.layers.{p}.mixer.suv_up"] = torch.rand(I, generator=g) + 0.2
            w[f"backbone.layers.{p}.mlp_alpha"] = torch.randn(H, generator=g) * 0.3
    return w


def _write_safetensors(path, tensors):
    from safetensors.torch import save_file
    save_file({k: v.contiguous() for k, v in tensors.items()}, path)


# -- the oracle ------------------------------------------------------------

def ref_l2(x, eps=1e-6, dim=-1):
    return x / torch.sqrt((x * x).sum(dim, keepdim=True) + eps)


def ref_rope(x, positions, base):
    """NEOX rotate-half RoPE through complex rotation (independent formulation)."""
    d = x.shape[-1]
    half = d // 2
    inv = base ** (-torch.arange(0, half, dtype=torch.float64) * 2.0 / d)
    angle = positions.to(torch.float64).view(-1, 1) * inv.view(1, -1)      # (Q, D/2)
    angle_full = torch.cat([angle, angle], dim=-1).to(torch.float64)       # NEOX duplicates
    cos = torch.cos(angle_full).to(x.dtype).view(-1, 1, d)   # (Q, 1, D)
    sin = torch.sin(angle_full).to(x.dtype).view(-1, 1, d)
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + rotated * sin


def ref_block(h, p, w, cfg, positions, mask, sliding_window=1024):
    """One physical block in the literal (unfolded) order, single batch."""
    prefix = f"backbone.layers.{p}"
    H, nh, nkv, hd = cfg["hidden_size"], 4, 1, 8
    init_std, alpha_init, gate_init = 0.0177, 0.05, 50.59644317626953
    residual = h
    k8 = p % 8
    if k8 in (1, 3, 5, 7):
        g = h @ w[f"{prefix}.mixer.gate_proj.weight"].T
        u = h @ w[f"{prefix}.mixer.up_proj.weight"].T
        g = g * (w[f"{prefix}.mixer.suv_gate"] * math.sqrt(H))
        u = u * (w[f"{prefix}.mixer.suv_up"] * math.sqrt(H))
        y = ref_l2(torch.nn.functional.silu(g) * u)
        f = y @ w[f"{prefix}.mixer.down_proj.weight"].T
    else:
        q = (h @ w[f"{prefix}.mixer.q_proj.weight"].T).view(-1, nh, hd)
        k = (h @ w[f"{prefix}.mixer.k_proj.weight"].T).view(-1, nkv, hd)
        v = (h @ w[f"{prefix}.mixer.v_proj.weight"].T).view(-1, nkv, hd)
        base = 1e7 if k8 == 6 else 1e4
        q = ref_rope(q, positions, base)
        k = ref_rope(k, positions, base)
        q = ref_l2(q) * ((w[f"{prefix}.mixer.sqk"] * math.sqrt(hd) / init_std) ** 2).view(nh, hd)
        k = ref_l2(k)
        gate = torch.sigmoid(
            (h @ w[f"{prefix}.mixer.g_proj.weight"].T)
            * ((w[f"{prefix}.mixer.s_gate"] * math.sqrt(gate_init) / init_std) ** 2))
        rep = nh // nkv
        kk = k.repeat_interleave(rep, dim=1).transpose(0, 1)   # (H, Q, D)
        vv = v.repeat_interleave(rep, dim=1).transpose(0, 1)
        qq = q.transpose(0, 1)                                  # (H, Q, D)
        logits = torch.einsum("hqd,hkd->hqk", qq, kk) / math.sqrt(hd) + mask
        probs = torch.softmax(logits, dim=-1)
        out = torch.einsum("hqk,hkd->hqd", probs, vv)
        out = out.transpose(0, 1).reshape(-1, nh * hd)
        f = ref_l2(out * gate) @ w[f"{prefix}.mixer.o_proj.weight"].T
    a = ref_l2(residual)
    b = ref_l2(f)
    alpha = (w[f"{prefix}.mlp_alpha" if k8 in (1, 3, 5, 7) else f"{prefix}.attn_alpha"]
             * alpha_init / init_std).abs()
    return ref_l2(a + (b - a) * alpha)


def ref_forward(w, ids, cfg, loops):
    """Whole-sequence forward in the literal order (single batch == per_loop)."""
    Q = len(ids)
    positions = torch.arange(Q)
    causal = torch.where(positions[:, None] >= positions[None, :],
                         torch.zeros(Q, Q), torch.full((Q, Q), float("-inf")))
    h = w["backbone.embeddings.weight"][torch.tensor(ids)].to(torch.float32)
    for p in range(16):
        h = ref_block(h, p, w, cfg, positions, causal)
    h_in = h
    for _ in range(loops):
        h = ref_l2(h + h_in)
        for p in range(16, 56):
            h = ref_block(h, p, w, cfg, positions, causal)
        h = ref_l2(h)
    for p in range(56, 72):
        h = ref_block(h, p, w, cfg, positions, causal)
    logits = h @ w["lm_head.weight"].T * (w["lm_head.logit_scale"] / 0.0177)
    return logits


def _build_model(tmp_path, tiny_weights, fold_scales, **extra):
    path = str(tmp_path / "tiny.safetensors")
    _write_safetensors(path, tiny_weights)
    cfg = dm.DoryANEConfig(**{**TINY, "fold_scales": fold_scales, **extra})
    model = dm.DoryANEForCausalLM(cfg)
    report = model.load_pretrained_weights(path)
    assert report["missing"] == [], report["missing"]
    model.eval()
    return model


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory, tiny_weights):
    return _build_model(tmp_path_factory.mktemp("tiny_folded"), tiny_weights, True)


def test_weight_loader_consumes_every_tensor(tmp_path, tiny_weights):
    model = _build_model(tmp_path, tiny_weights, True)
    assert model.embed_tokens.weight.dtype == torch.float32
    # folding is exact and visible: the MLP gate rows carry the suv scale
    p = 1
    raw = tiny_weights[f"backbone.layers.{p}.mixer.gate_proj.weight"]
    suv = tiny_weights[f"backbone.layers.{p}.mixer.suv_gate"] * math.sqrt(32)
    folded = model.model.layers[1].mixer.gate_proj.weight.detach().view(64, 32)
    assert torch.allclose(folded, raw * suv.view(-1, 1), rtol=1e-5, atol=1e-6)


def _damped(tiny_weights, scale=1e-3):
    """The same checkpoint with a near-zero residual alpha.

    A random-weight Dory amplifies a 1e-7 rounding difference by ~1.8x per
    block (the nGPT residual extrapolates), so a *chained* comparison of the
    whole model is meaningless with a normal alpha.  Damping alpha keeps the
    chain contractive, which isolates the wiring -- masks, caches, loop entry
    and exit, chunk order -- while the per-block test above checks the math.
    """
    out = dict(tiny_weights)
    for key, value in tiny_weights.items():
        if key.endswith(".attn_alpha") or key.endswith(".mlp_alpha"):
            out[key] = value * scale
    return out


def test_every_block_matches_the_reference(tiny_model, tiny_weights):
    """Every one of the 72 physical blocks reproduces the literal formula.

    The module block and the oracle are both fed the oracle's input, so this
    measures one block, not accumulated chaos.
    """
    ids = [3, 127, 64, 9, 12, 88, 5, 41]
    Q = len(ids)
    positions = torch.arange(Q)
    causal = torch.where(positions[:, None] >= positions[None, :],
                         torch.zeros(Q, Q), torch.full((Q, Q), float("-inf")))
    h = tiny_weights["backbone.embeddings.weight"][torch.tensor(ids)].to(torch.float32)
    model = tiny_model.model
    cfg = tiny_model.config
    mask = tiny_model.build_mask(0, Q)
    mask_full = mask.narrow(-1, 0, cfg.context_length)
    mask_sliding = mask.narrow(-1, cfg.context_length, cfg.sliding_rows)
    cur = torch.tensor([0], dtype=torch.int32)
    cp = torch.tensor([0], dtype=torch.int32)
    worst = 0.0
    for p in range(72):
        want = ref_block(h, p, tiny_weights, TINY, positions, causal)
        got = model.layers[p](h.unsqueeze(0), mask_full, mask_sliding,
                              model.kv_cache_full, model.kv_cache_sliding,
                              cur, cp).squeeze(0)
        worst = max(worst, (want - got).abs().max().item())
        # atol 1e-4: q is multiplied by sq (~1e3 with the real sqk), so the
        # softmax is saturated and a 1e-7 rounding difference in q shows up as
        # 1e-4 in the weighted average of v.
        assert torch.allclose(got, want, rtol=1e-4, atol=1e-4), (p, worst)
        h = want
    assert worst < 1e-5, worst


def test_full_sequence_forward_matches_the_reference(tmp_path, tiny_weights):
    """Chained end-to-end (damped alpha): chunks, loops, entry/exit, head."""
    weights = _damped(tiny_weights)
    model = _build_model(tmp_path, weights, fold_scales=True)
    ids = [3, 127, 64, 9, 12, 88, 5, 41]
    got = model.prefill(ids, 0, loops=2).detach()
    want = ref_forward(weights, ids, TINY, loops=2)
    assert torch.allclose(got, want, rtol=1e-3, atol=1e-3), \
        (got - want).abs().max().item()


@pytest.mark.parametrize("loops", [1, 2, 4])
def test_loop_count_is_just_an_argument(tmp_path, tiny_weights, loops):
    weights = _damped(tiny_weights)
    model = _build_model(tmp_path, weights, fold_scales=True)
    ids = [7, 11, 99, 3]
    got = model.prefill(ids, 0, loops=loops).detach()
    want = ref_forward(weights, ids, TINY, loops=loops)
    assert torch.allclose(got, want, rtol=1e-3, atol=1e-3)


def test_folding_scales_is_exact_in_fp32(tmp_path, tiny_weights):
    """Folded and unfolded FP32 forms agree to FP32 rounding, end to end."""
    weights = _damped(tiny_weights)         # contractive, so nothing amplifies
    folded = _build_model(tmp_path, weights, fold_scales=True)
    unfolded = dm.DoryANEForCausalLM(dm.DoryANEConfig(
        **{**TINY, "fold_scales": False}))
    path = str(tmp_path / "tiny.safetensors")
    _write_safetensors(path, weights)
    unfolded.load_pretrained_weights(path)
    ids = [5, 17, 91, 2]
    a = folded.prefill(ids, 0, loops=2).detach()
    b = unfolded.prefill(ids, 0, loops=2).detach()
    scale = b.abs().max().item()
    assert (a - b).abs().max().item() / scale < 2e-5, (a - b).abs().max().item()


def _block_pair(tmp_path, tiny_weights, block):
    folded = _build_model(tmp_path, tiny_weights, fold_scales=True)
    path = str(tmp_path / "tiny.safetensors")
    _write_safetensors(path, tiny_weights)
    unfolded = dm.DoryANEForCausalLM(dm.DoryANEConfig(**{**TINY, "fold_scales": False}))
    unfolded.load_pretrained_weights(path)
    return folded, unfolded


@pytest.mark.parametrize("block", [0, 1, 6, 22, 55])
def test_folding_scales_is_exact_per_block_in_fp32(tmp_path, tiny_weights, block):
    """The folded weight equals the unfolded weight times its scale, in FP32."""
    folded, unfolded = _block_pair(tmp_path, tiny_weights, block)
    ids = [5, 17, 91, 2]
    h = tiny_weights["backbone.embeddings.weight"][torch.tensor(ids)].unsqueeze(0)
    m = folded.build_mask(0, len(ids))
    masks = (m.narrow(-1, 0, 64), m.narrow(-1, 64, folded.config.sliding_rows))
    args = (h, *masks, folded.model.kv_cache_full, folded.model.kv_cache_sliding,
            torch.tensor([0]), torch.tensor([0]))
    uargs = (h, *masks, unfolded.model.kv_cache_full, unfolded.model.kv_cache_sliding,
             torch.tensor([0]), torch.tensor([0]))
    a = folded.model.layers[block](*args)
    b = unfolded.model.layers[block](*uargs)
    assert (a - b).abs().max().item() < 1e-5, block

# ===========================================================================
# Part 3 -- D5 structural guarantees and the decode path
# ===========================================================================

def test_gqa_attention_matches_a_materialised_repeat(tiny_model):
    """The reshape-only GQA is numerically identical to repeat_kv + einsum."""
    attn = tiny_model.model.layers[0].mixer
    bsz, q_len = 1, 5
    hidden = torch.randn(bsz, q_len, tiny_model.config.hidden_size)
    mask = tiny_model.build_mask(0, q_len)
    mask_full = mask.narrow(-1, 0, tiny_model.config.context_length)
    tiny_model.model.reset_caches()
    got = attn(hidden, mask_full, tiny_model.model.kv_cache_full, attn.slot,
               attn.num_slots, torch.tensor([0]), torch.tensor([0]))

    # oracle: repeat K/V, bmm over q heads
    with torch.no_grad():
        conv_in = hidden.permute(0, 2, 1).unsqueeze(2)
        q = attn.q_proj(conv_in).squeeze(2).permute(0, 2, 1)
        k = attn.k_proj(conv_in).squeeze(2).permute(0, 2, 1)
        v = attn.v_proj(conv_in).squeeze(2).permute(0, 2, 1)
        gate = torch.sigmoid(attn.g_proj(conv_in).squeeze(2).permute(0, 2, 1))
        q = attn.rope(q.view(bsz, q_len, attn.num_heads, attn.head_dim).transpose(1, 2), 0, q_len)
        k = attn.rope(k.view(bsz, q_len, attn.num_kv_heads, attn.head_dim).transpose(1, 2), 0, q_len)
        q = attn.q_norm(q) * attn.sq.view(1, attn.num_heads, 1, attn.head_dim)
        k = attn.k_norm(k)
        key_cache, value_cache = dc.read_cache_slot(
            tiny_model.model.kv_cache_full, attn.slot, attn.num_slots, mask_full.shape[-1])
        kk = key_cache.repeat_interleave(attn.rep, dim=1)
        vv = value_cache.repeat_interleave(attn.rep, dim=1)
        scores = torch.matmul(q, kk.transpose(-1, -2)) * attn.scale + mask_full
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, vv)
        out = out.transpose(1, 2).reshape(bsz, q_len, -1)
        want = _from_conv_layout_for_test(attn.o_proj(_to_conv_layout_for_test(attn.out_norm(out * gate))))
    assert torch.allclose(got, want, rtol=1e-4, atol=1e-5)


def _to_conv_layout_for_test(x):
    return x.permute(0, 2, 1).unsqueeze(2)


def _from_conv_layout_for_test(x):
    return x.squeeze(2).permute(0, 2, 1)


def test_attention_never_materialises_a_repeat_of_kv(tiny_model):
    """No op inside the attention block creates an H x K x D copy of the cache.

    The largest tensor the block may create is a logits tensor of size
    B * H * Q * K; a repeat of K/V would be H * K * D, which for the real
    model (16 heads, 24,576 rows, 256 dims) is 400 MB per layer per step.
    """
    from torch.utils._python_dispatch import TorchDispatchMode

    class Recorder(TorchDispatchMode):
        def __init__(self):
            self.sizes = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            if isinstance(out, torch.Tensor) and out.is_floating_point():
                self.sizes.append((str(func), out.numel()))
            return out

    attn = tiny_model.model.layers[0].mixer
    hidden = torch.randn(1, 4, tiny_model.config.hidden_size)
    mask = tiny_model.build_mask(0, 4).narrow(-1, 0, tiny_model.config.context_length)
    tiny_model.model.reset_caches()
    with Recorder() as rec:
        attn(hidden, mask, tiny_model.model.kv_cache_full, attn.slot, attn.num_slots,
             torch.tensor([0]), torch.tensor([0]))
    k_rows = mask.shape[-1]
    h, kv, d = attn.num_heads, attn.num_kv_heads, attn.head_dim
    repeated_kv = h * k_rows * d
    assert max(size for _, size in rec.sizes) < repeated_kv, \
        sorted(rec.sizes, key=lambda kv: -kv[1])[:5]


def test_cached_decode_equals_one_batch_prefill_at_one_loop(tmp_path, tiny_weights):
    """With one loop, last_loop and per_loop coincide: stepping == one batch."""
    model = _build_model(tmp_path, _damped(tiny_weights), fold_scales=True)
    ids = [4, 8, 15, 16, 23, 42]
    one_batch = model.prefill(ids, 0, loops=1).detach()
    model.reset_caches()
    stepped = torch.cat([model.decode_step(tok, i, loops=1) for i, tok in enumerate(ids)],
                        dim=1)
    # identical math, different matmul shapes: the difference is float32
    # rounding relative to the logit scale (a few 1e-7 relative)
    worst = (stepped - one_batch).abs().max().item()
    assert worst < 1e-5 * one_batch.abs().max().item(), worst


def test_last_loop_batching_differs_from_one_batch_at_two_loops(tiny_model):
    """The decode path really is last_loop: its result differs from per_loop.

    This is the guard that the cache path is not silently the exact path.
    """
    ids = [4, 8, 15, 16, 23, 42]
    tiny_model.set_loops(2)
    tiny_model.reset_caches()
    one_batch = tiny_model.prefill(ids, 0, loops=2).detach()
    tiny_model.reset_caches()
    stepped = torch.cat([tiny_model.decode_step(tok, i, loops=2) for i, tok in enumerate(ids)],
                        dim=1)
    assert not torch.allclose(stepped, one_batch, rtol=1e-3, atol=1e-3)


def test_reference_last_loop_semantics(tiny_model):
    """Decode against an explicit per-position final-loop cache.

    The reference keeps, per attention block, the K/V of the *last* loop of
    every earlier position and concatenates the current loop's row -- the
    definition of last_loop -- using the module's own blocks, so any
    difference is a cache/mask/loop-wiring difference, not rounding.
    """
    ids = [4, 8, 15, 16, 23]
    model = tiny_model
    cfg = model.config
    model.set_loops(2)
    model.reset_caches()
    got = [model.decode_step(tok, i, loops=2) for i, tok in enumerate(ids)]

    # oracle on scratch caches: one linear cache per physical attention block
    full = torch.zeros_like(model.model.kv_cache_full)
    sliding = torch.zeros_like(model.model.kv_cache_sliding)
    model.model.reset_caches()
    want = []
    for p, tok in enumerate(ids):
        cur = torch.tensor([p], dtype=torch.int32)
        mask = model.build_mask(p, 1)
        mf = mask.narrow(-1, 0, cfg.context_length)
        ms = mask.narrow(-1, cfg.context_length, cfg.sliding_rows)
        h = model.embed_tokens(torch.tensor([[tok]]))
        for i in range(cfg.input_layer_count):
            h = model.model.layers[i](h, mf, ms, full, sliding, cur, cur)
        h_in = h
        for _ in range(2):
            h = model.model.l2(h + h_in)
            for i in range(cfg.input_layer_count,
                           cfg.input_layer_count + cfg.recurrent_layer_count):
                h = model.model.layers[i](h, mf, ms, full, sliding, cur, cur)
            h = model.model.l2(h)
        for i in range(cfg.input_layer_count + cfg.recurrent_layer_count,
                       cfg.num_hidden_layers):
            h = model.model.layers[i](h, mf, ms, full, sliding, cur, cur)
        want.append(model.logits(h))
    for a, b in zip(got, want):
        assert torch.equal(a, b), (a - b).abs().max().item()

def test_changing_the_loop_count_resets_the_caches(tiny_model):
    """Spec D4: a different loop count invalidates the cache and starts clean."""
    ids = [1, 2, 3, 4, 5]
    tiny_model.set_loops(2)
    tiny_model.prefill(ids, 0, loops=2)
    assert tiny_model.model.kv_cache_full.abs().sum() > 0
    assert tiny_model.model.kv_cache_sliding.abs().sum() > 0

    assert tiny_model.set_loops(4) is True                     # a change resets
    assert tiny_model.model.kv_cache_full.abs().sum() == 0
    assert tiny_model.model.kv_cache_sliding.abs().sum() == 0

    assert tiny_model.set_loops(4) is False                    # same count, no reset
    tiny_model.set_loops(2)
    tiny_model.prefill(ids, 0, loops=2)

    # a stale cache would contaminate the next run: after the reset the result
    # must equal a run that never saw the old cache
    tiny_model.set_loops(4)
    after_change = tiny_model.prefill(ids, 0, loops=4).detach()
    fresh = dm.DoryANEForCausalLM(tiny_model.config)
    fresh.load_state_dict(tiny_model.state_dict())
    fresh.eval()
    assert torch.equal(after_change, fresh.prefill(ids, 0, loops=4).detach())


def test_loop_count_outside_1_to_4_is_refused(tiny_model):
    for bad in (0, 5):
        with pytest.raises(ValueError):
            tiny_model.set_loops(bad)

@pytest.mark.parametrize("parts", [1, 4, 8, 16])
def test_split_lm_head_matches_a_single_head(tmp_path, tiny_weights, parts):
    """Spec D6: N head parts concatenate to the single-head result, for any N."""
    split = _build_model(tmp_path, tiny_weights, fold_scales=True, lm_head_parts=parts)
    single = dm.DoryANEForCausalLM(dm.DoryANEConfig(
        **{**TINY, "fold_scales": True, "lm_head_parts": 1}))
    path = str(tmp_path / "tiny.safetensors")
    _write_safetensors(path, tiny_weights)
    single.load_pretrained_weights(path)
    ids = [3, 5, 7, 11]
    a = split.prefill(ids, 0, loops=2).detach()
    b = single.prefill(ids, 0, loops=2).detach()
    assert torch.allclose(a, b, rtol=1e-4, atol=1e-4), (a - b).abs().max().item()


def test_full_cache_is_addressed_by_absolute_position(tmp_path, tiny_weights):
    """Caught a real bug: a full cache must not use the sliding ring offset.

    Two configs, same weights, same tokens: the small ring (256 rows) makes
    cache_pos = position mod 256 diverge from the position, the big ring
    (1,088 rows) keeps them equal for these 300 tokens.  The sliding window is
    64 in both, so the attention sets -- and therefore the results -- must be
    identical.  A full cache written at the ring offset breaks that equality,
    which is exactly the long-context failure the 4,094-row gate exposed.
    """
    weights = _damped(tiny_weights)      # contractive, so rounding cannot mask a wiring bug
    path = str(tmp_path / "tiny.safetensors")
    _write_safetensors(path, weights)
    small = dm.DoryANEForCausalLM(dm.DoryANEConfig(
        **dict(TINY, context_length=512, sliding_window=64, batch_size=4, fold_scales=True)))
    big = dm.DoryANEForCausalLM(dm.DoryANEConfig(
        **dict(TINY, context_length=512, sliding_window=64, batch_size=1024,
               fold_scales=True)))
    small.load_pretrained_weights(path)
    big.load_pretrained_weights(path)
    assert small.config.sliding_rows == 256 < 512
    assert big.config.sliding_rows > 300

    ids = [((i * 37) % 120) + 1 for i in range(300)]
    a = torch.cat([small.decode_step(tok, i, loops=1) for i, tok in enumerate(ids)], dim=1)
    b = torch.cat([big.decode_step(tok, i, loops=1) for i, tok in enumerate(ids)], dim=1)
    # the bug this guards against put the wrong rows in the full cache, which
    # moves the logits by ~1e-3 even with a damped residual; correct wiring
    # agrees to ~1e-6
    worst = (a - b).abs().max().item()
    assert worst < 1e-5 * b.abs().max().item(), (worst, b.abs().max().item())

def test_reset_caches_keeps_the_loop_selection(tmp_path, tiny_weights):
    """A harness bug: reset_caches() cleared the loop count, so a gate that
    selected 1 or 4 loops silently ran the default 2."""
    model = _build_model(tmp_path, tiny_weights, fold_scales=True)
    ids = [3, 5, 7, 11]
    one = model.prefill(ids, 0, loops=1).detach()
    model.reset_caches()                       # must not change the loop count
    again = model.prefill(ids, 0).detach()     # no explicit loops argument
    assert torch.allclose(one, again, atol=0), (one - again).abs().max().item()

    four = model.prefill(ids, 0, loops=4).detach()
    model.reset_caches()
    still_four = model.prefill(ids, 0).detach()
    assert torch.allclose(four, still_four, atol=0), (four - still_four).abs().max().item()
    assert not torch.allclose(one, four, atol=1e-3), "loops 1 and 4 must differ"
