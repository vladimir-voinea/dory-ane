#!/usr/bin/env python3
"""Tests for the Dory1.0-4B reference model (anemll/models/dory_model.py).

Dory is the first scoreable slice of the ANE port, so these tests run the real
thing: NVIDIA's BF16 safetensors and the reference scorer, no mocked maths.
The seam under test is next-token log-probabilities -- the same seam the ANE
port and the converted Core ML chunks are judged at.

Fast tests (no weights needed):

    python tests/test_dory_model.py

Full suite (weights + llama.cpp references + the reference scorer):

    DORY_WEIGHTS=~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16/snapshots/<rev> \
    DORY_EVAL_DIR=/Users/vladimir/dev/dory-ane-artifacts/DANE-3 \
    python tests/test_dory_model.py

or with pytest: PYTHONPATH=. pytest tests/test_dory_model.py -v
"""

from __future__ import annotations

import json
import os
import sys
import hashlib
from typing import Dict, Optional

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anemll.models.dory_model import (  # noqa: E402
    DoryAttention,
    DoryConfig,
    DoryForCausalLM,
    DoryL2Norm,
    DoryRotaryEmbedding,
    L2_EPS,
    MAX_LOOPS,
)

HOME = os.path.expanduser("~")
DEFAULT_WEIGHTS = (
    HOME + "/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16"
    "/snapshots/b366b87f762ea334ba3eb55912031baba1515810"
)
DEFAULT_EVAL_DIR = "/Users/vladimir/dev/dory-ane-artifacts/DANE-3"

WEIGHTS = os.environ.get("DORY_WEIGHTS", DEFAULT_WEIGHTS)
EVAL_DIR = os.environ.get("DORY_EVAL_DIR", DEFAULT_EVAL_DIR)
REF_DIR = os.environ.get("DORY_REF_DIR", "/Users/vladimir/dev/dory-ane-artifacts/refs")

# Gates from the ticket (spec.md, Testing Decisions).
MIN_TOP1 = 99.0
MAX_KLD = 0.005

_CACHE: Dict[str, object] = {}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def tiny_config(**overrides) -> DoryConfig:
    """A small Dory of the same shape, for tests that do not need 4B weights."""
    kwargs = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=256,
        n_input_layers=2,
        n_recurrent_layers=4,
        n_output_layers=2,
        sliding_window=4,
        n_recurrent_loops=2,
        initializer_range=0.0177,
        ngpt_alpha_init=0.05,
    )
    kwargs.update(overrides)
    return DoryConfig(**kwargs)


def config_from_weights() -> DoryConfig:
    path = os.path.join(WEIGHTS, "config.json")
    if not os.path.isfile(path):
        _skip("no config.json under DORY_WEIGHTS=%s" % WEIGHTS)
    return DoryConfig.from_json(path)


def _skip(message: str):
    """Skip under pytest; stop the standalone runner with a clear message."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        import pytest  # type: ignore

        pytest.skip(message)
    print("SKIP: " + message)
    raise SystemExit(0)


def loaded_model(loop_count_labels: bool = True) -> DoryForCausalLM:
    """The real 4B model, loaded once per process (kept in memory)."""
    if "model" in _CACHE:
        return _CACHE["model"]  # type: ignore[return-value]
    if not os.path.isdir(WEIGHTS):
        _skip("no Dory weights at %s (set DORY_WEIGHTS)" % WEIGHTS)
    config = config_from_weights()
    model = DoryForCausalLM(config, dtype=torch.bfloat16)
    model.load_pretrained_weights(WEIGHTS, dtype=torch.bfloat16)
    model.eval()
    _CACHE["model"] = model
    return model


def parameter_fingerprint(model: torch.nn.Module) -> str:
    """Cheap checksum of every parameter, to prove a forward changed no weights."""
    digest = hashlib.sha1()
    for name, param in sorted(model.named_parameters()):
        flat = param.detach().reshape(-1)
        digest.update(name.encode())
        digest.update(flat[:8].float().numpy().tobytes())
        digest.update(flat[-8:].float().numpy().tobytes())
        digest.update(str(tuple(param.shape)).encode())
    return digest.hexdigest()


def reference_scorer():
    """The reference scorer from DANE-2 (anemll/eval), once it is on the branch."""
    for module_path in ("anemll.eval.score", "anemll.eval.reference_kld"):
        try:
            module = __import__(module_path, fromlist=["*"])
        except ImportError:
            continue
        if hasattr(module, "read_kld") and hasattr(module, "score"):
            return module
    return None


def reference_path(loop_count: int) -> Optional[str]:
    candidate = os.path.join(EVAL_DIR, "ref-eval-4k-loop%d.kld" % loop_count)
    return candidate if os.path.isfile(candidate) else None


# ---------------------------------------------------------------------------
# fast tests: configuration and the layout llama.cpp describes
# ---------------------------------------------------------------------------

def test_config_reproduces_the_llama_cpp_layout():
    config = config_from_weights()
    assert config.pattern == "S-S-S-*-" * 9
    assert config.num_hidden_layers == 72
    assert (config.n_input_layers, config.n_recurrent_layers, config.n_output_layers) == (16, 40, 16)
    assert len(config.attention_layer_indices) == 36
    assert len(config.mlp_layer_indices) == 36
    assert len(config.sliding_layer_indices) == 27
    assert config.full_attention_layer_indices == [6, 14, 22, 30, 38, 46, 54, 62, 70]
    # theta: 1e4 on sliding blocks, 1e7 on full-attention blocks
    assert config.rope_theta == 10_000_000.0
    assert config.rope_theta_sliding == 10_000.0
    # rope profile 2 marks the sliding blocks (llama.cpp: profile 2 -> theta_2)
    profile = config.rope_profile
    assert profile[0] == 2 and profile[6] == 1
    assert sum(1 for i, p in enumerate(profile) if p != 1) == 27
    assert config.sliding_window == 1024
    assert config.head_dim == 256 and config.num_attention_heads == 16
    assert config.num_key_value_heads == 4
    assert abs(config.initializer_range - 0.0177) < 1e-9
    assert abs(config.ngpt_alpha_init - 0.05) < 1e-9
    assert abs(config.gate_init - 2560 ** 0.5) < 1e-6


def test_tensor_inventory_is_the_507_tensor_checkpoint():
    config = config_from_weights()
    model = DoryForCausalLM(config, dtype=torch.float32)
    inventory = model.tensor_inventory()
    n_attention = len(config.attention_layer_indices)
    n_mlp = len(config.mlp_layer_indices)
    assert len(inventory) == 3 + 8 * n_attention + 6 * n_mlp == 507
    assert inventory["model.embeddings.weight"] == (131072, 2560)
    assert inventory["lm_head.weight"] == (131072, 2560)
    assert inventory["lm_head.logit_scale"] == (131072,)
    assert inventory["model.layers.0.mixer.q_proj.weight"] == (4096, 2560)
    assert inventory["model.layers.0.mixer.k_proj.weight"] == (1024, 2560)
    assert inventory["model.layers.0.mixer.o_proj.weight"] == (2560, 4096)
    assert inventory["model.layers.1.mixer.gate_proj.weight"] == (7168, 2560)
    assert inventory["model.layers.1.mixer.suv_gate"] == (7168,)


def test_l2_norm_is_llama_cpp_l2_with_eps_inside_the_sqrt():
    x = torch.randn(5, 257)
    norm = DoryL2Norm(257, L2_EPS)
    got = norm(x)
    expected = x / torch.sqrt((x * x).sum(dim=-1, keepdim=True) + L2_EPS)
    assert torch.allclose(got, expected, atol=1e-6), (got - expected).abs().max()
    # a zero vector stays finite instead of dividing by zero
    assert torch.isfinite(norm(torch.zeros(1, 257))).all()


def test_rope_is_neox_with_two_bases():
    config = config_from_weights()
    from anemll.models.dory_model import DoryRotaryEmbedding

    rotary = DoryRotaryEmbedding(config)
    positions = torch.tensor([0, 1, 3])
    cos_f, sin_f = rotary(positions, sliding=False)
    cos_s, sin_s = rotary(positions, sliding=True)
    dim = config.head_dim
    inv_freq = 1.0 / (10_000_000.0 ** (torch.arange(0, dim, 2).float() / dim))
    angles = positions.float().reshape(-1, 1) * inv_freq.reshape(1, -1)
    assert torch.allclose(cos_f, torch.cat([angles.cos(), angles.cos()], dim=-1), atol=1e-6)
    assert torch.allclose(sin_f, torch.cat([angles.sin(), angles.sin()], dim=-1), atol=1e-6)
    # position 0 is unrotated in both bases; later positions differ between them
    assert torch.allclose(cos_s[0], torch.ones(dim)) and torch.allclose(sin_s[0], torch.zeros(dim))
    assert not torch.allclose(cos_f[2, :8], cos_s[2, :8])


def test_virtual_layer_plan_covers_the_loops():
    config = tiny_config()
    model = DoryForCausalLM(config, dtype=torch.float32)
    for loops, expected in ((1, 8), (2, 12), (4, 20)):
        plan = model.model.virtual_layer_plan(loops)
        assert len(plan) == expected
        entries = [i for i, (_, is_entry, _) in enumerate(plan) if is_entry]
        exits = [i for i, (_, _, is_exit) in enumerate(plan) if is_exit]
        assert entries == [config.n_input_layers + loop * config.n_recurrent_layers for loop in range(loops)]
        assert exits == [e + config.n_recurrent_layers - 1 for e in entries]
        assert plan[0][0] == 0 and plan[-1][0] == config.num_hidden_layers - 1


def test_sliding_window_attention_matches_the_definition():
    """A window of 2 must equal a hand-written causal+window numpy-style reference."""
    config = tiny_config(sliding_window=2, num_hidden_layers=8, n_recurrent_layers=4)
    attn = DoryAttention(config, layer_idx=0, dtype=torch.float32)  # block 0 slides
    assert attn.is_sliding and attn.sliding_window == 2
    torch.manual_seed(0)
    with torch.no_grad():
        for name, param in attn.named_parameters():
            param.uniform_(-0.05, 0.05)
        attn.sqk.fill_(0.03)
        attn.s_gate.fill_(0.03)
    seq = 5
    hidden = torch.randn(1, seq, config.hidden_size)
    cos, sin = DoryRotaryEmbedding(config)(torch.arange(seq), sliding=True)
    out = attn(hidden, cos, sin)[0]

    # same maths, done without any chunking or masking shortcuts
    from anemll.models.dory_model import DoryL2Norm, apply_rotary_pos_emb

    with torch.no_grad():
        q = torch.nn.functional.linear(hidden, attn.q_proj.weight).view(1, seq, attn.num_heads, attn.head_dim).transpose(1, 2)
        k = torch.nn.functional.linear(hidden, attn.k_proj.weight).view(1, seq, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
        v = torch.nn.functional.linear(hidden, attn.v_proj.weight).view(1, seq, attn.num_kv_heads, attn.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q = DoryL2Norm(attn.head_dim)(q) * (attn.sqk * attn.qk_scale).pow(2).view(1, attn.num_heads, 1, attn.head_dim)
        k = DoryL2Norm(attn.head_dim)(k)
        k = k.repeat_interleave(attn.num_kv_groups, dim=1)
        v = v.repeat_interleave(attn.num_kv_groups, dim=1)
        scores = torch.matmul(q, k.transpose(-1, -2)) * attn.softmax_scale
        allowed = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
        for i in range(seq):
            for j in range(seq):
                if i - j >= 2:
                    allowed[i, j] = False
        scores = scores.masked_fill(~allowed, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        expected = torch.matmul(probs, v).transpose(1, 2).reshape(1, seq, attn.q_dim)
        gate = torch.sigmoid(
            torch.nn.functional.linear(hidden, attn.g_proj.weight) * (attn.s_gate * attn.gate_scale).pow(2)
        )
        expected = torch.nn.functional.linear(
            DoryL2Norm(attn.q_dim)(expected * gate), attn.o_proj.weight
        )[0]
    assert torch.allclose(out, expected, atol=1e-6), (out - expected).abs().max()


def test_loop_count_is_only_an_argument():
    """Same weights, same model object, different loop counts."""
    torch.manual_seed(0)
    config = tiny_config()
    model = DoryForCausalLM(config, dtype=torch.float32)
    with torch.no_grad():
        for param in model.parameters():
            param.uniform_(-0.02, 0.02)
    model.eval()
    ids = torch.arange(6)
    before = parameter_fingerprint(model)
    with torch.inference_mode():
        logits = {loops: model(ids, loop_count=loops) for loops in (1, 2, 4)}
    assert parameter_fingerprint(model) == before, "a forward pass changed the weights"
    assert all(tuple(v.shape) == (1, 6, 256) for v in logits.values())
    assert not torch.allclose(logits[1], logits[2])
    assert not torch.allclose(logits[2], logits[4])
    # the default loop count comes from the config, not from the weights
    assert torch.allclose(logits[config.n_recurrent_loops], model(ids))


def test_unsupported_loop_counts_are_rejected():
    config = tiny_config()
    model = DoryForCausalLM(config, dtype=torch.float32)
    ids = torch.arange(4)
    for loops in (0, MAX_LOOPS + 1, -1):
        try:
            model(ids, loop_count=loops)
        except ValueError:
            continue
        raise AssertionError("loop_count=%d should have been rejected" % loops)


# ---------------------------------------------------------------------------
# heavy tests: the real checkpoint, the real reference, the real scorer
# ---------------------------------------------------------------------------

def test_every_checkpoint_tensor_is_accounted_for():
    model = loaded_model()
    assert len(model.loaded_tensor_names) == 507
    assert model.missing_tensor_names == []
    assert model.unused_tensor_names == []
    # and the weights really landed in the module
    state = model.state_dict()
    assert set(model.loaded_tensor_names) <= set(state)
    for name in ("model.embeddings.weight", "model.layers.0.mixer.sqk", "lm_head.logit_scale"):
        assert torch.isfinite(state[name].float()).all()


def test_logits_are_finite_and_loop_count_stable_on_real_weights():
    model = loaded_model()
    tokens = torch.tensor(
        [1, 2217, 1103, 1851, 3905, 526, 1558, 1826, 1733, 2497, 1103, 1016, 2674, 13, 11],
        dtype=torch.long,
    )
    fingerprints = []
    with torch.inference_mode():
        for loops in (1, 2, 4):
            logits = model(tokens, loop_count=loops, positions=[tokens.shape[-1] - 1])
            assert torch.isfinite(logits).all()
            assert logits.shape == (1, 1, model.config.vocab_size)
            fingerprints.append(parameter_fingerprint(model))
    assert len(set(fingerprints)) == 1


def test_embedding_rows_are_unit_norm():
    """nGPT keeps the stream on the unit sphere: embedding rows are unit norm."""
    model = loaded_model()
    norms = model.model.embeddings.weight.float().norm(dim=-1)
    assert torch.allclose(norms[:2000], torch.ones(2000), atol=1e-2), (float(norms.min()), float(norms.max()))


def test_every_block_ends_on_the_unit_sphere():
    """Every block's nGPT residual ends in an L2, so |h| == 1 after every physical block."""
    model = loaded_model()
    tokens = torch.tensor([1, 2217, 1103, 1851, 3905, 526], dtype=torch.long)
    with torch.inference_mode():
        hidden = model.model.forward_embeddings(tokens)
        assert torch.allclose(hidden.norm(dim=-1), torch.ones(1, tokens.shape[-1]), atol=1e-3)
        for physical in range(model.config.num_hidden_layers):
            hidden = model.model.process_layers(hidden, start_layer=physical, end_layer=physical + 1)
            norms = hidden.norm(dim=-1)
            assert torch.allclose(norms, torch.ones_like(norms), atol=1e-3), (physical, norms.tolist())


def test_single_token_logits_are_almost_loop_count_invariant():
    """Anchor from the math notes: with one token each loop pass only re-processes
    that token, so the next-token distribution barely moves with the loop count."""
    model = loaded_model()
    tokens = torch.tensor([1], dtype=torch.long)
    with torch.inference_mode():
        base = torch.log_softmax(model(tokens, loop_count=1, positions=[0])[0, 0].float(), dim=-1)
        for loops in (2, 4):
            other = torch.log_softmax(model(tokens, loop_count=loops, positions=[0])[0, 0].float(), dim=-1)
            assert int(base.argmax()) == int(other.argmax()), (loops, int(base.argmax()), int(other.argmax()))
            print('    single-token max dlogp vs %d loops: %.4f' % (loops, float((base - other).abs().max())))


def candidate_logprobs(model, eval_set, loops: int):
    """Per-position log-probabilities for DANE-2's evaluation set.

    Each context is a fresh forward with positions restarting at 0, which is how
    the reference chunks are scored.
    """
    import numpy as np

    logprobs = np.empty((eval_set.n_rows, model.config.vocab_size), dtype=np.float32)
    with torch.inference_mode():
        for chunk in range(eval_set.n_chunks):
            start = chunk * eval_set.chunk_size
            ids = torch.as_tensor(
                eval_set.token_ids[start:start + eval_set.chunk_size], dtype=torch.long
            )
            local = np.arange(eval_set.first_scored, eval_set.chunk_size - 1)
            logits = model(ids, loop_count=loops, positions=local)[0]
            logprobs[chunk * eval_set.rows_per_chunk:(chunk + 1) * eval_set.rows_per_chunk] = (
                torch.log_softmax(logits.float(), dim=-1).numpy()
            )
            del logits
    return logprobs


def test_exact_mode_scores_at_every_loop_count():
    """Top-1 >= 99% and mean KLD <= 0.005 against llama.cpp per_loop (exact mode).

    The reference is per_loop with ONE ubatch covering the whole 4096-token
    context: any split ubatch makes llama.cpp batch-dependent (measured: at
    ctx 4096, -ub 1024 differs from -ub 4096 by mean KLD 0.0076), so only the
    single-ubatch run is a full recompute to compare a recompute against.
    """
    import numpy as np
    from anemll.eval import kld as kld_module
    from anemll.eval.eval_set import load_eval_set

    eval_set = load_eval_set()
    model = loaded_model()
    for loops in (1, 2, 4):
        path = os.path.join(REF_DIR, "bf16_exact_loop%d.kld" % loops)
        if not os.path.isfile(path):
            _skip("no exact reference %s (set DORY_REF_DIR)" % path)
        reference = kld_module.read_llama_cpp_kld(path)
        logprobs = candidate_logprobs(model, eval_set, loops)
        candidate = kld_module.array_source(
            logprobs, tokens=eval_set.token_ids, positions=eval_set.scored_positions(),
            name="dory-fp32-loop%d" % loops,
        )
        result = kld_module.score_sources(reference, candidate)
        print(result.summary())
        # Diagnosis: where the remaining difference sits, by row and by bucket.
        ref_rows = reference.logp_block(0, reference.n_rows)
        matches = np.argmax(ref_rows, axis=1) == np.argmax(logprobs, axis=1)
        positions = eval_set.scored_positions()
        edges = np.linspace(positions.min(), positions.max() + 1, 7).astype(int)
        buckets = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (positions >= lo) & (positions < hi)
            if sel.any():
                buckets.append("%d-%d %s" % (lo, hi - 1, "ok" if matches[sel].all() else
                                             "%.4f" % (1.0 - matches[sel].mean())))
        print("  top-1 by position bucket: " + "  ".join(buckets))
        del ref_rows, logprobs, candidate
        assert 100.0 * result.top1_agreement >= MIN_TOP1, result.summary()
        assert result.mean_kld <= MAX_KLD, result.summary()


def test_greedy_generation_matches_llama_cpp():
    """First 32 greedy tokens from a chat-template prompt, loops=2, vs llama.cpp."""
    from transformers import AutoTokenizer

    expected_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "fixtures", "dory_greedy", "greedy-loop2.json"
    )
    if not os.path.isfile(expected_path):
        expected_path = os.path.join(EVAL_DIR, "greedy-loop2.json")
    if not os.path.isfile(expected_path):
        _skip("no llama.cpp greedy reference (tests/fixtures/dory_greedy/greedy-loop2.json)")
    if not os.path.isfile(os.path.join(WEIGHTS, "tokenizer.json")):
        _skip("no tokenizer under DORY_WEIGHTS")
    with open(expected_path) as handle:
        expected = json.load(handle)
    tokenizer = AutoTokenizer.from_pretrained(WEIGHTS)
    prompt_ids = expected["prompt_tokens"]
    # same prompt tokens on both sides, else the comparison is meaningless
    assert tokenizer.decode(prompt_ids) == expected["prompt_text"]
    assert expected.get("llamacpp_prompt_tokens") in (None, len(prompt_ids)), (
        "llama.cpp tokenized the prompt into %s tokens, we have %d"
        % (expected.get("llamacpp_prompt_tokens"), len(prompt_ids))
    )
    model = loaded_model()
    with torch.inference_mode():
        got = model.greedy_generate(torch.tensor(prompt_ids), max_new_tokens=32, loop_count=2)
    print("    llama.cpp : %s" % expected["greedy_tokens"])
    print("    our model : %s" % got)
    assert got == expected["greedy_tokens"], (
        "greedy mismatch:\n got %s\n want %s" % (got, expected["greedy_tokens"])
    )


def main():
    fast = [
        test_config_reproduces_the_llama_cpp_layout,
        test_tensor_inventory_is_the_507_tensor_checkpoint,
        test_l2_norm_is_llama_cpp_l2_with_eps_inside_the_sqrt,
        test_rope_is_neox_with_two_bases,
        test_virtual_layer_plan_covers_the_loops,
        test_sliding_window_attention_matches_the_definition,
        test_loop_count_is_only_an_argument,
        test_unsupported_loop_counts_are_rejected,
    ]
    heavy = [
        test_every_checkpoint_tensor_is_accounted_for,
        test_logits_are_finite_and_loop_count_stable_on_real_weights,
        test_exact_mode_scores_at_every_loop_count,
        test_greedy_generation_matches_llama_cpp,
    ]
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    tests = fast + heavy if which == "all" else [t for t in fast + heavy if which in t.__name__]
    failures = 0
    for test in tests:
        print("=" * 70)
        print(test.__name__)
        try:
            test()
            print("  ok")
        except SystemExit:
            print("  skipped")
        except AssertionError as exc:
            failures += 1
            print("  FAILED: %s" % exc)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print("  ERROR: %s: %s" % (type(exc).__name__, exc))
    print("=" * 70)
    print("%d test(s) failed" % failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
