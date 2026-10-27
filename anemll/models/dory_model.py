"""Dory1.0-4B (NVIDIA Nemotron-Labs Dory) implementation for ANEMLL.

Dory is a looped (depth-recurrent) 4B decoder: 72 physical blocks split into
16 input + 40 recurrent + 16 output blocks, where the recurrent stack is run a
chosen number of times (the loop count, 1..4) with the hidden state of the
first pass fed back in. Every block is either gated attention or an MLP, and
the residual stream uses nGPT-style normalisation.

This module is the full-precision (FP32) reference implementation that the
ANEMLL conversion pipeline and the divergence harness compare against
llama.cpp + PR #30306. It runs a whole sequence in one shot -- no KV cache, a
full recompute -- so batching cannot change the result. The ANE-safe form
(FP16, 1x1 convs, Core ML state caches, last_loop semantics) is built on top of
this class in the same style as Gemma3Model.

The graph implemented here is exactly llama.cpp's (src/models/dory.cpp); where
this docstring and that file disagree, that file wins. Per virtual layer:

  * loop entry (first virtual layer of every loop):  h = L2(h + h_in)
  * MLP block:      g = W_gate h * (suv_gate * sqrt(n_embd))
                    u = W_up   h * (suv_up   * sqrt(n_embd))
                    h = W_down L2(silu(g) * u)
  * attention block: q,k,v = W_{q,k,v} h ; NEOX RoPE (all head_dim dims,
                    theta = 1e4 sliding / 1e7 full)
                    q = L2_per_head(q) * (sqk * sqrt(head_dim) / init_std)^2
                    k = L2_per_head(k)
                    gate = sigmoid(W_g h * (s_gate * sqrt(gate_init) / init_std)^2)
                    h = W_o L2(softmax(q k^T / sqrt(head_dim)) v * gate)
  * nGPT residual (every block):  a = L2(h_in_block), b = L2(h_block_out),
                    alpha = |alpha_param * alpha_init / init_std|
                    h = L2(a + (b - a) * alpha)
  * loop exit (last virtual layer of every loop):  h = L2(h)
  * logits = lm_head(h) * (logit_scale / init_std)

L2 is llama.cpp's build_gdn_l2_norm(x, 1e-6), i.e. x / sqrt(sum(x^2) + 1e-6).
Sliding-window blocks let a query see the last sliding_window (1024) positions
including itself.
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import safetensors
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Defaults, following the conventions of the other ANEMLL model classes.
# ---------------------------------------------------------------------------

MODEL_DTYPE = torch.float32    # dtype the maths runs in (the seam that is scored)
WEIGHT_DTYPE = torch.bfloat16  # dtype the checkpoint weights are kept in
TEST_DEVICE = "cpu"
CONTEXT_LENGTH = 4096
MAX_LOOPS = 4                  # llama.cpp rejects n_loops > 4 for Dory
L2_EPS = 1e-6                  # llama.cpp: build_gdn_l2_norm(x, 1e-6f)

# One motif of the block layout, repeated nine times over the 72 physical blocks.
_BLOCK_MOTIF: Tuple[str, ...] = (
    "sliding_attention", "mlp", "sliding_attention", "mlp",
    "sliding_attention", "mlp", "full_attention", "mlp",
)

# llama.cpp's layer_pattern characters (see notes/gguf-facts.md).
_PATTERN_CHARS = {"sliding_attention": "S", "full_attention": "*", "mlp": "-"}
_CHARS_TO_TYPES = {v: k for k, v in _PATTERN_CHARS.items()}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class DoryConfig:
    """Dory configuration, read from the Hugging Face config.json.

    Accepts NVIDIA's public config keys (layer_types, nested rope_parameters,
    head_dim, n_recurrent_loops) as well as the flattened keys the GGUF
    converter emits (hybrid_layer_pattern, rope_theta / rope_theta_2, init_std /
    alpha_init / gate_init), so the same object serves the reference model and
    the converter.
    """

    def __init__(self, **kwargs):
        self.architectures = kwargs.get("architectures", ["DoryForCausalLM"])
        self.model_type = kwargs.get("model_type", "dory")

        # Tokenizer / specials
        self.bos_token_id = kwargs.get("bos_token_id", 1)
        eos = kwargs.get("eos_token_ids", kwargs.get("eos_token_id", 11))
        self.eos_token_ids = [int(v) for v in (eos if isinstance(eos, (list, tuple)) else [eos])]
        self.eos_token_id = self.eos_token_ids[0]
        self.pad_token_id = kwargs.get("pad_token_id", 11)

        # Geometry
        self.hidden_size = int(kwargs.get("hidden_size", 2560))
        self.intermediate_size = int(kwargs.get("intermediate_size", 7168))
        self.num_hidden_layers = int(kwargs.get("num_hidden_layers", 72))
        self.num_attention_heads = int(kwargs.get("num_attention_heads", 16))
        self.num_key_value_heads = int(kwargs.get("num_key_value_heads", 4))
        self.head_dim = int(kwargs.get("head_dim", 256))
        self.vocab_size = int(kwargs.get("vocab_size", 131072))
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.attention_bias = kwargs.get("attention_bias", False)
        self.mlp_bias = kwargs.get("mlp_bias", False)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", False)
        self.max_position_embeddings = int(kwargs.get("max_position_embeddings", 262144))

        # Dory / nGPT constants. llama.cpp: init_std, alpha_init, gate_init.
        self.initializer_range = float(kwargs.get("initializer_range", kwargs.get("init_std", 0.0177)))
        self.ngpt_alpha_init = float(kwargs.get("ngpt_alpha_init", kwargs.get("alpha_init", 0.05)))
        gate_init = kwargs.get("attention_output_gate_logit_scale_init", kwargs.get("gate_init", None))
        self.gate_init = math.sqrt(self.hidden_size) if gate_init is None else float(gate_init)
        # eps of the L2 norm (llama.cpp build_gdn_l2_norm(..., 1e-6))
        self.rms_norm_eps = float(kwargs.get("rms_norm_eps", L2_EPS))

        # Recurrence
        self.n_input_layers = int(kwargs.get("n_input_layers", 16))
        self.n_recurrent_layers = int(kwargs.get("n_recurrent_layers", 40))
        self.n_output_layers = int(kwargs.get("n_output_layers", 16))
        self.n_recurrent_loops = int(kwargs.get("n_recurrent_loops", kwargs.get("num_loops", 2)))
        self.max_loops = int(kwargs.get("max_loops", MAX_LOOPS))

        # Sliding window and RoPE
        self.sliding_window = int(kwargs.get("sliding_window", kwargs.get("swa_window_size", 1024)))
        rope = dict(kwargs.get("rope_parameters") or {})
        full_rope = dict(rope.get("full_attention") or {})
        sliding_rope = dict(rope.get("sliding_attention") or {})
        # theta_1 (full attention) and theta_2 (sliding attention) in llama.cpp.
        self.rope_theta = float(kwargs.get("rope_theta", full_rope.get("rope_theta", 10000000.0)))
        self.rope_theta_sliding = float(
            kwargs.get("rope_theta_2", kwargs.get("rope_local_theta",
                       sliding_rope.get("rope_theta", 10000.0)))
        )
        self.partial_rotary_factor = float(
            kwargs.get("partial_rotary_factor", full_rope.get("partial_rotary_factor", 1.0))
        )

        # Context / cache conventions shared with the other ANEMLL models
        self.context_length = int(kwargs.get("context_length", CONTEXT_LENGTH))
        self.state_length = int(kwargs.get("state_length", self.context_length))
        self.use_cache = bool(kwargs.get("use_cache", True))
        self.batch_size = int(kwargs.get("batch_size", 64))

        # Block layout: Hugging Face layer_types, the converter's pattern
        # string, or the canonical motif.
        layer_types = kwargs.get("layer_types")
        pattern = kwargs.get("hybrid_layer_pattern")
        if layer_types is not None:
            layer_types = list(layer_types)
        elif pattern is not None:
            layer_types = [_CHARS_TO_TYPES[c] for c in pattern]
        else:
            layer_types = [self._block_motif(i) for i in range(self.num_hidden_layers)]
        self.layer_types = layer_types

        self._validate()

    # -- layout helpers -----------------------------------------------------

    @staticmethod
    def _block_motif(index: int) -> str:
        return _BLOCK_MOTIF[index % len(_BLOCK_MOTIF)]

    def _validate(self) -> None:
        n = self.num_hidden_layers
        if len(self.layer_types) != n:
            raise ValueError(
                "layer_types has %d entries, expected %d" % (len(self.layer_types), n)
            )
        bad = [t for t in self.layer_types if t not in _PATTERN_CHARS]
        if bad:
            raise ValueError("unsupported layer type(s): %s" % sorted(set(bad)))
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")
        if self.n_input_layers + self.n_recurrent_layers + self.n_output_layers != n:
            raise ValueError(
                "n_input_layers + n_recurrent_layers + n_output_layers "
                "(%d+%d+%d) must equal num_hidden_layers (%d)"
                % (self.n_input_layers, self.n_recurrent_layers, self.n_output_layers, n)
            )
        if not 1 <= self.n_recurrent_loops <= self.max_loops:
            raise ValueError("n_recurrent_loops must be in 1..%d" % self.max_loops)

    @property
    def pattern(self) -> str:
        """llama.cpp's layer_pattern: one character per physical block."""
        return "".join(_PATTERN_CHARS[t] for t in self.layer_types)

    @property
    def rope_profile(self) -> List[int]:
        """llama.cpp's rope_profile: 2 = sliding-window theta, 1 = full theta."""
        return [2 if t == "sliding_attention" else 1 for t in self.layer_types]

    @property
    def attention_layer_indices(self) -> List[int]:
        return [i for i, t in enumerate(self.layer_types) if t != "mlp"]

    @property
    def full_attention_layer_indices(self) -> List[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "full_attention"]

    @property
    def sliding_layer_indices(self) -> List[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "sliding_attention"]

    @property
    def mlp_layer_indices(self) -> List[int]:
        return [i for i, t in enumerate(self.layer_types) if t == "mlp"]

    @property
    def recurrent_start(self) -> int:
        return self.n_input_layers

    @property
    def recurrent_stop(self) -> int:
        return self.n_input_layers + self.n_recurrent_layers

    def resolve_loop_count(self, loop_count: Optional[int] = None) -> int:
        """The loop count for one forward pass -- an argument, never a property of the weights."""
        loops = self.n_recurrent_loops if loop_count is None else int(loop_count)
        if not 1 <= loops <= self.max_loops:
            raise ValueError("loop_count must be in 1..%d, got %d" % (self.max_loops, loops))
        return loops

    def num_virtual_layers(self, loop_count: Optional[int] = None) -> int:
        loops = self.resolve_loop_count(loop_count)
        return self.n_input_layers + self.n_recurrent_layers * loops + self.n_output_layers

    @classmethod
    def from_json(cls, json_file: str) -> "DoryConfig":
        with open(json_file, "r") as handle:
            config_dict = json.load(handle)
        # Dory does not nest its text config, but accept a wrapper if present.
        if "text_config" in config_dict:
            text_cfg = dict(config_dict["text_config"])
            for key in ("bos_token_id", "eos_token_id", "pad_token_id", "architectures", "model_type"):
                if key in config_dict and key not in text_cfg:
                    text_cfg[key] = config_dict[key]
            config_dict = text_cfg
        # NVIDIA's config.json carries no token ids; generation_config.json does
        # (eos_token_id = [2, 11], i.e. </s> and <|im_end|>).
        gen_path = os.path.join(os.path.dirname(os.path.abspath(json_file)), 'generation_config.json')
        if os.path.isfile(gen_path):
            with open(gen_path, 'r') as handle:
                generation = json.load(handle)
            for key in ('bos_token_id', 'eos_token_id', 'pad_token_id'):
                if key in generation and key not in config_dict:
                    config_dict[key] = generation[key]
        return cls(**config_dict)


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------

class DoryL2Norm(nn.Module):
    """L2 normalisation over the last dimension: llama.cpp's build_gdn_l2_norm.

    llama.cpp computes rms_norm(x, eps / n) * (1 / sqrt(n)), which is exactly
    x / sqrt(sum(x^2) + eps). Following ANEMLL's convention the sum of squares
    is never formed directly: it goes through layer_norm on [x, -x] (whose mean
    is zero, so the mean subtraction is a no-op and the variance is mean(x^2)),
    and then the 1 / sqrt(n) factor is applied. That keeps the op ANE-safe for
    the FP16 port while staying numerically identical to the reference.
    """

    def __init__(self, dim: int, eps: float = L2_EPS):
        super().__init__()
        self.dim = int(dim)
        self.eps = float(eps)
        self.inv_sqrt_dim = 1.0 / math.sqrt(self.dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = hidden_states
        doubled = torch.cat([x, -x], dim=-1)
        normed = F.layer_norm(
            doubled,
            normalized_shape=(2 * self.dim,),
            weight=None,
            bias=None,
            eps=self.eps / self.dim,
        )
        return normed[..., : self.dim] * self.inv_sqrt_dim

    def extra_repr(self) -> str:
        return "dim=%d, eps=%g" % (self.dim, self.eps)


class DoryRotaryEmbedding(nn.Module):
    """NEOX (rotate-half) RoPE tables for Dory's two theta values.

    Dory rotates all head_dim dimensions and uses theta 1e4 on sliding-window
    blocks and 1e7 on full-attention blocks -- llama.cpp keeps both bases, as
    Gemma 3 does. Note the GGUF key names are inverted relative to the blocks
    they apply to (dory.rope.freq_base_2 is the sliding base); the mapping used
    here is the one in dory.cpp: profile 2 -> theta_2 (sliding), 1 -> theta_1.
    """

    def __init__(self, config: DoryConfig):
        super().__init__()
        self.dim = config.head_dim
        inv_freq_full = 1.0 / (
            config.rope_theta ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        inv_freq_sliding = 1.0 / (
            config.rope_theta_sliding ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)
        )
        self.register_buffer("inv_freq_full", inv_freq_full, persistent=False)
        self.register_buffer("inv_freq_sliding", inv_freq_sliding, persistent=False)

    def forward(self, position_ids: torch.Tensor, sliding: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq_sliding if sliding else self.inv_freq_full
        pos = position_ids.to(torch.float32).reshape(-1, 1)
        freqs = pos * inv_freq.reshape(1, -1)          # [T, head_dim / 2]
        emb = torch.cat((freqs, freqs), dim=-1)        # [T, head_dim]
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """NEOX rotate-half: pairs dimension i with i + head_dim/2."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(0)   # [1, T, head_dim], broadcast over the head axis
    sin = sin.unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class DoryMLP(nn.Module):
    """Dory MLP block.

    h = W_down L2(silu(W_gate h * suv_gate * sqrt(n_embd))
                  * (W_up   h * suv_up   * sqrt(n_embd)))
    """

    def __init__(self, config: DoryConfig, dtype: torch.dtype = WEIGHT_DTYPE):
        super().__init__()
        self.config = config
        hidden, ffn = config.hidden_size, config.intermediate_size
        self.gate_proj = nn.Linear(hidden, ffn, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden, ffn, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(ffn, hidden, bias=False, dtype=dtype)
        self.suv_gate = nn.Parameter(torch.empty(ffn, dtype=dtype))
        self.suv_up = nn.Parameter(torch.empty(ffn, dtype=dtype))
        self.out_norm = DoryL2Norm(ffn, config.rms_norm_eps)
        self.input_scale = math.sqrt(hidden)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ct = hidden_states.dtype
        g = F.linear(hidden_states, self.gate_proj.weight.to(ct))
        u = F.linear(hidden_states, self.up_proj.weight.to(ct))
        g = g * (self.suv_gate.to(ct) * self.input_scale)
        u = u * (self.suv_up.to(ct) * self.input_scale)
        return F.linear(self.out_norm(F.silu(g) * u), self.down_proj.weight.to(ct))


class DoryAttention(nn.Module):
    """Gated attention with L2-normalised q/k, the sqk scale and NEOX RoPE.

    Ours is the full recompute form: attention is causal, and sliding-window
    blocks additionally mask keys older than the window. The per_loop /
    last_loop KV-cache distinction does not appear here -- a recompute of a
    sequence is per_loop by construction, which is the 'exact' reference mode.
    """

    def __init__(self, config: DoryConfig, layer_idx: int, dtype: torch.dtype = WEIGHT_DTYPE):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.hidden_size = config.hidden_size
        self.q_dim = self.num_heads * self.head_dim
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        self.sliding_window = config.sliding_window if self.is_sliding else None
        self.rope_theta = config.rope_theta_sliding if self.is_sliding else config.rope_theta
        self.softmax_scale = 1.0 / math.sqrt(self.head_dim)
        self.qk_scale = math.sqrt(self.head_dim) / config.initializer_range
        self.gate_scale = math.sqrt(config.gate_init) / config.initializer_range

        self.q_proj = nn.Linear(self.hidden_size, self.q_dim, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(self.q_dim, self.hidden_size, bias=False, dtype=dtype)
        self.g_proj = nn.Linear(self.hidden_size, self.q_dim, bias=False, dtype=dtype)
        self.sqk = nn.Parameter(torch.empty(self.q_dim, dtype=dtype))
        self.s_gate = nn.Parameter(torch.empty(self.q_dim, dtype=dtype))
        self.q_norm = DoryL2Norm(self.head_dim, config.rms_norm_eps)
        self.k_norm = DoryL2Norm(self.head_dim, config.rms_norm_eps)
        self.attn_out_norm = DoryL2Norm(self.q_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ct = hidden_states.dtype
        batch, seq_len, _ = hidden_states.shape

        q = F.linear(hidden_states, self.q_proj.weight.to(ct))
        k = F.linear(hidden_states, self.k_proj.weight.to(ct))
        v = F.linear(hidden_states, self.v_proj.weight.to(ct))

        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # llama.cpp applies RoPE first, then the per-head L2 norms, then sqk.
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q = self.q_norm(q) * (self.sqk.to(ct) * self.qk_scale).pow(2).view(
            1, self.num_heads, 1, self.head_dim
        )
        k = self.k_norm(k)

        gate = F.linear(hidden_states, self.g_proj.weight.to(ct))
        gate = torch.sigmoid(gate * (self.s_gate.to(ct) * self.gate_scale).pow(2))

        attn = self._attend(q, k, v, ct, attn_bias=attn_bias)
        attn = attn.transpose(1, 2).reshape(batch, seq_len, self.q_dim)
        attn = self.attn_out_norm(attn * gate)
        return F.linear(attn, self.o_proj.weight.to(ct))

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dtype: torch.dtype,
        attn_bias: Optional[torch.Tensor] = None,
        query_chunk: int = 512,
    ) -> torch.Tensor:
        """Causal (+ sliding-window) attention, computed in query chunks.

        Chunking bounds the score tensor on a full-sequence recompute without
        changing the result: keys older than the window, and keys after the
        query, are masked to -inf before the softmax.
        """
        batch, num_heads, seq_len, head_dim = q.shape
        out = torch.empty_like(q)
        window = self.sliding_window
        for q0 in range(0, seq_len, query_chunk):
            q1 = min(seq_len, q0 + query_chunk)
            k0 = 0 if window is None else max(0, q0 - window + 1)
            kb = k[:, :, k0:q1]
            vb = v[:, :, k0:q1]
            if self.num_kv_groups > 1:
                kb = kb.repeat_interleave(self.num_kv_groups, dim=1)
                vb = vb.repeat_interleave(self.num_kv_groups, dim=1)
            scores = torch.matmul(q[:, :, q0:q1], kb.transpose(-1, -2)) * self.softmax_scale
            qi = torch.arange(q0, q1, device=q.device).unsqueeze(1)
            ki = torch.arange(k0, q1, device=q.device).unsqueeze(0)
            allowed = ki <= qi
            if window is not None:
                allowed = allowed & (ki > (qi - window))
            scores = scores.masked_fill(~allowed, float("-inf"))
            if attn_bias is not None:
                scores = scores + attn_bias
            probs = torch.softmax(scores, dim=-1).to(dtype)
            out[:, :, q0:q1] = torch.matmul(probs, vb)
        return out


class DoryDecoderLayer(nn.Module):
    """One physical block: attention or MLP, then the nGPT residual update."""

    def __init__(self, config: DoryConfig, layer_idx: int, dtype: torch.dtype = WEIGHT_DTYPE):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.mixer_type = config.layer_types[layer_idx]
        hidden = config.hidden_size
        if self.mixer_type == "mlp":
            self.mixer = DoryMLP(config, dtype=dtype)
            self.mlp_alpha = nn.Parameter(torch.empty(hidden, dtype=dtype))
            self.attn_alpha = None
        else:
            self.mixer = DoryAttention(config, layer_idx, dtype=dtype)
            self.attn_alpha = nn.Parameter(torch.empty(hidden, dtype=dtype))
            self.mlp_alpha = None
        self.residual_norm = DoryL2Norm(hidden, config.rms_norm_eps)   # a = L2(residual)
        self.mixer_norm = DoryL2Norm(hidden, config.rms_norm_eps)      # b = L2(block output)
        self.out_norm = DoryL2Norm(hidden, config.rms_norm_eps)        # L2(a + (b - a) * alpha)
        self.alpha_scale = config.ngpt_alpha_init / config.initializer_range

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        if self.mixer_type == "mlp":
            mixed = self.mixer(hidden_states)
            alpha_param = self.mlp_alpha
        else:
            mixed = self.mixer(hidden_states, cos, sin, attn_bias)
            alpha_param = self.attn_alpha
        a = self.residual_norm(residual)
        b = self.mixer_norm(mixed)
        alpha = torch.abs(alpha_param.to(residual.dtype) * self.alpha_scale)
        return self.out_norm(a + (b - a) * alpha)


class DoryModel(nn.Module):
    """The Dory backbone: embeddings, the looped block stack, final hidden state.

    The structure is the one the ANE chunks need (see notes/anemll-conventions.md
    sections 1.3 / 8.1-8.3): every component boundary is its own method, so the
    converter can trace embeddings / input blocks / recurrent blocks / output
    blocks as separate chunks and drive the recurrent chunk once per loop:

        hidden = model.forward_embeddings(input_ids)
        hidden = model.forward_input_blocks(hidden, position_ids)
        h_in   = hidden                     # Dory's h_in, the recurrent entry state
        for _ in range(loop_count):
            hidden = model.forward_recurrent(hidden, h_in, position_ids)
        hidden = model.forward_output_blocks(hidden, position_ids)

    Nothing about the weights depends on the loop count, so one loaded model
    serves every loop count; the recurrent component takes no loop argument,
    because overwriting the current rows each pass reproduces llama.cpp's
    last_loop semantics by construction.
    """

    def __init__(self, config: DoryConfig, dtype: torch.dtype = WEIGHT_DTYPE):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.layers = nn.ModuleList(
            DoryDecoderLayer(config, i, dtype=dtype) for i in range(config.num_hidden_layers)
        )
        self.rotary_emb = DoryRotaryEmbedding(config)
        self.loop_entry_norm = DoryL2Norm(config.hidden_size, config.rms_norm_eps)
        self.loop_exit_norm = DoryL2Norm(config.hidden_size, config.rms_norm_eps)
        self.compute_dtype = MODEL_DTYPE

    # -- component boundaries ----------------------------------------------

    def forward_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token embedding for a [T] or [1, T] id tensor, in compute dtype."""
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError("input_ids must be [T] or [1, T], got %s" % (tuple(input_ids.shape),))
        if input_ids.shape[-1] == 0:
            raise ValueError("input_ids must contain at least one token")
        return self.embeddings(input_ids).to(self.compute_dtype)

    def _rope_tables(self, position_ids: torch.Tensor) -> Dict[bool, Tuple[torch.Tensor, torch.Tensor]]:
        return {
            sliding: self.rotary_emb(position_ids, sliding=sliding)
            for sliding in (True, False)
        }

    def process_layers(
        self,
        hidden_states: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        start_layer: int = 0,
        end_layer: Optional[int] = None,
        attn_bias: Optional[torch.Tensor] = None,
        rope: Optional[Dict[bool, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        """Run physical blocks [start_layer, end_layer) once, in order.

        This is the call a chunk wrapper makes; the FP32 reference builds the
        causal/sliding mask itself, while the ANE wrapper will pass its
        precomputed mask as attn_bias.
        """
        if end_layer is None:
            end_layer = self.config.num_hidden_layers
        if not 0 <= start_layer <= end_layer <= self.config.num_hidden_layers:
            raise ValueError("invalid layer range %d..%d" % (start_layer, end_layer))
        seq_len = hidden_states.shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device)
        position_ids = position_ids.reshape(-1)
        if position_ids.shape[-1] != seq_len:
            raise ValueError("position_ids must have one entry per token")
        if rope is None:
            rope = self._rope_tables(position_ids)
        for physical in range(start_layer, end_layer):
            layer = self.layers[physical]
            cos, sin = rope[layer.mixer_type == "sliding_attention"]
            hidden_states = layer(hidden_states, cos, sin, attn_bias)
        return hidden_states

    def loop_entry(self, hidden_states: torch.Tensor, input_h: torch.Tensor) -> torch.Tensor:
        """h = L2(h + h_in) -- every loop starts by adding the entry state back."""
        return self.loop_entry_norm(hidden_states + input_h)

    def loop_exit(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The extra L2 at the end of every loop."""
        return self.loop_exit_norm(hidden_states)

    def forward_input_blocks(self, hidden_states, position_ids=None, attn_bias=None):
        cfg = self.config
        return self.process_layers(hidden_states, position_ids, 0, cfg.n_input_layers, attn_bias)

    def forward_recurrent(self, hidden_states, input_h, position_ids=None, attn_bias=None, rope=None):
        """One loop pass over the recurrent blocks: entry norm, blocks, exit norm."""
        cfg = self.config
        hidden_states = self.loop_entry(hidden_states, input_h)
        hidden_states = self.process_layers(
            hidden_states, position_ids,
            cfg.n_input_layers, cfg.n_input_layers + cfg.n_recurrent_layers,
            attn_bias, rope,
        )
        return self.loop_exit(hidden_states)

    def forward_output_blocks(self, hidden_states, position_ids=None, attn_bias=None):
        cfg = self.config
        return self.process_layers(
            hidden_states, position_ids,
            cfg.n_input_layers + cfg.n_recurrent_layers, cfg.num_hidden_layers, attn_bias,
        )

    # -- whole-sequence forward --------------------------------------------

    def virtual_layer_plan(self, loop_count: Optional[int] = None) -> List[Tuple[int, bool, bool]]:
        """(physical_block, is_loop_entry, is_loop_exit) for every virtual layer."""
        cfg = self.config
        loops = cfg.resolve_loop_count(loop_count)
        plan: List[Tuple[int, bool, bool]] = []
        for i in range(cfg.n_input_layers):
            plan.append((i, False, False))
        for _loop in range(loops):
            for i in range(cfg.n_recurrent_layers):
                plan.append((
                    cfg.n_input_layers + i,
                    i == 0,
                    i == cfg.n_recurrent_layers - 1,
                ))
        for i in range(cfg.n_output_layers):
            plan.append((cfg.n_input_layers + cfg.n_recurrent_layers + i, False, False))
        return plan

    def forward(
        self,
        input_ids: torch.Tensor,
        loop_count: Optional[int] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Whole-sequence forward, full recompute, loop count as an argument."""
        loops = self.config.resolve_loop_count(loop_count)
        hidden = self.forward_embeddings(input_ids)
        seq_len = hidden.shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device)
        position_ids = position_ids.reshape(-1)
        rope = self._rope_tables(position_ids)

        hidden = self.forward_input_blocks(hidden, position_ids, None)
        input_h = hidden                      # h_in: the state entering the recurrent stack
        for _loop in range(loops):
            hidden = self.forward_recurrent(hidden, input_h, position_ids, None, rope)
        return self.forward_output_blocks(hidden, position_ids, None)


class DoryLMHead(nn.Module):
    """Untied LM head with Dory's per-vocabulary logit scale."""

    def __init__(self, config: DoryConfig, dtype: torch.dtype = WEIGHT_DTYPE):
        super().__init__()
        self.config = config
        self.weight = nn.Parameter(torch.empty(config.vocab_size, config.hidden_size, dtype=dtype))
        self.logit_scale = nn.Parameter(torch.empty(config.vocab_size, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        ct = hidden_states.dtype
        logits = F.linear(hidden_states, self.weight.to(ct))
        # llama.cpp: logits * (logit_scale / init_std)
        return logits * self.logit_scale.to(ct) / self.config.initializer_range


class DoryForCausalLM(nn.Module):
    """Dory with its LM head: loads NVIDIA's BF16 safetensors and scores text.

    forward returns next-token logits for a whole sequence in one recompute,
    optionally only for the rows named by positions (which is how candidate
    log-probability files are produced without materialising seq_len x vocab).
    """

    config_class = DoryConfig

    def __init__(self, config: DoryConfig, dtype: torch.dtype = WEIGHT_DTYPE):
        super().__init__()
        self.config = config
        self.model = DoryModel(config, dtype=dtype)
        self.lm_head = DoryLMHead(config, dtype=dtype)
        # Bookkeeping filled in by load_pretrained_weights.
        self.loaded_tensor_names: List[str] = []
        self.missing_tensor_names: List[str] = []
        self.unused_tensor_names: List[str] = []

    # -- loading ------------------------------------------------------------

    def tensor_inventory(self) -> Dict[str, Tuple[int, ...]]:
        """Every weight this class consumes, mapped to its checkpoint shape."""
        cfg = self.config
        hidden, ffn = cfg.hidden_size, cfg.intermediate_size
        q_dim = cfg.num_attention_heads * cfg.head_dim
        kv_dim = cfg.num_key_value_heads * cfg.head_dim
        expected: Dict[str, Tuple[int, ...]] = {
            "model.embeddings.weight": (cfg.vocab_size, hidden),
            "lm_head.weight": (cfg.vocab_size, hidden),
            "lm_head.logit_scale": (cfg.vocab_size,),
        }
        for i in range(cfg.num_hidden_layers):
            prefix = "model.layers.%d" % i
            if cfg.layer_types[i] == "mlp":
                expected[prefix + ".mixer.gate_proj.weight"] = (ffn, hidden)
                expected[prefix + ".mixer.up_proj.weight"] = (ffn, hidden)
                expected[prefix + ".mixer.down_proj.weight"] = (hidden, ffn)
                expected[prefix + ".mixer.suv_gate"] = (ffn,)
                expected[prefix + ".mixer.suv_up"] = (ffn,)
                expected[prefix + ".mlp_alpha"] = (hidden,)
            else:
                expected[prefix + ".mixer.q_proj.weight"] = (q_dim, hidden)
                expected[prefix + ".mixer.k_proj.weight"] = (kv_dim, hidden)
                expected[prefix + ".mixer.v_proj.weight"] = (kv_dim, hidden)
                expected[prefix + ".mixer.o_proj.weight"] = (hidden, q_dim)
                expected[prefix + ".mixer.g_proj.weight"] = (q_dim, hidden)
                expected[prefix + ".mixer.sqk"] = (q_dim,)
                expected[prefix + ".mixer.s_gate"] = (q_dim,)
                expected[prefix + ".attn_alpha"] = (hidden,)
        return expected

    @staticmethod
    def _checkpoint_key(name: str) -> str:
        """Map a checkpoint tensor name onto this module's state_dict name.

        NVIDIA's Hugging Face export wraps the backbone in "backbone."
        (DoryForCausalLM); ANEMLL's model classes call that stack "model." and
        the converter builds its Core ML state names from it, so the prefix is
        rewritten here instead of in every module.
        """
        for prefix in ("backbone.", "model."):
            if name.startswith(prefix):
                return "model." + name[len(prefix):]
        return name

    @staticmethod
    def _checkpoint_keys(model_path: str) -> Tuple[List[str], List[str]]:
        """All tensor names in the checkpoint file(s), and the files themselves."""
        if os.path.isdir(model_path):
            files = sorted(
                os.path.join(model_path, f)
                for f in os.listdir(model_path)
                if f.endswith(".safetensors")
            )
            if not files:
                raise FileNotFoundError("no .safetensors file in %s" % model_path)
        elif os.path.isfile(model_path):
            files = [model_path]
        else:
            raise FileNotFoundError(model_path)
        keys: List[str] = []
        for path in files:
            with safetensors.safe_open(path, framework="pt") as handle:
                keys.extend(handle.keys())
        return keys, files

    def load_pretrained_weights(
        self,
        model_path: str,
        dtype: Optional[torch.dtype] = None,
        allow_missing: Optional[bool] = None,
        verbose: bool = True,
    ) -> bool:
        """Load NVIDIA's BF16 safetensors, failing loudly on any unaccounted tensor.

        model_path is either a directory holding *.safetensors or the file
        itself. The expected inventory is checked against the 507-tensor count
        (3 globals + 36 MLP blocks x 6 + 36 attention blocks x 8) before any
        tensor is read, every checkpoint tensor must be consumed, and every
        model weight must be filled.
        """
        if allow_missing is None:
            allow_missing = os.environ.get("ANEMLL_ALLOW_MISSING_WEIGHTS", "").lower() in ("1", "true", "yes")
        target_dtype = dtype or self.lm_head.weight.dtype
        expected = self.tensor_inventory()
        expected_total = 3 + 36 * 6 + 36 * 8
        if len(expected) != expected_total:
            raise RuntimeError(
                "Dory tensor inventory is %d, expected %d "
                "(3 globals + 36 MLP x 6 + 36 attention x 8)"
                % (len(expected), expected_total)
            )

        keys, files = self._checkpoint_keys(model_path)
        by_model_name: Dict[str, str] = {}
        for key in keys:
            mapped = self._checkpoint_key(key)
            if mapped in by_model_name:
                raise RuntimeError(
                    "checkpoint tensors %r and %r both map to %r"
                    % (by_model_name[mapped], key, mapped)
                )
            by_model_name[mapped] = key

        matched = sorted(set(by_model_name) & set(expected))
        missing = sorted(set(expected) - set(by_model_name))
        unused = sorted(set(by_model_name) - set(expected))
        self.loaded_tensor_names = matched
        self.missing_tensor_names = missing
        self.unused_tensor_names = unused
        if verbose:
            print("Dory weights: %d of %d expected tensors found in %d file(s)"
                  % (len(matched), len(expected), len(files)))
            if missing:
                print("Missing Dory tensors (%d): %s" % (len(missing), missing[:8]))
            if unused:
                print("Unused checkpoint tensors (%d): %s"
                      % (len(unused), [by_model_name[k] for k in unused[:8]]))
        if (missing or unused) and not allow_missing:
            raise RuntimeError(
                "Dory checkpoint does not match the model: %d missing, %d unused tensors "
                "(set ANEMLL_ALLOW_MISSING_WEIGHTS=1 to load anyway)" % (len(missing), len(unused))
            )

        state: Dict[str, torch.Tensor] = {}
        for path in files:
            with safetensors.safe_open(path, framework="pt") as handle:
                for key in handle.keys():
                    mapped = self._checkpoint_key(key)
                    if mapped not in expected or mapped in state:
                        continue
                    tensor = handle.get_tensor(key)
                    shape = tuple(tensor.shape)
                    if shape != expected[mapped]:
                        raise RuntimeError(
                            "Dory tensor %s has shape %s, expected %s"
                            % (key, shape, expected[mapped])
                        )
                    state[mapped] = tensor.to(target_dtype)
        result = self.load_state_dict(state, strict=False)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                "Dory load mismatch: missing %s, unexpected %s"
                % (result.missing_keys[:8], result.unexpected_keys[:8])
            )
        return True

    # -- inference ----------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        loop_count: Optional[int] = None,
        positions: Optional[Sequence[int]] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Logits for the whole sequence, or only for the rows named by positions."""
        hidden = self.model(input_ids, loop_count=loop_count, position_ids=position_ids)
        if positions is not None:
            idx = torch.as_tensor(list(positions), dtype=torch.long, device=hidden.device)
            hidden = hidden.index_select(1, idx)
        return self.lm_head(hidden)

    @torch.inference_mode()
    def logits_next_token(self, input_ids: torch.Tensor, loop_count: Optional[int] = None) -> torch.Tensor:
        """Logits predicting the token after the end of input_ids."""
        seq_len = input_ids.reshape(-1).shape[-1]
        return self.forward(input_ids, loop_count=loop_count, positions=[seq_len - 1])[0, -1]

    @torch.inference_mode()
    def greedy_generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        loop_count: Optional[int] = None,
        eos_token_ids: Optional[Iterable[int]] = None,
    ) -> List[int]:
        """Greedy decoding by full recompute (no cache) -- the reference path."""
        tokens = [int(t) for t in input_ids.reshape(-1).tolist()]
        stop = set(int(t) for t in (self.config.eos_token_ids if eos_token_ids is None else eos_token_ids))
        generated: List[int] = []
        for _ in range(max_new_tokens):
            logits = self.logits_next_token(torch.tensor(tokens, dtype=torch.long), loop_count=loop_count)
            nxt = int(torch.argmax(logits).item())
            generated.append(nxt)
            tokens.append(nxt)
            if nxt in stop:
                break
        return generated
