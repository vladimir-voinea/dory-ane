"""The folding gate: the ANE form's static scales, folded in FP32, reproduce the
unfolded order to FP32 rounding -- on NVIDIA's real BF16 weights.

Spec D5 folds four per-output-channel scales into weights:

    suv_gate * sqrt(2560)                     -> MLP gate_proj rows
    suv_up   * sqrt(2560)                     -> MLP up_proj rows
    (s_gate * sqrt(gate_init) / init_std)^2   -> attention g_proj rows
    logit_scale / init_std                    -> lm_head rows

Each is a positive per-output-channel factor, so folding is exact algebra; this
script measures the FP32 rounding it actually costs on the real tensors.

    python scripts/dory_ane_folding_check.py --blocks 0 1 6 7 16 17 22 55
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

from anemll.models.dory_ane_model import DoryANEConfig, DoryANEModel, DoryL2Norm

WEIGHTS = os.path.expanduser(
    "~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16/"
    "snapshots/b366b87f762ea334ba3eb55912031baba1515810/model.safetensors")


def subset_model(blocks, fold):
    cfg = DoryANEConfig(context_length=64, batch_size=4, fold_scales=fold,
                        dtype=torch.float32, compute_mode="conv")
    return DoryANEModel(cfg, block_indices=blocks)


def load_blocks(model, path, blocks):
    """Load only the requested physical blocks, in FP32, folded or unfolded."""
    from safetensors import safe_open
    cfg = model.config
    with safe_open(path, framework="pt", device="cpu") as f:
        for layer in model.layers:
            p = layer.block
            prefix = f"backbone.layers.{p}"
            alpha_name = f"{prefix}.mlp_alpha" if layer.is_mlp else f"{prefix}.attn_alpha"
            layer.alpha.data.copy_(layer.compute_alpha(f.get_tensor(alpha_name), cfg))
            if layer.is_mlp:
                mlp = layer.mixer
                suv_gate = f.get_tensor(f"{prefix}.mixer.suv_gate").float()
                suv_up = f.get_tensor(f"{prefix}.mixer.suv_up").float()
                ksuv = math.sqrt(float(cfg.hidden_size))
                for name, scale in (("gate_proj", suv_gate * ksuv), ("up_proj", suv_up * ksuv)):
                    w = f.get_tensor(f"{prefix}.mixer.{name}.weight").float()
                    if cfg.fold_scales:
                        w = w * scale.view(-1, 1)
                    else:
                        getattr(mlp, "suv_gate" if name == "gate_proj" else "suv_up").data.copy_(scale)
                    getattr(mlp, name).weight.data.copy_(w.view(*w.shape, 1, 1))
                down = f.get_tensor(f"{prefix}.mixer.down_proj.weight").float()
                mlp.down_proj.weight.data.copy_(down.view(*down.shape, 1, 1))
            else:
                attn = layer.mixer
                sqk = f.get_tensor(f"{prefix}.mixer.sqk").float()
                attn.sq.data.copy_(attn.compute_sq(sqk, cfg))
                sg = attn.compute_sg(f.get_tensor(f"{prefix}.mixer.s_gate").float(), cfg)
                for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    w = f.get_tensor(f"{prefix}.mixer.{name}.weight").float()
                    getattr(attn, name).weight.data.copy_(w.view(*w.shape, 1, 1))
                g = f.get_tensor(f"{prefix}.mixer.g_proj.weight").float()
                if cfg.fold_scales:
                    g = g * sg.view(-1, 1)
                else:
                    attn.sg.data.copy_(sg)
                attn.g_proj.weight.data.copy_(g.view(*g.shape, 1, 1))
    model.eval()


def run_blocks(model, hidden, cur, cache_pos):
    from anemll.utils import dory_caches as dc
    cfg = model.config
    q_len = hidden.shape[1]
    mask = dc.build_causal_mask(0, q_len, cfg.context_length, cfg.sliding_rows)
    mask_full = mask.narrow(-1, 0, cfg.context_length)
    mask_sliding = mask.narrow(-1, cfg.context_length, cfg.sliding_rows)
    h = hidden
    for layer in model.layers:
        h = layer(h, mask_full, mask_sliding, model.kv_cache_full, model.kv_cache_sliding,
                  cur, cache_pos)
    return h


def lm_head_fold_check(path, hidden, vocab_rows):
    """Folded lm_head rows vs raw rows times logit_scale / init_std, in FP32."""
    from safetensors import safe_open
    with safe_open(path, framework="pt", device="cpu") as f:
        W = f.get_tensor("lm_head.weight")[vocab_rows].float()
        s = f.get_tensor("lm_head.logit_scale")[vocab_rows].float() / 0.0177
    folded = W * s.view(-1, 1)
    h = hidden.float().reshape(-1)
    a = folded @ h
    b = (W @ h) * s
    denom = b.abs().max().clamp_min(1e-30)
    return {"max_abs": float((a - b).abs().max()), "max_rel": float(((a - b).abs() / denom).max())}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--blocks", type=int, nargs="+", default=[0, 1, 6, 7, 16, 17, 22, 55])
    ap.add_argument("--hidden-state", default=os.path.expanduser(
        "~/dev/dory-ane-artifacts/DANE-4/hidden_last.pt"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    torch.set_num_threads(4)
    blocks = list(args.blocks)
    cfg = DoryANEConfig(context_length=64, batch_size=4, dtype=torch.float32)
    torch.manual_seed(0)
    ids = torch.tensor([[7, 11, 99, 3, 42, 8, 15, 16]])
    hidden = torch.randn(1, ids.shape[1], cfg.hidden_size) * 0.1
    # a unit-norm start, like the real embedding rows
    hidden = hidden / hidden.norm(dim=-1, keepdim=True)

    results = {}
    outs = {}
    for fold in (True, False):
        model = subset_model(blocks, fold)
        load_blocks(model, args.weights, blocks)
        cur = torch.tensor([0], dtype=torch.int32)
        with torch.inference_mode():
            outs[fold] = run_blocks(model, hidden, cur, cur)
        del model
    diff = (outs[True] - outs[False]).abs()
    denom = outs[False].abs().max().clamp_min(1e-30)
    results["blocks"] = {
        "blocks": blocks,
        "max_abs": float(diff.max()),
        "max_rel": float((diff / denom).max()),
        "output_scale": float(denom),
    }

    # real hidden state if the FP16 model left one behind, else a real embedding row
    if os.path.exists(args.hidden_state):
        h = torch.load(args.hidden_state, map_location="cpu")
    else:
        from safetensors import safe_open
        with safe_open(args.weights, framework="pt", device="cpu") as f:
            h = f.get_tensor("backbone.embeddings.weight")[100]
        sys.stderr.write("note: no saved hidden state, using a real embedding row\n")
    results["lm_head"] = lm_head_fold_check(args.weights, h, slice(0, 8192))

    print(json.dumps(results, indent=2))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)
    ok = results["blocks"]["max_rel"] < 1e-5 and results["lm_head"]["max_rel"] < 1e-5
    print("FOLDING GATE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
