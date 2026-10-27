"""ANEMLL's FP16 preflight, applied to the ANE-safe Dory module.

`anemll/utils/fp16_preflight.sh` drives `fp16_compatibility_check.py`, which
loads the model with `transformers.AutoModelForCausalLM`.  Dory has no
transformers class (model_type `dory`, no remote code), so that script cannot
load it.  This runner applies the *same* criteria to the same weights and to
this port's own FP16 forward:

1. weight range -- every tensor's absmax must be <= FP16_MAX (ANEMLL's check,
   `analyze_weights` in anemll/utils/fp16_compatibility_check.py);
2. precision run -- forward hooks on every block and every mixer count NaN,
   Inf and values above FP16_MAX per layer over the evaluation text.

    python scripts/dory_ane_fp16_preflight.py --tokens 64
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from anemll.models.dory_ane_model import DoryANEConfig, DoryANEForCausalLM
from anemll.utils.fp16_compatibility_check import FP16_MAX

WEIGHTS = os.path.expanduser(
    "~/.cache/huggingface/hub/models--nvidia--NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16/"
    "snapshots/b366b87f762ea334ba3eb55912031baba1515810/model.safetensors")


def check_weights(path):
    from safetensors import safe_open
    worst = []
    total = 0
    exceeding = 0
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            total += t.numel()
            absmax = t.float().abs().max().item()
            if absmax > FP16_MAX:
                exceeding += 1
            worst.append((absmax, key, tuple(t.shape)))
    worst.sort(reverse=True)
    return {
        "total_params": total,
        "weights_exceed_fp16": exceeding,
        "weight_max_abs": worst[0][0] if worst else 0.0,
        "weight_max_layer": worst[0][1] if worst else None,
        "top10": [{"name": k, "absmax": v, "shape": list(s)} for v, k, s in worst[:10]],
        "fp16_max": FP16_MAX,
    }


def run_precision(path, n_tokens, context_length, threads):
    torch.set_num_threads(threads)
    cfg = DoryANEConfig(context_length=context_length, batch_size=64, fold_scales=True,
                        dtype=torch.float16, compute_mode="matmul")
    model = DoryANEForCausalLM(cfg)
    report = model.load_pretrained_weights(path)
    model.eval()

    stats = {"nan": 0, "inf": 0, "overflow": 0, "max_abs": 0.0, "by_layer": []}

    def hook(name):
        def _hook(_module, _inputs, output):
            t = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(t) or not t.is_floating_point():
                return
            f = t.float()
            nan = int(torch.isnan(f).sum())
            inf = int(torch.isinf(f).sum())
            over = int((f.abs() > FP16_MAX).sum())
            mx = float(f.abs().max())
            if nan or inf or over:
                stats["by_layer"].append({"layer": name, "nan": nan, "inf": inf,
                                          "overflow": over, "absmax": mx})
            stats["nan"] += nan
            stats["inf"] += inf
            stats["overflow"] += over
            stats["max_abs"] = max(stats["max_abs"], mx)
        return _hook

    captured = {}

    def capture_output(_module, _inputs, output):
        captured["hidden"] = output.detach().float().clone()

    handles = [model.chunk_output.register_forward_hook(capture_output)]
    for layer in model.model.layers:
        handles.append(layer.register_forward_hook(hook(f"block{layer.block}")))
        handles.append(layer.mixer.register_forward_hook(hook(f"block{layer.block}.mixer")))
        if layer.is_mlp:
            # the G7 seam: silu(g)*u over 7,168 channels is the largest tensor
            # in the model, and it is what the MLP L2 normalises
            def pre_norm(_module, inputs, name=f"block{layer.block}.mlp_prenorm"):
                hook(name)(None, None, inputs[0])
            handles.append(layer.mixer.norm.register_forward_pre_hook(pre_norm))

    from anemll.eval import load_eval_set
    evalset = load_eval_set()
    ids = [int(t) for t in evalset.token_ids[:n_tokens]]
    with torch.inference_mode():
        model.reset_caches()
        for pos, tok in enumerate(ids):
            model.decode_step(tok, pos, loops=2)
        logits = model.decode_step(ids[-1], len(ids), loops=2)
    for h in handles:
        h.remove()

    if "hidden" in captured:
        torch.save(captured["hidden"], os.path.expanduser(
            "~/dev/dory-ane-artifacts/DANE-4/hidden_last.pt"))
    stats["logits_finite"] = bool(torch.isfinite(logits).all())
    stats["consumed"] = report["consumed"]
    stats["missing"] = report["missing"]
    return stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--context-length", type=int, default=256)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    print("=" * 70)
    print("WEIGHT ANALYSIS")
    print("=" * 70)
    weights = check_weights(args.weights)
    print(f"Total parameters: {weights['total_params']:,}")
    print(f"Max weight magnitude: {weights['weight_max_abs']:.4f}")
    print(f"Max weight layer: {weights['weight_max_layer']}")
    print(f"Weights exceeding FP16 range: {weights['weights_exceed_fp16']}")
    print("All weights within FP16 range" if not weights["weights_exceed_fp16"]
          else "SOME WEIGHTS EXCEED FP16 RANGE")

    print("=" * 70)
    print("PRECISION RUN (FP16 forward, hooks on every block)")
    print("=" * 70)
    prec = run_precision(args.weights, args.tokens, args.context_length, args.threads)
    print(f"tensors consumed: {prec['consumed']} (missing {len(prec['missing'])})")
    print(f"NaN: {prec['nan']}  Inf: {prec['inf']}  > FP16_MAX: {prec['overflow']}")
    print(f"max |activation|: {prec['max_abs']:.2f} (FP16_MAX = {FP16_MAX:.0f})")
    for row in prec["by_layer"]:
        print("  ", row)

    ok = (weights["weights_exceed_fp16"] == 0 and prec["nan"] == 0 and prec["inf"] == 0
          and prec["overflow"] == 0 and prec["logits_finite"])
    print()
    print(f"ANE Compatible: {'PASS' if ok else 'FAIL'}")
    payload = {"weights": weights, "precision": prec, "verdict": "PASS" if ok else "FAIL"}
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
