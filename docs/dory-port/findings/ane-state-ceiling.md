# The Core ML state budget on the Apple Neural Engine

*Measured while trying to run a 24,576-token KV cache on the ANE. Three ceilings, and the
first two are misleading on their own.*

## Summary

On macOS with coremltools 8.3.0, whether a **stateful** `mlprogram` loads depends on three
things, and they interact:

1. **A single state may not exceed ~24 MiB.** 24 MiB loads, 32 MiB fails (3/3 attempts each).
2. **Total resident state is capped at roughly 34–45 MiB — but only when the graph does real
   work.** A trivial graph declaring 45 states totalling 999 MiB loads fine. A graph that
   actually runs attention over those states fails once the total passes ~34 MiB.
3. **The number of states is not the limit** — 60 tiny states load, 45 tiny states load, and
   36 states of 0.94 MiB (33.8 MiB total) load while 36 states of 1.25 MiB (45 MiB total) fail.
   Count only became a suspect because count and bytes moved together in the earlier probes.

| graph | states | total state | result |
|---|---|---|---|
| trivial write/read | 1 | 24.0 MiB | loads |
| trivial write/read | 1 | 36.0 MiB | **fails** |
| trivial write/read | 45 | 999.0 MiB | loads |
| trivial write/read | 63 | 63.0 MiB | **fails** |
| **real attention** | 1 | 24.0 MiB | **loads, predicts** |
| **real attention** | 2 | 48.0 MiB | **fails** |
| **real attention** | 36 | 33.8 MiB | **loads** |
| **real attention** | 36 | 45.0 MiB | **fails** |
| **real attention** | 60 | 3.8 MiB | **loads** |
| real attention + 64 MiB of weights | 1 | 24.0 MiB | loads |

The last two rows are the ones that matter for a design: **weights are not the constraint**
(64 MiB of weights alongside a 24 MiB state is fine), and **carrier count is not the
constraint** (60 carriers are fine when they are small). What a working graph cannot do is
keep more than ~34 MiB of state resident.

## Why the trivial-graph numbers mislead

A stateful model whose only op is "read a row, lerp it, write it back" will happily declare
999 MiB across 45 states. That is a real measurement, and it is useless as a design rule:
as soon as the same model multiplies a query against the cache, the budget collapses to
~34 MiB. Any conclusion drawn from the 999 MiB result — including "45 states is fine" —
is true only of graphs that never touch the data.

This is the trap worth publishing. We built an entire per-(layer, head) cache layout around
the trivial-graph numbers, verified it numerically, and it still would not load, because the
number that mattered was the one measured with the real graph.

## What that means for a long-context model

A 24,576-token attention cache for one (layer, head) — K and V, 256 dims, fp16 — is exactly
**24.0 MiB**, and a single program holding exactly that loads and computes attention over it.
Two of them (48 MiB) do not. So for a model with 9 full-attention layers × 4 heads, the only
shape that fits is **one Core ML model per (layer, head)**: 36 programs, each carrying one
24 MiB cache, plus programs for everything that carries no state at all (MLPs, projections).

At shorter contexts the arithmetic is kinder: a 4,096-token per-head cache is 4 MiB, so eight
of them fit in one program, and the model can be chunked by layer groups instead of by head.

## Reproducing

```python
# one program, one 24K per-head cache, real attention over it -> loads
ROWS, D = 24576, 256
cache = torch.zeros(2, 1, ROWS, D, dtype=torch.float16)       # K and V for ONE head
# ... write one row, then matmul(q, K^T) + mask, softmax, matmul(probs, V)
states = [ct.StateType(name="c", wrapped_type=ct.TensorType(shape=(2, 1, ROWS, D),
                                                            dtype=np.float16))]
```

Double the state count (two such caches, 48 MiB) and the same graph fails with
`error code: -14`; shrink the rows and it loads again. The transition is between 33.8 MiB
and 45.0 MiB of total state for this graph.

## Two adjacent traps

1. **A sandboxed shell cannot load a stateful model at all.** Core ML writes its compiled
   model cache under `~/Library/Caches/com.apple.CoreML`; if that path is not writable the
   load fails with the same generic `-14`, and you will spend a day blaming the platform.
2. **`ct.models.MLModel(path)` constructs lazily.** It does not build an execution plan, so
   "it loads" proves the file is valid, not that it can execute. Always construct →
   `make_state()` → `predict()`.
