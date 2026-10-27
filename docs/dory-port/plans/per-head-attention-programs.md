# The architecture the measurements force

*Written after the state-budget measurement. This supersedes the five-component design for
long contexts; at 4,096 the five-component design still applies.*

## The constraint

A Core ML program that does real work may keep roughly **34 MiB of state** resident
(measured: 33.8 MiB loads, 45.0 MiB fails; one 24 MiB state loads, two 48 MiB do not).
Weights do not compete with that budget — 64 MiB of weights alongside a 24 MiB state loads —
and the number of carriers is not the constraint (60 tiny carriers load).

## What that means at 24,576 tokens

| cache | size per (layer, head) | carriers that fit per program |
|---|---|---|
| full attention, 24,576 rows × 256 dim × K+V | **24.0 MiB** | **1** |
| sliding window, 1,280 rows × 256 dim × K+V | **1.25 MiB** | ~24 |

Dory has 9 full-attention layers × 4 heads = 36 full caches (864 MiB) and 27 sliding layers
× 4 heads = 108 sliding caches (135 MiB). The only shape that fits:

* **36 attention programs**, one per (layer, head), each holding exactly one 24 MiB cache.
* **~6 sliding programs**, each holding ~24 sliding caches (30 MiB).
* **MLP and projection programs** carry no state at all and are limited only by the per-file
  weight budget, so they chunk by layer group as they do today.

## Cost to expect

Every program costs a dispatch. One token through one loop pass is therefore ~36 + ~6 + a few
MLP dispatches ≈ 50 dispatches, and at two loop passes ≈ 100 per token. At 0.15-0.30 ms of
dispatch overhead each, that is 15-30 ms per token — **35-70 tok/s**, before any compute.
That is the ceiling this architecture imposes, and it is worth measuring before building it:
the first experiment is a dispatch-cost microbenchmark, not a conversion.

## The alternative worth pricing first

If dispatch overhead makes the above too slow, the lever is **smaller caches**, not fewer
programs: an int8 cache halves every number (one (layer, head) becomes 12 MiB, so two heads
fit per program, halving the dispatch count), and a sliding-only design at 24K costs 135 MiB
total — but both change the model's numerics, so they need the KLD gate behind them.

## What is already proven

* One program holding one 24 MiB 24K cache and computing real attention over it: **loads and
  predicts** (probe `ms_probe`, state (2,1,24576,256), q·Kᵀ + mask + softmax + probs·V).
* Weights up to 64 MiB alongside that state: **loads**.
* Per-(layer, head) numerics: the refactor in `ticket/DANE-5` is bit-identical to the
  aggregated layout at 4,096 and equal at 24,576, so splitting by head changes nothing in
  the math — only the packaging.

## Measured dispatch cost (the number that decides TPS)

One predict on a 24K per-head attention program — state (2,1,24576,256), real q·Kᵀ + mask +
softmax + probs·V — costs **1.167 ms** (20 iterations, warmed, CPU_AND_NE).

That single number prices the whole design:

* 36 full-attention programs per loop pass => **42 ms**; at two loop passes **84 ms**.
* Adding the sliding and MLP dispatches, one token at 24,576 tokens is **~90-120 ms**,
  i.e. **8-12 tok/s**, and that is with the model already converted and correct.

So **24K and high TPS are in direct conflict on this stack.** The work per token at 24K is
not the problem — 248 MB of cache read per loop pass is ~2.5 ms of bandwidth — the problem is
that the cache has to be spread across dozens of programs, and each program costs about a
millisecond of latency that does not shrink with its size.

Levers, in order of expected effect, all of which change either the model or the numerics:

1. **int8 cache** — halves every cache, so two heads fit per program: 18 programs, ~21 ms per
   loop pass, and roughly double the throughput. Needs the KLD gate to price the quality cost.
2. **One loop pass at long context** — halves the dispatch count outright; Dory's loop count is
   a runtime choice, and the KLD penalty of 2 loops -> 1 loop is measurable with the existing
   reference (loops-1 floors are already recorded).
3. **Shorter context** — at 8,192 a per-head cache is 8 MiB, so four heads fit per program:
   9 programs instead of 36, and ~4x the throughput.

What this rules out is the thing worth ruling out: a single-program or few-program 24K design.
The state budget forbids it, and no packaging trick changes that.

## Implemented in ticket/DANE-5 (the opt-in plan)

The plan above is emitted by `anemll/ane_converter/dory_per_head.py` when the
converter is run with `--per-head-programs`; the five-component path is
untouched and still the default (4,096 stays exactly what it was).

### The programs a 24K conversion emits (102 files)

| kind | count | state per program | computes |
|---|---|---|---|
| `embeddings` | 1 | none | token ids -> hidden states, and `gate_0` |
| `entry_gate` | 1 | none | the loop entry's `l2(h + h_in)` and `gate_16`, per pass |
| `sliding_attn_L<b>` | 27 | **1 state (2, 4, 1280, 256) fp16 = 5.0 MiB** (that layer's own carrier) | one sliding layer's attention and nothing else: q/k/v of all 16 query heads, ring write and read, **one KV head scored per pass**, the raw contribution as **four** `contribution_h0..h3` tensors (one per KV head, 1024 channels each); **no gate, no tail**, and no 4096-wide tensor at any point |
| `sepilogue_L<b>` | 27 | none | the four contributions in, gate in as a leaf, `cat(contributions) * gate`, block `b`'s tail (out_norm over 4096, o_proj, residual blend), the MLP at `b+1`, and the next attention layer's gate out |
| `attn_full_<k>_h<h>` | 36 | **1 state (2, 1, 24576, 256) fp16 = 24.0 MiB** | one KV head of full-attention block 8k+6: its q/k/v/g rows, cache write, attention, contribution |
| `epilogue_<k>` | 9 | none | the tail of block 8k+6 (out_norm over 4096, o_proj, residual blend), the MLP at 8k+7, and the next layer's gate |
| `lm_head` | 1 | none | logits |

A sliding layer is split exactly the way a full-attention layer already was: the
attention is its own stateful program, and everything that happens after the
attention — the gate, the tail, the MLP — is one unstated program.

Declared state, total: **36 x 24.00 MiB + 27 x 5.00 MiB = 999.0 MiB**, largest
single state 24.00 MiB (the measured budget is ~34 MiB per program; the plan
validator refuses any program over 30 MiB). `--slots-per-sliding-carrier` is 1
here: one program per sliding layer owns that layer's carrier, so a carrier
shared by three layer-slots would be declared by three programs.

### The rule that decides the sliding layers: no second conv chain on a stateful program

Measured on the real per-head program, one variant at a time: it converts as a
passthrough, with the k/v cache write, with the scores, with q_proj. It stops
converting the moment it carries a **second convolution chain that joins the
attention output** — `g_proj -> sigmoid -> out * gate` fails, and an `out + gate`
variant fails identically, so it is not the elementwise op. `out * sigmoid(out)`
converts, so it is not the sigmoid either. `out * gate` with the gate arriving as
an **input leaf** converts, and so does a program that takes the gate, applies
it, finishes the layer and emits the **next** layer's gate as a second output.

That is why the sliding layers cannot share a program: each of the three layers
in a period computes its own gate, and the old `sliding_<k>` group (3 sliding
layers + 3 MLPs, 15.0 MiB) never loaded for exactly that reason. It is kept
behind `program_specs(config, sliding="group")` so the difference stays
measurable.

Handing the gate in and the next one out was necessary but not sufficient. The
one-program-per-sliding-layer shape built this way (`sliding_L0`: 175 MB, one
5.0 MiB state, 384 ops, 8 convolutions — q/k/v/o, the next layer's g_proj, and
the MLP's three) still did **not** build a plan at 24,576. Neither the state nor
any single ingredient was the problem: the same shape without the MLP, and the
same shape with fewer convolutions, load. What did not fit is the whole
"attention + tail + MLP" chain in one stateful program.

So a sliding layer is now split in two, which is what the full-attention path
already does:

    sliding_attn_L<b>   one state (that layer's carrier), q/k/v, the raw
                        contribution out as four 1024-wide tensors -- one
                        convolution chain, and nothing that joins it to the
                        output
    sepilogue_L<b>      unstated: cat(contributions) * gate, out_norm, o_proj,
                        the nGPT blend, the MLP, and gate_<b+2> out

Splitting in two was still not enough.  A bisect of the program on the ANE
clears the ingredients one at a time -- one KV group scored (1024 wide), all
four groups computed and summed into 1024 columns, no gate, no mask, no RoPE, a
512-row carrier: every one of those loads -- while the shipped
**single 4096-wide `contribution` output** fails.  Four outputs alone did not
fix it either: the same compute with four 1024-wide outputs that are
`torch.chunk()` of the 4096 tensor still builds no plan.  So the rule is not
about the output width but about the intermediate: **a stateful program may not
materialise a 4096-wide tensor anywhere**, and slicing one after the fact
cannot help.  `sliding_attn_L0` therefore scores one KV head per pass, from that
head's rows of the carrier, and emits its four 1024-wide tensors directly --
nothing wide is ever built (`mixer.contrib_per_head`).  The four outputs remain
the interface the full-attention path already has between `attn_full_<k>_h<h>`
and `epilogue_<k>`, so the two halves of a sliding layer still meet on the
surface the rest of the plan uses.

The gate chain is unchanged in shape — every attention layer's gate is still
emitted by the program that produces the hidden state it depends on, and taken
as a leaf by exactly one program — and `validate_program_specs` still proves
both halves.

Both older shapes stay selectable for measurement (`sliding="layer"` is the
one-program `sliding_L<b>`, `sliding="group"` the three-layer group); neither is
what the default plan builds.

Dory's layout is `sliding, mlp, sliding, mlp, sliding, mlp, full, mlp`. Three
sliding caches can be *declared* in one program, but a program that only does
attention must then be **called once per layer**, so it saves no dispatch at all
— and dispatch is the only reason to group. One call can only advance three
layers if the MLPs between them are in the same program. The largest contiguous
sliding-only run is therefore blocks [8k, 8k+6) (3 sliding + 3 MLP, 15.0 MiB);
a run across a period boundary would have to carry a full-attention layer as
well, whose per-head caches are 24.0 MiB each.

**Dispatches per loop pass: 56** (5 periods x (3 sliding attention programs +
3 slide epilogues + 4 head programs + 1 epilogue) + the entry gate, which runs
once per pass); at the measured 1.167 ms per 24K attention predict, the 20
full-attention head programs inside the zone are 23 ms per pass, and the rest is
small-tensor work. Splitting each sliding layer costs 3 more dispatches per
period than the unsplit shape (102 files instead of 75); the unsplit shape does
not load, so that is the price of the split, not of a choice.

Per pass, the recurrent zone (blocks 16-55, five periods) is:

    entry_gate                         1
    sliding_attn_L<b>                  15   (3 per period)
    sepilogue_L<b>                     15   (3 per period)
    attn_full_<k>_h<h>                 20   (4 per period)
    epilogue_<k>                        5   (1 per period)
    ---------------------------------------
    total                              56

### The command

```bash
# the stateful half: one sliding layer's attention, its own 5.0 MiB carrier
python -m anemll.ane_converter.dory_converter \
  --model <dory-bf16-dir> --output out-24k \
  --context-length 24576 --batch-size 64 \
  --per-head-programs --rope-tables-as-inputs --lut-blocks 6 \
  --only sliding_attn_L0 --dry-run       # then drop --dry-run

# the unstated half: gate, layer tail, the MLP, the next gate
... same flags ... --only sepilogue_L0 --dry-run
```

`--per-head-programs` implies `--heads-per-state 1` and
`--slots-per-sliding-carrier 1` (a value other than 1 is refused: one program
per sliding layer declares that layer's own carrier).

`sliding_attn_L0` declares exactly one state, `model.kv_cache_sliding_0_0`
(2, 4, 1280, 256) FLOAT16 = 5.0 MiB, inputs `hidden_states (1, seq, 2560)`,
`causal_mask (1, 1, seq, 25856)`, `current_pos`, `cache_pos`,
`cos_sliding`, `sin_sliding (1, 1, seq, 256)`, and **four** outputs
`contribution_h0 .. contribution_h3`, `(1, seq, 1024)` each. This is the
shape the ticket exists to make loadable: one cache, one convolution chain, and
outputs that join nothing.

`sepilogue_L0` declares **no** state and takes `hidden_states (1, seq, 2560)`,
the same four `contribution_h0..h3 (1, seq, 1024)`, and `gate
(1, seq, 4096)`; it emits `h_next (1, seq, 2560)` and `gate_2 (1, seq,
4096)`.

Any one program can be built with `--only <name>`; the names are in the plan
(`sliding_attn_L<b>` and `sepilogue_L<b>` for a sliding layer,
`attn_full_<k>_h<h>` for a KV head, `epilogue_<k>`, `entry_gate`,
`embeddings`, `lm_head`).

### What is proven, and what is not

* **Proven in torch** (tests/test_per_head_programs.py, fixed seed): four head
  programs + the epilogue reproduce the aggregated layer's output at 4,096 and
  at 24,576, across steps with the caches carried, including a step at position
  20,000 where the softmax is genuinely soft. Argmax of a readout is identical.
* **Proven in torch, the whole chain** (same file): over one full 8-block period
  — embeddings -> `sliding_attn_L0` -> `sepilogue_L0` -> `sliding_attn_L2` ->
  `sepilogue_L2` -> `sliding_attn_L4` -> `sepilogue_L4` -> four head programs ->
  `epilogue_0` — the plan reproduces the aggregated model at 4,096 and at
  24,576, rel 1e-3 on the readout logits and identical argmax at every scored
  position (the late step is at 20,000). `gate_0` is asserted bit-identical to
  the gate the aggregated block 0 computes for itself, and `sliding_attn_L0`'s
  four outputs are asserted **bit-identical** to the aggregated layer's own
  attention output (taken with a gate of ones and `finish` bypassed, which is
  the tensor the aggregated layer computes), both joined and slice by slice:
  each `contribution_h<i>` is asserted **equal** (not close) to channels
  `[i*1024, (i+1)*1024)` of the aggregated tensor, at 4,096 **and** 24,576, on a
  prefill step and on a late decode step -- max abs 0.0 on all four -- so the
  four single-head passes are the batched pass's channel blocks bit for bit, and
  the slice order is the concatenation order the epilogue uses.  A second test
  watches the matmuls: two per KV head, none carrying the four-head batch, which
  is what "never builds the wide tensor" means in the graph.
* **Proven by conversion** (tests/test_per_head_convert.py, a synthetic 72-block
  checkpoint at a 512-token context, so it costs seconds): `sliding_attn_L0`
  converts to a package declaring `hidden_states, causal_mask, current_pos,
  cache_pos, cos_sliding, sin_sliding` in, `contribution_h0..h3` out -- four
  outputs of 16 channels each in that 8-head/4-KV-head toy config, (1, seq,
  1024) at the real shapes, read back from the mlpackage so the graph is proven
  to emit four slices and not four names for one wide tensor: the traced program
  has **8 matmuls** (two per KV head), 4 softmaxes and no `split` op, and no
  operation in it produces a 4096-wide value -- and exactly one
  state `model_kv_cache_sliding_0_0` (2, 4, 512, 8) FLOAT16 (at 24,576 that
  state is (2, 4, 1280, 256) = 5.0 MiB); `sepilogue_L0` converts with
  `hidden_states, contribution_h0..h3, gate` in and `h_next, gate_2` out and no
  state.
  `entry_gate` converts with `hidden_states, h_in` in and `h_entry, gate_16` out
  and no state; `epilogue_0` with four head inputs plus `gate` in and
  `output_hidden_states, gate_8` out and no state; the prefill function declares
  the same names and state as infer.
  The MIL pipeline accepts each one (the same `-14` execution-plan warning every
  package on this shell produces). Palettization is off in that test because
  coremltools' k-means palettizer needs scikit-learn, which the dev venv lacks.
* **Not proven**: that it *loads* and *predicts* at 24K on the ANE, and the
  placement gate's verdict for the real shapes. Both need a load, which this
  ticket cannot do. The nan-cache probe already showed the shape loads at
  (2,1,24576,256); what remains is the real weight set in that state.
* **Not built**: the runtime that drives 102 programs and one MLState per
  attention program. The meta records the whole plan under
  `per_head_programs` (blocks, kv head, gate in/out, calls per token, states
  with sizes, inputs, outputs) plus a `gate_chain` map from `gate_<block>` to
  the program that emits it, precisely so that runtime is a loop over the
  manifest.
