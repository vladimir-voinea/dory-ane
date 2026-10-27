# Dory on the Apple Neural Engine — status

**Dory1.0-4B runs on the Neural Engine at a 24,576-token context** and produces coherent text.
This file is what a reader needs before the detail in `findings/` and `plans/`.

## The measured quality/speed curve

Four configurations of the same 102-program plan at 24,576 tokens, differing only in weight format.
Decode throughput is at `loops = 1`; KLD is against the aggregated model in torch fp16, scored over
two windows of the fixed evaluation set.

| weights | tok/s | KLD 64–71 | top-1 | KLD 120–127 | top-1 | sample output |
|---|---|---|---|---|---|---|
| all fp16 | 5.93 | 0.056 | 100 % | 0.019 | 87.5 % | *"The capital of France is Paris."* |
| attention fp16, MLP LUT6 | 7.64 | 0.459 | 75 % | 0.053 | 87.5 % | rambles |
| MLP fp16, attention LUT6 | 5.86 | 0.099 | 100 % | 0.017 | 87.5 % | rambles |
| all LUT6 | **10.71** | 0.632 | 62.5 % | 0.077 | 100 % | rambles |

The Pareto frontier is the two ends: full precision buys faithfulness, 4-bit buys 1.8× the throughput.
Neither hybrid beats either end, because the MLP holds most of the bytes and most of the error.

**The rates.** At 24,576 tokens with `loops = 1`: load 21 s, prefill **130 tok/s** steady state,
decode **11.27 tok/s** on the LUT6 set and 5.93 on fp16. Two one-off costs sit outside those rates and
are reported separately: the first token after a phase switch takes 80–100 s, because switching from
the prefill packages to the infer packages opens 102 more models and builds their execution plans.

Both of those numbers were wrong in earlier notes, in opposite directions, and the reasons are worth
keeping. Prefill measured 0.5 tok/s because the runtime reloaded each package per dispatch — a bug
here, not a property of the port. Decode measured 0.38 tok/s because the one-off phase switch was
being divided by the token count — a reporting bug. Steady state is what the numbers above say.

**Beyond 12K of context** the plan still tracks the reference at absolute position 12,288 — where the
full-attention caches are half full and every sliding ring has wrapped repeatedly: top-1 agreement
100 %, perplexity 1.016 against the reference's 1.029, KLD 0.156 against llama.cpp bf16 (a different
implementation, so absolute log-probabilities differ in scale; the argmax does not).

State is **fp16 and 999.0 MiB in total** — 36 full-attention caches of 24.00 MiB and 27 sliding caches
of 5.00 MiB. Peak RSS at 24K is 2.7 GiB; load takes 45–80 s.

## Can you actually talk to it?

Yes. `scripts/dory_chat.py` drives the model's own chat template with sampling and incremental
turns; a six-turn conversation on the LUT6 set at 24K went:

| turn | reply |
|---|---|
| *Hi! Who are you?* | "Hello! I am a large language model, trained by researchers from NVIDIA. I can help with a wide variety of tasks such as answering questions, writing, coding, translating languages…" |
| *What is the capital of France?* | "The capital of France is Paris." |
| *And what river runs through it?* | "The Seine River runs through Paris." |
| *Write a two-line poem about rain.* | "Grey clouds drift low, / wiping the world clean and slow." |
| *What is 17 times 4?* | "17 times 4 is 68." |
| *Summarise our conversation.* | "We discussed my identity as an NVIDIA language model, confirmed that Paris is the capital of France and is crossed by the Seine River, shared a two-line poem about rain, and calculated that 17 times 4 equals 68." |

That last turn is the one that matters: it carries every earlier fact, so the conversation state is
real and not an illusion of one prompt. Longer form holds up too — a 191-token story about a
lighthouse keeper came out structured and readable, and a list-versus-tuple answer was organised
correctly. Turns take 2.4–7 s (3–9 tok/s) once the first is warm.

Two settings matter for usability:

* **Sample, do not take the argmax.** Greedy decoding on a model this size degenerates into loops
  that make a working model look broken.
* **`enable_thinking=False` for chat.** With thinking on, the model reasons correctly — it derived
  80 km/h from 60 km in 45 minutes, with a self-check — but it wrote 320 tokens without closing the
  `</think>` block, and on the following turn it invented a system prompt it had never been given and
  argued with it. The template defaults to thinking, so a chat UI has to turn it off or budget for it.

Known rough edges, from the same transcripts: a list-versus-tuple answer claimed tuples are more
compact "because it stores a single reference to each element", which is not a real distinction, and
the lighthouse story has one small continuity slip between the sinking hold and the empty hold.

## The four shape rules that decide whether a program loads

Every one of these was found by converting a variant and loading it on the ANE, never by reasoning
about the model. They are the real content of this port.

1. **A stateful program may not carry a second convolution chain that joins its output.** Gated
   attention is exactly that shape, so the gate must arrive as an *input leaf* or be applied by a
   program with no state. Adding instead of multiplying fails identically — the rule is about two
   live chains meeting, not about the elementwise op.
2. **~34 MiB of state is the resident budget of a program that does real work.** 24.00 MiB loads,
   45.0 MiB fails; the *count* of buffers is not a limit, and weights do not compete with state.
3. **A stateful program may not materialise a 4096-wide tensor anywhere** — not at its output and not
   transiently. Slicing one into four outputs after building it still fails; scoring one KV head per
   pass and never assembling the four loads.
4. **Gates may arrive as inputs and leave as second outputs.** A convolution branch that ends at an
   output is fine; a branch that stays live across the state path and joins it is not.

## The plan (24,576 tokens, 102 programs)

| program | count | state each | role |
|---|---|---|---|
| `attn_full_<k>_h<h>` | 36 | 24.00 MiB | one KV head of a full-attention layer |
| `epilogue_<k>` | 9 | — | head concatenation, gate, out_norm, o_proj, residual, MLP |
| `sliding_attn_L<b>` | 27 | 5.00 MiB | one sliding layer's attention, four 1024-wide outputs |
| `sepilogue_L<b>` | 27 | — | gate, layer tail, MLP, and the next layer's gate |
| `embeddings`, `entry_gate`, `lm_head` | 3 | — | boundaries and vocabulary |

One `MLState` per stated program lives for the whole run, and the gate chain is threaded from
`meta.yaml` rather than hard-coded. The LM head cannot be a single convolution — a 131,072-wide
output exceeds the ANE's 65,536-element axis limit — so the vocabulary is split and the runtime
rejoins the slices in order.

## Impossible on this toolchain

- **An int8 KV cache.** Core ML states are fp16 only: `State only support fp16 dtype`. int8, uint8
  and int16 are rejected as inputs too, so there is no integer carrier to pack into. The cache
  cannot be halved here, and that is what caps throughput.
- **The aggregated multi-layer module.** Every such program returns error −14 on this hardware, which
  is why the 102-program plan exists at all.

## Reproducing

```bash
python -m anemll.ane_converter.dory_converter \
  --model <dory-bf16-dir> --output out-24k --context-length 24576 --batch-size 64 \
  --per-head-programs --heads-per-state 1 --slots-per-sliding-carrier 1 \
  --rope-tables-as-inputs --lut-blocks none          # fp16; use 6 for 4-bit

python -m anemll.runtime.dory_ane_runtime run \
  --model-dir out-24k --prompt "The capital of France is" --max-tokens 24 --loops 1
```

A conversion without palettization takes about seven minutes; with LUT6 it takes hours, so measure
first and compress last.
