# Dory 1.0 4B on the Apple Neural Engine

**NVIDIA's Dory1.0-4B runs on the Neural Engine of a Mac at a 24,576-token context**,
with a prefix-caching server, sampling, tool calling and streaming — the behaviours you
expect from a normal inference engine, on a laptop NPU.

| | |
|---|---|
| decode, `loops=1` | **11.3 tok/s** (4-bit) · 5.9 tok/s (fp16) |
| prefill (batch) | 130 tok/s |
| context | 24,576 tokens, 999 MiB of ANE state |
| quality | KLD 0.019–0.056 vs the reference model (fp16), top-1 87.5–100 % |
| load | 21 s |

It holds a conversation, calls tools, and answers correctly. Two hardware constraints
shape everything: a Core ML state cannot be copied or truncated (so the cache is per
session and reused by extension), and ~102 packages is the resident ceiling (so the
prefill and decode package sets take turns rather than living together).

## What you need

* a Mac with an Apple Neural Engine, macOS 14 or newer
* Python 3.12
* the converted model: `huggingface-cli download vladimir-voinea/dory-1.0-4b-ane-coreml`

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run it

```bash
# a chat server with an OpenAI-compatible API and a browser UI at http://127.0.0.1:8008/
python scripts/dory_serve.py --model-dir ./dory-1.0-4b-ane-coreml --port 8008

# or talk to it in a terminal
python scripts/dory_chat.py --model-dir ./dory-1.0-4b-ane-coreml \
    --turns "Hi! Who are you?" "What is the capital of France?"
```

Anything that speaks the OpenAI API can use it:

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8008/v1", api_key="no-key")
print(client.chat.completions.create(model="dory-1.0-4b-ane",
      messages=[{"role": "user", "content": "Hello!"}]).choices[0].message.content)
```

### Using it from pi (or any agent harness)

Add a provider to `~/.pi/agent/models.json`:

```json
{"providers": {"dory-ane": {
  "api": "openai-completions",
  "apiKey": "no-key",
  "baseUrl": "http://127.0.0.1:8008/v1",
  "compat": {"maxTokensField": "max_tokens",
             "reasoningContentField": "reasoning_content",
             "supportsDeveloperRole": false,
             "supportsReasoningEffort": false},
  "models": [{"id": "dory-1.0-4b-ane", "name": "Dory 1.0 4B (ANE, 24k)",
              "contextWindow": 24576, "maxTokens": 2048, "input": ["text"]}]}}}
```

```bash
pi --provider dory-ane --model dory-1.0-4b-ane "hello"
```

Plain turns are quick. **Agent turns are slow** — an agent sends a multi-thousand-token
system prompt with its tool definitions on the first request, and prefill on this path
runs at ~10 tok/s, so expect minutes for the first turn of a session and much faster
ones after the prefix cache catches up.

## Converting the model yourself

```bash
HF_HOME=./hf-cache python -m anemll.ane_converter.dory_converter \
  --model /path/to/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16 \
  --output out-24k --context-length 24576 --batch-size 64 \
  --per-head-programs --heads-per-state 1 --slots-per-sliding-carrier 1 \
  --rope-tables-as-inputs --lut-blocks 4
```

Without palettization (`--lut-blocks none`) the conversion takes ~7 minutes; with 4-bit
palettes it takes hours, so measure first and compress last.

## The four rules that decide whether a program loads

Each was found by converting a variant and loading it, never by reasoning about the
model. They are the real content of this port; [docs/dory-port/STATUS.md](docs/dory-port/STATUS.md)
has the measurements.

1. A stateful program may not carry a second convolution chain that joins its output —
   so gated attention needs its gate as an *input leaf*.
2. ~34 MiB of state per working program; 24 MiB loads, 45 MiB fails. The count of
   buffers is not a limit, and weights do not compete with state.
3. A stateful program may not materialise a 4096-wide tensor anywhere, not even
   transiently — so attention scores one KV head per pass.
4. Gates may arrive as inputs and leave as extra outputs; a branch that ends at an
   output is fine, a branch that stays live across the state path is not.

## Layout

```
anemll/ane_converter/   Dory -> Core ML conversion, per-head program planning
anemll/models/          the PyTorch reference, for KLD comparisons
anemll/runtime/         the ANE runtime, engine (sessions + prefix cache), sampling, tools
anemll/eval/            KLD scoring against a reference
scripts/                dory_serve.py, dory_chat.py, scoring tools
docs/dory-port/         measured status and the serving reference
tests/                  unit tests; the runtime ones use a toy plan and need no ANE
```

## Honest limits

* prefill and decode cannot share memory: both package sets resident fails with error −14
* an int8 KV cache is impossible — Core ML states are fp16 only
* no continuous batching, paged attention or speculative decoding: all need a mutable,
  copyable block table
* a tool call does not round-trip verbatim through the chat template, so a tool turn
  refills its prefix instead of reusing it

## Provenance

The model is a conversion and 4-bit palettization of
[nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16);
see its licence. The converter and runtime build on [Anemll](https://github.com/Anemll/Anemll).