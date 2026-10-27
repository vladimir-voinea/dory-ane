# Serving Dory on the ANE

An OpenAI-shaped HTTP server with the behaviours a normal engine has: a prefix cache,
sampling, multi-turn chat with the real template (including reasoning), and streaming.

```bash
python scripts/dory_serve.py --model-dir dane5-artifacts/set24k --port 8008 \
    --loops 1 --max-sessions 2
```

Startup takes about 2 minutes (21 s to load, then the infer packages are opened so the
first token is not charged for it).

## Endpoints

| method | path | what it does |
|---|---|---|
| GET | `/health` | engine state, live sessions and their token counts |
| GET | `/v1/models` | the one model this process serves |
| POST | `/v1/chat/completions` | chat, `stream: true` for SSE |
| POST | `/v1/completions` | raw completion |
| POST | `/v1/sessions` | create a session explicitly |
| DELETE | `/v1/sessions/<id>` | drop a session and free its caches |

Request fields beyond the OpenAI ones: `session_id`, `enable_thinking`, `top_k`, `min_p`,
`repetition_penalty`. Responses carry `session_id` and a `stats` block with
`prefix_reused`, `prefill_seconds` and `decode_tokens_per_second`, so the cache is
observable rather than a claim.

## The prefix cache, precisely

Each session owns its KV caches (`999 MiB` of Neural Engine state). A turn feeds only the
tokens past what the session already holds, so conversation history is never
re-prefilled. Measured on a three-turn conversation:

| turn | prompt tokens | reused | new tokens fed | decode |
|---|---|---|---|---|
| 1 | 23 | 0 | 23 | 8.58 tok/s |
| 2 | 49 | 30 | 19 | 7.15 tok/s |
| 3 | 81 | 62 | 19 | 7.42 tok/s |

A request naming no session adopts the longest cached prefix it extends. Two limits are
honest consequences of the hardware: **a cache cannot be truncated or copied**, so a
prompt that shares a prefix but diverges midway starts a fresh session rather than
borrowing it; and **each live session costs 999 MiB of ANE state**, so the cache holds a
couple of conversations, not hundreds. `--max-sessions` sets it, LRU evicts.

## Reasoning

`enable_thinking: true` uses the template's thinking mode. When the model closes its
`</think>` block the trace lands in `reasoning_content` and the answer in `content`:

```json
{"choices":[{"message":{"role":"assistant",
  "reasoning_content":"...1288 chars of working...","content":"144"}}]}
```

If the token budget runs out before the block closes, the trace stays in `content`, which
is what the model actually produced. Budget generously — a short arithmetic question spent
413 tokens thinking, and the template defaults to thinking on, so set
`enable_thinking: false` for ordinary chat.

## Streaming

`stream: true` sends `text/event-stream` chunks in the OpenAI shape and `[DONE]` at the
end. A client that disconnects mid-stream is handled: the server hands the caches back to
the session instead of stranding them in the runtime (verified by abandoning a stream with
`head -c 200` and finding the session still listed and reusable).

## Tool calling

An agent gets what it gets from any other engine. `tools` in the request are rendered by the
model's own template, and the `<tool_call><function=NAME><parameter=KEY>VALUE</parameter>`
blocks it answers with are parsed into OpenAI `tool_calls` (arguments as a JSON string, string
values preserved so the tool decides its own types). Text before the first block stays as
`content`; `finish_reason` becomes `tool_calls`.

A real loop, measured end to end on the ANE:

```
turn 1  29.8 s  finish_reason=tool_calls  {"city": "Paris"}   (content: null)
        the client runs get_weather and sends role:"tool" back
turn 2  32.6 s  finish_reason=stop
        "The weather in Paris right now is light rain with a temperature of 18°C."
```

Streaming sends the call as one `delta.tool_calls` chunk followed by a chunk with
`finish_reason: "tool_calls"`, which is what agent clients reassemble from.

**Why a tool turn costs ~30 s rather than ~3 s.** The prefix cache matches tokens, and a tool
turn does not round-trip verbatim: the model emits the call as text, the client sends it back as
a structured `tool_calls` message, and the template re-renders it with its own whitespace. The
cached tokens and the new prompt therefore agree only up to the assistant turn —
`common_prefix: 271` of a 296-token prompt in the measurement above — and a cache cannot be
truncated back to that boundary. Plain chat, where the client echoes the reply exactly, does
reuse: 30 of 49 tokens on turn two and 62 of 81 on turn three.

**Never switch phases on a request.** Batch prefill runs at 130 tok/s but needs the prefill
packages, and the infer packages must be dropped first (both halves resident fails with error
−14; verified again at 204 models). That switch costs 100–270 s, so the server feeds prompts
token by token through the decode path at ~10 tok/s instead — a 300-token prompt costs ~30 s and
no switch. `--prefill-batch N` opts back in for a single long prompt.

## What is deliberately not here

* **Continuous batching.** The ANE serves one model set at a time and every program is a
  serial dispatch; requests are serialized behind a lock. Batch inference would need
  separate programs, not a scheduler.
* **Speculative decoding, prefix sharing across divergent prompts, paged attention.**
  All three need a mutable, truncatable block table, and an `MLState` is neither copyable
  nor truncatable.
* **Multiple models.** One process holds one model set; 999 MiB of state is already the
  budget for a couple of conversations.