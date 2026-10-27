#!/usr/bin/env python3
"""The acceptance checks for the Dory server, run against a live server.

    python -m anemll.openai_server --model-dir <converted-dir> --port 8000 &
    python scripts/dory_server_smoke.py --base-url http://127.0.0.1:8000/v1

This is the user-facing seam from the spec's Testing Decisions: an OpenAI
client and nothing lower.  Each check prints PASS, FAIL or WARN with the value
it saw, and the exit code is 0 only when every check passed.

1. `GET /v1/models`: context length, loop range, default loop count, and one
   alias per loop count.
2. a streamed reply with `reasoning_content` and `content` separated.
3. a request with `tools` returning a well-formed tool call.
4. the loop count set by the model alias and by the `dory_loops` body field,
   with the choice visible in the response metadata.
5. the server answering on loopback only.

It needs the `openai` package (`pip install openai`), which the port itself
does not.
"""

from __future__ import annotations

import argparse
import json
import socket

try:
    import openai
except ImportError:                                   # pragma: no cover
    raise SystemExit("this script needs the openai package: pip install openai")

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {"type": "object",
                       "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    },
}]

TOOL_PROMPT = "What is the weather in Santa Clara right now? Use the tool."

results = []


def check(name, ok, detail):
    results.append({"check": name, "ok": bool(ok), "detail": detail})
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")


def warn(name, detail):
    results.append({"check": name, "ok": True, "warning": True, "detail": detail})
    print(f"WARN  {name}: {detail}")


def dory_block(response):
    """The port's metadata block, which OpenAI's schema does not define."""
    extra = getattr(response, "model_extra", None) or {}
    return extra.get("dory") or getattr(response, "dory", None) or {}


def check_models(client, model, expect_context):
    models = list(client.models.list())
    ids = [entry.id for entry in models]
    by_id = {entry.id: entry for entry in models}
    base = by_id.get(model)
    if base is None:
        check("models", False, f"{model} is not listed; the ids are {ids}")
        return
    context = getattr(base, "context_length", None) or getattr(base, "max_model_len", None)
    aliases = {f"{model}-{n}loop" if n == 1 else f"{model}-{n}loops": n for n in (1, 2, 4)}
    missing = [alias for alias in aliases if alias not in by_id]
    check("models.context_length", context == expect_context,
          f"context_length={context} (expected {expect_context})")
    check("models.aliases", not missing,
          f"{len(ids)} ids: {', '.join(ids)}" + (f"; missing {missing}" if missing else ""))
    loops = dory_block(base)
    check("models.loops",
          loops.get("loop_range") == [1, 4] and loops.get("default_loops") == 2,
          f"loop_range={loops.get('loop_range')} default_loops={loops.get('default_loops')}")


def check_stream(client, model, prompt):
    stream = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], stream=True)
    reasoning, content, mixed, dory = [], [], 0, {}
    for chunk in stream:
        delta = chunk.choices[0].delta
        reasoning_piece = getattr(delta, "reasoning_content", None)
        if reasoning_piece:
            reasoning.append(reasoning_piece)
        if delta.content:
            content.append(delta.content)
        if reasoning_piece and delta.content:
            mixed += 1
        if dory_block(chunk):
            dory = dory_block(chunk)
    reasoning_text = "".join(reasoning)
    content_text = "".join(content)
    check("stream.reasoning", bool(reasoning_text.strip()),
          f"{len(reasoning_text)} characters of reasoning")
    check("stream.content", bool(content_text.strip()),
          f"{len(content_text)} characters of content")
    check("stream.separated", bool(reasoning_text.strip()) and mixed == 0,
          f"{mixed} chunks carried both channels")
    check("stream.metadata", dory.get("loops") is not None,
          f"dory={json.dumps(dory, sort_keys=True)}")
    print("      reasoning:", json.dumps(reasoning_text[:120]))
    print("      content  :", json.dumps(content_text[:120]))


def check_tools(client, model):
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": TOOL_PROMPT}], tools=TOOLS)
    message = response.choices[0].message
    calls = message.tool_calls or []
    finish = response.choices[0].finish_reason
    if not calls:
        check("tools.call", False,
              f"no tool call; finish_reason={finish} content={json.dumps(message.content)[:120]}")
        return
    call = calls[0]
    try:
        arguments = json.loads(call.function.arguments)
    except (TypeError, ValueError):
        arguments = None
    check("tools.call", call.function.name == "get_weather" and isinstance(arguments, dict),
          f"{call.function.name}({call.function.arguments}) id={call.id} "
          f"finish_reason={finish}")


def check_loop_choices(client, model, prompt, loops):
    # a different loop count from the previous request, or the server would have
    # nothing to reset
    alias_loops = 1 if loops != 1 else 4
    alias = f"{model}-{alias_loops}loop" if alias_loops == 1 else f"{model}-{alias_loops}loops"
    by_alias = client.chat.completions.create(
        model=alias, messages=[{"role": "user", "content": prompt}], max_tokens=8)
    block = dory_block(by_alias)
    check("loops.alias",
          block.get("loops") == alias_loops and block.get("loops_source") == "alias",
          f"model={alias} dory={json.dumps(block, sort_keys=True)}")
    by_body = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], max_tokens=8,
        extra_body={"dory_loops": loops})
    body = dory_block(by_body)
    check("loops.body", body.get("loops") == loops and body.get("loops_source") == "body",
          f"dory_loops={loops} dory={json.dumps(body, sort_keys=True)}")
    check("loops.cache_reset", body.get("cache_reset") is True,
          f"switching loops reset the cache: cache_epoch={body.get('cache_epoch')}")


def check_bind(base_url, model):
    host = base_url.split("//", 1)[-1].split("/", 1)[0].rsplit(":", 1)
    name, port = host[0], int(host[1])
    loopback = name in ("127.0.0.1", "localhost", "::1")
    try:
        lan = socket.gethostbyname(socket.gethostname())
    except OSError:                                   # pragma: no cover
        lan = ""
    refused = None
    if lan and not lan.startswith("127."):
        try:
            socket.create_connection((lan, port), timeout=2).close()
            refused = False
        except OSError:
            refused = True
    check("bind.loopback", loopback, f"the client is talking to {name}:{port}")
    if refused is None:
        warn("bind.network", "no non-loopback address to probe from")
    else:
        check("bind.network", refused,
              f"a connection to {lan}:{port} is refused, so only loopback is listening")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="dory-ane")
    parser.add_argument("--prompt", default="What is 2+2? Answer in one sentence.")
    parser.add_argument("--loops", type=int, default=4,
                        help="loop count for the alias/body check")
    parser.add_argument("--expect-context", type=int, default=24576)
    parser.add_argument("--json", action="store_true", help="print the results as JSON")
    args = parser.parse_args(argv)

    client = openai.OpenAI(base_url=args.base_url, api_key="local", max_retries=0)
    try:
        check_models(client, args.model, args.expect_context)
        check_stream(client, args.model, args.prompt)
        check_tools(client, args.model)
        check_loop_choices(client, args.model, args.prompt, args.loops)
        check_bind(args.base_url, args.model)
    except openai.APIError as error:
        check("server", False, f"{type(error).__name__}: {error}")
    finally:
        client.close()

    failures = [result for result in results if not result["ok"]]
    if args.json:
        print(json.dumps(results, indent=2))
    print(f"\n{len(results) - len(failures)}/{len(results)} checks passed")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
