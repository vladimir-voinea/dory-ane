#!/usr/bin/env python3
"""OpenAI-shaped HTTP server for Dory on the ANE.

Endpoints:
  GET  /health              engine and session state
  GET  /v1/models           the one model this process serves
  POST /v1/chat/completions chat, streaming or not, with a real chat template
  POST /v1/completions      raw completion
  POST /v1/sessions         create or reset a session explicitly
  DELETE /v1/sessions/<id>  drop a session and free its caches

Sessions are the prefix cache: a request carrying the same session_id (or the same
leading tokens) reuses the caches and feeds only the delta, so history is never
re-prefilled.  Inference is serialized behind one lock because the Neural Engine
serves one model set at a time.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anemll.runtime.dory_engine import DoryEngine  # noqa: E402
from anemll.runtime.dory_sampling import SamplingParams  # noqa: E402


UI_PATH = Path(__file__).resolve().parent / "dory_ui.html"
LOG_REQUESTS = bool(os.environ.get("DORY_LOG_REQUESTS"))
HEARTBEAT_SECONDS = float(os.environ.get("DORY_HEARTBEAT") or 5.0)


def normalise_messages(messages) -> list:
    """Accept the OpenAI content forms a real client sends.

    Clients differ: some send content as a string, some as a list of parts
    ({"type": "text", "text": ...}), some include image parts this model cannot
    read.  Flattening here keeps the template happy and lets a UI talk to it.
    """
    out = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if isinstance(content, list):
            pieces = []
            for part in content:
                if isinstance(part, str):
                    pieces.append(part)
                elif isinstance(part, dict):
                    if part.get("text"):
                        pieces.append(str(part["text"]))
                    elif part.get("type") not in ("text", None):
                        pieces.append(f"[{part.get('type')} omitted]")
            content = "".join(pieces)
        out.append({"role": role, "content": content if content is not None else ""})
    return out


def params_from(body: Dict[str, Any]) -> SamplingParams:
    """Accept OpenAI spellings and this engine's extras."""
    stop = body.get("stop") or []
    if isinstance(stop, str):
        stop = [stop]
    return SamplingParams(
        max_tokens=int(body.get("max_tokens") or body.get("max_completion_tokens") or 256),
        temperature=float(body.get("temperature", 0.0)),
        top_p=float(body.get("top_p", 1.0)),
        top_k=int(body.get("top_k", 0) or 0),
        min_p=float(body.get("min_p", 0.0) or 0.0),
        repetition_penalty=float(body.get("repetition_penalty", 1.0) or 1.0),
        frequency_penalty=float(body.get("frequency_penalty", 0.0) or 0.0),
        presence_penalty=float(body.get("presence_penalty", 0.0) or 0.0),
        stop=list(stop),
        seed=body.get("seed"),
        ignore_eos=bool(body.get("ignore_eos", False)),
    )


class Handler(BaseHTTPRequestHandler):
    engine: DoryEngine
    model_name: str
    protocol_version = "HTTP/1.1"
    _write_lock = threading.Lock()

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    # -- plumbing ------------------------------------------------------
    def _json(self, code: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _raw_chunk(self, data: bytes) -> None:
        """Write one HTTP chunk (the body is Transfer-Encoding: chunked)."""
        with self._write_lock:
            self.wfile.write(("%X\r\n" % len(data)).encode() + data + b"\r\n")
            self.wfile.flush()

    def _raw_chunk_end(self) -> None:
        """The zero-length chunk that ends a chunked body."""
        with self._write_lock:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    def _heartbeat(self, stop) -> None:
        """Keep the body alive while a prompt is being read.

        Sending the headers first and then going quiet is what a long prompt looks
        like from the client's side: pi and Node's fetch both aborted a request whose
        body had been silent for minutes while the prompt was still being fed.  An SSE
        comment every few seconds is legal, ignored by parsers, and keeps the
        connection from timing out.
        """
        while not stop.wait(HEARTBEAT_SECONDS):
            try:
                self._raw_chunk(b": reading prompt\n\n")
            except Exception:  # noqa: BLE001 - the client is gone; the loop will notice
                return

    def _sse(self, payload: Any) -> None:
        """Send one server-sent event as a proper HTTP chunk.

        Framing this by hand and terminating with a zero-length chunk matters: a body
        with no Content-Length and no chunked framing ends only when the socket closes,
        and clients that wait for the body to end rather than parsing our [DONE] sat on
        a finished reply for eight minutes (pi) or needed the tab reloaded (browser).
        """
        text = payload if isinstance(payload, str) else json.dumps(payload)
        self._raw_chunk(("data: " + text + "\n\n").encode())

    def _read_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        body = json.loads(raw or b"{}")
        if LOG_REQUESTS:
            short = {k: v for k, v in body.items() if k != "messages"}
            short["n_messages"] = len(body.get("messages") or [])
            short["n_tools"] = len(body.get("tools") or [])
            sys.stderr.write("request: " + json.dumps(short)[:600] + "\n")
        return body

    # -- routes --------------------------------------------------------
    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, DELETE, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        route = self.path.split("?", 1)[0]        # clients append cache-busters
        if route in ("/", "/index.html", "/chat"):
            body = UI_PATH.read_bytes() if UI_PATH.exists() else b"chat UI missing"
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/health"):
            self._json(200, {
                "status": "ok",
                "model": self.model_name,
                "sessions": [{"id": s.id, "tokens": s.length, "turns": s.turns}
                             for s in self.engine.sessions.values()],
                "max_sessions": self.engine.max_sessions,
                "context_length": self.engine.runtime.plan.context_length,
                "loops": self.engine.runtime.loops,
            })
            return
        if self.path.startswith("/v1/models"):
            self._json(200, {"object": "list", "data": [{"id": self.model_name,
                                                         "object": "model"}]})
            return
        self._json(404, {"error": {"message": "no such path"}})

    def do_DELETE(self) -> None:
        if self.path.startswith("/v1/sessions/"):
            sid = self.path.rsplit("/", 1)[-1]
            self.engine.sessions.pop(sid, None)
            self._json(200, {"deleted": sid})
            return
        self._json(404, {"error": {"message": "no such path"}})

    def do_POST(self) -> None:
        if self.path.startswith("/v1/sessions"):
            body = self._read_body()
            session = self.engine.new_session(body.get("session_id"))
            self._json(200, {"id": session.id, "tokens": session.length})
            return
        if self.path.startswith("/v1/chat/completions"):
            self._chat(self._read_body())
            return
        if self.path.startswith("/v1/completions"):
            self._completion(self._read_body())
            return
        self._json(404, {"error": {"message": "no such path"}})
    # -- inference -----------------------------------------------------
    def _stream(self, events, *, chat: bool, body: Dict[str, Any]) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        # Both signals, deliberately: clients disagree about which one ends the reply.
        # A chunked body has an explicit terminator, and closing the socket after it
        # also ends the body for parsers that only watch for EOF (pi did).
        self.send_header("Connection", "close")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        reply_id = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        last_text, last_reasoning = "", ""
        self._raw_chunk(b": connected\n\n")
        stop = threading.Event()
        beat = threading.Thread(target=self._heartbeat, args=(stop,), daemon=True)
        beat.start()
        try:
            for event in events:
                if event.finished:
                    # A tool call arrives as one complete delta: the model writes it as
                    # a single <tool_call> block, so there is nothing to stream
                    # piecemeal, and clients that reassemble fragments handle one.
                    for index, call in enumerate(getattr(event, "tool_calls", []) or []):
                        delta = {"tool_calls": [
                            {"index": index, "id": call["id"], "type": "function",
                             "function": {"name": call["function"]["name"],
                                          "arguments": call["function"]["arguments"]}}]}
                        if index == 0:
                            # OpenAI opens the assistant turn in the same chunk
                            delta = {"role": "assistant", "content": None, **delta}
                        self._sse({
                            "id": reply_id, "object": "chat.completion.chunk",
                            "created": created, "model": self.model_name,
                            "choices": [{"index": 0, "delta": delta,
                                         "finish_reason": None}],
                        })
                    stats = event.stats or {}
                    usage = {"prompt_tokens": stats.get("prompt_tokens", 0),
                             "completion_tokens": stats.get("completion_tokens", 0),
                             "total_tokens": stats.get("prompt_tokens", 0)
                                           + stats.get("completion_tokens", 0)}
                    self._sse({
                        "id": reply_id, "object": "chat.completion.chunk",
                        "created": created, "model": self.model_name,
                        "choices": [{"index": 0, "delta": {},
                                     "finish_reason": event.finish_reason}],
                        "stats": event.stats,
                    })
                    # OpenAI sends usage in its own final chunk with an EMPTY choices
                    # array when stream_options.include_usage is set, and a client that
                    # asked for it can wait for exactly that (pi did: it took the tool
                    # call, then sat on a finished body).
                    wants_usage = bool((body.get("stream_options") or {}).get("include_usage"))
                    if wants_usage:
                        self._sse({"id": reply_id, "object": "chat.completion.chunk",
                                   "created": created, "model": self.model_name,
                                   "choices": [], "usage": usage})
                    break
                delta: Dict[str, Any] = {}
                if event.text and event.text != last_text:
                    delta["content"] = event.text[len(last_text):]
                    last_text = event.text
                if event.reasoning and event.reasoning != last_reasoning:
                    delta["reasoning_content"] = event.reasoning[len(last_reasoning):]
                    last_reasoning = event.reasoning
                if not delta:
                    continue
                self._sse({"id": reply_id, "object": "chat.completion.chunk",
                           "created": created, "model": self.model_name,
                           "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
            self._sse("[DONE]")
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            # Stop the heartbeat BEFORE the terminating chunk: a beat that lands after
            # the zero-length chunk writes into a finished body and leaves clients
            # waiting forever (this is what kept pi hanging after a good reply).
            stop.set()
            beat.join(timeout=2.0)
            try:
                self._raw_chunk_end()
            except Exception:  # noqa: BLE001 - the client is gone
                pass
            self.close_connection = True
            # A client can leave mid-stream.  Close the generator first: it holds the
            # engine lock for the length of the turn, so a suspended generator makes
            # every later request wait on a turn whose client is gone (measured: the
            # next request sat for 51 s on a turn nobody was reading).
            close = getattr(events, "close", None)
            if close is not None:
                try:
                    close()
                except (BrokenPipeError, ConnectionResetError, ValueError):
                    pass
            # and hand the caches back to the session instead of stranding them
            self.engine.reclaim()

    def _collect(self, events):
        text = reasoning = ""
        stats = None
        finish = "length"
        calls = []
        for event in events:
            if event.finished:
                stats, finish = event.stats, event.finish_reason
                calls = getattr(event, "tool_calls", []) or []
                # the final event carries the authoritative text: with a tool call,
                # the raw <tool_call> block is parsed OUT of it, not left in
                text = getattr(event, "content", text)
            else:
                text = event.text or text
                reasoning = event.reasoning or reasoning
        return text, reasoning, stats, finish, calls

    def _chat(self, body: Dict[str, Any]) -> None:
        messages = normalise_messages(body.get("messages") or [])
        if not messages:
            self._json(400, {"error": {"message": "messages is required"}})
            return
        params = params_from(body)
        stream = bool(body.get("stream", False))
        session_id = body.get("session_id")
        think = bool(body.get("enable_thinking", False))
        tools = body.get("tools")
        events = self.engine.chat(messages, params, session_id=session_id,
                                  enable_thinking=think, tools=tools)
        if stream:
            self._stream(events, chat=True, body=body)
            return
        text, reasoning, stats, finish, calls = self._collect(events)
        clean = text.split("<|im_end|>")[0]
        message: Dict[str, Any] = {"role": "assistant", "content": clean}
        if calls:
            message["tool_calls"] = calls
            if not clean:
                message["content"] = None
        if reasoning:
            message["reasoning_content"] = reasoning
        self._json(200, {
            "id": "chatcmpl-" + uuid.uuid4().hex[:24],
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.model_name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": (stats or {}).get("prompt_tokens", 0),
                      "completion_tokens": (stats or {}).get("completion_tokens", 0),
                      "total_tokens": (stats or {}).get("prompt_tokens", 0)
                                      + (stats or {}).get("completion_tokens", 0)},
            "session_id": (stats or {}).get("session_id"),
            "stats": stats,
        })

    def _completion(self, body: Dict[str, Any]) -> None:
        prompt = body.get("prompt") or ""
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        params = params_from(body)
        ids = self.engine.tokenizer(prompt, return_tensors=None)["input_ids"]
        events = self.engine.generate(ids, params, session_id=body.get("session_id"))
        if body.get("stream"):
            self._stream(events, chat=False, body=body)
            return
        text, _, stats, finish, _calls = self._collect(events)
        self._json(200, {
            "id": "cmpl-" + uuid.uuid4().hex[:24],
            "object": "text_completion",
            "created": int(time.time()), "model": self.model_name,
            "choices": [{"index": 0, "text": text.split("<|im_end|>")[0],
                         "finish_reason": finish}],
            "usage": {"prompt_tokens": (stats or {}).get("prompt_tokens", 0),
                      "completion_tokens": (stats or {}).get("completion_tokens", 0),
                      "total_tokens": (stats or {}).get("prompt_tokens", 0)
                                      + (stats or {}).get("completion_tokens", 0)},
            "stats": stats,
        })


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8008)
    p.add_argument("--loops", type=int, default=1)
    p.add_argument("--max-sessions", type=int, default=2,
                   help="how many conversations keep their caches (999 MiB each)")
    p.add_argument("--model-name", default="dory-1.0-4b-ane")
    p.add_argument("--prefill-batch", type=int, default=1 << 30,
                   help="feed prompts of this many tokens through the batch prefill "
                        "packages; that switches the resident model set (100-270 s), so "
                        "the default never does it and feeds token by token instead")
    args = p.parse_args(argv)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    engine = DoryEngine(args.model_dir, loops=args.loops,
                        max_sessions=args.max_sessions, tokenizer=tokenizer,
                        prefill_threshold=args.prefill_batch)
    print("loading ...", flush=True)
    print("loaded in %.1f s" % engine.load(), flush=True)

    Handler.engine = engine
    Handler.model_name = args.model_name
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print("serving %s on http://%s:%d" % (args.model_name, args.host, args.port), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())