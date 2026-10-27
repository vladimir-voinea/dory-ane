from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from anemll.runtime.dory_ane_runtime import DoryProgramRuntime
from anemll.runtime.dory_sampling import SamplingParams, Sampler
from anemll.runtime.dory_tools import normalise_tools, parse_tool_calls


@dataclass
class Session:
    """One conversation's token history and its private KV caches."""

    id: str
    tokens: List[int] = field(default_factory=list)
    states: Dict[str, object] = field(default_factory=dict)
    created: float = field(default_factory=time.time)
    used: float = field(default_factory=time.time)
    turns: int = 0

    @property
    def length(self) -> int:
        return len(self.tokens)


@dataclass
class GenerationEvent:
    token_id: Optional[int] = None
    text: str = ""
    reasoning: str = ""
    finished: bool = False
    finish_reason: Optional[str] = None
    stats: Optional[dict] = None
    content: str = ""
    tool_calls: List[dict] = field(default_factory=list)
class DoryEngine:
    """Sessions + prefix reuse + sampling in front of one DoryProgramRuntime.

    A cache here cannot be truncated or copied, so a session is reused only by
    extending it: a prompt that shares a prefix but diverges midway gets a fresh
    session.  Every live session costs 999 MiB of Neural Engine state, so the
    cache stays small -- see max_sessions.
    """

    def __init__(self, model_dir: str, *, loops: int = 1, max_sessions: int = 2,
                 tokenizer=None, verbose: bool = False, prefill_threshold: int = 1 << 30):
        self.runtime = DoryProgramRuntime(model_dir, loops=loops, verbose=verbose)
        self.tokenizer = tokenizer
        self.max_sessions = max(1, int(max_sessions))
        # Packing a batch would switch the resident model set (opening the prefill
        # packages and dropping the infer ones, 100-270 s here, and both halves cannot
        # be resident at once).  An agent re-sends a growing conversation every turn, so
        # a switch per turn is unusable; feeding token by token through the infer path
        # costs ~10 tok/s and no switch.  Set this high to keep that default, or low
        # (e.g. 256) to prefer batch prefill for one long prompt.
        self.prefill_threshold = int(prefill_threshold)
        self.sessions: Dict[str, Session] = {}
        self._active_session: Optional[Session] = None
        self.last_common_prefix: int = 0
        self._lock = threading.RLock()
        self._loaded = False

    def load(self) -> float:
        with self._lock:
            if self._loaded:
                return self.runtime.load_seconds
            seconds = self.runtime.load()
            self.runtime.prepare_decode()
            self._loaded = True
            return seconds

    def reclaim(self) -> None:
        """Take the caches back into the session that was mid-turn.

        A caller that walks away mid-stream (a closed curl, a dropped browser tab)
        abandons the generator, so the code after the loop never runs and the runtime
        is left holding a session's caches.  The server calls this in a finally; a
        session that got reclaimed simply reports turns=0 and can be reused normally.
        """
        with self._lock:
            session = self._active_session
            if session is None:
                return
            if self.runtime._states:
                session.states = self.runtime.take_states()
            session.used = time.time()
            self._active_session = None

    def _evict_if_needed(self) -> None:
        while len(self.sessions) >= self.max_sessions:
            victim = min(self.sessions.values(), key=lambda s: s.used)
            del self.sessions[victim.id]

    def new_session(self, session_id: Optional[str] = None) -> Session:
        with self._lock:
            self._evict_if_needed()
            sid = session_id or hashlib.sha1(
                ("%s%d" % (time.time(), len(self.sessions))).encode()).hexdigest()[:16]
            session = Session(id=sid, states=self.runtime.new_states())
            self.sessions[sid] = session
            return session

    def _adopt(self, tokens: Sequence[int]) -> Tuple[Session, int]:
        best: Optional[Session] = None
        wanted = list(tokens)
        for session in self.sessions.values():
            if session.length and session.length <= len(wanted):
                if session.tokens == wanted[: session.length]:
                    if best is None or session.length > best.length:
                        best = session
        if best is None:
            return self.new_session(), 0
        best.used = time.time()
        return best, best.length
    def generate(self, prompt_tokens: Sequence[int], params: SamplingParams, *,
                 session_id: Optional[str] = None) -> Iterator[GenerationEvent]:
        with self._lock:
            self.load()
            tokens = [int(t) for t in prompt_tokens]
            if not tokens:
                raise ValueError("the prompt is empty")
            if session_id is not None:
                session = self.sessions.get(session_id) or self.new_session(session_id)
                common = 0
                for old, new in zip(session.tokens, tokens):
                    if old != new:
                        break
                    common += 1
                reused = session.length if session.length and common >= session.length else 0
                if reused == 0:
                    self.last_common_prefix = common
                    session.tokens = []
                    session.states = self.runtime.new_states()
            else:
                session, reused = self._adopt(tokens)

            self.runtime.install_states(session.states)
            self._active_session = session
            took = time.perf_counter()
            delta = tokens[reused:]
            if not delta:
                raise ValueError("the prompt adds nothing to the cached session")
            logits = self.runtime.feed(delta, reused, prefill_threshold=self.prefill_threshold)
            prefill_seconds = time.perf_counter() - took
            session.tokens = list(tokens)
            # The caches belong to the session, not to the runtime, so they have to go
            # back even when the caller walks away mid-stream (a closed curl, a browser
            # tab).  Without this the runtime keeps them and the session is left holding
            # a stale dict -- which shows up as a session with turns=0 and a prefix that
            # cannot be reused.  Measured exactly that in the first server test.

            sampler = Sampler(params)
            produced: List[int] = []
            started = time.perf_counter()
            handed_back = False
            finish = "length"
            in_reasoning = False
            text = ""
            reasoning = ""

            for _ in range(params.max_tokens):
                position = len(session.tokens)
                if position >= self.runtime.plan.context_length:
                    finish = "length"
                    break
                token = sampler.next_token(logits[-1], session.tokens)
                if token in params.stop_token_ids:
                    finish = "stop"
                    break
                if not params.ignore_eos and self._is_eos(token):
                    finish = "stop"
                    break
                piece = self._decode([token])
                session.tokens.append(token)
                produced.append(token)

                if "</think>" in piece:
                    in_reasoning = False
                    whole = self._decode(produced)
                    head, text = whole.split("</think>", 1)
                    reasoning = head.replace(" thinking", "")
                    yield GenerationEvent(text=text, reasoning=reasoning)
                elif piece.strip() == " thinking":
                    in_reasoning = True
                    yield GenerationEvent()
                elif in_reasoning:
                    reasoning += piece
                    yield GenerationEvent(reasoning=reasoning)
                else:
                    text += piece
                    yield GenerationEvent(token_id=token, text=text)

                if any(s and text.endswith(s) for s in params.stop):
                    finish = "stop"
                    break
                logits = np.stack([self.runtime.decode(token, position)])

            session.states = self.runtime.take_states()
            handed_back = True
            self._active_session = None
            session.used = time.time()
            session.turns += 1
            elapsed = time.perf_counter() - started
            yield GenerationEvent(
                finished=True, finish_reason=finish,
                stats={"prompt_tokens": len(tokens), "prefix_reused": reused,
                       "common_prefix": self.last_common_prefix,
                       "completion_tokens": len(produced),
                       "prefill_seconds": round(prefill_seconds, 3),
                       "decode_seconds": round(elapsed, 3),
                       "decode_tokens_per_second":
                           round(len(produced) / elapsed, 2) if elapsed else None,
                       "session_id": session.id})

    def _is_eos(self, token: int) -> bool:
        if self.tokenizer is None:
            return False
        return token == getattr(self.tokenizer, "eos_token_id", None)

    def _decode(self, ids: Sequence[int]) -> str:
        if self.tokenizer is None:
            return ""
        return self.tokenizer.decode(list(ids), skip_special_tokens=False)

    def chat(self, messages: Sequence[dict], params: SamplingParams, *,
             session_id: Optional[str] = None, enable_thinking: bool = False,
             tools: Optional[list] = None) -> Iterator[GenerationEvent]:
        """Chat through the model's own template, with tool calls parsed out.

        The template renders the tool definitions and asks the model to answer with
        <tool_call> blocks.  Those are turned into the OpenAI tool_calls shape here, so
        a client sees what it would see from any other engine -- and the text before
        the first call is kept as content, because the model often reasons first.
        """
        if self.tokenizer is None:
            raise RuntimeError("chat() needs a tokenizer")
        prepared = normalise_tools(tools)
        kwargs = {"tokenize": False, "add_generation_prompt": True,
                  "enable_thinking": enable_thinking}
        if prepared:
            kwargs["tools"] = prepared
        prompt = self.tokenizer.apply_chat_template(list(messages), **kwargs)
        ids = self.tokenizer(prompt, return_tensors=None)["input_ids"]

        last_text = ""
        final: Optional[GenerationEvent] = None
        for event in self.generate(ids, params, session_id=session_id):
            if event.finished:
                final = event
                continue
            if event.text:
                last_text = event.text
            yield event
        if final is None:
            return
        content, calls = parse_tool_calls(last_text)
        final.content = content
        final.tool_calls = calls
        if calls:
            final.finish_reason = "tool_calls"
        yield final
