"""Talk to a converted Dory model on the Neural Engine.

Spec D9: embeddings -> input -> recurrent x N -> output -> LM head per decode
token, and per 64-token prefill batch, with coremltools predictions carrying one
shared MLState.  Spec D4: the loop count is chosen at run time (1-4) and changing
it invalidates the cache, because older rows keep the *final* loop's K/V.

Three entry points:

    python -m anemll.dory_chat chat  --model-dir DIR [--loops 2] [--prompt TEXT]
    python -m anemll.dory_chat bench --model-dir DIR --context 1024 --loops 1,2,4
    python -m anemll.dory_chat score --model-dir DIR --reference REF --out CAND

The chat loop streams tokens, stops on EOS, and takes commands mid-session:

    /loops 4     change the loop count (resets the KV cache, and says so)
    /reset       drop the conversation and the cache
    /quit        leave

Reasoning is on by default, through Dory's own chat template (enable_thinking),
as spec D9 and the ticket ask.  --no-think turns it off for a run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from anemll.ane_converter.dory_converter import read_meta
from anemll.utils import dory_caches as dc

class ContextOverflowError(ValueError):
    """A prompt (plus the room generation needs) does not fit the context.

    Its own class so the CLI can turn it into one clear line and exit code 2:
    the ticket asks for a clear error, not a crash and not silent truncation.
    """

    def __init__(self, message: str, *, used: int = 0, context: int = 0):
        super().__init__(message)
        self.used = int(used)
        self.context = int(context)


def format_tokens(count: int) -> str:
    return f"{count:,}"


__all__ = ["ContextOverflowError", "DoryRuntime", "ChatTokenizer", "main",
           "build_parser"]


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _coreml():
    # imported where it is used: coremltools takes a second to load, while the
    # pure helpers (masks, sampling, templates) must stay importable without it.
    import coremltools as ct
    return ct


def resolve_model_path(model_dir: Path, file_name: str) -> Path:
    """The file the meta names, or its sibling if only the other form exists."""
    path = Path(model_dir) / file_name
    if path.exists():
        return path
    for suffix in (".mlpackage", ".mlmodelc"):
        alternative = path.with_suffix(suffix)
        if alternative.exists():
            return alternative
    raise FileNotFoundError(f"neither {path.name} nor its .mlpackage/.mlmodelc sibling "
                            f"exists in {model_dir}")


def _check_coreml_cache() -> None:
    """Fail with the real cause when Core ML cannot write its model cache.

    Loading a .mlpackage makes Core ML compile it and cache the result under
    ~/Library/Caches/com.apple.CoreML.  If that directory is not writable (a
    sandboxed shell, a restricted container, a read-only home) Core ML reports
    the unhelpful "Failed to build the model execution plan ... error code -14";
    the cause is the denied write, not the model.  Say so, with the path and the
    fix, instead of letting -14 masquerade as a broken package.
    """
    global _COREML_CACHE_OK
    if _COREML_CACHE_OK is not None:
        return
    cache = Path.home() / "Library" / "Caches" / "com.apple.CoreML"
    try:
        cache.mkdir(parents=True, exist_ok=True)
        probe = cache / ".dory_cache_probe"
        probe.write_text("ok")
        probe.unlink()
        _COREML_CACHE_OK = True
    except Exception as exc:  # noqa: BLE001 - any failure means "not writable"
        raise RuntimeError(
            f"Core ML cache not writable at {cache}: {exc}. Core ML must write its "
            "compiled-model cache there to load a model; when the write is denied it "
            "reports error -14, which looks like a broken package but is a permission "
            "problem. Run in a shell that can write that directory (or grant Full Disk "
            "Access to the process), then retry.") from exc


_COREML_CACHE_OK = None


def load_function(path: Path, function_name: str = None, compute_units=None):
    """Load one package, which holds exactly one function.

    Dory ships one package per function (infer and prefill separately): welded
    multifunction packages are rejected by this macOS build, so there is no
    function to select here.  function_name is kept only so call sites read
    clearly.  A .mlmodelc is the cached compile; a .mlpackage is compiled by
    Core ML on first load.
    """
    ct = _coreml()
    if compute_units is None:
        compute_units = ct.ComputeUnit.CPU_AND_NE
    path = Path(path)
    _check_coreml_cache()
    if path.suffix == ".mlmodelc":
        return ct.models.CompiledMLModel(str(path), compute_units)
    return ct.models.MLModel(str(path), compute_units=compute_units)


class DoryRuntime:
    """The five components, driven the way spec D6 says they are called."""

    #: None unless the package ships the exported RoPE rows (rope_tables_as_inputs).
    #: A class default so a runtime built without __init__ (the tests) still answers.
    rope_tables = None

    def __init__(self, model_dir, compute_units=None, verbose=True):
        self.model_dir = Path(model_dir)
        self.meta = read_meta(self.model_dir / "meta.yaml")
        self.params = self.meta["parameters"]
        self.rope_tables = self._load_rope_tables()
        self.verbose = verbose
        self.compute_units = compute_units
        self.models = {}
        self.load_seconds = 0.0
        started = time.time()
        self._load_models()
        self.load_seconds = time.time() - started
        self._state = None
        self._loops = int(self.meta["loops"]["default"])
        self.reset_state()
        self.predict_seconds = 0.0
        self.predict_calls = 0

    def _load_rope_tables(self):
        """The cos/sin rows the converter exported beside the package, if any.

        The converter writes them from the model's own rotary module, so the
        runtime slices rows and never recomputes the formula: one source of
        truth, and the graph gets exactly the values its constant table held.
        """
        if not self.params.get("rope_tables_as_inputs"):
            return None
        declared = self.params.get("rope_tables") or {}
        name = declared.get("file")
        if not name:
            raise RuntimeError(
                "the meta asks for RoPE tables as inputs but declares no "
                "parameters.rope_tables.file; reconvert with this converter version")
        path = self.model_dir / name
        if not path.is_file():
            raise RuntimeError(
                f"{path} is missing: the package was converted with "
                f"rope_tables_as_inputs and the runtime needs those rows")
        with np.load(path) as data:
            tables = {key: np.asarray(data[key]) for key in declared["keys"]}
        shape = (int(declared["rows"]), int(declared["head_dim"]))
        for key, array in tables.items():
            if array.shape != shape:
                raise RuntimeError(f"{path}: {key} has shape {array.shape}, expected {shape}")
            if array.dtype != np.float16:
                tables[key] = array.astype(np.float16)
        return tables

    def _rope_inputs(self, declared, start: int, q_len: int) -> dict:
        """The rows this batch needs, sliced from the exported tables."""
        if self.rope_tables is None:
            return {}
        begin, end = int(start), int(start) + int(q_len)
        out = {}
        for name in declared:
            if not (name.startswith("cos_") or name.startswith("sin_")):
                continue
            table = self.rope_tables[name]
            if end > int(table.shape[0]):
                raise ContextOverflowError(
                    f"a {q_len}-token batch at {start} needs RoPE rows up to {end} and the "
                    f"exported table has {int(table.shape[0])}")
            out[name] = np.ascontiguousarray(
                table[begin:end].reshape(1, 1, end - begin, int(table.shape[1])))
        return out

    # -- wiring ------------------------------------------------------------
    def _load_models(self):
        for name in self.meta["run_order"]:
            entry = self.meta["components"][name]
            loaded = {}
            for function in entry["functions"]:
                path = resolve_model_path(self.model_dir, entry["files"][function])
                loaded[function] = load_function(path, None, self.compute_units)
                if self.verbose:
                    print(f"loaded {name}/{function}: {path.name}", file=sys.stderr)
            self.models[name] = loaded

    @property
    def state_owner(self) -> str:
        return self.meta["run_order"][0]

    def reset_state(self) -> None:
        """A fresh MLState -- what a loop-count change must do (spec D4)."""
        self._state = self.models[self.state_owner]["infer"].make_state()

    @property
    def loops(self) -> int:
        return self._loops

    def set_loops(self, loops: int) -> bool:
        """Select the loop count; returns True when the cache was reset."""
        loops = int(loops)
        lo, hi = self.meta["loops"]["min"], self.meta["loops"]["max"]
        if not lo <= loops <= hi:
            raise ValueError(f"loop count must be {lo}..{hi}, got {loops}")
        changed = loops != self._loops
        if changed:
            self._loops = loops
            self.reset_state()
        return changed

    # -- tensors -----------------------------------------------------------
    def build_mask(self, start: int, q_len: int) -> np.ndarray:
        return dc.build_causal_mask(
            start, q_len, self.params["context_length"],
            self.params["sliding_rows"], self.params["sliding_window"],
            dtype=torch.float16).numpy()

    def _write_inputs(self, entry: dict, start: int, q_len: int) -> dict:
        """The per-tile write index and gate a tiled conversion asks for.

        Mirrors DoryANEForCausalLM.tile_write_inputs: the host owns the decision
        (which full-attention tile owns these rows), the graph only lerps.  A
        batch that straddles a tile boundary is refused, because one write per
        tile cannot express it.
        """
        names = entry["inputs"]
        if "write_index_1" not in names:
            return {}
        tiles = self.params.get("full_cache_tiles") or [int(self.params["context_length"])]
        offsets, start_pos = [], 0
        for rows in tiles:
            offsets.append(start_pos)
            start_pos += int(rows)
        end = int(start) + int(q_len)
        owners = [(offset, int(rows)) for offset, rows in zip(offsets, tiles)
                  if int(start) >= offset and end <= offset + int(rows)]
        if not owners:
            raise ContextOverflowError(
                f"a {q_len}-token batch at {start} straddles the full-attention tile "
                f"boundary {tuple(tiles)}; the tiled path writes one tile per call")
        owner = owners[0]
        out = {}
        for tile, (offset, rows) in enumerate(zip(offsets, tiles), start=1):
            inside = 1.0 if (offset, int(rows)) == owner else 0.0
            index = (int(start) - owner[0]) if inside else 0
            out[f"write_index_{tile}"] = np.array([index], dtype=np.int32)
            out[f"write_gate_{tile}"] = np.array([inside], dtype=np.float16)
        return out

    def _predict(self, name: str, inputs: dict, function: str) -> dict:
        started = time.time()
        out = self.models[name][function].predict(inputs, state=self._state)
        self.predict_seconds += time.time() - started
        self.predict_calls += 1
        return out

    def _blocks(self, hidden, start, q_len, progress=None):
        """Run every block component in run order, looping the recurrent ones."""
        mask = self.build_mask(start, q_len)
        pos = np.array([start], dtype=np.int32)
        rows = int(self.params["sliding_rows"])
        # The sliding ring is written as one contiguous slice, so a batch that
        # straddles its seam would be silently clamped by the traced graph (and
        # raises in eager PyTorch).  dc.cache_positions is the guard the model
        # uses in eager mode; the runtime has to apply it too, because a resumed
        # prefill can start at any position, not just a batch boundary.
        try:
            ring_pos = int(dc.cache_positions(int(start), int(q_len), rows))
        except ValueError as exc:
            raise ValueError(
                f"a {q_len}-token batch at position {start} would straddle the "
                f"{rows}-row sliding ring ({int(start) % rows} + {q_len} > {rows}); "
                f"resume at a multiple of the batch size, or feed a smaller batch"
            ) from exc
        cache_pos = np.array([ring_pos], dtype=np.int32)
        function = "prefill" if q_len > 1 else "infer"
        order = [name for name in self.meta["run_order"]
                 if self.meta["components"][name]["kind"] == "blocks"]
        for name in order:
            declared = self.meta["components"][name].get("inputs") or []
            if (any(n.startswith("cos_") or n.startswith("sin_") for n in declared)
                    and self.rope_tables is None):
                raise RuntimeError(
                    f"component {name} was converted with rope tables as inputs and this "
                    f"package carries none; reconvert, or convert without that option")
        rope_inputs = self._rope_inputs(
            [n for name in order
             for n in (self.meta["components"][name].get("inputs") or [])
             if n.startswith("cos_") or n.startswith("sin_")], start, q_len)
        hidden = np.ascontiguousarray(hidden, dtype=np.float16)
        index = 0
        while index < len(order):
            entry = self.meta["components"][order[index]]
            if entry["calls_per_token"] == "loops":
                group = []
                while (index < len(order)
                       and self.meta["components"][order[index]]["calls_per_token"] == "loops"):
                    group.append(order[index])
                    index += 1
                h_in = hidden
                for _ in range(self._loops):
                    for name in group:
                        inputs = {"hidden_states": hidden, "causal_mask": mask,
                                  "current_pos": pos, "cache_pos": cache_pos}
                        inputs.update(self._write_inputs(self.meta["components"][name],
                                                         start, q_len))
                        inputs.update({key: value for key, value in rope_inputs.items()
                                       if key in (self.meta["components"][name].get("inputs") or [])})
                        if self.meta["components"][name]["loop_entry"]:
                            inputs["h_in"] = h_in
                        hidden = self._predict(name, inputs, function)["output_hidden_states"]
                continue
            inputs = {"hidden_states": hidden, "causal_mask": mask,
                      "current_pos": pos, "cache_pos": cache_pos}
            inputs.update(self._write_inputs(entry, start, q_len))
            inputs.update({key: value for key, value in rope_inputs.items()
                           if key in (entry.get("inputs") or [])})
            hidden = self._predict(order[index], inputs, function)["output_hidden_states"]
            index += 1
        return hidden


    def embeddings(self, input_ids) -> np.ndarray:
        ids = np.asarray(input_ids, dtype=np.int32).reshape(1, -1)
        function = "prefill" if ids.shape[1] > 1 else "infer"
        return self._predict("embeddings", {"input_ids": ids}, function)["hidden_states"]

    def lm_head(self, hidden) -> np.ndarray:
        entry = self.meta["components"]["lm_head"]
        function = "prefill" if hidden.shape[1] > 1 else "infer"
        out = self._predict("lm_head", {"hidden_states": hidden}, function)
        if self.params["argmax_in_model"]:
            best_val, best_id = None, None
            offset = 0
            for part, size in enumerate(self.params["lm_head_chunk_sizes"], start=1):
                value = out[f"argmax_val_{part}"]
                index = out[f"argmax_idx_{part}"] + offset
                if best_val is None:
                    best_val, best_id = value, index
                else:
                    take = value > best_val
                    best_val = np.where(take, value, best_val)
                    best_id = np.where(take, index, best_id)
                offset += size
            return best_id.astype(np.int64)
        parts = [out[name] for name in entry["outputs"]]
        return np.concatenate(parts, axis=-1)

    # -- the two call paths -------------------------------------------------
    def forward(self, input_ids, start: int):
        """One call of the whole chain; returns logits as (seq, vocab)."""
        hidden = self.embeddings(input_ids)
        hidden = self._blocks(hidden, start, len(input_ids))
        return self.lm_head(hidden)[0]

    def check_context(self, prompt_tokens: int, reserve: int = 1) -> None:
        """Refuse a prompt that cannot fit, before any work is done.

        The context is the number of positions the caches hold; one position per
        generated token is used as the loop above advances, so the room left for
        generation is context - prompt_tokens.  The message names the numbers and
        the way out rather than letting a slice run off the end of a cache.
        """
        context = int(self.params["context_length"])
        used = int(prompt_tokens) + int(reserve)
        if used > context:
            raise ContextOverflowError(
                f"the prompt is {format_tokens(prompt_tokens)} tokens and this model's "
                f"context is {format_tokens(context)} tokens: "
                f"{format_tokens(max(used - context, 0))} token(s) too long. Shorten the "
                f"prompt (or lower --max-tokens), or convert a model with a larger "
                f"--context-length; the prompt is not truncated silently.",
                used=int(prompt_tokens), context=context)

    def prefill(self, input_ids, start: int = 0, progress=None):
        """Feed a prompt in 64-token batches; returns logits for every position.

        A 24K prompt is 384 predicts per loop pass, so the caller gets a progress
        callback every few batches: a long prefill must never look hung (spec
        user story 12).  The tail shorter than one batch goes through the
        one-token function, which is the only path the traced graphs provide.
        """
        ids = [int(t) for t in np.asarray(input_ids).reshape(-1)]
        self.check_context(start + len(ids))
        batch = int(self.params["batch_size"] or 1)
        total = max(len(ids), 1)
        step = max(1, min(16, total // 8 or 1))
        rows = []
        started = time.time()
        done = 0
        while done + batch <= len(ids):
            rows.append(self.forward(ids[done:done + batch], start + done))
            done += batch
            if progress is not None and (done % step == 0 or done == len(ids)):
                progress(done, len(ids), time.time() - started)
        while done < len(ids):
            rows.append(self.forward([ids[done]], start + done))
            done += 1
            if progress is not None:
                progress(done, len(ids), time.time() - started)
        if not rows:
            return np.zeros((0, self.params["vocab_size"]), dtype=np.float32)
        return np.concatenate(rows, axis=0)

    def decode(self, token_id: int, pos: int) -> np.ndarray:
        context = int(self.params["context_length"])
        if pos >= context:
            raise ContextOverflowError(
                f"the context is full: position {format_tokens(pos)} of "
                f"{format_tokens(context)}. Stop generating or raise --context-length.",
                used=pos, context=context)
        return self.forward([int(token_id)], pos)[0]


    # -- helpers -----------------------------------------------------------
    def stats(self) -> dict:
        return {"load_seconds": self.load_seconds,
                "predict_seconds": self.predict_seconds,
                "predict_calls": self.predict_calls}


# ---------------------------------------------------------------------------
# Tokenizer and sampling
# ---------------------------------------------------------------------------

#: ChatML with an open reasoning block -- the documented fallback (see
#: FALLBACK_TEMPLATE in dory_converter): Dory's weights carry the special tokens
#: but ship no template, and NVIDIA's README runs llama.cpp with reasoning on.
CHATML_FALLBACK = (
    "{% if messages[0]['role'] == 'system' %}<|im_start|>system\n"
    "{{ messages[0]['content'] }}<|im_end|>\n{% endif %}"
    "{% for message in messages %}"
    "{% if not (loop.first and message['role'] == 'system') %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking %} thinking\n{% endif %}{% endif %}"
)


def render_template(template_text: str, messages, reasoning: bool = True,
                    tools=None) -> str:
    """Render a Jinja chat template with the variables a HF template expects.

    `tools` is the OpenAI `tools` array: Dory's template renders it as
    `<tools>/<function>/<parameter>` XML inside the system block and teaches the
    model the `<tool_call>` reply format, so a request that carries tools has to
    pass them through or the model never learns the tools exist.
    """
    from jinja2 import Environment, BaseLoader

    environment = Environment(loader=BaseLoader(), trim_blocks=True, lstrip_blocks=True,
                              keep_trailing_newline=True)
    template = environment.from_string(template_text)
    return template.render(
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=bool(reasoning),
        tools=list(tools) if tools else [],
        bos_token="<s>",
        eos_token="<|im_end|>",
    )


class ChatTokenizer:
    """The BF16 repository's tokenizer and chat template (spec D9).

    Order of preference: a chat_template.jinja beside the models (written by the
    converter when the weights had one), then a chat_template inside
    tokenizer_config.json, then the documented ChatML fallback -- the ticket
    needs reasoning on by default either way, and the prompt format is recorded
    in the meta file rather than hidden in code.
    """

    def __init__(self, model_dir, meta):
        from transformers import AutoTokenizer

        self.meta = meta
        self.model_dir = Path(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
        self.template_text, self.template_source = self._resolve_template()

    def _resolve_template(self):
        name = (self.meta.get("tokenizer") or {}).get("chat_template")
        if name:
            path = self.model_dir / name
            if path.is_file():
                return path.read_text(encoding="utf-8"), str(path.name)
        config = getattr(self.tokenizer, "chat_template", None)
        if isinstance(config, str) and config.strip():
            return config, "tokenizer_config.json"
        fallback = (self.meta.get("tokenizer") or {}).get("chat_template_fallback")
        return CHATML_FALLBACK, fallback or "chatml-think"

    def apply_chat_template(self, messages, reasoning: bool = True, tools=None) -> str:
        """Render the conversation with reasoning on by default."""
        return render_template(self.template_text, messages, reasoning=reasoning,
                               tools=tools)

    def encode(self, text: str) -> np.ndarray:
        return np.asarray(self.tokenizer.encode(text, add_special_tokens=False),
                          dtype=np.int32)

    def decode(self, ids) -> str:
        return self.tokenizer.decode(list(int(i) for i in ids), skip_special_tokens=True)

    @property
    def eos_ids(self) -> list:
        ids = set(int(t) for t in (self.meta["tokenizer"].get("eos_token_ids") or []))
        if not ids:
            value = self.tokenizer.eos_token_id
            if isinstance(value, (list, tuple)):
                ids.update(int(v) for v in value)
            elif value is not None:
                ids.add(int(value))
        ids.add(int(self.meta["tokenizer"]["eos_token_id"]))
        return sorted(ids)


def sample_token(logits, temperature: float = 0.0, top_p: float = 1.0, rng=None):
    """Greedy at temperature 0, otherwise temperature + nucleus sampling."""
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    if temperature is None or temperature <= 0:
        return int(np.argmax(logits))
    rng = rng or np.random.default_rng()
    scaled = logits / float(temperature)
    scaled -= scaled.max()
    probs = np.exp(scaled)
    probs /= probs.sum()
    if top_p is not None and 0.0 < float(top_p) < 1.0:
        order = np.argsort(-probs)
        cumulative = np.cumsum(probs[order])
        keep = order[: int(np.searchsorted(cumulative, float(top_p)) + 1)]
        mask = np.zeros_like(probs, dtype=bool)
        mask[keep] = True
        probs = np.where(mask, probs, 0.0)
        probs /= probs.sum()
    return int(rng.choice(probs.shape[0], p=probs))


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

def build_messages(prompt: str, system: str = None) -> list:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


class StreamPrinter:
    """Write only the text that is new, so no partial character is emitted."""

    def __init__(self, tokenizer, out=None):
        self.tokenizer = tokenizer
        self.out = out or sys.stdout
        self.printed = 0

    def update(self, token_ids) -> None:
        piece = self.tokenizer.decode(token_ids)
        if len(piece) > self.printed:
            self.out.write(piece[self.printed:])
            self.out.flush()
            self.printed = len(piece)

    def done(self) -> None:
        self.out.write(chr(10))


def generate(runtime, tokenizer, prompt, max_tokens=256, temperature=0.0, top_p=1.0,
             reasoning=True, system=None, stream=True, out=sys.stdout,
             progress=True):
    """Prefill the prompt, then stream tokens until EOS or max_tokens."""
    text = tokenizer.apply_chat_template(build_messages(prompt, system), reasoning=reasoning)
    prompt_ids = tokenizer.encode(text)
    context = int(runtime.params["context_length"])
    runtime.check_context(len(prompt_ids), reserve=1)
    runtime.reset_state()
    started = time.time()
    long_prompt = len(prompt_ids) > 2 * int(runtime.params.get("batch_size") or 64)
    printer_progress = _progress_printer() if (progress and long_prompt) else None
    logits = runtime.prefill(prompt_ids, 0, progress=printer_progress)
    prefill_seconds = time.time() - started
    generated = []
    printer = StreamPrinter(tokenizer, out) if stream else None
    # pos is the row the NEXT generated token occupies: the token sampled from
    # the prompt's last logits sits at len(prompt_ids), not after it.  The server
    # does the same (openai_server/service.py feeds decode at the current
    # position and increments afterwards); incrementing first would write every
    # key one row late and leave the true row a zero key the mask admits.
    pos = len(prompt_ids)
    eos = set(tokenizer.eos_ids)
    while len(generated) < max_tokens and pos < context - 1:
        next_id = sample_token(logits[-1], temperature, top_p)
        if next_id in eos:
            break
        generated.append(next_id)
        if printer is not None:
            printer.update(generated)
        if len(generated) >= max_tokens:
            break
        logits = runtime.decode(next_id, pos).reshape(1, -1)
        pos += 1
    seconds = time.time() - started
    if printer is not None:
        printer.done()
    return {
        "text": tokenizer.decode(generated),
        "prompt_tokens": int(len(prompt_ids)),
        "generated_tokens": int(len(generated)),
        "prefill_seconds": prefill_seconds,
        "seconds": seconds,
        "decode_tok_s": (len(generated) / max(seconds - prefill_seconds, 1e-9)),
        "loops": runtime.loops,
    }


def run_session(args) -> int:
    """Load the model, then either answer one prompt or open a session."""
    runtime = DoryRuntime(args.model_dir, verbose=args.verbose)
    tokenizer = ChatTokenizer(Path(args.model_dir), runtime.meta)
    runtime.set_loops(args.loops)
    print(f"loops {runtime.loops}, context {runtime.params['context_length']}, "
          f"reasoning {'on' if not args.no_think else 'off'}", file=sys.stderr)
    if args.prompt:
        return _answer_once(args, runtime, tokenizer)
    return _converse(args, runtime, tokenizer)


def _answer_once(args, runtime, tokenizer) -> int:
    result = generate(runtime, tokenizer, args.prompt, args.max_tokens,
                      args.temperature, args.top_p, not args.no_think,
                      args.system, args.stream, getattr(args, "progress", True))
    print(json.dumps({k: v for k, v in result.items() if k != "text"}, indent=2),
          file=sys.stderr)
    return 0


def _converse(args, runtime, tokenizer) -> int:
    """The interactive loop: /loops N, /reset, /quit."""
    print("commands: /loops N  /reset  /quit", file=sys.stderr)
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(file=sys.stderr)
            return 0
        if not line:
            continue
        if line in ("/quit", "/exit"):
            return 0
        if line == "/reset":
            runtime.reset_state()
            print("context and KV cache reset", file=sys.stderr)
            continue
        if line.startswith("/loops"):
            parts = line.split()
            if len(parts) != 2 or not parts[1].isdigit():
                print("usage: /loops 1|2|3|4", file=sys.stderr)
                continue
            wanted = int(parts[1])
            changed = runtime.set_loops(wanted)
            if changed:
                print(f"loop count {wanted}; KV cache reset -- older rows keep the "
                      f"previous loop count's K/V, so the conversation starts empty "
                      f"(spec D4)", file=sys.stderr)
            else:
                print(f"loop count already {wanted}; cache kept", file=sys.stderr)
            continue
        print("dory> ", end="", flush=True)
        try:
            generate(runtime, tokenizer, line, args.max_tokens, args.temperature,
                     args.top_p, not args.no_think, args.system, args.stream,
                     getattr(args, "progress", True))
        except ContextOverflowError as error:
            print(f"error: {error}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Benchmarks and scoring
# ---------------------------------------------------------------------------

def _progress_printer(out=sys.stderr):
    """A prefill progress line: batch count, percentage and tok/s so far."""
    state = {"last": 0.0}

    def report(done: int, total: int, elapsed: float) -> None:
        now = time.time()
        if done < total and now - state["last"] < 0.25:
            return
        state["last"] = now
        rate = done / elapsed if elapsed > 0 else 0.0
        print(f"\rprefill {done:,}/{total:,} tokens ({100.0 * done / max(total, 1):.0f}%, "
              f"{rate:,.0f} tok/s)", end="" if done < total else "\n", file=out, flush=True)

    return report


def _uptime() -> dict:
    try:
        return {"loadavg": list(open("/proc/loadavg").read().split()[:3])}
    except Exception:
        pass
    try:
        import subprocess
        text = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
        return {"uptime": text}
    except Exception:
        return {}


def filler_tokens(context: int, tokenizer=None, text=None) -> np.ndarray:
    """A prompt of exactly context tokens to decode after."""
    if text:
        ids = tokenizer.encode(text)
    else:
        ids = np.ones(max(context, 1), dtype=np.int32)
    if len(ids) < context:
        ids = np.tile(ids, int(np.ceil(context / max(len(ids), 1))))
    return ids[:context].astype(np.int32)


def bench_workload(context: int, steps: int) -> int:
    """How many prompt tokens a bench at this context may prefill.

    The benchmark decodes a number of tokens after the prefill, and check_context
    reserves one position for that, so the prompt has to leave that many positions
    free.  Prefilling the whole context raised ContextOverflowError before the
    first measurement, which is what made the 24K bench cell unmeasurable.
    """
    context, steps = int(context), int(steps)
    keep = context - max(steps, 1)
    if keep < 1:
        raise ContextOverflowError(
            f"a {format_tokens(context)}-token context leaves no room for {steps} decode "
            f"step(s); the benchmark needs context_length > tokens")
    return keep


def resident_memory() -> dict:
    """Peak resident set size of this process, in bytes, plus the machine's swap.

    ps and vmmap are not reachable from a sandboxed session, so the process
    measures itself: ru_maxrss is the peak RSS in bytes on macOS, which is the
    number the ticket asks for (the model's own weights and states dominate it).
    """
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    machine = {}
    try:
        import subprocess
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True,
                             text=True).stdout.strip()
        machine["swap"] = out
    except Exception:
        pass
    return {"peak_rss_bytes": int(peak), "peak_rss_gib": round(peak / 1024 ** 3, 2),
            "machine": machine}


def benchmark(args) -> int:
    """Decode tok/s at a chosen context, for one or more loop counts."""
    runtime = DoryRuntime(args.model_dir, verbose=args.verbose)
    tokenizer = ChatTokenizer(Path(args.model_dir), runtime.meta)
    context = int(args.context)
    if context > int(runtime.params["context_length"]):
        raise ContextOverflowError(
            f"--context {format_tokens(context)} is beyond the converted context "
            f"{format_tokens(int(runtime.params['context_length']))}")
    steps = int(args.tokens)
    prompt = filler_tokens(bench_workload(context, steps), tokenizer, args.filler_text)
    printer = _progress_printer() if args.progress else None
    results = []
    for loops in [int(v) for v in str(args.loops).split(",")]:
        runtime.set_loops(loops)
        runtime.reset_state()
        started = time.time()
        logits = runtime.prefill(prompt, 0, progress=printer)
        prefill_seconds = time.time() - started
        window = []
        positions = [int(len(prompt))]
        token = int(np.argmax(logits[-1]))
        for _ in range(steps):
            started = time.time()
            logits = runtime.decode(token, positions[-1])
            window.append(time.time() - started)
            token = int(np.argmax(logits))
            positions.append(positions[-1] + 1)
        warm = window[len(window) // 4:] or window
        mean = sum(warm) / len(warm)
        result = {"loops": loops, "context": context,
                  "prompt_tokens": int(len(prompt)),
                  "decode_headroom": int(context - len(prompt)),
                  "prefill_seconds": prefill_seconds,
                  "prefill_tok_s": len(prompt) / max(prefill_seconds, 1e-9),
                  "decode_tok_s": 1.0 / mean, "decode_ms": mean * 1e3,
                  "tokens": len(warm), "resident": resident_memory(),
                  "machine": _uptime()}
        results.append(result)
        print(json.dumps(result), file=sys.stderr)
    print(json.dumps(results, indent=2))
    return 0


def log_softmax_rows(logits) -> np.ndarray:
    """Row-wise log-softmax in float32: the distribution DANE-2 scores."""
    block = np.asarray(logits, dtype=np.float32)
    block = block - block.max(axis=-1, keepdims=True)
    return block - np.log(np.exp(block).sum(axis=-1, keepdims=True))


def collect_logprobs(runtime, ids, chunk_size: int, max_windows: int = 0,
                     progress=None):
    """Feed the evaluation text in windows; return (logprobs, positions).

    Decode mode means the candidate batches the way it runs -- a 64-token
    prefill per batch, then the tail one token at a time.  The scored rows are
    the second half of each window, llama-perplexity's convention and what
    anemll.eval.kld.scored_positions produces.
    """
    chunk_size = int(chunk_size)
    n_chunks = len(ids) // chunk_size
    if n_chunks < 1:
        raise ValueError(f"{len(ids)} tokens is shorter than one "
                         f"{chunk_size}-token window")
    lo, hi = chunk_size // 2, chunk_size - 1
    logprobs, positions = [], []
    for index in range(n_chunks):
        if max_windows and index >= int(max_windows):
            break
        start = index * chunk_size
        runtime.reset_state()
        logits = runtime.prefill(ids[start:start + chunk_size], 0)
        logprobs.append(log_softmax_rows(logits[lo:hi]))
        positions.append(np.arange(start + lo, start + hi, dtype=np.int32))
        if progress is not None:
            progress(index + 1, n_chunks)
    return np.concatenate(logprobs), np.concatenate(positions)


def score(args) -> int:
    """Score the Core ML pipeline in decode mode against a reference (spec gates)."""
    from anemll.eval.kld import score_files, write_candidate

    runtime = DoryRuntime(args.model_dir, verbose=args.verbose)
    ids = np.loadtxt(Path(args.tokens), dtype=np.int64).reshape(-1).astype(np.int32)
    chunk = int(args.chunk_size)
    runtime.set_loops(int(args.loops))
    progress = (lambda done, total: print(f"window {done}/{total}", file=sys.stderr)) \
        if args.verbose else None
    logprobs, positions = collect_logprobs(runtime, ids, chunk, args.max_windows, progress)
    payload = {
        "model": "dory-coreml",
        "loops": int(args.loops),
        "kv_mode": "last_loop",
        "batch_size": int(runtime.params["batch_size"]),
        "n_ctx": chunk,
        "trace": "coreml",
        "machine": _uptime(),
    }
    candidate = write_candidate(args.out, logprobs, positions, ids, payload)
    print(f"wrote {candidate}", file=sys.stderr)
    if args.reference:
        result = score_files(args.reference, candidate)
        print(result.summary())
        print(json.dumps(result.as_dict(), indent=2))
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dory on the Neural Engine")
    sub = parser.add_subparsers(dest="command", required=True)

    chat = sub.add_parser("chat", help="talk to a converted model")
    chat.add_argument("--model-dir", required=True)
    chat.add_argument("--prompt", default=None, help="one prompt, then exit")
    chat.add_argument("--loops", type=int, default=2, help="loop count at start (1-4)")
    chat.add_argument("--max-tokens", type=int, default=256)
    chat.add_argument("--temperature", type=float, default=0.0)
    chat.add_argument("--top-p", type=float, default=1.0)
    chat.add_argument("--system", default=None)
    chat.add_argument("--no-think", action="store_true", help="turn reasoning off")
    chat.add_argument("--no-stream", dest="stream", action="store_false")
    chat.add_argument("--verbose", action="store_true")
    chat.add_argument("--no-progress", dest="progress", action="store_false",
                      help="hide the prefill progress line")
    chat.set_defaults(func=run_session)

    bench = sub.add_parser("bench", help="decode tok/s at a context and loop counts")
    bench.add_argument("--model-dir", required=True)
    bench.add_argument("--context", type=int, default=1024)
    bench.add_argument("--loops", default="1,2,4")
    bench.add_argument("--tokens", type=int, default=32)
    bench.add_argument("--filler-text", default=None)
    bench.add_argument("--verbose", action="store_true")
    bench.add_argument("--no-progress", dest="progress", action="store_false")
    bench.set_defaults(func=benchmark)

    sc = sub.add_parser("score", help="score this pipeline against a decode reference")
    sc.add_argument("--model-dir", required=True)
    sc.add_argument("--tokens", default="tests/fixtures/dory_eval/eval_tokens.txt")
    sc.add_argument("--chunk-size", type=int, default=1024)
    sc.add_argument("--max-windows", type=int, default=0)
    sc.add_argument("--loops", type=int, default=2)
    sc.add_argument("--out", required=True)
    sc.add_argument("--reference", default=None)
    sc.add_argument("--verbose", action="store_true")
    sc.set_defaults(func=score)
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ContextOverflowError as error:
        # one clear line, exit code 2: never a traceback, never silent truncation
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
