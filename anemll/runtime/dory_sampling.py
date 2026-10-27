"""Sampling for the Dory ANE runtime -- the knobs a normal engine exposes.

Kept separate from the runtime so the runtime stays a thin dispatcher and so the
sampler can be tested without a Neural Engine: everything here is numpy on a
float vector.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class SamplingParams:
    """The subset of vLLM's SamplingParams that makes sense on one laptop ANE.

    Defaults are greedy, because a caller that says nothing should get the most
    predictable thing rather than a random one.
    """

    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop: List[str] = field(default_factory=list)
    stop_token_ids: List[int] = field(default_factory=list)
    ignore_eos: bool = False
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature cannot be negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p has to be in (0, 1]")
        if self.max_tokens < 0:
            raise ValueError("max_tokens cannot be negative")

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


class Sampler:
    """Turns logits into the next token id, honouring the penalties and filters."""

    def __init__(self, params: SamplingParams):
        self.params = params
        self.rng = np.random.default_rng(params.seed)

    def _apply_penalties(self, logits: np.ndarray, history: Sequence[int]) -> np.ndarray:
        p = self.params
        if not history:
            return logits
        out = logits.astype(np.float64, copy=True)
        if p.repetition_penalty != 1.0:
            seen = np.fromiter(set(int(t) for t in history), dtype=np.int64,
                               count=len(set(int(t) for t in history)))
            if seen.size:
                # vLLM's convention: divide positives, multiply negatives
                values = out[seen]
                out[seen] = np.where(values > 0, values / p.repetition_penalty,
                                     values * p.repetition_penalty)
        if p.frequency_penalty or p.presence_penalty:
            ids, counts = np.unique(np.asarray(history, dtype=np.int64), return_counts=True)
            if p.frequency_penalty:
                out[ids] -= p.frequency_penalty * counts
            if p.presence_penalty:
                out[ids] -= p.presence_penalty
        return out

    def _filter(self, logits: np.ndarray) -> np.ndarray:
        p = self.params
        out = logits.astype(np.float64, copy=True)
        if p.top_k and p.top_k > 0 and p.top_k < out.size:
            cut = np.partition(out, -p.top_k)[-p.top_k]
            out[out < cut] = -np.inf
        shifted = out - np.max(out)
        probs = np.exp(shifted)
        probs /= probs.sum()
        if p.top_p < 1.0:
            order = np.argsort(-probs)
            cumulative = np.cumsum(probs[order])
            keep = order[:int(np.searchsorted(cumulative, p.top_p)) + 1]
            mask = np.zeros_like(probs, dtype=bool)
            mask[keep] = True
            probs = np.where(mask, probs, 0.0)
            probs /= probs.sum()
        if p.min_p > 0.0:
            cutoff = p.min_p * probs.max()
            probs = np.where(probs >= cutoff, probs, 0.0)
            total = probs.sum()
            probs = probs / total if total > 0 else probs
        return probs

    def next_token(self, logits: np.ndarray, history: Sequence[int] = ()) -> int:
        adjusted = self._apply_penalties(np.asarray(logits, dtype=np.float64), history)
        if self.params.greedy:
            return int(np.argmax(adjusted))
        probs = self._filter(adjusted)
        return int(self.rng.choice(probs.size, p=probs))
