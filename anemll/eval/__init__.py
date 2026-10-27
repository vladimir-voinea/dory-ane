"""Evaluation utilities for parity work against a llama.cpp reference.

See docs/dory-eval.md for the candidate log-probability format and the
reference-scoring methodology.
"""

from .eval_set import EvalSet, load_eval_set
from .kld import (
    ArtifactError,
    LogprobSource,
    ScoreResult,
    read_source,
    score_arrays,
    score_files,
    score_sources,
    write_candidate,
)

__all__ = [
    "ArtifactError",
    "EvalSet",
    "LogprobSource",
    "ScoreResult",
    "load_eval_set",
    "read_source",
    "score_arrays",
    "score_files",
    "score_sources",
    "write_candidate",
]
