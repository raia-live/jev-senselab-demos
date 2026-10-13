"""A small embedding store over past cases.

Embeddings: ``BAAI/bge-small-en-v1.5`` through ``fastembed`` (CPU, local, no key).
Ranking: cosine similarity on the case text, discounted by age (``recency_half_life``
episodes). ``filter_fields`` restricts candidates to past cases that share those
structured fields with the current one. Two configurations ship, see :data:`CONFIGS`.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.scenario import Case, Outcome

try:  # the baseline is optional: `uv sync --extra baseline`
    from fastembed import TextEmbedding  # type: ignore

    _AVAILABLE = True
except Exception:  # noqa: BLE001
    TextEmbedding = None  # type: ignore
    _AVAILABLE = False

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def available() -> bool:
    return _AVAILABLE


@dataclass(frozen=True)
class RetrievalConfig:
    name: str
    k: int = 8
    recency_half_life: float = 30.0
    filter_fields: tuple[str, ...] = ()
    include_fields: bool = True


CONFIGS: dict[str, RetrievalConfig] = {
    "text-k8": RetrievalConfig("text-k8", k=8, recency_half_life=30.0),
    "meta-k8": RetrievalConfig("meta-k8", k=8, recency_half_life=30.0, filter_fields=("situation",)),
}


@dataclass
class _Row:
    episode: int
    case: Case
    action: str
    outcome: Outcome
    vector: np.ndarray = field(repr=False)


class RetrievalStore:
    _model: Any = None
    # One embedding model for the process, used by one thread at a time: the arms run
    # side by side and the ONNX session is not safe to share without this.
    _lock = threading.Lock()

    def __init__(self, config: RetrievalConfig) -> None:
        if not _AVAILABLE:
            raise RuntimeError("fastembed is not installed; run `uv sync --extra baseline`")
        self.config = config
        self.rows: list[_Row] = []
        with RetrievalStore._lock:
            if RetrievalStore._model is None:
                RetrievalStore._model = TextEmbedding(model_name=MODEL_NAME)

    def _embed(self, text: str) -> np.ndarray:
        with RetrievalStore._lock:
            vec = next(iter(RetrievalStore._model.embed([text])))
        v = np.asarray(vec, dtype=np.float32)
        n = float(np.linalg.norm(v)) or 1.0
        return v / n

    def _doc(self, case: Case) -> str:
        if self.config.include_fields:
            fields = " ".join(f"{k}={v}" for k, v in case.fields.items())
            return f"{case.text}\n{fields}"
        return case.text

    def add(self, episode: int, case: Case, action: str, outcome: Outcome) -> None:
        self.rows.append(_Row(episode, case, action, outcome, self._embed(self._doc(case))))

    def neighbours(self, episode: int, case: Case) -> list[tuple[float, _Row]]:
        if not self.rows:
            return []
        q = self._embed(self._doc(case))
        scored: list[tuple[float, _Row]] = []
        for row in self.rows:
            if any(row.case.fields.get(f, row.case.situation if f == "situation" else None)
                   != case.fields.get(f, case.situation if f == "situation" else None)
                   for f in self.config.filter_fields):
                continue
            sim = float(np.dot(q, row.vector))
            age = max(0, episode - row.episode)
            recency = 0.5 + 0.5 * math.pow(2.0, -age / self.config.recency_half_life)
            scored.append((sim * recency, row))
        scored.sort(key=lambda t: -t[0])
        return scored[: self.config.k]

    def render(self, episode: int, case: Case) -> str:
        hits = self.neighbours(episode, case)
        if not hits:
            return ""
        lines = ["Similar past cases (most relevant first):"]
        for _, row in hits:
            age = episode - row.episode
            verdict = "PASS" if row.outcome.success else f"FAIL ({row.outcome.severity})"
            fields = ", ".join(f"{k}={v}" for k, v in row.case.fields.items())
            lines.append(f"- {age} cases ago [{fields}] -> decide:{row.action} -> {verdict}: {row.outcome.note}")
        return "\n".join(lines)


__all__ = ["CONFIGS", "MODEL_NAME", "RetrievalConfig", "RetrievalStore", "available"]
