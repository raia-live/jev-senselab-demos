"""Jev (TypeSafe AI System One) as a typed decision function.

One ``Choice`` question per case: the state carries the case, the structured fields, the
written policy and a ``memory`` slot; the answer is an action with calibrated
probabilities and a confidence. Jev is stateless and fast; it does not learn from how a
decision turns out — that is what the memory slot is for.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from typesafe_sdk import Choice, RetryPolicy, TypeSafeAPIError, TypeSafeClient

from . import settings as S


@dataclass
class Decision:
    action: str
    confidence: float
    probabilities: dict[str, float]
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    state_bytes: int
    error: str | None = None
    basis: str | None = None
    basis_confidence: float | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_row(self) -> dict[str, Any]:
        top = sorted(self.probabilities.items(), key=lambda kv: -kv[1])[:3]
        return {
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "top_probabilities": {k: round(v, 4) for k, v in top},
            "basis": self.basis,
            "jev_model": self.model,
            "jev_input_tokens": self.input_tokens,
            "jev_output_tokens": self.output_tokens,
            "jev_latency_ms": round(self.latency_ms, 1),
            "state_bytes": self.state_bytes,
            "jev_error": self.error,
        }


class Jev:
    """A thin client around ``TypeSafeClient.system_one`` with the model pinned."""

    def __init__(self, api_key: str, *, model: str = S.JEV_MODEL, timeout: float = 20.0) -> None:
        self.model = model
        self._client = TypeSafeClient(
            api_key=api_key, model=model, timeout=timeout,
            retry=RetryPolicy(max_retries=3, backoff_initial=0.5, backoff_max=6.0),
        )

    def close(self) -> None:
        self._client.close()

    def smoke(self) -> list[str]:
        """The models the key can use; raises if the key is bad or out of credit."""
        return [m.name for m in self._client.models.list().models]

    def decide(self, state: dict[str, Any], *, instructions: str, criteria: dict[str, str],
               question: str = "action", basis: dict[str, str] | None = None) -> Decision:
        """Ask for the action; with *basis*, also ask which memory entry (by tag) the
        decision rested on. Both questions go in the one call, so the citation costs a few
        output tokens and no extra round trip."""
        payload = json.dumps(state, ensure_ascii=False)
        questions = {question: Choice(instructions=instructions, criteria=criteria)}
        if basis:
            questions["basis"] = Choice(instructions=BASIS_INSTRUCTIONS, criteria=basis)
        t0 = time.perf_counter()
        try:
            resp = self._client.system_one(state=state, questions=questions)
        except TypeSafeAPIError as exc:
            latency = (time.perf_counter() - t0) * 1000
            # A failed call is a decision not taken: fall back to the first action so the
            # episode still scores, and keep the error on the row.
            first = next(iter(criteria))
            return Decision(first, 0.0, {}, self.model, 0, 0, latency, len(payload.encode()),
                            error=f"{type(exc).__name__}: {exc}"[:200])
        latency = (time.perf_counter() - t0) * 1000
        ans = resp.choices[question]
        usage = resp.usage
        cited = resp.choices.get("basis") if basis else None
        return Decision(
            action=ans.choice,
            confidence=float(ans.confidence),
            probabilities={k: float(v) for k, v in ans.probabilities.items()},
            model=resp.model,
            input_tokens=int(usage.input_tokens or 0),
            output_tokens=int(usage.output_tokens or 0),
            latency_ms=latency,
            state_bytes=len(payload.encode()),
            basis=(cited.choice if cited is not None else None),
            basis_confidence=(float(cited.confidence) if cited is not None else None),
        )


BASIS_INSTRUCTIONS = (
    "Which entry in the 'memory' field (by its [M#] tag) most determined your decision? "
    "Answer 'none' if the memory did not change what the written policy alone would give."
)

__all__ = ["BASIS_INSTRUCTIONS", "Decision", "Jev"]
