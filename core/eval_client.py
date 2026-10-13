"""The SenseLab repair-loop API (``/api/v1/eval``), the subset the demos drive.

The loop itself — judging, behaviour clustering, incidents, proposing a fix, the Tier 1
replay, the canary decision, shipping, verifying, rolling back — runs on the SenseLab
side. This client only asks it to grade a trace, lists the agent's fixes, and advances
the canary the way the hourly sweep would, so the loop can close inside one run.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from . import settings as S


class EvalApiError(RuntimeError):
    def __init__(self, status: int, detail: Any, path: str) -> None:
        super().__init__(f"{path} -> {status}: {str(detail)[:300]}")
        self.status = status
        self.detail = detail
        self.path = path


class EvalClient:
    def __init__(self, api_key: str, *, base_url: str = S.DEFAULT_AMFS_URL, timeout: float = 180.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url, headers={"X-AMFS-API-Key": api_key},
            timeout=httpx.Timeout(timeout, connect=10.0),
        )

    def close(self) -> None:
        self._client.close()

    # -- plumbing ------------------------------------------------------------------------

    def _call(self, method: str, path: str, *, json: Any = None, params: dict[str, Any] | None = None,
              retries: int = 4, timeout: float | None = None) -> Any:
        for attempt in range(retries + 1):
            kw: dict[str, Any] = {}
            if timeout is not None:
                kw["timeout"] = httpx.Timeout(timeout, connect=10.0)
            try:
                resp = self._client.request(method, path, json=json, params=params, **kw)
            except httpx.TransportError as exc:
                if attempt < retries:
                    time.sleep(min(1.5 * (2 ** attempt), 12.0))
                    continue
                raise EvalApiError(0, f"{type(exc).__name__}: {exc}", path) from exc
            if resp.status_code in (429, 502, 503, 504) and attempt < retries:
                wait = min(max(1.0, float(resp.headers.get("Retry-After", 1.0))) * (1.5 ** attempt), 20.0)
                time.sleep(wait)
                continue
            if resp.status_code >= 400:
                try:
                    detail = resp.json()
                except ValueError:
                    detail = resp.text[:300]
                raise EvalApiError(resp.status_code, detail, path)
            if not resp.content:
                return {}
            return resp.json()
        raise RuntimeError("unreachable")

    # -- capability probe ----------------------------------------------------------------

    def probe(self) -> dict[str, Any]:
        """Can this key run the repair loop? ``{"ok", "readiness", "budget", "error"}``."""
        out: dict[str, Any] = {"ok": False, "readiness": None, "budget": None, "error": None}
        try:
            # A 429 here is load, not a missing capability; give it a few tries before saying no.
            out["readiness"] = self._call("GET", "/api/v1/eval/readiness", retries=4, timeout=20.0)
            out["budget"] = self._call("GET", "/api/v1/eval/budget", retries=4, timeout=20.0)
        except EvalApiError as exc:
            out["error"] = f"{exc.status}: {str(exc.detail)[:200]}"
            return out
        ready = bool((out["readiness"] or {}).get("ready"))
        out["ok"] = ready
        if not ready:
            out["error"] = (out["readiness"] or {}).get("reason") or "eval readiness is false"
        return out

    def budget(self) -> dict[str, Any]:
        return self._call("GET", "/api/v1/eval/budget")

    # -- judges --------------------------------------------------------------------------

    def ensure_judge(self, judge_id: str, agent_id: str, name: str, prompt: str, *,
                     mode: str = "continuous", sampling_rate: float = 1.0,
                     model: str | None = None) -> dict[str, Any]:
        try:
            return self._call("GET", f"/api/v1/eval/judges/{judge_id}", params={"agent_id": agent_id})
        except EvalApiError as exc:
            if exc.status != 404:
                raise
        body: dict[str, Any] = {"id": judge_id, "agent_id": agent_id, "name": name, "prompt": prompt,
                                "scoring_type": "binary", "mode": mode, "sampling_rate": sampling_rate, "enabled": True}
        if model:
            body["model"] = model
        try:
            return self._call("POST", "/api/v1/eval/judges", json=body)
        except EvalApiError as exc:
            if exc.status == 409:
                return self._call("GET", f"/api/v1/eval/judges/{judge_id}", params={"agent_id": agent_id})
            raise

    def judge(self, trace_id: str, judge_id: str, *, agent_id: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"trace_id": trace_id, "judge_id": judge_id, "persist": True, "force": False}
        if agent_id:
            body["agent_id"] = agent_id
        return self._call("POST", "/api/v1/eval/judge", json=body, retries=2, timeout=240.0)

    def verdicts(self, agent_id: str, *, limit: int = 200, **params: Any) -> dict[str, Any]:
        q: dict[str, Any] = {"agent_id": agent_id, "limit": limit}
        q.update({k: v for k, v in params.items() if v is not None})
        return self._call("GET", "/api/v1/eval/verdicts", params=q)

    # -- fixes and the canary ------------------------------------------------------------

    def list_fixes(self, agent_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"agent_id": agent_id, "limit": min(100, limit)}
            if cursor:
                params["cursor"] = cursor
            page = self._call("GET", "/api/v1/eval/fixes", params=params)
            if isinstance(page, list):
                items.extend(i for i in page if isinstance(i, dict))
                break
            items.extend(i for i in (page.get("items") or []) if isinstance(i, dict))
            cursor = page.get("next_cursor") or page.get("cursor")
            if not cursor or len(items) >= limit:
                break
        return items

    def get_fix(self, fix_id: str) -> dict[str, Any]:
        return self._call("GET", f"/api/v1/eval/fixes/{fix_id}")

    def fix_events(self, fix_id: str, limit: int = 100) -> dict[str, Any]:
        return self._call("GET", f"/api/v1/eval/fixes/{fix_id}/events", params={"limit": limit})

    def fix_tests(self, fix_id: str) -> dict[str, Any]:
        return self._call("GET", f"/api/v1/eval/fixes/{fix_id}/tests")

    def test_fix(self, fix_id: str) -> dict[str, Any]:
        return self._call("POST", f"/api/v1/eval/fixes/{fix_id}/test", params={"now": "true"})

    def prevention(self, fix_id: str) -> dict[str, Any]:
        return self._call("GET", f"/api/v1/eval/fixes/{fix_id}/prevention")

    def recheck(self, fix_id: str) -> dict[str, Any]:
        return self._call("POST", f"/api/v1/eval/fixes/{fix_id}/recheck")

    def rollback(self, fix_id: str, reason: str = "") -> dict[str, Any]:
        return self._call("POST", f"/api/v1/eval/fixes/{fix_id}/rollback", json={"reason": reason[:2000]})

    def agent_canary(self, agent_id: str) -> dict[str, Any]:
        return self._call("GET", f"/api/v1/eval/agents/{agent_id}/canary")

    def start_canary(self, fix_id: str, *, pct: int, min_outcomes: int, timeout_days: int = 1) -> dict[str, Any]:
        body = {"pct": pct, "min_outcomes": min_outcomes, "timeout_days": timeout_days}
        return self._call("POST", f"/api/v1/eval/fixes/{fix_id}/canary", json=body)

    def stop_canary(self, fix_id: str, reason: str = "") -> dict[str, Any]:
        return self._call("POST", f"/api/v1/eval/fixes/{fix_id}/canary/stop", json={"reason": reason[:2000]})

    def evaluate_canary(self, fix_id: str) -> dict[str, Any]:
        return self._call("POST", f"/api/v1/eval/fixes/{fix_id}/canary/evaluate")

    # -- incidents, behaviours, settings -------------------------------------------------

    def incidents(self, agent_id: str, *, limit: int = 100) -> dict[str, Any]:
        return self._call("GET", "/api/v1/eval/incidents", params={"agent_id": agent_id, "limit": limit})

    def behaviors(self, agent_id: str) -> dict[str, Any]:
        return self._call("GET", "/api/v1/eval/behaviors", params={"agent_id": agent_id})

    def repairs_summary(self, agent_id: str) -> dict[str, Any]:
        return self._call("GET", "/api/v1/eval/repairs/summary", params={"agent_id": agent_id})

    def get_repair_settings(self, agent_id: str) -> dict[str, Any]:
        return self._call("GET", f"/api/v1/eval/agents/{agent_id}/repair-settings")

    def set_repair_settings(self, agent_id: str, **fields: Any) -> dict[str, Any]:
        body = {k: v for k, v in fields.items() if v is not None}
        return self._call("PUT", f"/api/v1/eval/agents/{agent_id}/repair-settings", json=body)

    def feedback_contract(self, agent_id: str, policy: str | None = None) -> dict[str, Any]:
        params = {"policy": policy} if policy else None
        return self._call("GET", f"/api/v1/eval/agents/{agent_id}/feedback-contract", params=params)


__all__ = ["EvalApiError", "EvalClient"]
