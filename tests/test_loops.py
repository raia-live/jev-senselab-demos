"""The repair loop watcher: what it asks the platform for, and when."""

from __future__ import annotations

from typing import Any

from core.loops import RepairLoop, fix_covers, fix_row
from demos import load


class FakeEval:
    """Enough of the eval API for the watcher: verdicts in, proposals out."""

    def __init__(self, verdicts: list[dict[str, Any]], fixes: list[dict[str, Any]] | None = None) -> None:
        self._verdicts = verdicts
        self._fixes = list(fixes or [])
        self.proposed: list[str] = []
        self.calls: list[str] = []

    def verdicts(self, agent_id: str, *, limit: int = 200, **_: Any) -> dict[str, Any]:
        self.calls.append("verdicts")
        return {"items": self._verdicts}

    def propose_fix(self, agent_id: str, *, verdict_id: str | None = None, behavior_id: str | None = None) -> dict[str, Any]:
        self.proposed.append(verdict_id or "")
        fix = {"id": f"fix-{len(self.proposed)}", "status": "testing", "lever": "knowledge", "source": "repair_agent",
               "branch": "repair/x", "correction": {"key": "policy-billing-small", "value": "rewritten", "replaces_version": 3}}
        self._fixes.append(fix)
        return fix

    def list_fixes(self, agent_id: str, *, limit: int = 200) -> list[dict[str, Any]]:
        return list(self._fixes)


def verdict(vid: str, key: str | None, *, fail: bool = True, at: str = "2026-10-14T09:00:00Z") -> dict[str, Any]:
    ev = [{"kind": "entry", "key": key, "entity_path": "e"}] if key else [{"kind": "span", "quote": "x"}]
    return {"id": vid, "verdict": "fail" if fail else "pass", "evidence": ev, "created_at": at}


def loop(api: Any, **kw: Any) -> RepairLoop:
    return RepairLoop(api, agent_id="a", scenario=load("call_handling"), judge=False, **kw)  # type: ignore[arg-type]


def test_two_failures_pinned_on_one_entry_ask_for_a_fix_once() -> None:
    api = FakeEval([
        verdict("v1", "policy-billing-small", at="2026-10-14T09:00:00Z"),
        verdict("v2", "policy-billing-small", at="2026-10-14T09:05:00Z"),
        verdict("v3", "policy-cancellation"),                 # one failure: not yet
        verdict("v4", "policy-outage", fail=False),           # a pass cites entries too; not a failure
        verdict("v5", None), verdict("v6", None),             # failures the judge pinned on nothing
    ])
    lp = loop(api)
    events: list[dict[str, Any]] = []
    lp._propose(api.list_fixes("a"), api.verdicts("a")["items"], lambda kind, fid, **d: events.append({"kind": kind, "fix": fid, **d}))
    assert api.proposed == ["v2"]  # the latest of the two, once
    assert events[-1]["kind"] == "propose" and events[-1]["key"] == "policy-billing-small" and events[-1]["failures"] == 2
    # Asked again with the fix now open for that key: nothing new, and nothing for the single failure.
    lp._propose(api.list_fixes("a"), api.verdicts("a")["items"], lambda *a, **k: None)
    assert api.proposed == ["v2"]


def test_an_open_or_shipped_fix_for_the_entry_stops_a_second_proposal() -> None:
    had = {"id": "f0", "status": "memory_shipped", "correction": {"key": "policy-billing-small", "value": "v"}}
    two = [verdict("v1", "policy-billing-small"), verdict("v2", "policy-billing-small", at="2026-10-14T09:05:00Z")]
    api = FakeEval(two, fixes=[had])
    lp = loop(api)
    lp._propose(api.list_fixes("a"), api.verdicts("a")["items"], lambda *a, **k: None)
    assert api.proposed == []
    # A dismissed one does not count as cover.
    api2 = FakeEval(two, fixes=[{**had, "status": "dismissed"}])
    loop(api2)._propose(api2.list_fixes("a"), api2.verdicts("a")["items"], lambda *a, **k: None)
    assert api2.proposed == ["v2"]


def test_a_draft_the_canary_stopped_is_tried_again_only_on_failures_graded_after_it() -> None:
    held = {"id": "f0", "status": "test_passed", "created_at": "2026-10-14T09:10:00Z",
            "canary": {"status": "failed", "pct": 50}, "correction": {"key": "policy-billing-small", "value": "transfer"}}
    before = [verdict("v1", "policy-billing-small", at="2026-10-14T09:00:00Z"),
              verdict("v2", "policy-billing-small", at="2026-10-14T09:05:00Z")]
    api = FakeEval(list(before), fixes=[held])
    lp = loop(api)
    lp._propose(api.list_fixes("a"), api.verdicts("a")["items"], lambda *a, **k: None)
    assert api.proposed == []  # the same two failures already produced that draft
    api._verdicts += [verdict("v3", "policy-billing-small", at="2026-10-14T09:20:00Z")]
    lp._propose(api.list_fixes("a"), api.verdicts("a")["items"], lambda *a, **k: None)
    assert api.proposed == []  # one failure since the draft: not yet
    api._verdicts += [verdict("v4", "policy-billing-small", at="2026-10-14T09:25:00Z")]
    lp._propose(api.list_fixes("a"), api.verdicts("a")["items"], lambda *a, **k: None)
    assert api.proposed == ["v4"]
    # The new draft is open now and covers the entry; a replay-test failure uncovers it again.
    lp._propose(api.list_fixes("a"), api.verdicts("a")["items"], lambda *a, **k: None)
    assert api.proposed == ["v4"]
    assert not fix_covers({"status": "test_failed", "correction": {"key": "k"}})
    assert fix_covers({"status": "canary", "canary": {"status": "running"}})
    assert fix_covers({"status": "memory_shipped", "canary": {"status": "passed"}})


def test_propose_after_zero_turns_it_off() -> None:
    api = FakeEval([verdict("v1", "policy-billing-small"), verdict("v2", "policy-billing-small")])
    loop(api, propose_after=0)._propose([], api.verdicts("a")["items"], lambda *a, **k: None)
    assert api.proposed == []


def test_a_failure_the_continuous_judge_skipped_is_asked_for_by_name() -> None:
    api = FakeEval([verdict("v1", "policy-billing-small")])
    lp = loop(api, look_every=2)
    lp.judge_on = True
    lp._pool = type("Pool", (), {"submit": lambda self, fn, *a: a})()  # type: ignore[assignment]
    lp.after_seal(0, "t-graded", reported_success=False)
    lp.after_seal(1, "t-skipped", reported_success=False)
    lp.after_seal(2, "t-fine", reported_success=True)
    graded = [{**verdict("v9", None), "trace_id": "t-graded"}]
    lp._judge_missing(graded)   # one session since the seal: not yet
    assert lp._asked == set()
    lp.after_seal(3, "t-later", reported_success=False)
    lp._judge_missing(graded)
    assert lp._asked == {"t-skipped"} and "t-graded" not in lp._unjudged and "t-later" in lp._unjudged


def test_fix_row_carries_the_memory_change() -> None:
    row = fix_row({"id": "f", "status": "testing", "lever": "knowledge", "situation": "billing_dispute/consumer/repeat/small",
                   "correction": {"key": "policy-billing-small", "entity_path": "e", "value": "new text", "replaces_version": 7},
                   "manifest": {"predicted_fixes": ["a"], "predicted_regressions": [], "rationale": "why"}})
    assert row["correction"] == {"key": "policy-billing-small", "entity_path": "e", "value": "new text", "replaces_version": 7}
    assert row["manifest"]["rationale"] == "why" and row["situation"].endswith("/small")
