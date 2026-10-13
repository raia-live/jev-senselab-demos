"""The repair loop, driven on the SenseLab side, looked at from the harness.

Per run, once: a judge for the agent (the scenario's rubric) and the agent's repair
settings (policy and canary knobs). Per episode: ask the judge to grade the sealed trace
(in the background, so the decision loop keeps moving). Every ``look_every`` sessions:
what the hourly sweep would do for this one agent — evaluate the running canary, start
the oldest fix waiting for the slot, re-check shipped fixes — and snapshot the fix
lifecycle onto the episode row.

Nothing that decides runs here. Behaviour clustering, incidents, attribution, the repair
agent's draft, the Tier 1 replay, the canary's statistics, ship, verify and rollback are
the product's.
"""

from __future__ import annotations

import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from .eval_client import EvalApiError, EvalClient
from .scenario import Scenario

JUDGE_ID = "jev-demo-decision-failure"

JUDGE_PROMPT = (
    "You grade one run of a decision agent. The agent read a {case_noun}, consulted its memory, "
    "and chose one action; a downstream review then reported how the decision turned out. "
    "Decide whether the run FAILED and, if so, name the failure in one short phrase that would "
    "be the same for every run that fails the same way — name the kind of {case_noun} and the "
    "action taken, for example 'mid-market EMEA lead routed to AE and lost' or 'partial-shipment "
    "invoice held and released late'. A run failed when the review reported a failure of any "
    "severity; a run whose review reported success is a PASS. Use only what the trace shows: the "
    "task as posed, the decide call and its arguments, the memory entries the agent read, the "
    "decision summary, the response and the outcome. Quote the review line your verdict rests on.\n\n"
    "When the run failed, say where the wrong action came from. Compare the action taken with the "
    "memory entries under memory_read: if an entry recommends, records or endorses the action the "
    "agent took for this kind of {case_noun} (a policy line, a lesson that says the action worked), "
    "add a criterion named 'followed-memory' with passed=false whose evidence_quote is the exact "
    "text of that entry's value (copy it verbatim, up to one sentence) and whose span_id is null — "
    "the evidence is the memory entry, not a span; a span_id on this criterion is wrong. If no "
    "entry read supports the action, add 'followed-memory' with passed=true and quote the decide "
    "call instead. Do not grade tone or verbosity.\n\n"
    "Scenario notes: {rubric}\n\nTRACE:\n{{{{trace}}}}"
)

OPEN_STATUSES = ("proposed", "testing", "test_passed", "test_failed", "test_inconclusive", "canary")
SHIPPED_STATUSES = ("memory_shipped", "verified", "regressed")
INCONCLUSIVE_CANARY_PCT_CAP = 10


def _canary_ready(f: dict[str, Any]) -> bool:
    status = f.get("status")
    if status in ("test_passed", "test_inconclusive"):
        return True
    if status != "proposed":
        return False
    if f.get("tested_at"):
        return True
    manifest = f.get("manifest") if isinstance(f.get("manifest"), dict) else {}
    return (not f.get("case_set_id") and not manifest.get("predicted_fixes")
            and not manifest.get("predicted_regressions"))


def _replay_awaited(f: dict[str, Any]) -> bool:
    rep = f.get("replay") if isinstance(f.get("replay"), dict) else None
    return bool(rep and rep.get("status") == "requested")


def fix_row(f: dict[str, Any]) -> dict[str, Any]:
    """The lifecycle fields of one fix, compact enough for an episode row."""
    canary = f.get("canary") if isinstance(f.get("canary"), dict) else None
    test = f.get("test") if isinstance(f.get("test"), dict) else None
    manifest = f.get("manifest") if isinstance(f.get("manifest"), dict) else None
    row: dict[str, Any] = {
        "id": str(f.get("id")), "status": f.get("status"), "lever": f.get("lever"),
        "source": f.get("source"), "branch": f.get("branch"), "behavior_id": f.get("behavior_id"),
        "incident_id": f.get("incident_id"), "created_at": f.get("created_at"),
        "shipped_at": f.get("shipped_at"), "rolled_back_at": f.get("rolled_back_at"),
        "title": (f.get("title") or f.get("summary") or "")[:200] if isinstance(f.get("title") or f.get("summary"), str) else None,
    }
    if manifest:
        row["manifest"] = {
            "predicted_fixes": len(manifest.get("predicted_fixes") or []),
            "predicted_regressions": len(manifest.get("predicted_regressions") or []),
            "expected_prevention": manifest.get("expected_prevention"),
        }
    if test:
        row["test"] = {k: test.get(k) for k in ("status", "passed", "repaired", "total", "regressed", "fidelity", "unreplayable")}
    if canary:
        row["canary"] = {
            "status": canary.get("status"), "pct": canary.get("pct"), "min_outcomes": canary.get("min_outcomes"),
            "evaluations": canary.get("evaluations"), "recurrence_p": canary.get("recurrence_p"),
            "success_p": canary.get("success_p"), "reason": canary.get("reason"),
            "canary": {k: (canary.get("canary") or {}).get(k) for k in ("traces", "graded", "recurred", "outcomes", "succeeded", "read")},
            "control": {k: (canary.get("control") or {}).get(k) for k in ("traces", "graded", "recurred", "outcomes", "succeeded")},
        }
    return row


class RepairLoop:
    def __init__(
        self,
        api: EvalClient,
        *,
        agent_id: str,
        scenario: Scenario,
        policy: str = "auto_after_canary",
        canary_pct: int = 50,
        canary_min_outcomes: int = 5,
        canary_timeout_sessions: int = 60,
        look_every: int = 5,
        judge: bool = True,
        judge_mode: str = "continuous",
        judge_workers: int = 1,
    ) -> None:
        self.api = api
        self.agent_id = agent_id
        self.scenario = scenario
        self.policy = policy
        self.canary_pct = canary_pct
        self.canary_min_outcomes = canary_min_outcomes
        self.canary_timeout_sessions = canary_timeout_sessions
        self.look_every = look_every
        self.judge_on = judge
        # ``continuous``: the platform grades each trace as it seals, off the request path.
        # ``on_demand``: this process asks for each verdict (slower; it queues behind the
        # grader on the server). Continuous is what a deployment would use.
        self.judge_mode = judge_mode
        self.sessions_sealed = 0
        self._sessions_at_look = 0
        self._looks = 0
        self._canary_started_at: dict[str, int] = {}
        self._kicked: set[str] = set()
        self._first_seen_look: dict[str, int] = {}
        self.fixes_seen: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self.verdicts: list[dict[str, Any]] = []
        self._pool = ThreadPoolExecutor(max_workers=judge_workers) if judge else None
        self._pending: list[Future[Any]] = []
        self.settings: dict[str, Any] = {}
        self.contract: dict[str, Any] | None = None

    # -- once per run --------------------------------------------------------------------

    def ready(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        prompt = JUDGE_PROMPT.format(case_noun=self.scenario.case_noun, rubric=self.scenario.judge_rubric[:1500])
        judge = None
        if self.judge_on:
            judge = self.api.ensure_judge(JUDGE_ID, self.agent_id, "Decision failure (demo rubric)", prompt,
                                          mode=self.judge_mode, sampling_rate=1.0)
        self.settings = self.api.set_repair_settings(
            self.agent_id,
            repair_policy=self.policy,
            canary_pct=self.canary_pct,
            canary_min_outcomes=self.canary_min_outcomes,
            canary_timeout_days=1,
            auto_propose=True,
            replay_webhook_url="",
        )
        try:
            self.contract = self.api.feedback_contract(self.agent_id, self.policy)
        except EvalApiError:
            self.contract = None
        info = {
            "agent_id": self.agent_id, "judge": (judge or {}).get("id") if isinstance(judge, dict) else None,
            "settings": {k: self.settings.get(k) for k in ("repair_policy", "canary_pct", "canary_min_outcomes",
                                                            "canary_timeout_days", "auto_propose", "replay_fidelity_floor")},
            "contract": self.contract, "ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        self.events.append({"t": time.time(), "kind": "agent_ready", **info})
        return info

    # -- per episode ---------------------------------------------------------------------

    def after_seal(self, episode: int, trace_id: str | None, *, reported_success: bool) -> None:
        self.sessions_sealed += 1
        if not self.judge_on or not trace_id or self._pool is None or self.judge_mode != "on_demand":
            return
        fut = self._pool.submit(self._judge, episode, trace_id, reported_success)
        self._pending.append(fut)

    def _judge(self, episode: int, trace_id: str, reported_success: bool) -> dict[str, Any]:
        t0 = time.perf_counter()
        rec: dict[str, Any] = {"episode": episode, "trace_id": trace_id, "reported_success": reported_success}
        try:
            got = self.api.judge(trace_id, JUDGE_ID, agent_id=self.agent_id)
            # {"trace_id", "verdicts": [{verdict, category, score, confidence, cost_usd, judge_model, ...}], "skipped": [...]}
            verdicts = got.get("verdicts") if isinstance(got, dict) else None
            v = verdicts[0] if verdicts else (got if isinstance(got, dict) and "verdict" in got else {})
            rec["verdict"] = v.get("verdict")
            rec["behavior"] = v.get("category") or v.get("behavior") or v.get("failure_label")
            rec["score"] = v.get("score")
            rec["confidence"] = v.get("confidence")
            rec["judge_model"] = v.get("judge_model") or v.get("model")
            rec["cost_usd"] = v.get("cost_usd")
            if not verdicts and isinstance(got, dict) and got.get("skipped"):
                rec["skipped"] = str(got["skipped"])[:200]
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {exc}"[:200]
        rec["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        self.verdicts.append(rec)
        return rec

    def judge_backlog(self) -> int:
        return sum(1 for f in self._pending if not f.done())

    # -- the look schedule ---------------------------------------------------------------

    def look(self, episode: int, *, force: bool = False) -> dict[str, Any]:
        """List the agent's fixes; on a due look, advance the canary as the sweep would."""
        t0 = time.perf_counter()
        due = force or (self.sessions_sealed - self._sessions_at_look) >= self.look_every
        note: dict[str, Any] = {"look": bool(due), "sessions_sealed": self.sessions_sealed,
                                "judge_backlog": self.judge_backlog()}
        try:
            fixes = self.api.list_fixes(self.agent_id)
        except Exception as exc:  # noqa: BLE001
            note["fixes_error"] = f"{type(exc).__name__}: {exc}"[:200]
            note["ms"] = round((time.perf_counter() - t0) * 1000, 1)
            return note
        snapshot = [fix_row(f) for f in fixes]
        changed = []
        for row in snapshot:
            prev = self.fixes_seen.get(row["id"])
            if prev is None or prev.get("status") != row["status"] or \
                    (prev.get("canary") or {}).get("status") != (row.get("canary") or {}).get("status"):
                changed.append(row)
                self.events.append({"t": time.time(), "sessions": self.sessions_sealed, "episode": episode,
                                    "kind": "fix_status", "fix": row})
            self.fixes_seen[row["id"]] = row
            self._first_seen_look.setdefault(row["id"], self._looks)
        if due:
            self._sessions_at_look = self.sessions_sealed
            self._looks += 1
            note["calls"] = self._advance(fixes, episode)
        note["fixes"] = {s: sum(1 for r in snapshot if r["status"] == s) for s in sorted({str(r["status"]) for r in snapshot})}
        note["changed"] = changed[:6]
        active = next((r for r in snapshot if r["status"] == "canary"), None)
        if active:
            note["canary_active"] = {"id": active["id"], **(active.get("canary") or {})}
        shipped = [r["id"] for r in snapshot if r["status"] in SHIPPED_STATUSES]
        if shipped:
            note["shipped"] = shipped
        note["ms"] = round((time.perf_counter() - t0) * 1000, 1)
        return note

    def _advance(self, fixes: list[dict[str, Any]], episode: int) -> int:
        calls = 0

        def log(kind: str, fix_id: str, **data: Any) -> None:
            self.events.append({"t": time.time(), "sessions": self.sessions_sealed, "episode": episode,
                                "kind": kind, "fix_id": fix_id, **data})

        # A proposed fix whose Tier 1 test has not been queued after two looks: ask for it.
        for f in fixes:
            fid = str(f["id"])
            if f.get("status") == "proposed" and not f.get("tested_at") and fid not in self._kicked:
                manifest = f.get("manifest") if isinstance(f.get("manifest"), dict) else {}
                if (manifest.get("predicted_fixes") or f.get("case_set_id")) and \
                        self._looks - self._first_seen_look.get(fid, self._looks) >= 2:
                    self._kicked.add(fid)
                    try:
                        after = self.api.test_fix(fid)
                        calls += 1
                        log("test_kick", fid, status=after.get("status"))
                    except EvalApiError as exc:
                        calls += 1
                        log("test_kick_error", fid, error=f"{exc.status}: {str(exc.detail)[:200]}")

        running = [f for f in fixes if f.get("status") == "canary"]
        slot_busy = bool(running)
        for f in running:
            fid = str(f["id"])
            self._canary_started_at.setdefault(fid, self.sessions_sealed)
            age = self.sessions_sealed - self._canary_started_at[fid]
            try:
                if age >= self.canary_timeout_sessions:
                    after = self.api.stop_canary(fid, reason=f"demo timeout: {age} sessions")
                    log("canary_timeout", fid, age=age, status=after.get("status"))
                    slot_busy = False
                else:
                    after = self.api.evaluate_canary(fid)
                    c = after.get("canary") or {}
                    log("canary_evaluate", fid, status=after.get("status"), canary_status=c.get("status"),
                        recurrence_p=c.get("recurrence_p"), success_p=c.get("success_p"),
                        canary_n=(c.get("canary") or {}).get("outcomes"), control_n=(c.get("control") or {}).get("outcomes"),
                        canary_graded=(c.get("canary") or {}).get("graded"), control_graded=(c.get("control") or {}).get("graded"),
                        canary_read=(c.get("canary") or {}).get("read"), reason=c.get("reason"))
                    if after.get("status") != "canary":
                        slot_busy = False
                    if self.policy == "review" and after.get("status") == "test_passed" and c.get("status") == "passed":
                        shipped = self.api._call("POST", f"/api/v1/eval/fixes/{fid}/approve-memory")  # noqa: SLF001
                        calls += 1
                        log("ship_by_hand", fid, status=shipped.get("status"))
                calls += 1
            except EvalApiError as exc:
                log("canary_error", fid, error=f"{exc.status}: {str(exc.detail)[:200]}")
            except Exception as exc:  # noqa: BLE001
                log("canary_error", fid, error=f"{type(exc).__name__}: {exc}"[:200])

        if not slot_busy and self.policy != "auto_after_replay":
            waiting = [f for f in fixes if _canary_ready(f) and f.get("branch") and not f.get("canary")
                       and not _replay_awaited(f)]
            waiting.sort(key=lambda f: (str(f.get("tested_at") or "9"), str(f.get("created_at") or "")))
            for f in waiting[:1]:
                fid = str(f["id"])
                # The demo asks for its configured split whatever the Tier 1 verdict, so the
                # loop can close inside one run; the server may cap an inconclusive fix at
                # INCONCLUSIVE_CANARY_PCT_CAP, and the response says what it granted.
                pct = self.canary_pct
                try:
                    after = self.api.start_canary(fid, pct=pct, min_outcomes=self.canary_min_outcomes, timeout_days=1)
                    calls += 1
                    self._canary_started_at[fid] = self.sessions_sealed
                    granted = (after.get("canary") or {}).get("pct") if isinstance(after.get("canary"), dict) else None
                    log("canary_start", fid, status=after.get("status"), pct=pct, granted_pct=granted,
                        from_status=f.get("status"))
                except EvalApiError as exc:
                    calls += 1
                    log("canary_start_refused", fid, error=f"{exc.status}: {str(exc.detail)[:200]}")

        for f in fixes:
            if f.get("status") == "memory_shipped":
                fid = str(f["id"])
                try:
                    after = self.api.recheck(fid)
                    calls += 1
                    prev = after.get("prevention") if isinstance(after.get("prevention"), dict) else None
                    log("recheck", fid, status=after.get("status"), prevention=prev)
                except EvalApiError as exc:
                    calls += 1
                    log("recheck_error", fid, error=f"{exc.status}: {str(exc.detail)[:200]}")
        return calls

    # -- end of run ----------------------------------------------------------------------

    def drain(self, timeout_s: float = 600.0) -> int:
        """Wait for the judge backlog. Returns how many verdicts were still pending."""
        if not self.judge_on:
            return 0
        deadline = time.time() + timeout_s
        if self.judge_mode != "on_demand":
            # The platform grades in the background; give it a little while to catch up so
            # the final look and the report see every verdict.
            while time.time() < deadline:
                graded = self._server_verdicts()
                if len(graded) >= self.sessions_sealed:
                    return 0
                time.sleep(10)
            return max(0, self.sessions_sealed - len(self._server_verdicts()))
        if self._pool is None:
            return 0
        for fut in self._pending:
            remaining = max(0.0, deadline - time.time())
            try:
                fut.result(timeout=remaining)
            except Exception:  # noqa: BLE001
                pass
        return self.judge_backlog()

    def _server_verdicts(self) -> list[dict[str, Any]]:
        """Every verdict the platform holds for this agent (continuous mode)."""
        items: list[dict[str, Any]] = []
        cursor = None
        for _ in range(10):
            try:
                page = self.api.verdicts(self.agent_id, limit=200, cursor=cursor)
            except Exception:  # noqa: BLE001
                break
            items += page.get("items") or []
            cursor = page.get("next_cursor")
            if not cursor:
                break
        return items

    def _verdict_summary(self) -> dict[str, Any]:
        if self.judge_mode == "on_demand":
            rows = self.verdicts
            def reported(v: dict[str, Any]) -> bool:
                return bool(v.get("reported_success"))
        else:
            rows = self._server_verdicts()
            self.verdicts = [{k: v.get(k) for k in ("trace_id", "verdict", "category", "score", "confidence",
                                                     "judge_model", "cost_usd", "agent_self_reported", "created_at")}
                             for v in rows]
            def reported(v: dict[str, Any]) -> bool:
                return str(v.get("agent_self_reported") or "").lower() == "success"
        graded = [v for v in rows if v.get("verdict") is not None]
        return {
            "graded": len(graded),
            "failed": sum(1 for v in graded if str(v.get("verdict")).lower() in ("fail", "failed", "false")),
            "errors": sum(1 for v in rows if v.get("error")),
            "judge_usd": round(sum(float(v.get("cost_usd") or 0) for v in rows), 4),
            "disagreements": sum(1 for v in graded if (str(v["verdict"]).lower() == "pass") != reported(v)),
            "categories": dict(Counter(str(v.get("category") or v.get("behavior") or "") for v in graded
                                       if str(v.get("verdict")).lower() in ("fail", "failed", "false")).most_common(8)),
        }

    def report(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "agent_id": self.agent_id, "policy": self.policy,
            "canary": {"pct": self.canary_pct, "min_outcomes": self.canary_min_outcomes,
                       "timeout_sessions": self.canary_timeout_sessions, "look_every": self.look_every},
            "sessions_sealed": self.sessions_sealed, "events": self.events, "judge_mode": self.judge_mode,
        }
        out["verdict_summary"] = self._verdict_summary()
        out["verdicts"] = self.verdicts
        try:
            fixes = self.api.list_fixes(self.agent_id)
            full = []
            for f in fixes:
                row = dict(f)
                for name, fn in (("events", lambda fid: (self.api.fix_events(fid) or {}).get("items")),
                                 ("tests", lambda fid: (self.api.fix_tests(fid) or {}).get("items"))):
                    try:
                        row[name] = fn(str(f["id"]))
                    except EvalApiError:
                        pass
                if f.get("status") in SHIPPED_STATUSES:
                    try:
                        row["prevention"] = self.api.prevention(str(f["id"]))
                    except EvalApiError:
                        pass
                full.append(row)
            out["fixes"] = full
            out["lifecycle"] = {s: sum(1 for f in fixes if f.get("status") == s) for s in sorted({str(f.get("status")) for f in fixes})}
        except Exception as exc:  # noqa: BLE001
            out["fixes_error"] = f"{type(exc).__name__}: {exc}"[:200]
        for name, fn in (("incidents", lambda: self.api.incidents(self.agent_id)),
                         ("behaviors", lambda: self.api.behaviors(self.agent_id)),
                         ("contract", lambda: self.api.feedback_contract(self.agent_id, self.policy)),
                         ("repairs_summary", lambda: self.api.repairs_summary(self.agent_id))):
            try:
                out[name] = fn()
            except Exception as exc:  # noqa: BLE001
                out[f"{name}_error"] = f"{type(exc).__name__}: {exc}"[:200]
        return out

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=False)


__all__ = ["JUDGE_ID", "JUDGE_PROMPT", "OPEN_STATUSES", "SHIPPED_STATUSES", "RepairLoop", "fix_row"]
