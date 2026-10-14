"""The voice server's page API, with a stand-in for the agent (no SenseLab, no Vapi)."""

import threading
from dataclasses import dataclass, field
from typing import Any

import pytest

pytest.importorskip("fastapi", reason="the voice server's tests need the voice extra: uv sync --extra voice")

from fastapi.testclient import TestClient  # noqa: E402

import voice.agent as agent  # noqa: E402
import voice.server as server  # noqa: E402
from voice.agent import Caller, caller_line


def test_the_decision_does_not_describe_a_stranger_with_a_known_accounts_defaults():
    # The same defaults reached Jev as "Caller on a consumer account, 14 months with us".
    assert caller_line(Caller(name="")) == "Caller not matched to any account (no history on file; a first contact)"
    known = caller_line(Caller(name="Tomas", tier="enterprise", tenure_months=39, customer_id="tomas-vega"))
    assert known == "Caller on a enterprise account, 39 months with us"


def test_a_stranger_gets_the_step_written_for_no_file():
    # Run 15: "resolve on the call" for a caller nobody matched went out as "say what the
    # charge is, from the caller's file", and the agent made one up ("an extra seat added
    # last month"). The step a stranger gets is the one written for no file.
    from types import SimpleNamespace

    b = server.Business.load()
    matched = b.then_for("resolve_in_bot", matched=True)
    assert matched.startswith("Answer it now, from the caller's file") and "If " not in matched
    stranger = b.then_for("resolve_in_bot", matched=False)
    assert stranger.startswith("No account matched this caller, so there is no file") and "do not guess" in stranger
    assert "the caller's file" not in stranger
    for action in ("transfer_to_agent", "schedule_callback", "issue_credit"):
        assert "No account matched this caller" in b.then_for(action, matched=False)
        assert "number on file" not in b.then_for(action, matched=False).replace("no number on file", "")
    assert b.then_for("made_up_action", matched=False) == ""  # an unknown action has no text either way
    d = SimpleNamespace(action="resolve_in_bot", then=stranger)
    result = server.tool_result(d, b, record="6 of 6 resolved on the call.")
    assert result.startswith("Decided: resolve on the call (resolve_in_bot). What happens now: No account matched")
    # The "nothing else" line is marked as an instruction, after a run in which it was read out.
    assert "Instructions for the assistant, not words for the caller: this is the one next step" in result
    assert "Why: 6 of 6 resolved on the call." in result


def test_variable_values_give_a_stranger_no_facts_to_repeat():
    # The Caller defaults (consumer, 14 months) are for a lookup that found an account. A
    # caller nobody matched must not hear them as facts about themselves.
    v = server.variable_values({"name": "", "tier": "consumer", "tenure_months": 14, "prior_contacts_30d": 0})
    assert v["name"] == "unknown" and v["history"] == ""
    assert v["account"].startswith("No account matched this caller") and "14" not in v["account"]
    v = server.variable_values({"name": "Bruno", "tier": "consumer", "tenure_months": 39, "prior_contacts_30d": 3, "history": "h"})
    assert v["account"] == "name Bruno, consumer plan, 3 contact(s) with support in the last 30 days, 39 months with us"
    assert v["history"] == "h"


@dataclass
class FakeAgent:
    calls: dict[str, Any] = field(default_factory=dict)
    notes: list[tuple[str, ...]] = field(default_factory=list)
    changed: bool = False

    def caller_from(self, m: dict[str, Any] | None) -> Caller:
        return Caller.from_mapping(m)

    def call_started(self, call_id: str, caller: Caller, *, call_type: str = "webCall", brief: str | None = None) -> None:
        self.calls[call_id] = {"caller": caller, "type": call_type, "brief": brief}

    def note_speech(self, call_id: str, role: str, status: str, *, t: float | None = None) -> None:
        self.notes.append(("speech", call_id, role, status, t))

    def note_transcript(self, call_id: str, role: str, text: str, *, t: float | None = None) -> None:
        self.notes.append(("transcript", call_id, role, text, t))


def test_page_registers_a_browser_call_and_relays_what_vapi_does_not_send(monkeypatch):
    fake = FakeAgent()
    monkeypatch.setattr(server, "agent", fake)
    monkeypatch.setitem(server.profiles, "p1", {"profile_id": "p1", "registered": 0, "name": "Bruno", "tier": "consumer",
                                                "prior_contacts_30d": 3, "tenure_months": 39, "customer_id": "bruno-andrade",
                                                "history": "the file", "first_message": "Hi Bruno"})
    c = TestClient(server.app)
    # The SDK has connected: the call is registered from the profile, history and all, and
    # the runs open ahead. On the classic stacks the status update that follows is a no-op.
    assert c.post("/api/calls/c1/start", json={"profile_id": "p1"}).json() == {"call_id": "c1", "known": False}
    assert fake.calls["c1"]["caller"].name == "Bruno" and fake.calls["c1"]["brief"] == "the file"
    assert c.post("/api/calls/c1/start", json={"profile_id": "p1"}).json()["known"] is True
    # An unknown profile is a stranger, not the defaults.
    c.post("/api/calls/c2/start", json={"profile_id": "nope"})
    assert fake.calls["c2"]["caller"].name == "" and fake.calls["c2"]["brief"] is None
    # Transcript and speech messages, in Vapi's shape; partial transcripts are dropped. The
    # page may say when it happened (GPT-Live: a stop is known only after the pause).
    c.post("/api/calls/c1/notes", json={"type": "transcript", "role": "user", "transcriptType": "partial", "transcript": "hel"})
    c.post("/api/calls/c1/notes", json={"type": "transcript", "role": "user", "transcriptType": "final", "transcript": "hello"})
    c.post("/api/calls/c1/notes", json={"type": "speech-update", "role": "assistant", "status": "started"})
    c.post("/api/calls/c1/notes", json={"type": "speech-update", "role": "assistant", "status": "stopped", "t": 1700000000.5})
    assert fake.notes == [("transcript", "c1", "user", "hello", None), ("speech", "c1", "assistant", "started", None),
                          ("speech", "c1", "assistant", "stopped", 1700000000.5)]
    assert c.get("/api/config").json()["stack"] == server.cfg.stack


class _Turns:
    """Just enough of a LiveAgent for note_turns: the calls, the lock, the event log."""

    def __init__(self, started: float) -> None:
        self.calls: dict[str, Any] = {"c1": {"id": "c1", "started": started, "live": []}}
        self._lock = threading.Lock()
        self.events: list[dict[str, Any]] = []

    def _event(self, kind: str, data: dict[str, Any]) -> None:
        self.events.append({"kind": kind, **data})


def test_turns_from_the_artifact_fill_in_what_the_page_did_not_relay():
    """On GPT-Live the caller's words reach the server only inside artifact.messages, with
    each tool call and at the end; the agent's own words the page relayed already. The rows
    become turn events at their real times, once each, and never repeat a line on the call."""
    ag = _Turns(started=1_700_000_000.0)
    ag.calls["c1"]["live"].append({"role": "assistant", "text": "Hi Bruno, it's John. What's going on today?", "t": 1_700_000_004.0})
    rows = [
        {"role": "bot", "message": "Hi Bruno, it's John.", "time": 1_700_000_000_500, "endTime": 1_700_000_002_000, "secondsFromStart": 0.5},
        {"role": "bot", "message": "What's going on today?", "time": 1_700_000_002_000, "endTime": 1_700_000_004_000, "secondsFromStart": 2.0},
        {"role": "user", "message": "You charged me twice.", "time": 1_700_000_005_000, "endTime": 1_700_000_007_000, "secondsFromStart": 5.0},
        {"role": "tool_calls", "message": "", "toolCalls": [{"id": "x"}]},
        {"role": "user", "message": "Hello?", "secondsFromStart": 9.0, "duration": 0.8},
    ]
    assert agent.LiveAgent.note_turns(ag, "c1", rows) == 2
    assert [e["kind"] for e in ag.events] == ["turn", "turn"]
    first, second = ag.events
    assert (first["role"], first["text"], first["t0"], first["t1"]) == ("user", "You charged me twice.", 1_700_000_005.0, 1_700_000_007.0)
    assert (second["role"], second["t0"], second["t1"]) == ("user", 1_700_000_009.0, 1_700_000_009.8)  # from secondsFromStart
    # The same rows again, plus one new one: only the new one is announced.
    rows.append({"role": "bot", "message": "Right, the two charges.", "time": 1_700_000_010_000, "endTime": 1_700_000_012_000})
    assert agent.LiveAgent.note_turns(ag, "c1", rows) == 1
    assert ag.events[-1]["text"] == "Right, the two charges." and len(ag.calls["c1"]["live"]) == 4
    assert agent.LiveAgent.note_turns(ag, "nope", rows) == 0 and agent.LiveAgent.note_turns(ag, "c1", None) == 0
