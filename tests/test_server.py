"""The voice server's page API, with a stand-in for the agent (no SenseLab, no Vapi)."""

import asyncio
import threading
import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("fastapi", reason="the voice server's tests need the voice extra: uv sync --extra voice")

from fastapi.testclient import TestClient  # noqa: E402

import voice.agent as agent  # noqa: E402
import voice.server as server  # noqa: E402
from voice.agent import Caller, caller_line
from voice.business import Business


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
    assert "Before the decision" not in result
    # The caller rang back about the same thing: the result says so before the new step, so
    # the agent owns it instead of telling them the earlier step went fine.
    back = ("They are back about the same thing, two minutes after the last step we took for them (booked a specialist "
            "callback). That step is on their record as not working, on that alone.")
    d = SimpleNamespace(action="transfer_to_agent", then="A colleague takes it.", rang_back=back)
    result = server.tool_result(d, b, record="r")
    assert f"Why: r Before the decision: {back} Own it in one line" in result
    result = server.tool_result(d, b, record="r", stack="realtime")
    assert f"Why, in plain words and without the numbers: r First: {back} Say in one line" in result


def test_one_turns_tool_calls_run_in_a_fixed_order():
    # The reasoner may list decide_next_step before update_file in one turn. The refusal
    # on file has to close the step it refuses before the new decision opens the next one.
    calls = [{"id": "3", "function": {"name": "decide_next_step"}}, {"id": "1", "name": "identify_caller"},
             {"id": "2", "function": {"name": "update_file"}}, {"id": "4", "function": {"name": "endCall"}},
             {"id": "5", "function": {"name": "update_file"}}]
    assert [tc["id"] for tc in server.tool_order(calls)] == ["1", "2", "5", "3", "4"]
    assert server.tool_order([]) == []


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

    def call_started(self, call_id: str, caller: Caller, *, call_type: str = "webCall", brief: str | None = None,
                     known_by: str | None = None) -> None:
        self.calls[call_id] = {"caller": caller, "type": call_type, "brief": brief, "known_by": known_by}

    @property
    def phonebook(self) -> Any:
        from voice.phonebook import PhoneBook

        if not hasattr(self, "_book"):
            self._book = PhoneBook(_Store(), "john", "jev-demos/voice")
        return self._book

    def update_file(self, call_id: str, params: dict[str, Any]) -> str:
        self.notes.append(("file", call_id, params.get("kind"), params.get("value")))
        return "On file: phone number 650 555 0199."

    def decide(self, call_id: str, params: dict[str, Any], *, caller: Caller | None = None) -> Any:
        self.notes.append(("decide", call_id, params.get("intent")))
        return SimpleNamespace(action="schedule_callback", situation="billing_dispute/consumer/repeat/small",
                               then="A billing specialist will call you within one business day.")

    def record_words(self, call_id: str, situation: str) -> str:
        return (f"For callers in this situation ({situation}), what has worked: a scheduled specialist callback (6 of 7). "
                "This caller's own record with these steps: booked a specialist callback 2 times (most recently 3 minutes ago); "
                "it did not hold 1 time, on their word, 1 still open.")

    def identify(self, call_id: str, params: dict[str, Any]) -> tuple[str, Caller | None]:
        if params.get("name") == "Bruno":
            who = Caller(name="Bruno", tier="consumer", prior_contacts_30d=3, tenure_months=39, customer_id="bruno-andrade")
            self.calls[call_id]["caller"] = who
            return "Matched: Bruno. the file", who
        return "No account matches the name", None

    def note_speech(self, call_id: str, role: str, status: str, *, t: float | None = None) -> None:
        self.notes.append(("speech", call_id, role, status, t))

    def note_transcript(self, call_id: str, role: str, text: str, *, t: float | None = None) -> None:
        self.notes.append(("transcript", call_id, role, text, t))

    def intro_for(self, caller: Caller) -> tuple[str, str]:
        return ("", f"Hi {caller.name}".strip())

    def set_listening(self, call_id: str, on: bool = True) -> None:
        self.calls[call_id]["listening"] = on

    def is_listening(self, call_id: str) -> bool:
        return bool(self.calls.get(call_id, {}).get("listening"))


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
    # Once the server follows the call's audio, who is speaking comes from there; the page's
    # speech notes are ignored and its words still land.
    fake.set_listening("c1")
    assert c.post("/api/calls/c1/notes", json={"type": "speech-update", "role": "user", "status": "started"}).json() == {"ignored": "the call's audio says who is speaking"}
    c.post("/api/calls/c1/notes", json={"type": "transcript", "role": "user", "transcriptType": "final", "transcript": "still here"})
    assert fake.notes[-1] == ("transcript", "c1", "user", "still here", None)


def test_the_server_follows_the_calls_audio_when_vapi_says_where(monkeypatch):
    """Every webhook message carries the call record; its monitor.listenUrl is where the
    audio streams. The listener starts once per call, from whichever message comes first
    (a phone call's assistant-request, a status update, or a tool call the server first
    hears of the call in), and is cancelled by the end-of-call report."""
    fake = FakeAgent()
    monkeypatch.setattr(server, "agent", fake)
    monkeypatch.setattr(server.cfg, "listen", True)
    monkeypatch.setattr(server.cfg, "assistant_id", "asst")
    monkeypatch.setattr(server.cfg, "webhook_secret", None)  # whatever the shell's .env says
    monkeypatch.setattr(server.cfg, "private_key", None)
    started: list[tuple[str, str]] = []

    async def fake_listen(call_id, url, on_speech, *, on_open=None):
        started.append((call_id, url))
        if on_open:
            on_open()
        on_speech("user", "started", 1_700_000_001.0)
        await asyncio.sleep(3600)  # until the call ends

    monkeypatch.setattr(server.li, "listen", fake_listen)
    server.listeners.clear()
    fake.end_call_later = lambda *a, **k: None  # type: ignore[attr-defined]
    fake.note_turns = lambda *a, **k: 0  # type: ignore[attr-defined]
    call = {"id": "p1", "type": "inboundPhoneCall", "monitor": {"listenUrl": "wss://listen/p1", "controlUrl": "wss://control/p1"}}
    monkeypatch.setattr(server.app.router, "on_startup", [])  # no real agent, no SenseLab
    monkeypatch.setattr(server.app.router, "on_shutdown", [])
    with TestClient(server.app) as c:  # one event loop for the whole call, as in the server
        c.post("/vapi/webhook", json={"message": {"type": "assistant-request", "call": call}})
        c.post("/vapi/webhook", json={"message": {"type": "status-update", "status": "in-progress", "call": call}})
        c.post("/vapi/webhook", json={"message": {"type": "tool-calls", "call": call, "toolCallList": []}})
        assert started == [("p1", "wss://listen/p1")]  # once, not three times
        assert fake.calls["p1"]["listening"] is True and ("speech", "p1", "user", "started", 1_700_000_001.0) in fake.notes
        # A call with no listen URL is left alone; a server told not to listen is too.
        c.post("/vapi/webhook", json={"message": {"type": "status-update", "status": "in-progress", "call": {"id": "p2", "type": "webCall"}}})
        monkeypatch.setattr(server.cfg, "listen", False)
        c.post("/vapi/webhook", json={"message": {"type": "status-update", "status": "in-progress", "call": {**call, "id": "p3"}}})
        assert [cid for cid, _ in started] == ["p1"] and set(server.listeners) == {"p1"}
        monkeypatch.setattr(server.cfg, "listen", True)
        # The end-of-call report stops it.
        c.post("/vapi/webhook", json={"message": {"type": "end-of-call-report", "call": call, "endedReason": "customer-ended-call"}})
        assert "p1" not in server.listeners
        # A browser call: the page knows only the id, so the record is fetched for where the
        # audio streams. The URL appears once the call is in progress, so the lookup is
        # retried a few times; a record that never gains one is logged and let be.
        records = {"w1": [{"id": "w1", "monitor": {"listenUrl": "wss://listen/w1"}}],
                   "w2": [{"id": "w2"}, {"id": "w2", "monitor": {"listenUrl": "wss://listen/w2"}}],
                   "w3": [{"id": "w3"}]}
        fetched: list[str] = []

        def fake_fetch(_key: str, cid: str) -> dict[str, Any]:
            fetched.append(cid)
            return records[cid].pop(0) if len(records[cid]) > 1 else records[cid][0]

        monkeypatch.setattr(server, "LOOKUP_AFTER_S", (0.01, 0.01, 0.01))
        monkeypatch.setattr(server.cfg, "private_key", "k")
        monkeypatch.setattr(server.va, "fetch_call", fake_fetch)
        c.post("/api/calls/w1/start", json={"profile_id": "nope"})
        c.post("/api/calls/w1/start", json={"profile_id": "nope"})
        c.post("/api/calls/w2/start", json={"profile_id": "nope"})
        c.post("/api/calls/w3/start", json={"profile_id": "nope"})
        for _ in range(100):  # the lookups run off the request's path
            if fetched.count("w3") == 3 and ("w2", "wss://listen/w2") in started:
                break
            time.sleep(0.02)
        assert fetched.count("w1") == 1 and fetched.count("w2") == 2 and fetched.count("w3") == 3
        assert ("w1", "wss://listen/w1") in started and ("w2", "wss://listen/w2") in started and "w3" not in server.listeners


def test_a_phone_number_is_remembered_once_we_know_who_it_is(monkeypatch):
    from voice.phonebook import PhoneBook

    fake = FakeAgent()
    monkeypatch.setattr(server, "agent", fake)
    monkeypatch.setattr(server.cfg, "assistant_id", "asst")
    monkeypatch.setattr(server.cfg, "webhook_secret", None)
    monkeypatch.setattr(server.cfg, "listen", False)
    store = _Store()
    fake._book = PhoneBook(store, "john", "jev-demos/voice")
    server.profiles.clear()
    fake.note_turns = lambda *a, **k: 0  # type: ignore[attr-defined]
    c = TestClient(server.app)
    ring = lambda cid, number: {"message": {"type": "assistant-request", "call": {  # noqa: E731
        "id": cid, "type": "inboundPhoneCall", "customer": {"number": number}}}}
    # A number we have never met, nobody registered on the page: a stranger.
    r = c.post("/vapi/webhook", json=ring("p1", "+15550001111")).json()
    assert fake.calls["p1"]["caller"].name == "" and fake.calls["p1"]["known_by"] is None
    assert r["assistantOverrides"]["variableValues"]["name"] == "unknown"
    # The caller says who they are; the name matches, the call is theirs, the number is now them.
    c.post("/vapi/webhook", json={"message": {"type": "tool-calls", "call": {"id": "p1", "customer": {"number": "+15550001111"}},
                                              "toolCallList": [{"id": "t1", "function": {"name": "identify_caller", "arguments": {"name": "Bruno"}}}]}})
    assert fake.phonebook.who("+15550001111") == ({"name": "Bruno", "tier": "consumer", "prior_contacts_30d": 3, "tenure_months": 39, "customer_id": "bruno-andrade"}, "number")
    assert [e for e, _k, _v in store.writes] == ["jev-demos/voice/phonebook"] and "+1555" not in str(store.writes)  # hashed
    # Next call from that number, nothing registered on the page: known from the number.
    r = c.post("/vapi/webhook", json=ring("p2", "+15550001111")).json()
    assert fake.calls["p2"]["caller"].name == "Bruno" and fake.calls["p2"]["known_by"] == "number"
    assert r["assistantOverrides"]["variableValues"]["name"] == "Bruno"
    # A caller registered on the page in the last ten minutes wins (the demo switches
    # callers that way), and teaches the book the new name for the number.
    server.profiles["q"] = {"profile_id": "q", "registered": time.time(), "name": "Priya", "tier": "business",
                            "prior_contacts_30d": 1, "tenure_months": 8, "customer_id": "priya-raman",
                            "history": "the file", "first_message": "Hi Priya"}
    c.post("/vapi/webhook", json=ring("p3", "+15550001111"))
    assert fake.calls["p3"]["caller"].name == "Priya" and fake.calls["p3"]["known_by"] == "page"
    assert fake.phonebook.who("+15550001111")[0]["customer_id"] == "priya-raman"
    # A stranger registered on purpose (the test calls) stays a stranger, and is not
    # written into the book; another number is still nobody.
    server.profiles["s"] = {"profile_id": "s", "registered": time.time() + 1, "name": "", "tier": "consumer",
                            "prior_contacts_30d": 0, "tenure_months": 0, "customer_id": "", "history": "", "first_message": "Hi"}
    c.post("/vapi/webhook", json=ring("p4", "+15550001111"))
    assert fake.calls["p4"]["caller"].name == "" and fake.phonebook.who("+15550001111")[0]["customer_id"] == "priya-raman"
    server.profiles.clear()
    c.post("/vapi/webhook", json=ring("p5", "+15550002222"))
    assert fake.calls["p5"]["caller"].name == ""
    # A detail for the file goes to the agent, and its words go back to the reasoner.
    r = c.post("/vapi/webhook", json={"message": {"type": "tool-calls", "call": {"id": "p2", "customer": {"number": "+15550001111"}},
                                                  "toolCallList": [{"id": "t3", "function": {"name": "update_file", "arguments": {"kind": "phone", "value": "650 555 0199"}}}]}}).json()
    assert ("file", "p2", "phone", "650 555 0199") in fake.notes and r["results"][0]["result"].startswith("On file: phone number")
    # A name nobody has leaves the call and the book alone.
    c.post("/vapi/webhook", json={"message": {"type": "tool-calls", "call": {"id": "p5", "customer": {"number": "+15550002222"}},
                                              "toolCallList": [{"id": "t2", "function": {"name": "identify_caller", "arguments": {"name": "Sam Ortiz"}}}]}})
    assert fake.phonebook.who("+15550002222") == ({}, None)
    # The book is read back from SenseLab at startup.
    again = PhoneBook(store, "john", "jev-demos/voice")
    assert again.load() == 1 and again.who("+1 (555) 000-1111")[0]["name"] == "Priya"


class _Store:
    """A stand-in for the SenseLab client: writes kept, listed back as entries."""

    def __init__(self) -> None:
        self.writes: list[tuple[str, str, dict[str, Any]]] = []

    def memory(self, _agent_id: str) -> "_Store":
        return self

    def write(self, entity: str, key: str, value: dict[str, Any], **_kw: Any) -> None:
        self.writes.append((entity, key, value))

    def list(self, entity: str) -> list[Any]:
        latest: dict[str, Any] = {}
        for e, k, v in self.writes:
            if e == entity:
                latest[k] = type("E", (), {"key": k, "value": v})()
        return list(latest.values())


class _Turns:
    """Just enough of a LiveAgent for note_turns: the calls, the lock, the event log."""

    def __init__(self, started: float) -> None:
        self.calls: dict[str, Any] = {"c1": {"id": "c1", "started": started, "live": []}}
        self._lock = threading.Lock()
        self.events: list[dict[str, Any]] = []

    def _event(self, kind: str, data: dict[str, Any]) -> None:
        self.events.append({"kind": kind, **data})


def test_the_decisions_answer_carries_the_callers_own_record(monkeypatch):
    # The "why" the speaker gets is the situation's record and this caller's own: a step
    # that already failed them is said as such, not as what works for most people.
    fake = FakeAgent()
    monkeypatch.setattr(server, "agent", fake)
    monkeypatch.setattr(server.cfg, "webhook_secret", None)
    monkeypatch.setattr(server.cfg, "listen", False)
    monkeypatch.setattr(server.cfg, "tool_floor_ms", 0)
    fake.note_turns = lambda *a, **k: 0  # type: ignore[attr-defined]
    fake.business = Business.load()  # type: ignore[attr-defined]
    fake.calls["w1"] = {"caller": Caller(name="Bruno", customer_id="bruno-andrade")}
    c = TestClient(server.app)
    r = c.post("/vapi/webhook", json={"message": {"type": "tool-calls", "call": {"id": "w1"}, "toolCallList": [
        {"id": "t9", "function": {"name": "decide_next_step", "arguments": {"intent": "billing_dispute", "caller_said": "nobody called"}}}]}}).json()
    out = r["results"][0]["result"]
    assert out.startswith("Decided: schedule a callback (schedule_callback). What happens now: A billing specialist")
    assert "Why: For callers in this situation" in out and "did not hold 1 time, on their word" in out
    assert fake.notes[-1] == ("decide", "w1", "billing_dispute")


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
