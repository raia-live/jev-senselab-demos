"""The test-call harness takes its scenarios from the business the server is running, and
turns each into one Vapi simulation. No server, no Vapi: both are stubbed."""

from typing import Any

import pytest

from voice import testcalls
from voice.business import Business, CallScenario

BUSINESS = Business.load().public()


def test_scenarios_come_from_the_running_business():
    scenarios = testcalls.scenarios_of(BUSINESS)
    assert [s.id for s in scenarios] == ["bruno", "stranger", "priya", "tomas", "marcus"]
    assert all(isinstance(s, CallScenario) for s in scenarios)
    assert [s.id for s in testcalls.by_id(["priya", "marcus"], scenarios)] == ["priya", "marcus"]
    assert testcalls.by_id(None, scenarios) is scenarios
    with pytest.raises(SystemExit, match="no scenario 'nope'"):
        testcalls.by_id(["nope"], scenarios)
    with pytest.raises(SystemExit, match=r"no \[\[scenarios\]\]"):
        testcalls.scenarios_of({**BUSINESS, "scenarios": []})


class _Api:
    def __init__(self) -> None:
        self.sent: list[tuple[str, str, dict[str, Any]]] = []

    def _call(self, method: str, path: str, **kw: Any) -> dict[str, Any]:
        self.sent.append((method, path, kw.get("json") or {}))
        return {"id": "run-1", "url": "https://dashboard.vapi.ai/simulations/run/run-1"}


def test_a_scenario_becomes_one_simulation_with_its_judge(monkeypatch):
    """Each TOML scenario is one simulation: the tester gets the script as its instructions,
    the judge entries become required boolean structured outputs, and the call is placed as
    the persona's twin (or as a stranger) with the opening the server would give that caller."""
    posted: list[tuple[str, dict[str, Any]]] = []

    class _Resp:
        def __init__(self, body: dict[str, Any]) -> None:
            self._body = body

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, Any]:
            return self._body

    def fake_post(url: str, json: dict[str, Any], timeout: float) -> _Resp:
        posted.append((url, json))
        if url.endswith("/api/twins"):
            return _Resp({})
        who = "a stranger" if json.get("stranger") else json["name"]
        return _Resp({"first_message": f"Hi {who}", "variable_values": {"profile_id": f"p-{who}", "name": who}})

    monkeypatch.setattr(testcalls.httpx, "post", fake_post)
    api = _Api()
    scenarios = testcalls.by_id(["bruno", "stranger"], testcalls.scenarios_of(BUSINESS))
    profile_ids: dict[str, str] = {}
    run = testcalls.start_run(api, "asst-1", "http://srv", BUSINESS, scenarios, chat=False, iterations=1, profile_ids=profile_ids)  # type: ignore[arg-type]
    assert run["id"] == "run-1"
    # Bruno calls as his twin, seeded first; the stranger registers as a stranger.
    assert posted[0] == ("http://srv/api/twins", {"persona_id": "bruno-andrade", "customer_id": "bruno-andrade-sim"})
    assert posted[1][1]["customer_id"] == "bruno-andrade-sim" and posted[1][1]["name"] == "Bruno"
    assert posted[2] == ("http://srv/api/profiles", {"stranger": True})
    assert profile_ids == {"jev-demo: bruno": "p-Bruno", "jev-demo: stranger": "p-a stranger"}
    method, path, body = api.sent[0]
    assert (method, path) == ("POST", "/eval/simulation/run")
    assert body["target"] == {"type": "assistant", "assistantId": "asst-1"} and body["transport"] == {"provider": "vapi.websocket"}
    bruno, stranger = body["simulations"]
    assert bruno["scenario"]["instructions"] == scenarios[0].script and bruno["scenario"]["instructions"].startswith("You are Bruno")
    assert bruno["scenario"]["targetOverrides"] == {"firstMessage": "Hi Bruno", "variableValues": {"profile_id": "p-Bruno", "name": "Bruno"}}
    evals = bruno["scenario"]["evaluations"]
    assert [e["structuredOutput"]["name"] for e in evals] == [n for n, _ in scenarios[0].judge]
    assert all(e["structuredOutput"]["schema"] == {"type": "boolean"} and e["value"] is True and e["required"] for e in evals)
    assert stranger["scenario"]["targetOverrides"]["firstMessage"] == "Hi a stranger"
    assert testcalls.TESTER in bruno["personality"]["assistant"]["model"]["messages"][0]["content"]


def test_a_scenario_whose_persona_the_server_lacks_is_refused(monkeypatch):
    monkeypatch.setattr(testcalls.httpx, "post", lambda *a, **k: pytest.fail("should not reach the server"))
    s = CallScenario(id="x", title="x", script="You are X. Steps: 1. Hi.", judge=(("n", "c"),), persona="ghost")
    with pytest.raises(SystemExit, match="persona 'ghost' is not in the running server's business"):
        testcalls.twin_profile("http://srv", BUSINESS, s)
