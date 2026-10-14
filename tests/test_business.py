"""The business file: the demo's reproduces the call_handling scenario, and another
business loads without any of it."""

import itertools
import tomllib
from pathlib import Path

import pytest

from demos.call_handling.scenario import SCENARIO, situation_of
from voice.agent import Caller
from voice.business import Business

DEMO = Business.load()


def test_demo_file_matches_the_scenario():
    assert DEMO.name == "Northwind Cloud" and DEMO.agent_name == "Riley"
    assert DEMO.simulated and DEMO.scenario.simulator is SCENARIO
    assert list(DEMO.action_ids) == SCENARIO.actions
    assert list(DEMO.policy) == SCENARIO.policy
    assert DEMO.scenario.jev_criteria == SCENARIO.jev_criteria
    assert DEMO.scenario.change_rule == SCENARIO.change_rule


def test_situation_labels_reproduce_the_scenario_everywhere():
    for intent, tier, prior, amount in itertools.product(DEMO.intent_ids, DEMO.tiers, [0, 1, 2, 3], [0, 49, 199, 200, 400]):
        fields = {"intent": intent, "tier": tier, "prior_contacts_30d": prior, "amount_usd": amount}
        assert DEMO.situation_of(fields) == situation_of(fields), fields


def test_candidate_fields_open_the_common_situations_first():
    dana = Caller(name="Dana", tier="consumer", prior_contacts_30d=3)
    labels = [DEMO.situation_of(f) for f in DEMO.candidate_fields(dana)]
    assert labels[0] == "billing_dispute/consumer/repeat/small"
    assert labels[1] == "cancellation/consumer"
    assert labels[-1] == "billing_dispute/consumer/repeat/large"
    assert len(labels) == len(set(labels)) == 6


def test_classify_gives_the_simulator_its_truth():
    known = DEMO.scenario.classify({"intent": "billing_dispute", "tier": "consumer", "prior_contacts_30d": 3, "amount_usd": 49},
                                   changed=False)
    assert known == {"truth": "schedule_callback", "is_tacit": True, "is_changed": False}
    cancel = {"intent": "cancellation", "tier": "consumer", "prior_contacts_30d": 0, "amount_usd": 0}
    assert DEMO.scenario.classify(cancel, changed=True)["truth"] == "resolve_in_bot"
    assert DEMO.scenario.classify(cancel, changed=False)["truth"] == "transfer_to_agent"


ANOTHER = """
[company]
name = "Harbor Dental"
agent_name = "Sam"

[caller]
tiers = ["patient", "plan"]

[[intents]]
id = "appointment"
label = "booking or moving an appointment"

[[intents]]
id = "bill"
label = "a bill they have a question about"
needs = ["amount_usd"]
ahead = [{ amount_usd = 80 }, { amount_usd = 900 }]

[fields]
amount_usd = "the amount in dollars"

[[actions]]
id = "book_online"
label = "Book it on the call"
jev = "Book or move the appointment now."

[[actions]]
id = "front_desk"
label = "Hand to the front desk"
jev = "Have the front desk call them back."

[[policy]]
key = "policy-appointments"
text = "Appointments: book on the call."

[outcomes]
mode = "reported"
"""


def test_another_business_loads_in_reported_mode(tmp_path: Path):
    b = Business.from_dict(tomllib.loads(ANOTHER), path=tmp_path / "harbor.toml")
    assert b.name == "Harbor Dental" and not b.simulated and b.personas == ()
    assert b.greeting.startswith("Hi, thanks for calling Harbor Dental, this is Sam.")
    assert b.fallback_intent == "bill" and b.default_fields("bill") == {"amount_usd": 80}
    assert b.situation_of({"intent": "bill", "tier": "plan", "amount_usd": 900}) == "bill/plan"
    assert b.situation_of({"intent": "appointment", "tier": "gold"}) == "appointment/patient"  # unknown tier: the default
    sc = b.scenario
    assert sc.actions == ["book_online", "front_desk"] and sc.simulator is None and sc.change_rule == ""
    assert sc.classify({"intent": "bill"}, changed=False) == {"truth": "", "is_tacit": False, "is_changed": False}
    with pytest.raises(NotImplementedError):
        sc.severity_of(None, "front_desk")  # type: ignore[arg-type]
    assert b.public()["change_rule"] is None and b.public()["outcomes_mode"] == "reported"


def test_simulated_mode_needs_matching_actions(tmp_path: Path):
    raw = tomllib.loads(ANOTHER)
    raw["outcomes"] = {"mode": "simulated", "simulator": "demos.call_handling.scenario:SCENARIO"}
    with pytest.raises(ValueError, match="must match the simulator"):
        _ = Business.from_dict(raw, path=tmp_path / "x.toml").scenario


def test_prompt_and_tools_come_from_the_file(tmp_path: Path):
    from voice.assistant import assistant_config, decide_tool, system_prompt

    b = Business.from_dict(tomllib.loads(ANOTHER), path=tmp_path / "harbor.toml")
    prompt = system_prompt(b)
    assert prompt.startswith("You are Sam, on the phones at Harbor Dental.")
    assert "booking or moving an appointment (appointment)" in prompt
    assert "for a bill they have a question about, the amount in dollars" in prompt
    assert "Northwind" not in prompt and "Riley" not in prompt
    tool = decide_tool(b)
    assert tool["function"]["parameters"]["properties"]["intent"]["enum"] == ["appointment", "bill"]
    assert "amount_usd" in tool["function"]["parameters"]["properties"]
    assert all(m["type"] != "request-start" for m in tool["messages"])  # nothing scripted
    cfg = assistant_config(b, server_url="https://x/vapi/webhook", secret=None, voice_id="matilda", voice_provider="11labs")
    assert cfg["firstMessage"] == b.greeting and "endCallMessage" not in cfg
    assert cfg["backchannelingEnabled"] is True
