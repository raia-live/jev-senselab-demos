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
    assert DEMO.name == "Northwind Cloud" and DEMO.agent_name == "John"
    assert DEMO.simulated and DEMO.scenario.simulator is SCENARIO
    assert list(DEMO.action_ids) == SCENARIO.actions
    # The scenario's rules, plus the ones only a caller with a file can trigger (a step we
    # promised that did not happen): the harness decides one call at a time and has no file.
    assert list(DEMO.policy)[:len(SCENARIO.policy)] == SCENARIO.policy
    assert [k for k, _ in DEMO.policy[len(SCENARIO.policy):]] == ["policy-broken-promise"]
    assert DEMO.scenario.jev_criteria == SCENARIO.jev_criteria
    assert DEMO.scenario.change_rule == SCENARIO.change_rule
    assert list(DEMO.policy_claims) == SCENARIO.policy_claims == DEMO.scenario.policy_claims


def test_every_situation_has_one_written_rule_and_it_says_what_the_policy_says():
    """Before the tacit rule is learned, the written rule for a call is what the policy text
    says: enterprise transfers, small disputes close in the bot, large ones transfer."""
    from core.scenario import Case

    for intent, tier, prior, amount in itertools.product(DEMO.intent_ids, DEMO.tiers, [0, 3], [49, 400]):
        fields = {"intent": intent, "tier": tier, "prior_contacts_30d": prior, "amount_usd": amount}
        case = Case(id="x", episode=0, text="", fields=fields, situation=DEMO.situation_of(fields), truth="", kind=intent)
        claim = DEMO.scenario.policy_claim(case)
        assert claim is not None, fields
        if tier == "enterprise" and intent != "how_to":
            assert claim.key == "policy-enterprise" and claim.action == "transfer_to_agent"
        elif intent == "billing_dispute":
            assert claim.key == ("policy-billing-large" if amount >= 200 else "policy-billing-small")
            assert claim.action == ("transfer_to_agent" if amount >= 200 else "resolve_in_bot")
        elif intent == "cancellation":
            assert claim.key == "policy-cancellation" and claim.action == "transfer_to_agent"
        else:
            assert claim.action == "resolve_in_bot"


def test_situation_labels_reproduce_the_scenario_everywhere():
    for intent, tier, prior, amount in itertools.product(DEMO.intent_ids, DEMO.tiers, [0, 1, 2, 3], [0, 49, 199, 200, 400]):
        fields = {"intent": intent, "tier": tier, "prior_contacts_30d": prior, "amount_usd": amount}
        assert DEMO.situation_of(fields) == situation_of(fields), fields


def test_candidate_fields_open_the_common_situations_first():
    bruno = Caller(name="Bruno", tier="consumer", prior_contacts_30d=3)
    labels = [DEMO.situation_of(f) for f in DEMO.candidate_fields(bruno)]
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
    assert "You are Sam, on the phones at Harbor Dental." in prompt
    assert "{{history}}" in prompt and "{{account}}" in prompt  # filled per call by the server
    assert "booking or moving an appointment (appointment)" in prompt
    assert "for a bill they have a question about, the amount in dollars" in prompt
    assert "Northwind" not in prompt and "John" not in prompt
    # The opening is built by the server; the prompt tells the model it has been said. The
    # decision waits for the caller's words, the close waits for the caller.
    assert "1. Open. The greeting has been said" in prompt and "endCall right after it, in the same turn" in prompt
    assert "say NOTHING in the turn you call a tool in" in prompt and "One tool call at a time" in prompt
    assert "call recall_customer" in prompt and "call recall_customer" not in system_prompt(b, recall=False)
    tool = decide_tool(b)
    assert tool["function"]["parameters"]["properties"]["intent"]["enum"] == ["appointment", "bill"]
    assert "amount_usd" in tool["function"]["parameters"]["properties"]
    assert "caller_said" in tool["function"]["parameters"]["required"]
    assert all(m["type"] != "request-start" for m in tool["messages"])  # nothing scripted
    # The default stack is GPT-Live: a speaker prompt and a reasoner prompt, the one decision
    # tool plus endCall, no transcriber or speaking plans (the model listens while it talks).
    live = assistant_config(b, server_url="https://x/vapi/webhook", secret=None)
    assert live["model"]["model"] == "gpt-live-1" and live["voice"] == {"provider": "openai", "voiceId": "cedar"}
    speaker, reasoner = live["model"]["speaker"]["instructions"], live["model"]["reasoner"]["instructions"]
    assert "You are Sam, on the phones at Harbor Dental." in speaker and "{{history}}" in speaker
    assert "Delegate to the backend when:" in speaker and "Saying goodbye does not end the call" in speaker
    assert "Never ask for a decision from the file alone" in speaker and "NEVER mention tools" in speaker
    assert "call decide_next_step" in reasoner and "appointment (booking or moving an appointment)" in reasoner
    assert "return a short question for the assistant to ask" in reasoner and "call endCall" in reasoner
    # What the fourth test run found: a goodbye the reasoner answered with text instead of
    # endCall, a second decision for a detail added to the same problem, and a stranger told
    # how long they had been a customer (the Caller defaults, read as facts).
    assert "Do not return text instead of calling it" in reasoner and "One decision per problem" in reasoner
    assert "One problem, one answer" in speaker and "{{account}}" in speaker and "{{tenure_months}}" not in speaker
    assert "If no account matched, you do not know this person" in speaker
    # The seventh run: three requests answered from the file and never decided. The rule is
    # now the first thing the speaker reads, with every intent named as a trigger, and the
    # reasoner checks for a goodbye before anything else; a 30 s hook ends a dropped one.
    assert speaker.index("THE ONE RULE") < speaker.index("# How you speak")
    assert "booking or moving an appointment (appointment)" in speaker.split("Delegate to the backend when:")[1]
    assert reasoner.index("call endCall, and do nothing else") < reasoner.index("Deciding:")
    assert live["hooks"][1]["options"]["timeoutSeconds"] == 30 and live["hooks"][1]["do"][1] == {"type": "tool", "tool": {"type": "endCall"}}
    # The eighth run: the reasoner dropped the tool's "Why" so the speaker had no answer to
    # "why would it work this time", and promised to "check" with no tool to check with.
    assert 'ALWAYS pass on the tool\'s "Why"' in reasoner and "Beyond that you have no lookup" in reasoner and "identify_caller is for one case" in reasoner
    assert "sort it out. There is nothing to check" in speaker and "Do not add a time, a date, a name or a job title" in speaker
    # The ninth run: a stranger was told the charge was "a setup fee" from a file that does not
    # exist, and Tomas asked who was calling and got no name although his file has one.
    assert 'never guess one ("a setup fee")' in speaker and "say the name" in speaker
    # Tenth and twelfth runs: "okay, that's fine" ended a call; "nothing else is on offer" was said out loud.
    assert "Never end the call on acceptance" in reasoner and "notes to you, never words for the" in speaker
    # Fifteenth run: a stranger was told the charge was "an extra seat" (the reasoner handed
    # on "explain what the charge is" from a text written for a caller with a file); "what
    # email do you have for me" was delegated and answered with "let me check" and silence;
    # a file question settled Marcus's call and the request never reached the backend.
    assert "If the tool says no account matched the caller, pass that on word for word" in reasoner
    assert "never as lines to read out" in reasoner and "if the file does not have it, say so and ask the caller" in reasoner
    assert '"Let me check" is never the reply' in speaker and "hand the request off in the same turn" in speaker
    assert "A request stated on this call always reaches the backend" in speaker
    # Sixteenth run: the fallback clause in the step text came out as a conditional read to
    # the caller ("the file either explains it or it doesn't"). The fallback lives in the
    # speaker's own rules; the step the reasoner relays is one plain instruction.
    assert "Two instructions, not one conditional sentence" in reasoner
    assert "For a charge, the first thing you say is the entry in your file that matches" in speaker
    assert "For a cancellation it is the offer the tool result names" in speaker
    # Twentieth run, on a clean file: "one was a duplicate and it's already been reversed",
    # about two charges the file did not have. The step text and the speaker both name the
    # file entry first, and a reversal the file does not record is called what it is.
    assert "never a reversal, refund or credit the file does not record" in speaker
    assert 'never "explain what the charge is"' in reasoner
    # Twenty-first run: the reasoner copied the step text's example entry ("two $19 payments")
    # into Marcus's instructions. The step text carries no example, and the reasoner is told
    # it has never seen the file.
    assert "never put words in its mouth" in reasoner
    # Twenty-second run: the clarifying question carried the answer ("that could be the three
    # seats you added"), the caller said "that's all", and the request never reached the backend.
    assert "the question is about what they need, never the answer to it" in speaker
    # Twenty-third run: a number read back with no question after it, the tester improvised
    # a line, and the speaker answered nothing for sixty seconds; no hook fires when the
    # customer spoke last.
    assert "in the same turn ask if there is anything else" in speaker and "Silence is never a reply" in speaker
    for a in Business.load().actions:
        for text in (a.then, a.then_no_file):
            assert "$" not in text and '"' not in text and "'your file" not in text, a.id
    assert [t.get("function", {}).get("name", t["type"]) for t in live["model"]["tools"]] == ["identify_caller", "update_file", "decide_next_step", "endCall"]
    assert all("messages" not in t for t in live["model"]["tools"])  # not spoken on GPT-Live
    assert not any(k in live for k in ("transcriber", "startSpeakingPlan", "stopSpeakingPlan", "backchannelingEnabled", "messagePlan"))
    assert live["hooks"][0]["on"] == "customer.speech.timeout"
    assert assistant_config(b, server_url="https://x", secret=None, voice_id="meridian")["voice"]["voiceId"] == "meridian"
    rt = assistant_config(b, server_url="https://x/vapi/webhook", secret=None, stack="realtime")
    assert rt["model"]["model"] == "gpt-realtime-2" and rt["voice"] == {"provider": "openai", "voiceId": "cedar"}
    assert rt["stopSpeakingPlan"]["numWords"] == 0  # no streaming transcriber: interrupt on voice, not words
    assert rt["firstMessageMode"] == "assistant-speaks-first"
    assert "transcriber" not in rt and rt["firstMessage"] == b.greeting and "endCallMessage" not in rt
    # No spoken filler during a tool call on the realtime stack; the pipeline keeps it. The
    # realtime stack has the one decision tool (two tool calls in a turn lose an answer
    # there); the pipeline has recall_customer as well.
    rt_msgs = [m["type"] for t in rt["model"]["tools"] for m in t.get("messages", [])]
    assert "request-response-delayed" not in rt_msgs and "request-failed" in rt_msgs
    assert [t.get("function", {}).get("name", t["type"]) for t in rt["model"]["tools"]] == ["decide_next_step", "endCall"]
    assert "recall_customer" not in rt["model"]["messages"][0]["content"]
    assert rt["backchannelingEnabled"] is True
    pl = assistant_config(b, server_url="https://x/vapi/webhook", secret=None, stack="pipeline", voice_id="sarah")
    assert pl["model"]["model"] == "gpt-4.1" and pl["voice"]["voiceId"] == "sarah" and pl["transcriber"]["provider"] == "deepgram"
    assert "request-response-delayed" in [m["type"] for t in pl["model"]["tools"] for m in t.get("messages", [])]
    assert [t.get("function", {}).get("name", t["type"]) for t in pl["model"]["tools"]] == ["identify_caller", "update_file", "recall_customer", "decide_next_step", "endCall"]
    with pytest.raises(ValueError):
        assistant_config(b, server_url="https://x", secret=None, stack="telepathy")


# -- test calls ([[scenarios]]) --------------------------------------------------------------

HARBOR_SCENARIOS = """
[[personas]]
id = "ana-reyes"
label = "Ana, moving a cleaning"
name = "Ana"
tier = "patient"
prior_contacts_30d = 1
tenure_months = 30
intent = "appointment"
amount_usd = 0
history = [[9, "Booked a cleaning for next Tuesday at 3."]]
script = ["Hi, I need to move my cleaning."]
why = "A booking the front desk should not need to touch."

[[scenarios]]
id = "ana"
title = "Ana moves her cleaning"
persona = "ana-reyes"
tags = ["known"]
script = \"\"\"
You are Ana. Facts: you have a cleaning next Tuesday at 3 and need it moved to Thursday. Steps:
1. After the greeting, say: 'Hi, I need to move my cleaning to Thursday.'
2. Whatever the agent proposes, say: 'That works. Thanks, bye.'
\"\"\"
[[scenarios.judge]]
name = "knew_the_booking"
check = "The agent referred to the existing Tuesday booking without being told about it."
[[scenarios.judge]]
name = "proper_close"
check = "The agent said goodbye only after the caller said she was done."

[[scenarios]]
id = "walk-in"
title = "Someone we have never seen asks about a bill"
decisions = 1
script = \"\"\"
You are a new patient. Steps:
1. After the greeting, say: 'I got a bill for eighty dollars and I don't know what it is for.'
2. Whatever the agent says, say: 'Okay, bye.'
\"\"\"
[[scenarios.judge]]
name = "did_not_invent"
check = "The agent did not state details of the bill it could not know."
"""


def test_the_demo_file_carries_its_test_calls():
    ids = [s.id for s in DEMO.scenarios]
    assert ids == ["bruno", "stranger", "priya", "tomas", "marcus"]
    personas = {p.id for p in DEMO.personas}
    for s in DEMO.scenarios:
        assert s.persona is None or s.persona in personas, s.id
        assert s.script.startswith("You are ") and "Steps:" in s.script, s.id
        assert len(s.judge) >= 4 and all(name and check for name, check in s.judge), s.id
        assert s.known == (s.persona is not None)
    stranger = next(s for s in DEMO.scenarios if s.id == "stranger")
    assert stranger.persona is None and not stranger.known
    assert next(s for s in DEMO.scenarios if s.id == "priya").decisions == 2
    # The page's config carries them, which is where voice.testcalls reads them from.
    pub = DEMO.public()["scenarios"]
    assert [p["id"] for p in pub] == ids and pub[0]["judge"][0] == {"name": "knew_the_caller", "check": DEMO.scenarios[0].judge[0][1]}


def test_another_business_writes_its_own_test_calls(tmp_path: Path):
    b = Business.from_dict(tomllib.loads(ANOTHER + HARBOR_SCENARIOS), path=tmp_path / "harbor.toml")
    assert [s.id for s in b.scenarios] == ["ana", "walk-in"]
    ana, walk_in = b.scenarios
    assert ana.persona == "ana-reyes" and ana.known and ana.decisions == 1 and ana.tags == ("known",)
    assert ana.judge == (("knew_the_booking", "The agent referred to the existing Tuesday booking without being told about it."),
                         ("proper_close", "The agent said goodbye only after the caller said she was done."))
    assert "1. After the greeting, say: 'Hi, I need to move my cleaning to Thursday.'" in ana.script
    assert walk_in.persona is None and not walk_in.known and walk_in.title == "Someone we have never seen asks about a bill"
    # What the server publishes is what the harness rebuilds, unchanged.
    from voice.business import CallScenario
    assert [CallScenario.from_row(r) for r in b.public()["scenarios"]] == list(b.scenarios)


def test_a_scenario_must_be_whole(tmp_path: Path):
    def load(extra: str) -> Business:
        return Business.from_dict(tomllib.loads(ANOTHER + extra), path=tmp_path / "x.toml")

    with pytest.raises(ValueError, match="persona 'nobody'"):
        load('[[scenarios]]\nid = "a"\nscript = "You are X. Steps: 1. Say hi."\npersona = "nobody"\n[[scenarios.judge]]\nname = "n"\ncheck = "c"\n')
    with pytest.raises(ValueError, match="needs a script"):
        load('[[scenarios]]\nid = "a"\n[[scenarios.judge]]\nname = "n"\ncheck = "c"\n')
    with pytest.raises(ValueError, match="at least one \\[\\[scenarios.judge\\]\\]"):
        load('[[scenarios]]\nid = "a"\nscript = "You are X. Steps: 1. Say hi."\n')
    with pytest.raises(ValueError, match="two scenarios with id 'a'"):
        load('[[scenarios]]\nid = "a"\nscript = "s"\n[[scenarios.judge]]\nname = "n"\ncheck = "c"\n'
             '[[scenarios]]\nid = "a"\nscript = "s"\n[[scenarios.judge]]\nname = "n"\ncheck = "c"\n')
    assert Business.from_dict(tomllib.loads(ANOTHER), path=tmp_path / "x.toml").scenarios == ()  # none is allowed
