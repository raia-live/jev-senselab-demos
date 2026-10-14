"""The per-caller memory the voice demo keeps in SenseLab: seeding, the brief, the note."""

import datetime as dt
from types import SimpleNamespace

from voice.agent import Caller
from voice.business import Business
from voice.customers import CustomerMemory, brief, contact_entry, opening, seed_entries

TODAY = dt.date(2026, 10, 14)
BUSINESS = Business.load()
PERSONAS = BUSINESS.personas


def test_personas_are_distinct_and_scripted():
    ids = [p.id for p in PERSONAS]
    assert len(ids) == len(set(ids))
    for p in PERSONAS:
        assert len(p.script) >= 3, p.id  # a conversation, not a phrase
        assert p.history, p.id


def test_seed_entries_are_dated_oldest_first():
    bruno = next(p for p in PERSONAS if p.id == "bruno-andrade")
    rows = seed_entries(bruno, today=TODAY)
    dates = [v["date"] for k, v in rows if k.startswith("contact-")]
    assert dates == sorted(dates)
    assert dates[0] == "2026-09-23" and dates[-1] == "2026-10-09"
    assert all(v["seeded"] for _, v in rows)
    assert rows[-1][0] == "notes"


def test_brief_reads_as_prose_with_recent_first():
    bruno = next(p for p in PERSONAS if p.id == "bruno-andrade")
    entries = [SimpleNamespace(key=k, value=v) for k, v in seed_entries(bruno, today=TODAY)]
    mem = CustomerMemory.from_entries("bruno-andrade", entries)
    assert [c["date"] for c in mem.contacts] == ["2026-10-09", "2026-10-02", "2026-09-23"]
    caller = Caller(name="Bruno", tier="consumer", prior_contacts_30d=3, tenure_months=39, customer_id="bruno-andrade")
    text = brief(caller, mem, record="For callers in this situation, what has worked: a scheduled specialist callback (6 of 6).",
                 today=TODAY)
    assert text.startswith("Caller: Bruno. Consumer plan, customer for 39 months. 3 contact(s)")
    assert "- 5 days ago: Called a third time" in text
    assert "- Sep 23:" in text
    assert "Notes: Mobile on file: (650) 555-0142." in text
    assert "what has worked: a scheduled specialist callback" in text
    assert "before decide_next_step has answered" in text


def test_opening_line_is_built_from_the_file():
    bruno = next(p for p in PERSONAS if p.id == "bruno-andrade")
    entries = [SimpleNamespace(key=k, value=v) for k, v in seed_entries(bruno, today=TODAY)]
    mem = CustomerMemory.from_entries("bruno-andrade", entries)
    caller = Caller(name="Bruno", tier="consumer", prior_contacts_30d=3, tenure_months=39, customer_id="bruno-andrade")
    line = opening(caller, mem, agent_name="John", company="Northwind Cloud", greeting=BUSINESS.greeting, today=TODAY)
    assert line == ("Hi Bruno, it's John at Northwind Cloud. I've got your file open here. I can see you've been in touch "
                    "three times this past month, most recently 5 days ago, so no need to start from the beginning. "
                    "What's going on today?")
    # Known, but nothing on record yet: named, no claims about history.
    assert opening(Caller(name="Lena"), None, agent_name="John", company="Northwind Cloud", greeting="g") == \
        "Hi Lena, it's John at Northwind Cloud. I've got your file open here. What can I do for you today?"
    # A stranger gets the business's greeting.
    assert opening(Caller(name=""), None, agent_name="John", company="N", greeting=BUSINESS.greeting) == BUSINESS.greeting
    assert opening(Caller(name="unknown"), None, agent_name="John", company="N", greeting="g") == "g"


def test_brief_for_an_unknown_caller():
    caller = Caller(name="Alex")
    text = brief(caller, None, today=TODAY)
    assert "No earlier contacts on record" in text


def test_contact_entry_from_an_outcome():
    key, value = contact_entry("fake-adf2e126", intent="billing_dispute", summary="charged $49 twice, third call about it",
                               action="schedule_callback", outcome_note="Specialist called back next morning; CSAT 5/5.",
                               worked=True, agent_name="Riley", action_words=BUSINESS.action_words(), today=TODAY)
    assert key == "contact-2026-10-14-f2e126"
    assert value["text"] == ("Called about billing dispute: charged $49 twice, third call about it. Riley booked a "
                             "specialist callback. What came of it: Specialist called back next morning; CSAT 5/5.")
    assert value["worked"] is True and not value.get("seeded")
    _, failed = contact_entry("x", intent="billing_dispute", summary="s", action="issue_credit", outcome_note="Credit reversed",
                              worked=False, today=TODAY)
    assert failed["text"].endswith("Credit reversed. It did not hold.")
    assert failed["text"].startswith("Called about billing dispute: s. The agent issue credit.")  # no words given: the id


def test_caller_customer_id_is_a_slug():
    c = Caller.from_mapping({"customer_id": "Bruno Andrade!"})
    assert c.customer_id == "brunoandrade"
    assert Caller.from_mapping({}).customer_id == ""


def test_caller_tier_is_checked_against_the_business():
    c = Caller.from_mapping({"tier": "platinum"}, tiers=BUSINESS.tiers, default_tier=BUSINESS.default_tier)
    assert c.tier == "consumer"
    assert Caller.from_mapping({"tier": "platinum"}).tier == "platinum"  # unchecked without a business
