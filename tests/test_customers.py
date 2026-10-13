"""The per-caller memory the voice demo keeps in SenseLab: seeding, the brief, the note."""

import datetime as dt
from types import SimpleNamespace

from voice.agent import Caller
from voice.customers import PERSONAS, CustomerMemory, brief, contact_entry, seed_entries

TODAY = dt.date(2026, 10, 14)


def test_personas_are_distinct_and_scripted():
    ids = [p.id for p in PERSONAS]
    assert len(ids) == len(set(ids))
    for p in PERSONAS:
        assert len(p.script) >= 3, p.id  # a conversation, not a phrase
        assert p.history, p.id


def test_seed_entries_are_dated_oldest_first():
    dana = next(p for p in PERSONAS if p.id == "dana-whitfield")
    rows = seed_entries(dana, today=TODAY)
    dates = [v["date"] for k, v in rows if k.startswith("contact-")]
    assert dates == sorted(dates)
    assert dates[0] == "2026-09-23" and dates[-1] == "2026-10-09"
    assert all(v["seeded"] for _, v in rows)
    assert rows[-1][0] == "notes"


def test_brief_reads_as_prose_with_recent_first():
    dana = next(p for p in PERSONAS if p.id == "dana-whitfield")
    entries = [SimpleNamespace(key=k, value=v) for k, v in seed_entries(dana, today=TODAY)]
    mem = CustomerMemory.from_entries("dana-whitfield", entries)
    assert [c["date"] for c in mem.contacts] == ["2026-10-09", "2026-10-02", "2026-09-23"]
    caller = Caller(name="Dana", tier="consumer", prior_contacts_30d=3, tenure_months=39, customer_id="dana-whitfield")
    text = brief(caller, mem, record="For callers in this situation, what has worked: a scheduled specialist callback (6 of 6).",
                 today=TODAY)
    assert text.startswith("Caller: Dana. Consumer plan, customer for 39 months. 3 contact(s)")
    assert "- 5 days ago: Called a third time" in text
    assert "- Sep 23:" in text
    assert "Notes: Prefers" in text
    assert "what has worked: a scheduled specialist callback" in text
    assert "before decide_next_step has answered" in text


def test_brief_for_an_unknown_caller():
    caller = Caller(name="Alex")
    text = brief(caller, None, today=TODAY)
    assert "No earlier contacts on record" in text


def test_contact_entry_from_an_outcome():
    key, value = contact_entry("fake-adf2e126", intent="billing_dispute", summary="charged $49 twice, third call about it",
                               action="schedule_callback", outcome_note="Specialist called back next morning; CSAT 5/5.",
                               worked=True, today=TODAY)
    assert key == "contact-2026-10-14-f2e126"
    assert value["text"] == ("Called about billing dispute: charged $49 twice, third call about it. Riley booked a "
                             "specialist callback. What came of it: Specialist called back next morning; CSAT 5/5.")
    assert value["worked"] is True and not value.get("seeded")
    _, failed = contact_entry("x", intent="billing_dispute", summary="s", action="issue_credit", outcome_note="Credit reversed",
                              worked=False, today=TODAY)
    assert failed["text"].endswith("Credit reversed. It did not hold.")


def test_caller_customer_id_is_a_slug():
    c = Caller.from_mapping({"customer_id": "Dana Whitfield!"})
    assert c.customer_id == "danawhitfield"
    assert Caller.from_mapping({}).customer_id == ""
