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


def test_seeding_adds_the_rows_an_account_is_missing_and_leaves_the_rest():
    # A row added to a persona after an account was seeded (Priya's two $19 payments) used
    # to never reach it: seeding skipped any account with entries. Now it adds what is
    # missing, by seed index, and touches nothing else, on another day as well.
    from voice.agent import LiveAgent

    priya = next(p for p in PERSONAS if p.id == "priya-raman")
    yesterday = TODAY - dt.timedelta(days=1)
    old_rows = [(k, v) for k, v in seed_entries(priya, today=yesterday) if not k.endswith("-seed3")]
    store = {k: v for k, v in old_rows}
    store["contact-2026-10-13-ab12cd"] = {"date": "2026-10-13", "text": "Called about cancellation: ...", "action": "transfer_to_agent", "worked": True}
    writes: list[str] = []

    class Mem:
        def list(self, entity):
            return [SimpleNamespace(key=k, value=v) for k, v in store.items()]

        def write(self, entity, key, value, **kw):
            store[key] = value
            writes.append(key)

    assert LiveAgent._seed_missing(None, Mem(), "x/customers/priya-raman", priya) == 1
    assert len(writes) == 1 and writes[0].endswith("-seed3")
    assert "two $19 monthly plan payments" in store[writes[0]]["text"]
    # Run again: nothing to add, nothing rewritten.
    assert LiveAgent._seed_missing(None, Mem(), "x/customers/priya-raman", priya) == 0
    assert len(writes) == 1 and "contact-2026-10-13-ab12cd" in store
    # A row reworded in business.toml (Marcus's seats now say what they bill as) is rewritten
    # under the key and date the account already has, not added beside the old wording.
    seed1_key = next(k for k in store if k.endswith("-seed1"))
    store[seed1_key] = {**store[seed1_key], "text": "Signed up on the monthly plan."}
    assert LiveAgent._seed_missing(None, Mem(), "x/customers/priya-raman", priya) == 1
    assert writes[-1] == seed1_key and store[seed1_key]["text"].startswith("Signed up on the monthly plan after")
    assert store[seed1_key]["date"] == seed1_key.split("-seed")[0].removeprefix("contact-")  # the account's date, not today's
    assert sum(k.endswith("-seed1") for k in store) == 1
    # An empty account gets every row.
    store.clear()
    writes.clear()
    assert LiveAgent._seed_missing(None, Mem(), "x/customers/priya-raman", priya) == len(seed_entries(priya))


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


def test_a_caller_who_rings_five_times_does_not_push_the_account_out_of_the_brief():
    # Tenth test run: Tomas's record held seven "today: Called about outage" lines, the six
    # newest filled the brief, and the seeded "Named account manager: Jordan Blake" and the
    # SSO incident fell off the end. Asked "who is calling me", John had no name.
    tomas = next(p for p in PERSONAS if p.id == "tomas-eriksen")
    rows = [SimpleNamespace(key=k, value=v) for k, v in seed_entries(tomas, today=TODAY)]
    for i in range(5):
        k, v = contact_entry(f"call-{i}", intent="outage", summary="Whole team locked out", action="transfer_to_agent",
                             outcome_note=f"Agent closed it in {9 + i} minutes; CSAT 4/5", worked=True, agent_name="John",
                             action_words={"transfer_to_agent": "transferred the call to an agent"}, today=TODAY)
        rows.append(SimpleNamespace(key=k, value=v))
    k, v = contact_entry("call-x", intent="billing_dispute", summary="A $12 line", action="resolve_in_bot",
                         outcome_note="Sorted on the call", worked=True, agent_name="John", today=TODAY)
    rows.append(SimpleNamespace(key=k, value=v))
    mem = CustomerMemory.from_entries("tomas-eriksen", rows)
    caller = Caller(name="Tomas", tier="enterprise", prior_contacts_30d=1, tenure_months=30, customer_id="tomas-eriksen")
    text = brief(caller, mem, today=TODAY)
    assert "Named account manager: Jordan Blake" in text and "SSO logins failed" in text
    assert "today: called 5 times about outage. Most recent: Called about outage: Whole team locked out." in text
    assert "Earlier the same day: Agent closed it in 10 minutes; CSAT 4/5. / Agent closed it in 11 minutes" in text
    assert text.count("Called about outage") == 1 and "Called about billing dispute: A $12 line" in text


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
