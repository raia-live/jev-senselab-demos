"""The per-caller memory the voice demo keeps in SenseLab: seeding, the brief, the note."""

import datetime as dt
import threading
from types import SimpleNamespace

from voice.agent import Caller
from voice.business import Business
from voice.customers import CustomerMemory, brief, call_entry, contact_entry, opening, seed_entries

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


def test_a_call_with_no_decision_goes_on_the_record_too():
    # The caller rang, asked what their last call was about, and hung up. Nothing to decide,
    # still a call they will remember making: the next call's brief says so, in their words.
    key, value = call_entry("fake-01a13944", caller_lines=["Hey John,  what was our last call about? ", " ", "Ok thanks, bye."],
                            agent_name="John", ended_reason="customer-ended-call", seconds=22, today=TODAY)
    assert key == "contact-2026-10-14-a13944"
    assert value["text"] == ("Called, no request to act on. Caller said: \"Hey John, what was our last call about?\"; "
                             "\"Ok thanks, bye\". John answered from the file; the caller hung up after 22 s.")
    assert value["action"] is None and value["worked"] is None
    _, silent = call_entry("x", caller_lines=[], agent_name="John", ended_reason="silence-timed-out", today=TODAY)
    assert silent["text"] == "Called and said nothing we could act on; the line went quiet."
    # It reads back as an earlier contact, newest first, and the brief carries it.
    mem = CustomerMemory.from_entries("bruno", [SimpleNamespace(key=key, value=value),
                                                SimpleNamespace(key="contact-2026-10-01-aaaaaa", value={"date": "2026-10-01", "text": "Called about outage: x. John did y. What came of it: z.", "action": "y", "worked": True})])
    assert [c["date"] for c in mem.contacts] == ["2026-10-14", "2026-10-01"]
    text = brief(Caller.from_mapping({"name": "Bruno", "customer_id": "bruno", "tier": "Team"}), mem, today=TODAY)
    assert "- today: Called, no request to act on. Caller said: \"Hey John, what was our last call about?\"" in text


def test_the_agent_notes_a_call_that_reached_no_decision():
    # Through the agent: call_ended with nothing pending writes the entry under the caller's
    # entity, from the lines on the call; a decided call leaves that to the seal.
    from voice.agent import LiveAgent

    writes: list[tuple[str, str, dict]] = []

    class _Mem:
        def write(self, entity, key, value, **_kw):
            writes.append((entity, key, value))

    class _Client:
        def memory(self, _agent_id):
            return _Mem()

    ag = LiveAgent.__new__(LiveAgent)
    ag.client, ag.agent_id, ag.entity_path, ag.business = _Client(), "john", "jev-demos/voice", BUSINESS
    ag.events = []
    ag._event = lambda kind, data: ag.events.append((kind, data))  # type: ignore[method-assign]
    call = {"id": "c-01a13944", "started": 1_700_000_000.0, "ended": 1_700_000_022.4, "caller": {"customer_id": "bruno-andrade"},
            "live": [{"role": "assistant", "text": "Hi Bruno"}, {"role": "user", "text": "what was our last call about?"}]}
    ag._note_call(call, "customer-ended-call")
    assert len(writes) == 1 and writes[0][0].endswith("/customers/bruno-andrade") and writes[0][1].startswith("contact-")
    assert "Caller said: \"what was our last call about?\"" in writes[0][2]["text"] and "after 22 s" in writes[0][2]["text"]
    assert ag.events[-1][0] == "customer_noted" and ag.events[-1][1]["call_id"] == "c-01a13944"
    ag._note_call({"id": "c-2", "caller": {}, "live": []}, None)  # no account matched: nothing to file it under
    assert len(writes) == 1


def test_a_caller_who_says_their_name_is_matched_to_an_account():
    from concurrent.futures import ThreadPoolExecutor

    from voice.agent import LiveAgent

    ag = LiveAgent.__new__(LiveAgent)
    ag.business, ag.entity_path, ag.agent_id = BUSINESS, "jev-demos/voice", "john"
    ag.calls, ag.events, ag._lock = {}, [], threading.Lock()
    ag._opening, ag._customer, ag._pool = {}, {}, ThreadPoolExecutor(1)
    ag._event = lambda kind, data: ag.events.append((kind, data))  # type: ignore[method-assign]
    ag._open_all = lambda call_id, caller: None  # type: ignore[method-assign]
    ag._fetch_customer = lambda caller: None  # type: ignore[method-assign]
    ag.intro_for = lambda caller: (f"Caller: {caller.name}.\n- today: Called about outage: x.", "Hi")  # type: ignore[method-assign]
    # The name: full, first, said loosely; and the ones that do not pick anyone out.
    assert ag.persona_named("Bruno").id == "bruno-andrade" and ag.persona_named("it's bruno").id == "bruno-andrade"
    assert ag.persona_named("Priya Raman") is None or ag.persona_named("Priya").id == "priya-raman"
    assert ag.persona_named("Sam Ortiz") is None and ag.persona_named("") is None
    # On a call nobody matched, the name makes the call theirs: caller, brief, runs, record.
    ag.calls["c1"] = {"id": "c1", "caller": {"name": "", "customer_id": ""}, "status": "in-progress"}
    text, who = ag.identify("c1", {"name": "Bruno"})
    assert who is not None and who.customer_id == "bruno-andrade" and who.tier == next(p for p in PERSONAS if p.id == "bruno-andrade").tier
    assert text.startswith("Matched: Bruno.") and "Called about outage" in text
    assert ag.calls["c1"]["caller"]["customer_id"] == "bruno-andrade" and ag.calls["c1"]["known_by"] == "name"
    assert "c1" in ag._opening and "c1" in ag._customer
    assert [k for k, _ in ag.events] == ["identified", "recall"] and ag.events[0][1]["matched"] is True
    text, who = ag.identify("c1", {"name": "Sam Ortiz"})
    assert who is None and text.startswith("No account matches the name 'Sam Ortiz'")
    assert ag.identify("nope", {"name": "Bruno"}) == ("No call to identify.", None)


def test_caller_customer_id_is_a_slug():
    c = Caller.from_mapping({"customer_id": "Bruno Andrade!"})
    assert c.customer_id == "brunoandrade"
    assert Caller.from_mapping({}).customer_id == ""


def test_caller_tier_is_checked_against_the_business():
    c = Caller.from_mapping({"tier": "platinum"}, tiers=BUSINESS.tiers, default_tier=BUSINESS.default_tier)
    assert c.tier == "consumer"
    assert Caller.from_mapping({"tier": "platinum"}).tier == "platinum"  # unchecked without a business
