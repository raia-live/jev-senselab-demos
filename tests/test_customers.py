"""The per-caller memory the voice demo keeps in SenseLab: seeding, the brief, the note."""

import datetime as dt
import threading
import time
from types import SimpleNamespace

from core.scenario import Outcome
from voice.agent import Caller
from voice.business import Business
from voice.customers import (
    CustomerMemory,
    brief,
    call_entry,
    contact_entry,
    detail_entry,
    opening,
    seed_entries,
    track_record,
)

TODAY = dt.date(2026, 10, 14)
BUSINESS = Business.load()
PERSONAS = BUSINESS.personas


def test_personas_are_distinct_and_scripted():
    ids = [p.id for p in PERSONAS]
    assert len(ids) == len(set(ids))
    for p in PERSONAS:
        assert len(p.script) >= 3, p.id  # a conversation, not a phrase
        assert p.history, p.id
    # A persona whose point is the demo world's switch says something else when outcomes
    # are reported and there is no switch; the rest read the same in both modes.
    priya = next(p for p in PERSONAS if p.id == "priya-raman")
    assert "switch" in priya.label and "switch" in priya.why
    shown = priya.for_mode(reported=True)
    assert "switch" not in shown.label and "switch" not in shown.why and shown.id == priya.id and shown.script == priya.script
    assert priya.for_mode(reported=False) is priya
    for p in PERSONAS:
        if p.id != "priya-raman":
            assert p.for_mode(reported=True) is p and "switch" not in p.why, p.id
    assert "switch" not in BUSINESS.public()["personas"][2]["label"] if not BUSINESS.simulated else True
    assert "for_mode" not in shown.public() and shown.public()["label"] == shown.label


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
    assert "today: called 5 times about outage. Most recent (today): Called about outage: Whole team locked out." in text
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
    # Reported mode, the call just ended: the line is on file before the outcome is, under
    # the same key, and says to ask.
    k3, open_ = contact_entry("fake-adf2e126", intent="billing_dispute", summary="s", action="schedule_callback", outcome_note="",
                              worked=None, agent_name="Riley", action_words=BUSINESS.action_words(), today=TODAY)
    assert k3 == key and open_["worked"] is None
    assert open_["text"] == ("Called about billing dispute: s. Riley booked a specialist callback. What came of it: not known yet; "
                             "ask, and put their answer on file.")


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
    ag._files, ag._files_dirty, ag._lock = {}, set(), threading.Lock()
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
    # A call with two decisions (the first refused, a second decided) keeps a line for each:
    # the first under the call's key, the second under one that differs.
    ag._lock = threading.Lock()
    bruno = {"customer_id": "bruno-andrade"}
    first = SimpleNamespace(id="c-01a13944:4", call_id="c-01a13944", caller=bruno, fields={"intent": "cancellation"},
                            text="Bruno, consumer. Caller said: \"cancel\" Wants out.", action="transfer_to_agent")
    second = SimpleNamespace(id="c-01a13944:5", call_id="c-01a13944", caller=bruno, fields={"intent": "cancellation"},
                             text="Bruno, consumer. Caller said: \"no transfer\" Refused.", action="resolve_in_bot")
    ag.decisions = [first, second]
    ag._note_customer(first, Outcome(success=False, severity="failure", note="Caller said on this call: no transfer", true_success=False, cost_usd=0.0))
    ag._note_customer(second, None)
    assert writes[1][1] == "contact-" + dt.date.today().isoformat() + "-a13944" and writes[1][2]["worked"] is False
    assert writes[2][1] == "contact-" + dt.date.today().isoformat() + "-139441" and writes[2][2]["worked"] is None
    assert writes[1][1] != writes[2][1]


def test_a_caller_who_says_their_name_is_matched_to_an_account():
    from concurrent.futures import ThreadPoolExecutor

    from voice.agent import LiveAgent

    ag = LiveAgent.__new__(LiveAgent)
    ag._files, ag._files_dirty = {}, set()
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


def test_a_detail_the_caller_gives_is_on_file_next_time():
    # The caller gives a number to reach them on; it is written the moment they confirm it,
    # and the next call's brief carries it as current, so John does not ask again.
    key, entry = detail_entry("fake-01a13972", kind="phone", value=" 650 555 0199 ", note="for the specialist's callback", today=TODAY)
    assert key == "detail-phone" and entry == {"kind": "phone", "value": "650 555 0199", "note": "for the specialist's callback",
                                               "date": "2026-10-14", "at": "2026-10-14T00:00:00+00:00", "call_id": "fake-01a13972"}
    k2, corr = detail_entry("fake-01a13972", kind="correction", value="The specialist callback never came.", today=TODAY)
    assert k2 == "note-2026-10-14-a13972" and corr["kind"] == "correction"
    assert detail_entry("x", kind="shoe size", value="44")[0].startswith("note-")  # an unknown kind is a note
    older = {"kind": "phone", "value": "650 555 0000", "note": "", "date": "2026-10-01", "call_id": "old"}
    mem = CustomerMemory.from_entries("bruno", [SimpleNamespace(key="detail-phone", value=older), SimpleNamespace(key=key, value=entry),
                                                SimpleNamespace(key=k2, value=corr),
                                                SimpleNamespace(key="contact-2026-10-01-aaaaaa", value={"date": "2026-10-01", "text": "Called about outage: x. John did y. What came of it: z."})])
    assert mem.details == {"phone": {"value": "650 555 0199", "note": "for the specialist's callback", "date": "2026-10-14"}}  # the latest
    assert [r["value"] for r in mem.remarks] == ["The specialist callback never came."]
    text = brief(Caller.from_mapping({"name": "Bruno", "customer_id": "bruno", "tier": "Team"}), mem, today=TODAY)
    assert "On file, from the caller: phone number 650 555 0199 (given today). These are current; use them, do not ask for them again." in text
    assert "Corrections and notes from earlier calls, most recent first:\n- today: The specialist callback never came." in text
    assert text.index("On file") < text.index("Earlier contacts")  # the details come before the history


def test_the_agent_puts_a_detail_on_the_callers_record_and_the_number_in_the_book():
    from voice.agent import LiveAgent
    from voice.phonebook import PhoneBook

    writes: list[tuple[str, str, dict]] = []

    class _Mem:
        def write(self, entity, key, value, **_kw):
            writes.append((entity, key, value))

        def list(self, entity):
            return []

    class _Client:
        def memory(self, _agent_id):
            return _Mem()

    ag = LiveAgent.__new__(LiveAgent)
    ag._files, ag._files_dirty = {}, set()
    ag.client, ag.agent_id, ag.entity_path, ag.business = _Client(), "john", "jev-demos/voice", BUSINESS
    ag.calls, ag.events, ag._lock = {}, [], threading.Lock()
    ag.phonebook = PhoneBook(_Client(), "john", "jev-demos/voice")
    ag._event = lambda kind, data: ag.events.append((kind, data))  # type: ignore[method-assign]
    ag.calls["c1"] = {"id": "c1", "caller": {"name": "Bruno", "tier": "consumer", "prior_contacts_30d": 3, "tenure_months": 39,
                                             "customer_id": "bruno-andrade"}, "customer_brief": "Caller: Bruno."}
    text = ag.update_file("c1", {"kind": "phone", "value": "six five zero, five five five, zero one nine nine", "note": "callback"})
    assert text.startswith("On file: phone number six five zero, five five five, zero one nine nine.")
    customer_writes = [w for w in writes if w[0].endswith("/customers/bruno-andrade")]
    assert customer_writes[0][1] == "detail-phone" and customer_writes[0][2]["value"].startswith("six five zero")
    assert ag.phonebook.who("+16505550199") == ({"name": "Bruno", "tier": "consumer", "prior_contacts_30d": 3, "tenure_months": 39,
                                                  "customer_id": "bruno-andrade"}, "number")  # a call from it is Bruno
    assert ag.calls["c1"]["customer_brief"].endswith("On file now (phone number, given on this call): six five zero, five five five, zero one nine nine.")
    assert ag.events[-1][0] == "file_updated" and ag.events[-1][1]["detail"] == "phone"
    assert ag.update_file("c1", {"kind": "phone", "value": ""}) == "Nothing to put on file: say what the detail is."
    ag.calls["c2"] = {"id": "c2", "caller": {"name": "", "customer_id": ""}}
    assert ag.update_file("c2", {"kind": "email", "value": "sam@example.com"}).startswith("No account matched this caller")
    assert len([w for w in writes if w[0].endswith("/customers/")]) == 0
    assert ag.update_file("nope", {"kind": "email", "value": "x"}) == "No call to put this on."


def test_what_the_caller_says_came_of_the_last_step_closes_it_on_their_word():
    # Outcomes are reported, so the callback booked on the earlier call is still open when
    # the caller rings back and says nobody called. Their word seals it as a failure, the
    # note goes on the file, and a step already sealed is left alone.
    from voice.agent import LiveAgent

    writes: list[tuple[str, str, dict]] = []
    sealed: list[tuple[str, bool, str, str]] = []

    class _Mem:
        def write(self, entity, key, value, **_kw):
            writes.append((entity, key, value))

    class _Client:
        def memory(self, _agent_id):
            return _Mem()

    ag = LiveAgent.__new__(LiveAgent)
    ag._files, ag._files_dirty = {}, set()
    ag.client, ag.agent_id, ag.entity_path, ag.business = _Client(), "john", "jev-demos/voice", BUSINESS
    ag.calls, ag.events, ag._lock, ag.phonebook = {}, [], threading.Lock(), None
    ag._event = lambda kind, data: ag.events.append((kind, data))  # type: ignore[method-assign]
    # The seal (trace, lesson, record) is slow and runs off the call's path; here it runs inline.
    ag._pool = SimpleNamespace(submit=lambda fn, *a, **kw: fn(*a, **kw))
    ag._seal = lambda p, outcome, source="review": sealed.append((p.decision.id, outcome.success, outcome.note, source))  # type: ignore[method-assign]
    bruno = {"name": "Bruno", "tier": "consumer", "prior_contacts_30d": 3, "tenure_months": 39, "customer_id": "bruno-andrade"}
    earlier = SimpleNamespace(decision=SimpleNamespace(id="c0:0", call_id="c0", at=100.0, caller=bruno, action="schedule_callback"))
    older = SimpleNamespace(decision=SimpleNamespace(id="b0:0", call_id="b0", at=50.0, caller=bruno, action="issue_credit"))
    priya = SimpleNamespace(decision=SimpleNamespace(id="p0:0", call_id="p0", at=200.0, caller={**bruno, "customer_id": "priya-raman"}, action="transfer_to_agent"))
    ag._pending = {"c0": [earlier], "b0": [older], "p0": [priya]}
    ag.calls["c1"] = {"id": "c1", "caller": bruno}
    text = ag.update_file("c1", {"kind": "outcome", "value": "nobody called me", "worked": False})
    assert sealed == [("c0:0", False, "Caller said on a later call: nobody called me", "caller")]  # the latest of theirs, not Priya's
    assert ("That closes the earlier step (booked a specialist callback) as did not work, on their word; "
            "the record for callers in their situation now says so.") in text
    assert "c0" not in ag._pending and ag._pending["b0"] == [older]  # off the open list; the other step stays open
    # Whatever it closed, the file update arranges nothing: the next step is the decision's.
    assert text.endswith("decide_next_step says what happens, and no step is promised before it has answered.")
    assert writes[-1][1].startswith("note-") and writes[-1][2]["kind"] == "outcome"
    # Jev sees it at the next decision on this call, under the own-record heading.
    assert ag.calls["c1"]["said_on_call"] == [{"line": 'they said the step from their last call (booked a specialist callback) did not '
                                                       'hold, in their words: "nobody called me".',
                                               "action": "schedule_callback", "worked": False}]
    # The caller refuses the step taken a minute ago on THIS call ("no, don't transfer me"):
    # that is the outcome of this call's own open step, and the result says to decide again.
    mine = SimpleNamespace(decision=SimpleNamespace(id="c1:0", call_id="c1", at=300.0, caller=bruno, action="transfer_to_agent"))
    ag._pending = {"c1": [mine]}
    text = ag.update_file("c1", {"kind": "outcome", "value": "No, I don't want to be transferred, cancel it now", "worked": False})
    assert sealed[-1] == ("c1:0", False, "Caller said on this call: No, I don't want to be transferred, cancel it now", "caller")
    assert "That closes the step taken on this call (transferred the call to an agent) as did not work, on their word" in text
    assert text.endswith("decide_next_step again, with what they said as the new fact: it is the only way a different step comes back.")
    assert ag._pending == {} and ag.calls["c1"]["said_on_call"][-1]["action"] == "transfer_to_agent"
    assert ag.calls["c1"]["said_on_call"][-1]["line"].startswith("they said the step taken a moment ago on this call "
                                                                                 "(transferred the call to an agent) did not hold")
    # Nothing open for this caller and no record fetched: the note still goes on file.
    ag._pending, ag._customer = {}, {}
    text = ag.update_file("c1", {"kind": "outcome", "value": "they did call, thanks", "worked": "true"})
    assert text.startswith("On file: what came of the last step, in the caller's words they did call, thanks.") and "closes" not in text
    assert len(sealed) == 2 and writes[-1][2]["value"] == "they did call, thanks"
    # Nothing open, but the file has the line the demo world wrote ("called back next
    # morning; CSAT 5/5"): the caller's word replaces it, under the same key, and the step
    # is marked as not holding. That is the 17:37 call: John told Bruno the specialist had
    # called the next morning, from a verdict the simulator reached without him.
    key, line = contact_entry("fake-01a13982", intent="billing_dispute", summary="s", action="schedule_callback",
                              outcome_note="Specialist called back next morning; CSAT 5/5", worked=True, agent_name="John",
                              action_words=BUSINESS.action_words(), today=TODAY)
    mem = CustomerMemory.from_entries("bruno-andrade", [SimpleNamespace(key=key, value=line)])
    fut = SimpleNamespace(result=lambda timeout=None: mem)
    ag._customer = {"c1": fut}
    text = ag.update_file("c1", {"kind": "outcome", "value": "nobody called back", "worked": False})
    assert "The file's line on that step (booked a specialist callback) now carries their word: it did not hold." in text
    amended = [w for w in writes if w[1] == key]
    assert len(amended) == 1 and amended[0][2]["worked"] is False and amended[0][2]["at"] == line["at"]
    assert amended[0][2]["text"] == ("Called about billing dispute: s. John booked a specialist callback. What came of it: Caller said "
                                     "on a later call: nobody called back. It did not hold.")
    assert "key" not in amended[0][2] and mem.contacts[0]["worked"] is False  # the in-hand record agrees
    assert ag.events[-1][0] == "customer_noted" and ag.events[-1][1]["amended"] is True
    assert len(sealed) == 2  # amending the file seals nothing
    assert ag.calls["c1"]["said_on_call"][-1] == {"line": 'they said the step from their last call (booked a specialist callback) did '
                                                          'not hold, in their words: "nobody called back".',
                                                  "action": "schedule_callback", "worked": False}
    # Already saying what the caller says: nothing to rewrite.
    n = len(writes)
    ag.update_file("c1", {"kind": "outcome", "value": "still nothing", "worked": False})
    assert len(writes) == n + 1 and writes[-1][1].startswith("note-")
    # An open step (outcome not in yet) is the one their word goes on, over a sealed one above it.
    k_open, v_open = contact_entry("fake-01a13990", intent="billing_dispute", summary="s2", action="schedule_callback",
                                   outcome_note="", worked=None, agent_name="John", action_words=BUSINESS.action_words(), today=TODAY)
    k_ok, v_ok = contact_entry("fake-01a13999", intent="billing_dispute", summary="s3", action="issue_credit",
                               outcome_note="Credit stuck", worked=True, agent_name="John", action_words=BUSINESS.action_words(), today=TODAY)
    mem = CustomerMemory.from_entries("bruno-andrade", [SimpleNamespace(key=k_ok, value=v_ok), SimpleNamespace(key=k_open, value=v_open)])
    ag._customer = {"c1": SimpleNamespace(result=lambda timeout=None: mem)}
    text = ag.update_file("c1", {"kind": "outcome", "value": "nobody called", "worked": False})
    assert "(booked a specialist callback) now carries their word: it did not hold" in text
    assert writes[-1][1] == k_open and writes[-1][2]["worked"] is False and writes[-2][1].startswith("note-")


def test_a_caller_ringing_back_about_the_same_thing_closes_the_open_step_as_not_working():
    # Outcomes are reported. Priya was transferred at 21:22 and rang back at 21:24 about the
    # same cancellation: nobody needs to press a button, the ring-back is the outcome. Every
    # open step of hers on that problem is sealed as a failure, Jev's view of her record says
    # so, and the result tells the agent to own it. Steps on other problems, other callers'
    # steps and this call's own stay open.
    from voice.agent import LiveAgent, _ago

    assert [_ago(s) for s in (10, 125, 1800, 7000, 3600 * 5, 86400 * 3, 86400 * 40)] == [
        "a minute", "two minutes", "30 minutes", "two hours", "five hours", "three days", "40 days"]
    writes: list[tuple[str, str, dict]] = []
    sealed: list[tuple[str, bool, str, str]] = []

    class _Mem:
        def write(self, entity, key, value, **_kw):
            writes.append((entity, key, value))

    class _Client:
        def memory(self, _agent_id):
            return _Mem()

    ag = LiveAgent.__new__(LiveAgent)
    ag._files, ag._files_dirty = {}, set()
    ag.client, ag.agent_id, ag.entity_path, ag.business = _Client(), "john", "jev-demos/voice", BUSINESS
    ag.calls, ag.events, ag._lock, ag._customer = {}, [], threading.Lock(), {}
    ag._event = lambda kind, data: ag.events.append((kind, data))  # type: ignore[method-assign]
    ag._pool = SimpleNamespace(submit=lambda fn, *a, **kw: fn(*a, **kw))
    ag._seal = lambda p, outcome, source="review": sealed.append((p.decision.id, outcome.success, outcome.note, source))  # type: ignore[method-assign]
    priya = {"name": "Priya", "tier": "consumer", "prior_contacts_30d": 1, "tenure_months": 20, "customer_id": "priya-raman"}
    now = time.time()

    def pend(did, call_id, at, caller, intent, action):
        return SimpleNamespace(decision=SimpleNamespace(id=did, call_id=call_id, at=at, caller=caller, action=action,
                                                        fields={"intent": intent}, situation=f"{intent}/consumer"))

    first = pend("p0:0", "p0", now - 130, priya, "cancellation", "transfer_to_agent")
    again = pend("p0:1", "p0", now - 100, priya, "cancellation", "transfer_to_agent")
    billing = pend("p1:0", "p1", now - 3000, priya, "billing_dispute", "schedule_callback")
    bruno = pend("b0:0", "b0", now - 60, {**priya, "name": "Bruno", "customer_id": "bruno-andrade"}, "cancellation", "transfer_to_agent")
    mine = pend("p2:0", "p2", now - 5, priya, "cancellation", "transfer_to_agent")
    ag._pending = {"p0": [first, again], "p1": [billing], "b0": [bruno], "p2": [mine]}
    ag.calls["p2"] = {"id": "p2", "caller": priya}
    text = ag._rang_back("p2", "priya-raman", "cancellation")
    assert text == ("They are back about the same thing, two minutes after the last step we took for them (transferred the "
                    "call to an agent). That step is on their record as not working, on that alone.")
    assert sealed == [("p0:0", False, "Rang back about the same thing two minutes later", "caller"),
                      ("p0:1", False, "Rang back about the same thing two minutes later", "caller")]
    assert ag._pending == {"p1": [billing], "b0": [bruno], "p2": [mine]}
    assert ag.calls["p2"]["said_on_call"] == [{"line": "they are back about the same thing, two minutes after the last step we took "
                                                       "for them (transferred the call to an agent); that step did not hold, on that alone.",
                                               "action": "transfer_to_agent", "worked": False}]
    # A stranger, a different problem, or nothing open: nothing happens. And once per call and
    # problem: a second decision on the same call (she refuses the step) does not mark it again.
    assert ag._rang_back("p2", "", "cancellation") == "" and ag._rang_back("p2", "priya-raman", "outage") == ""
    assert ag._rang_back("p2", "priya-raman", "cancellation") == "" and len(sealed) == 2
    assert ag.calls["p2"]["rang_back_for"] == ["cancellation", "outage"]
    ag.calls["p3"] = {"id": "p3", "caller": priya}
    ag._pending.pop("p2")  # this call's own step, closed some other way
    assert ag._rang_back("p3", "priya-raman", "cancellation") == "" and len(sealed) == 2  # nothing open, no file in hand
    # Nothing open in this process (the server was restarted since), but her file carries the
    # step with its outcome not in yet: the file's line gets the ring-back instead.
    ag.calls["p4"] = {"id": "p4", "caller": priya}
    k_open, v_open = contact_entry("fake-01a13a5d", intent="cancellation", summary="not using it enough", action="transfer_to_agent",
                                   outcome_note="", worked=None, agent_name="John", action_words=BUSINESS.action_words(),
                                   today=TODAY, now=dt.datetime.now(dt.UTC) - dt.timedelta(days=2))
    k_older, v_older = contact_entry("fake-01a13a5c", intent="cancellation", summary="first try", action="schedule_callback",
                                     outcome_note="", worked=None, agent_name="John", action_words=BUSINESS.action_words(),
                                     today=TODAY, now=dt.datetime.now(dt.UTC) - dt.timedelta(days=5))
    k_other, v_other = contact_entry("fake-01a13a00", intent="billing_dispute", summary="s", action="schedule_callback",
                                     outcome_note="", worked=None, agent_name="John", action_words=BUSINESS.action_words(), today=TODAY)
    mem = CustomerMemory.from_entries("priya-raman", [SimpleNamespace(key=k_open, value=v_open), SimpleNamespace(key=k_other, value=v_other),
                                                      SimpleNamespace(key=k_older, value=v_older)])
    ag._customer = {"p4": SimpleNamespace(result=lambda timeout=None: mem)}
    n = len(writes)
    text = ag._rang_back("p4", "priya-raman", "cancellation")
    assert text.startswith("They are back about the same thing, two days after the last step we took for them (transferred the call to an agent).")
    # Both open steps on that problem get it (oldest first); the billing one is left alone.
    assert [w[1] for w in writes[n:]] == [k_older, k_open] and all(w[2]["worked"] is False for w in writes[n:])
    assert writes[-1][2]["text"].endswith("What came of it: Caller said on a later call: they rang back about it two days later. It did not hold.")
    assert writes[-2][2]["text"].endswith("they rang back about it five days later. It did not hold.")
    assert ag.events[-1][0] == "customer_noted" and ag.events[-1][1]["amended"] is True and len(sealed) == 2
    assert ag.calls["p4"]["said_on_call"][0]["line"].startswith("they are back about the same thing, two days after")
    assert ag.calls["p4"]["said_on_call"][0]["action"] == "transfer_to_agent"
    ag.calls["p5"] = {"id": "p5", "caller": priya}
    ag._customer["p5"] = ag._customer["p4"]
    assert ag._rang_back("p5", "priya-raman", "cancellation") == ""  # the line now says so; nothing left open on it


def test_calls_on_one_day_sort_by_the_minute_and_the_brief_says_how_long_ago():
    # Two calls on one day: the one at 17:36 is the most recent, whatever order the store
    # returns them in, and the brief says "3 minutes ago" rather than "today", so John can
    # say "we spoke a few minutes ago" instead of counting contacts at a caller who just hung up.
    t1 = dt.datetime(2026, 10, 14, 8, 23, tzinfo=dt.UTC)
    t2 = dt.datetime(2026, 10, 14, 8, 36, tzinfo=dt.UTC)
    now = dt.datetime(2026, 10, 14, 8, 39, tzinfo=dt.UTC)
    k1, v1 = contact_entry("call-aaaaa1", intent="billing_dispute", summary="first", action="schedule_callback",
                           outcome_note="Specialist called back next morning", worked=True, agent_name="John",
                           action_words=BUSINESS.action_words(), today=TODAY, now=t1)
    k2, v2 = contact_entry("call-bbbbb2", intent="billing_dispute", summary="second", action="schedule_callback",
                           outcome_note="", worked=None, agent_name="John", action_words=BUSINESS.action_words(), today=TODAY, now=t2)
    assert v1["at"] == "2026-10-14T08:23:00+00:00"
    mem = CustomerMemory.from_entries("bruno-andrade", [SimpleNamespace(key=k1, value=v1), SimpleNamespace(key=k2, value=v2)])
    assert [c["key"] for c in mem.contacts] == [k2, k1]
    caller = Caller(name="Bruno", tier="consumer", prior_contacts_30d=3, tenure_months=39, customer_id="bruno-andrade")
    text = brief(caller, mem, today=TODAY, now=now)
    assert ("- today: called 2 times about billing dispute. Most recent (3 minutes ago): Called about billing dispute: second. "
            "John booked a specialist callback. What came of it: not known yet; ask, and put their answer on file. "
            "Earlier the same day: Specialist called back next morning.") in text
    # The opening says we just spoke, instead of the count they heard three minutes ago.
    line = opening(caller, mem, agent_name="John", company="Northwind Cloud", greeting="g", today=TODAY, now=now)
    assert line == "Hi Bruno, it's John at Northwind Cloud. We spoke 3 minutes ago; I've still got your file open here. What's going on?"
    assert opening(caller, mem, agent_name="John", company="N", greeting="g", today=TODAY, now=t2 + dt.timedelta(seconds=50)).startswith(
        "Hi Bruno, it's John at N. We spoke just now;")
    # Two hours on, back to the count.
    later = opening(caller, mem, agent_name="John", company="N", greeting="g", today=TODAY, now=now + dt.timedelta(hours=2))
    assert "most recently today" in later
    # The caller's own record with the steps, for the decision and for the "why".
    assert track_record(mem, BUSINESS.action_words(), today=TODAY, now=now) == (
        "This caller's own record with these steps: booked a specialist callback 2 times (most recently 3 minutes ago); "
        "it held 1 time, 1 still open.")
    mem.contacts[0]["worked"] = False
    assert track_record(mem, BUSINESS.action_words(), today=TODAY, now=now).endswith(
        "it did not hold 1 time, on their word, held 1 time.")
    assert track_record(None) == "" and track_record(CustomerMemory(customer_id="x")) == ""
    # A seeded row (date only) sorts as the start of its day, after what was written that day.
    seeded = SimpleNamespace(key="contact-2026-10-14-seed1", value={"date": "2026-10-14", "text": "Signed up.", "seeded": True})
    mem2 = CustomerMemory.from_entries("b", [seeded, SimpleNamespace(key=k1, value=v1)])
    assert [c["key"] for c in mem2.contacts] == [k1, "contact-2026-10-14-seed1"]


def test_caller_customer_id_is_a_slug():
    c = Caller.from_mapping({"customer_id": "Bruno Andrade!"})
    assert c.customer_id == "brunoandrade"
    assert Caller.from_mapping({}).customer_id == ""


def test_caller_tier_is_checked_against_the_business():
    c = Caller.from_mapping({"tier": "platinum"}, tiers=BUSINESS.tiers, default_tier=BUSINESS.default_tier)
    assert c.tier == "consumer"
    assert Caller.from_mapping({"tier": "platinum"}).tier == "platinum"  # unchecked without a business


def test_a_step_that_did_not_hold_for_this_caller_is_named_as_not_to_be_taken_again():
    from dataclasses import replace

    from voice.agent import LiveAgent

    ag = LiveAgent.__new__(LiveAgent)
    ag._files, ag._files_dirty = {}, set()
    ag.business = BUSINESS
    memory = CustomerMemory(customer_id="priya-raman", contacts=[
        {"key": "contact-2026-10-14-a", "date": "2026-10-14", "text": "Called about cancellation: transferred the call to an agent.",
         "action": "transfer_to_agent", "worked": False, "seeded": False},
        {"key": "contact-2026-10-01-b", "date": "2026-10-01", "text": "Called about billing dispute: booked a specialist callback.",
         "action": "schedule_callback", "worked": True, "seeded": False},
        {"key": "contact-2026-09-01-seed0", "date": "2026-09-01", "text": "Opened the account.", "seeded": True},
    ])
    # From the file alone: the transfer did not hold; the callback did and is not named.
    assert ag._not_again_line(memory, []) == (
        "Not to be taken again for this caller: transferred the call to an agent. Each did not hold for them, and a step "
        "that did not hold for a caller is not booked for them again, whatever it has done for others. Choose the best of "
        "the other steps for their situation.")
    # What this call has shown is added, once per step, in the order it came.
    said = [{"line": "they refused a credit", "action": "issue_credit", "worked": False},
            {"line": "rang back", "action": "transfer_to_agent", "worked": False}]
    assert ag._not_again_line(memory, said).startswith(
        "Not to be taken again for this caller: transferred the call to an agent; applied a credit.")
    # Their word on this call that the step held lifts it.
    held = [{"line": "they said the agent did call", "action": "transfer_to_agent", "worked": True}]
    assert ag._not_again_line(memory, held) == ""
    # Nothing failed, nothing said: no line. A stranger: no line.
    assert ag._not_again_line(CustomerMemory(customer_id="x"), []) == ""
    assert ag._not_again_line(None, []) == ""
    # The business can keep the rule off.
    ag.business = replace(BUSINESS, no_repeat_failed_steps=False)
    assert ag._not_again_line(memory, said) == ""


def test_the_step_words_for_a_cancellation_are_the_offer_not_the_charge():
    on_call = BUSINESS.then_for("resolve_in_bot", matched=True, intent="cancellation")
    assert on_call.startswith("Handle the cancellation on this call") and "half price" in on_call
    assert "charge" not in BUSINESS.then_for("resolve_in_bot", matched=True, intent="cancellation").split(".")[0]
    # A charge keeps the billing words; no intent keeps them too; a stranger gets the no-file words whatever the intent.
    assert BUSINESS.then_for("resolve_in_bot", matched=True, intent="billing_dispute") == BUSINESS.action("resolve_in_bot").then
    assert BUSINESS.then_for("resolve_in_bot", matched=True) == BUSINESS.action("resolve_in_bot").then
    assert BUSINESS.then_for("resolve_in_bot", matched=False, intent="cancellation") == BUSINESS.action("resolve_in_bot").then_no_file
    # Other actions have no per-intent words and fall back to their one wording.
    assert BUSINESS.then_for("transfer_to_agent", matched=True, intent="cancellation") == BUSINESS.action("transfer_to_agent").then


def test_a_detail_both_sides_said_on_the_call_is_filed_at_hang_up():
    # Bruno's 11:15 call: he gave an email, John read it back and said "I've got it", and
    # never called update_file. At hang-up the transcript puts it on his file, once; a
    # number John read out of the file and Bruno only nodded at is not a detail he gave.
    from concurrent.futures import Future

    from voice.agent import LiveAgent, confirmed_details

    lines = [("assistant", "Hi Bruno, it's John at Northwind Cloud. I've got your file open here."),
             ("user", "You can use the one in the file"),
             ("assistant", "I don't actually have an email on file here. What's the best one to use?"),
             ("user", "bruno at example dot com"), ("assistant", "Bruno at example dot com."), ("user", "Yep"),
             ("assistant", "A colleague will reply to bruno at example dot com. I have (650) 555-0142 as your mobile, still right?"),
             ("user", "Yes, that's right. Thanks, bye")]
    assert confirmed_details(lines) == [("email", "bruno@example.com")]
    assert confirmed_details([("user", "my number is 650-555-0142"),
                              ("assistant", "Six five zero, five five five, zero one four two, right?"), ("user", "yes")]) == [("phone", "6505550142")]
    assert confirmed_details([("user", "it's at bruno dot com, I'm at home"), ("assistant", "okay")]) == []

    writes: list[tuple[str, str, dict]] = []

    class _Mem:
        def write(self, entity, key, value, **_kw):
            writes.append((entity, key, value))

    class _Client:
        def memory(self, _agent_id):
            return _Mem()

    ag = LiveAgent.__new__(LiveAgent)
    ag._files, ag._files_dirty, ag._lock = {}, set(), threading.Lock()
    ag.client, ag.agent_id, ag.entity_path, ag.business, ag.phonebook = _Client(), "john", "jev-demos/voice", BUSINESS, None
    ag.events = []
    ag._event = lambda kind, data: ag.events.append((kind, data))  # type: ignore[method-assign]
    call = {"id": "c-bruno", "caller": {"customer_id": "bruno-andrade", "name": "Bruno"},
            "live": [{"role": r, "text": t} for r, t in lines], "customer_brief": "Caller: Bruno."}
    fut: Future = Future()
    fut.set_result(CustomerMemory(customer_id="bruno-andrade", notes="Mobile on file: (650) 555-0142."))
    ag._file_details_from_call(call, fut)
    assert [(k, v["kind"], v["value"]) for _, k, v in writes] == [("detail-email", "email", "bruno@example.com")]
    assert writes[0][0] == "jev-demos/voice/customers/bruno-andrade"
    assert ag.events == [("file_updated", {"call_id": "c-bruno", "customer_id": "bruno-andrade", "detail": "email", "key": "detail-email",
                                           "value": "bruno@example.com", "note": "given on the call and read back; filed at hang-up",
                                           "source": "transcript"})]
    assert call["customer_brief"].endswith("On file now (email, given on this call): bruno@example.com.")
    assert ag._files_dirty == {"bruno-andrade"}
    # Already on file (the agent did call update_file this time, or an earlier call had it): nothing is written twice.
    writes.clear()
    ag.events.clear()
    call["file_updates"] = [{"kind": "email", "value": "Bruno@example.com"}]
    ag._file_details_from_call(call, fut)
    assert writes == []
    del call["file_updates"]
    fut2: Future = Future()
    fut2.set_result(CustomerMemory(customer_id="bruno-andrade", details={"email": {"value": "bruno@example.com", "date": "2026-10-01"}}))
    ag._file_details_from_call(call, fut2)
    assert writes == []
    # A stranger has no file to put it on.
    ag._file_details_from_call({"id": "c-x", "caller": {}, "live": call["live"]}, None)
    assert writes == []
