"""What the account remembers about a caller, kept in SenseLab per customer.

The situation record (``seen-*``, ``learned-*`` on the entity) is what SenseLab knows about
callers *like* this one. This module is the other half: what it knows about *this* caller,
from their earlier contacts. It lives at ``<entity>/customers/<customer_id>`` as one entry per
contact, written by the agent after each call from the sealed outcome, and read at the start
of the next call so the agent can pick the conversation up where it was left.

The demo callers (``[[personas]]`` in ``voice/business.toml``) come with a seeded history,
so a first demo call has something to know. Those entries are marked ``seeded``; everything
after them is written by the agent from what actually happened on a call. Call twice as the
same person to see the second kind appear.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import asdict, dataclass, field, replace
from typing import Any


@dataclass
class Persona:
    """A demo caller: the account facts, the seeded history, and a script for the person
    playing them (several turns, so the call is a conversation, not a phrase)."""

    id: str
    label: str
    name: str
    tier: str
    prior_contacts_30d: int
    tenure_months: int
    intent: str
    amount_usd: float
    history: list[tuple[int, str]]  # (days ago, what happened)
    notes: str
    script: list[str]
    why: str
    pushback: str = ""
    # What the label and the "why" say when outcomes are reported, not simulated: a persona
    # whose point is the demo world's switch has no switch to point at then. Empty means
    # the same words in both modes.
    label_reported: str = ""
    why_reported: str = ""

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Persona:
        """One ``[[personas]]`` table from the business file."""
        return cls(
            id=re.sub(r"[^a-z0-9-]", "", str(row["id"]).lower())[:40], label=str(row.get("label") or row.get("name") or row["id"]),
            name=str(row.get("name") or row["id"]), tier=str(row.get("tier") or ""),
            prior_contacts_30d=int(row.get("prior_contacts_30d") or 0), tenure_months=int(row.get("tenure_months") or 0),
            intent=str(row.get("intent") or ""), amount_usd=float(row.get("amount_usd") or 0),
            history=[(int(h[0]), str(h[1])) for h in row.get("history") or ()],
            notes=str(row.get("notes") or ""), script=[str(x) for x in row.get("script") or ()],
            why=str(row.get("why") or ""), pushback=str(row.get("pushback") or ""),
            label_reported=str(row.get("label_reported") or ""), why_reported=str(row.get("why_reported") or ""),
        )

    def for_mode(self, reported: bool) -> Persona:
        """The persona as the page should show it in this outcomes mode."""
        if not reported or not (self.label_reported or self.why_reported):
            return self
        return replace(self, label=self.label_reported or self.label, why=self.why_reported or self.why)

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("history")
        return d




def customer_entity(entity_path: str, customer_id: str) -> str:
    return f"{entity_path}/customers/{customer_id}"


def _date(days_ago: int, today: dt.date | None = None) -> str:
    return ((today or dt.date.today()) - dt.timedelta(days=days_ago)).isoformat()


def _stamp(today: dt.date | None = None, now: dt.datetime | None = None) -> str:
    """When an entry was written, to the second, so two calls on one day sort in the order
    they happened and the brief can say "12 minutes ago". A fixed *today* without *now*
    (tests) stamps midnight."""
    if now is not None:
        return now.astimezone(dt.UTC).isoformat(timespec="seconds")
    if today is not None:
        return f"{today.isoformat()}T00:00:00+00:00"
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _minutes_since(at: str, now: dt.datetime | None = None) -> float | None:
    try:
        then = dt.datetime.fromisoformat(at)
    except ValueError:
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=dt.UTC)
    return ((now or dt.datetime.now(dt.UTC)) - then).total_seconds() / 60


def seed_entries(persona: Persona, today: dt.date | None = None) -> list[tuple[str, dict[str, Any]]]:
    """The seeded history as (key, value) rows, oldest first."""
    rows: list[tuple[str, dict[str, Any]]] = []
    for i, (days_ago, text) in enumerate(sorted(persona.history, key=lambda h: -h[0]), start=1):
        rows.append((f"contact-{_date(days_ago, today)}-seed{i}", {"date": _date(days_ago, today), "text": text, "seeded": True}))
    if persona.notes:
        rows.append(("notes", {"date": _date(0, today), "text": persona.notes, "seeded": True}))
    return rows


def contact_entry(call_id: str, *, intent: str, summary: str, action: str, outcome_note: str, worked: bool | None,
                  agent_name: str = "The agent", action_words: dict[str, str] | None = None,
                  today: dt.date | None = None, now: dt.datetime | None = None) -> tuple[str, dict[str, Any]]:
    """The entry the agent writes after a call, from the sealed outcome. *action_words* says
    how each action reads in a history line ("booked a specialist callback"). *worked* None
    is a decision whose outcome is not in yet (reported mode): the line says so, under the
    same key, and is rewritten when the outcome lands."""
    date = _date(0, today)
    did = (action_words or {}).get(action) or action.replace("_", " ")
    came = outcome_note.rstrip(".") if worked is not None else "not known yet; ask, and put their answer on file"
    text = (f"Called about {intent.replace('_', ' ')}: {summary.rstrip('.')}. {agent_name} {did}. "
            f"What came of it: {came}." + (" It did not hold." if worked is False else ""))
    return (f"contact-{date}-{re.sub(r'[^A-Za-z0-9]', '', call_id)[-6:]}",
            {"date": date, "at": _stamp(today, now), "text": text, "action": action, "worked": worked})


def amend_contact(contact: dict[str, Any], *, worked: bool, words: str) -> dict[str, Any]:
    """The same contact line with the caller's word on what came of it in place of what the
    file said. Written back under the contact's own key: the caller's word wins over a
    verdict the system reached without them."""
    came = f"Caller said on a later call: {words.rstrip('.')}."
    text = _CAME_OF_IT.sub(lambda _m: f"What came of it: {came}", contact["text"]).rstrip()
    if "What came of it:" not in text:
        text = f"{text} What came of it: {came}"
    if not worked and not text.endswith("It did not hold."):
        text += " It did not hold."
    out = {k: v for k, v in contact.items() if k not in ("key", "seeded")}
    return {**out, "text": text, "worked": worked}


_ENDINGS = {"customer-ended-call": "the caller hung up", "assistant-ended-call": "{agent} ended the call",
            "silence-timed-out": "the line went quiet", "customer-did-not-answer": "the caller did not answer"}


def call_entry(call_id: str, *, caller_lines: list[str], agent_name: str = "The agent", ended_reason: str | None = None,
               seconds: int = 0, today: dt.date | None = None, now: dt.datetime | None = None) -> tuple[str, dict[str, Any]]:
    """The entry for a call that reached no decision: the caller rang, asked or said
    something, and nothing needed deciding (or they hung up before saying why they called).
    Every call goes on the record, so the next one can pick up the thread ("earlier today
    you asked what your last call was about") and a caller is never told we have no trace
    of a call they remember making. Carries the caller's own first words, since there is
    no decision to summarise them by."""
    date = _date(0, today)
    said = [s for s in (" ".join(str(x).split()) for x in caller_lines) if s]
    how = _ENDINGS.get(ended_reason or "", "the call ended").format(agent=agent_name)
    after = f" after {seconds} s" if seconds else ""
    if not said:
        text = f"Called and said nothing we could act on; {how}{after}."
    else:
        quote = "; ".join(f"\"{s[:120].rstrip('.')}\"" for s in said[:2])
        text = f"Called, no request to act on. Caller said: {quote}. {agent_name} answered from the file; {how}{after}."
    return (f"contact-{date}-{re.sub(r'[^A-Za-z0-9]', '', call_id)[-6:]}",
            {"date": date, "at": _stamp(today, now), "text": text, "action": None, "worked": None})


DETAIL_KINDS = {"phone": "phone number", "email": "email", "name": "preferred name", "correction": "correction to the file",
                "outcome": "what came of the last step, in the caller's words", "other": "note"}


def detail_entry(call_id: str, *, kind: str, value: str, note: str = "", today: dt.date | None = None,
                 now: dt.datetime | None = None) -> tuple[str, dict[str, Any]]:
    """A detail the caller gave for their file, written the moment it is confirmed on the
    call (``update_file``): a number to reach them on, an email, what to call them, or a
    correction to what the file says ("the callback never came"). One key per kind for the
    details that replace each other (the phone number on file is the latest one given;
    SenseLab keeps the earlier versions); corrections and notes each get their own."""
    kind = kind if kind in DETAIL_KINDS else "other"
    date = _date(0, today)
    value, note = " ".join(str(value).split())[:200], " ".join(str(note).split())[:300]
    key = f"detail-{kind}" if kind in ("phone", "email", "name") else f"note-{date}-{re.sub(r'[^A-Za-z0-9]', '', call_id)[-6:]}"
    return key, {"kind": kind, "value": value, "note": note, "date": date, "at": _stamp(today, now), "call_id": call_id}


@dataclass
class CustomerMemory:
    """What came back for one caller."""

    customer_id: str
    contacts: list[dict[str, Any]] = field(default_factory=list)  # newest first
    notes: str = ""
    details: dict[str, dict[str, Any]] = field(default_factory=dict)  # kind -> {value, note, date}
    remarks: list[dict[str, Any]] = field(default_factory=list)  # corrections and notes, newest first
    ms: int = 0

    @classmethod
    def from_entries(cls, customer_id: str, entries: list[Any], *, ms: int = 0) -> CustomerMemory:
        contacts: list[dict[str, Any]] = []
        details: dict[str, dict[str, Any]] = {}
        remarks: list[dict[str, Any]] = []
        notes = ""
        for e in entries:
            key, value = str(getattr(e, "key", "")), getattr(e, "value", None)
            if not isinstance(value, dict):
                continue
            if key == "notes":
                notes = str(value.get("text") or "")
            elif key.startswith("contact-"):
                contacts.append({"key": key, "date": str(value.get("date") or ""), "at": str(value.get("at") or ""),
                                 "text": str(value.get("text") or ""), "seeded": bool(value.get("seeded")),
                                 "action": value.get("action"), "worked": value.get("worked")})
            elif key.startswith("detail-") and value.get("value"):
                kind = key[len("detail-"):]
                if str(value.get("date") or "") >= str(details.get(kind, {}).get("date") or ""):
                    details[kind] = {"value": str(value["value"]), "note": str(value.get("note") or ""), "date": str(value.get("date") or "")}
            elif key.startswith("note-") and (value.get("value") or value.get("note")):
                remarks.append({"kind": str(value.get("kind") or "other"), "value": str(value.get("value") or ""),
                                "note": str(value.get("note") or ""), "date": str(value.get("date") or ""),
                                "at": str(value.get("at") or "")})
        # Newest first, to the second where the entry says; a seeded row (date only) sorts as
        # the start of its day.
        contacts.sort(key=_when_key, reverse=True)
        remarks.sort(key=_when_key, reverse=True)
        return cls(customer_id=customer_id, contacts=contacts, notes=notes, details=details, remarks=remarks, ms=ms)

    def public(self) -> dict[str, Any]:
        return asdict(self)


def _when_key(row: dict[str, Any]) -> str:
    return str(row.get("at") or "") or f"{row.get('date', '')}T00:00:00"


def _days_ago(iso: str, today: dt.date) -> int:
    try:
        return (today - dt.date.fromisoformat(iso)).days
    except ValueError:
        return 10**6


def _nice_date(iso: str, today: dt.date | None = None) -> str:
    try:
        d = dt.date.fromisoformat(iso)
    except ValueError:
        return iso
    today = today or dt.date.today()
    days = (today - d).days
    if days == 0:
        return "today"
    if days == 1:
        return "yesterday"
    if days < 7:
        return f"{days} days ago"
    return d.strftime("%b %-d") + ("" if d.year == today.year else f", {d.year}")


def _nice_when(row: dict[str, Any], today: dt.date | None = None, now: dt.datetime | None = None) -> str:
    """``_nice_date`` with the minutes for a row written today: "just now", "12 minutes
    ago", else "today"."""
    when = _nice_date(str(row.get("date") or ""), today)
    if when == "today" and row.get("at"):
        mins = _minutes_since(str(row["at"]), now)
        if mins is not None and 0 <= mins < 2:
            return "just now"
        if mins is not None and mins < 90:
            return f"{int(mins)} minutes ago"
    return when


_TOPIC = re.compile(r"^Called about ([^:]+):")
_CAME_OF_IT = re.compile(r"What came of it: (.+)$")


def history_lines(contacts: list[dict[str, Any]], today: dt.date | None = None, *, recent_groups: int = 4,
                  now: dt.datetime | None = None) -> list[str]:
    """The earlier contacts as lines, most recent first, sized for a prompt however long the
    record gets. The account's own history (the seeded entries: the plan, the named account
    manager, the incident two weeks ago) is always there in full. What the agent wrote after
    its own calls is folded: several calls on one day about one thing become one line saying
    how many, what was done and what came of it, and only the latest *recent_groups* of those
    are kept. Otherwise a caller who rings five times about an outage pushes their account
    manager's name out of the brief, and the agent cannot name them when asked."""
    today = today or dt.date.today()
    seeded = [c for c in contacts if c.get("seeded")]
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for c in contacts:
        if c.get("seeded"):
            continue
        m = _TOPIC.match(c["text"])
        groups.setdefault((c["date"], m.group(1) if m else ""), []).append(c)
    rows: list[tuple[str, str]] = [(c["date"], f"{_nice_date(c['date'], today)}: {c['text']}") for c in seeded]
    for (date, topic), cs in list(groups.items())[:recent_groups]:
        if len(cs) == 1 or not topic:
            rows.append((_when_key(cs[0]), f"{_nice_when(cs[0], today, now)}: {cs[0]['text']}"))
            continue
        latest = cs[0]["text"]
        outcomes = []
        for c in cs:
            m = _CAME_OF_IT.search(c["text"])
            if m and m.group(1) not in outcomes:
                outcomes.append(m.group(1))
        rows.append((_when_key(cs[0]), f"{_nice_date(date, today)}: called {len(cs)} times about {topic}. "
                                       f"Most recent ({_nice_when(cs[0], today, now)}): {latest} "
                                       f"Earlier the same day: {' / '.join(outcomes[1:4]) or 'the same'}"))
    rows.sort(key=lambda r: r[0], reverse=True)
    return [r[1] for r in rows]


def track_record(memory: CustomerMemory | None, action_words: dict[str, str] | None = None, *,
                 today: dt.date | None = None, now: dt.datetime | None = None) -> str:
    """This caller's own record with the steps we have taken for them, for the decision: how
    many times each was taken and how each fared, the caller's word included. What has worked
    for callers like them is one thing; a step that has already failed this caller twice is
    another, and the decision should see both."""
    if memory is None:
        return ""
    by_action: dict[str, list[dict[str, Any]]] = {}
    for c in memory.contacts:
        if c.get("action") and not c.get("seeded"):
            by_action.setdefault(str(c["action"]), []).append(c)
    if not by_action:
        return ""
    parts = []
    for action, cs in by_action.items():
        did = (action_words or {}).get(action) or action.replace("_", " ")
        n = len(cs)
        held = sum(1 for c in cs if c.get("worked") is True)
        failed = sum(1 for c in cs if c.get("worked") is False)
        open_ = n - held - failed
        how = []
        if failed:
            how.append(f"did not hold {failed} time{'s' if failed != 1 else ''}, on their word")
        if held:
            how.append(f"held {held} time{'s' if held != 1 else ''}")
        if open_:
            how.append(f"{open_} still open")
        parts.append(f"{did} {n} time{'s' if n != 1 else ''} (most recently {_nice_when(cs[0], today, now)}); "
                     f"it {', '.join(how)}")
    return "This caller's own record with these steps: " + "; ".join(parts) + "."


def brief(caller: Any, memory: CustomerMemory | None, *, record: str = "", today: dt.date | None = None,
          now: dt.datetime | None = None) -> str:
    """What the conversational model gets back from ``recall_customer``: the account facts, the
    earlier contacts and what came of them, the notes, and what has worked for callers in the
    same situation. Prose it can draw on, not a form to read out."""
    tier = f"{caller.tier} plan" if caller.tier and "plan" not in caller.tier else (caller.tier or "customer")
    contacts = memory.contacts if memory else []
    today = today or dt.date.today()
    recent = sum(1 for c in contacts if _days_ago(c["date"], today) <= 30)
    who = caller.name or "unknown (no account matched this call)"
    lines = [f"Caller: {who}. {tier.capitalize()}, customer for {caller.tenure_months} months. "
             f"{max(caller.prior_contacts_30d, recent)} contact(s) with support in the last 30 days."]
    if memory and memory.details:
        given = [f"{DETAIL_KINDS.get(k, k)} {d['value']} (given {_nice_date(d['date'], today)})" for k, d in memory.details.items()]
        lines.append(f"On file, from the caller: {'; '.join(given)}. These are current; use them, do not ask for them again.")
    if memory and memory.remarks:
        lines.append("Corrections and notes from earlier calls, most recent first:")
        lines.extend(f"- {_nice_when(r, today, now)}: {(r['value'] + ('. ' + r['note'] if r['note'] else '')).rstrip('.')}."
                     for r in memory.remarks[:4])
    if contacts:
        lines.append("Earlier contacts, most recent first:")
        lines.extend(f"- {c}" for c in history_lines(contacts, today, now=now))
    else:
        lines.append("No earlier contacts on record: this is the first time we hear from them.")
    if memory and memory.notes:
        lines.append(f"Notes: {memory.notes}")
    if record:
        lines.append(record)
    lines.append("Use this to show you know what has happened: pick up the thread and refer to one or two specifics in "
                 "your own words. Do not read it out as a list. The history is the past; what the caller needs today "
                 "is whatever they have told you on this call. Do not promise a fix, a transfer, a credit or a "
                 "callback before decide_next_step has answered.")
    return "\n".join(lines)


def opening(caller: Any, memory: CustomerMemory | None, *, agent_name: str, company: str, greeting: str,
            today: dt.date | None = None, now: dt.datetime | None = None) -> str:
    """The first thing the caller hears, built from the file rather than left to the model:
    a known caller is greeted by name and told, in a clause, that their history is in hand
    (how many times they have been in touch, how recently), then asked what is going on
    today. It presumes nothing about today's reason. A stranger gets the business's greeting.
    The conversational model takes over from the caller's first words, with the same history
    in its prompt."""
    if not caller.name or caller.name.lower() == "unknown":
        return greeting
    contacts = memory.contacts if memory else []
    today = today or dt.date.today()
    recent = [c for c in contacts if _days_ago(c["date"], today) <= 30]
    n = max(len(recent), int(caller.prior_contacts_30d or 0))
    lead = f"Hi {caller.name}, it's {agent_name} at {company}."
    if contacts and n:
        when = _nice_when(contacts[0], today, now)
        if when == "just now" or when.endswith("minutes ago"):
            # Back within the hour: say so, and skip the count they heard last time.
            return f"{lead} We spoke {when}; I've still got your file open here. What's going on?"
        when = _nice_date(contacts[0]["date"], today)
        when = when if when in ("today", "yesterday") or when.endswith("ago") else f"on {when}"
        times = {1: "once", 2: "twice", 3: "three times", 4: "four times", 5: "five times"}.get(n, f"{n} times")
        return (f"{lead} I've got your file open here. I can see you've been in touch {times} this past month, "
                f"most recently {when}, so no need to start from the beginning. What's going on today?")
    return f"{lead} I've got your file open here. What can I do for you today?"


__all__ = ["DETAIL_KINDS", "CustomerMemory", "Persona", "amend_contact", "brief", "call_entry", "contact_entry", "customer_entity",
           "detail_entry", "opening", "seed_entries", "track_record"]
