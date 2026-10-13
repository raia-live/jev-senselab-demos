"""What the account remembers about a caller, kept in SenseLab per customer.

The situation record (``seen-*``, ``learned-*`` on the entity) is what SenseLab knows about
callers *like* this one. This module is the other half: what it knows about *this* caller,
from their earlier contacts. It lives at ``<entity>/customers/<customer_id>`` as one entry per
contact, written by the agent after each call from the sealed outcome, and read at the start
of the next call so Riley can pick the conversation up where it was left.

The five demo callers come with a seeded history, so a first demo call has something to
know. Those entries are marked ``seeded``; everything after them is written by the agent
from what actually happened on a call. Call twice as the same person to see the second
kind appear.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import asdict, dataclass, field
from typing import Any

ACTION_WORDS = {
    "resolve_in_bot": "handled it on the call",
    "transfer_to_agent": "transferred the call to an agent",
    "schedule_callback": "booked a specialist callback",
    "issue_credit": "applied a credit",
}


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

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("history")
        return d


PERSONAS: list[Persona] = [
    Persona(
        id="dana-whitfield", label="Dana, third call about the same $49 (consumer)",
        name="Dana", tier="consumer", prior_contacts_30d=3, tenure_months=39, intent="billing_dispute", amount_usd=49,
        history=[
            (21, "Called about a $49 charge that appeared twice on the September invoice. Handled on the call: told "
                 "the duplicate would drop off the next invoice. What came of it: it did not; the charge posted again."),
            (12, "Called again about the same $49 duplicate. A $49 credit was applied on the call. What came of it: "
                 "finance reversed the credit eight days later as unsupported, and the caller was not told."),
            (5, "Called a third time, upset that the credit had vanished. Transferred to an agent; 31 minutes in the "
                "queue. The agent said a billing specialist would call back. What came of it: no callback was made."),
        ],
        notes="Prefers to be called back on the mobile number on file, afternoons.",
        script=[
            "Hi. I'm calling about my bill. Again.",
            "Yes, that one. And last time someone promised to call me back and nobody did.",
            "Why would it work this time? That's exactly what they told me last week.",
            "Fine. Will the credit come back too, or is that gone?",
            "Okay. Thanks.",
        ],
        why=("The written policy says a small billing dispute is resolved on the call. For Dana that has been tried, "
             "and a credit, and a transfer; each time she called back. Only the outcomes of those calls can teach the "
             "agent that, and only a memory of Dana lets Riley say so instead of asking her to explain a fourth time."),
        pushback="Riley has what was tried and what came of it, so 'why would it work this time' gets a real answer.",
    ),
    Persona(
        id="marcus-oyelaran", label="Marcus, first question about a $29 line (business)",
        name="Marcus", tier="business", prior_contacts_30d=0, tenure_months=18, intent="billing_dispute", amount_usd=29,
        history=[
            (540, "Account opened on the business plan, five seats."),
            (61, "Added three seats through the admin page (five to eight). No contact with support."),
        ],
        notes="",
        script=[
            "Hi, there's a $29 line on this month's invoice I don't recognise.",
            "Oh. Is that the extra seats? I did add some people a while back.",
            "Right, that makes sense then. That's all, thanks.",
        ],
        why="A plain case: the bot explains the charge. The history says where the $29 came from.",
    ),
    Persona(
        id="priya-raman", label="Priya, cancelling (consumer; flip the switch)",
        name="Priya", tier="consumer", prior_contacts_30d=0, tenure_months=11, intent="cancellation", amount_usd=0,
        history=[
            (330, "Signed up on the monthly plan after a two-week trial."),
            (40, "Usage review: two logins in the last sixty days, down from near daily in the spring. No contact with support."),
        ],
        notes="",
        script=[
            "Hi, I want to cancel my account. I'm just not using it enough.",
            "What would the offer be?",
            "Hmm. Okay, I'll give it two more months then.",
        ],
        why=("With the retention offer off, a human saves the account. Turn it on and the bot's offer saves it instead; "
             "the written policy still says transfer. Watch SenseLab notice the switch."),
    ),
    Persona(
        id="tomas-eriksen", label="Tomas, enterprise outage (the expensive mistake)",
        name="Tomas", tier="enterprise", prior_contacts_30d=1, tenure_months=30, intent="outage", amount_usd=0,
        history=[
            (900, "Enterprise plan, 240 seats. Named account manager: Jordan Blake."),
            (16, "SSO logins failed for about forty minutes. Escalated to engineering; root cause was a certificate "
                 "rotation at the identity provider. Post-incident review sent to the account manager."),
        ],
        notes="Wants a named person and a time, not a status page.",
        script=[
            "Our whole team is locked out. The dashboard has been down for twenty minutes.",
            "Is this the SSO thing again? We had this two weeks ago.",
            "Okay. Who's calling me, and when?",
        ],
        why="Enterprise goes to a named person. A bot answer here is the $5,000 mistake.",
    ),
    Persona(
        id="lena-hoffmann", label="Lena, a how-to (nothing to learn)",
        name="Lena", tier="consumer", prior_contacts_30d=0, tenure_months=5, intent="how_to", amount_usd=0,
        history=[(150, "Trial converted to the monthly plan. No contact with support.")],
        notes="",
        script=["Hi, how do I export my data to a spreadsheet?", "Where's that setting?", "Got it. Thanks."],
        why="The bot answers. Nothing for memory to add, and the page says so.",
    ),
]

PERSONA_BY_ID = {p.id: p for p in PERSONAS}


def customer_entity(entity_path: str, customer_id: str) -> str:
    return f"{entity_path}/customers/{customer_id}"


def _date(days_ago: int, today: dt.date | None = None) -> str:
    return ((today or dt.date.today()) - dt.timedelta(days=days_ago)).isoformat()


def seed_entries(persona: Persona, today: dt.date | None = None) -> list[tuple[str, dict[str, Any]]]:
    """The seeded history as (key, value) rows, oldest first."""
    rows: list[tuple[str, dict[str, Any]]] = []
    for i, (days_ago, text) in enumerate(sorted(persona.history, key=lambda h: -h[0]), start=1):
        rows.append((f"contact-{_date(days_ago, today)}-seed{i}", {"date": _date(days_ago, today), "text": text, "seeded": True}))
    if persona.notes:
        rows.append(("notes", {"date": _date(0, today), "text": persona.notes, "seeded": True}))
    return rows


def contact_entry(call_id: str, *, intent: str, summary: str, action: str, outcome_note: str, worked: bool,
                  today: dt.date | None = None) -> tuple[str, dict[str, Any]]:
    """The entry the agent writes after a call, from the sealed outcome."""
    date = _date(0, today)
    text = (f"Called about {intent.replace('_', ' ')}: {summary.rstrip('.')}. Riley {ACTION_WORDS.get(action, action)}. "
            f"What came of it: {outcome_note.rstrip('.')}." + ("" if worked else " It did not hold."))
    return f"contact-{date}-{re.sub(r'[^A-Za-z0-9]', '', call_id)[-6:]}", {"date": date, "text": text, "action": action, "worked": worked}


@dataclass
class CustomerMemory:
    """What came back for one caller."""

    customer_id: str
    contacts: list[dict[str, Any]] = field(default_factory=list)  # newest first
    notes: str = ""
    ms: int = 0

    @classmethod
    def from_entries(cls, customer_id: str, entries: list[Any], *, ms: int = 0) -> CustomerMemory:
        contacts: list[dict[str, Any]] = []
        notes = ""
        for e in entries:
            key, value = str(getattr(e, "key", "")), getattr(e, "value", None)
            if not isinstance(value, dict):
                continue
            if key == "notes":
                notes = str(value.get("text") or "")
            elif key.startswith("contact-"):
                contacts.append({"date": str(value.get("date") or ""), "text": str(value.get("text") or ""),
                                 "seeded": bool(value.get("seeded")), "action": value.get("action"), "worked": value.get("worked")})
        contacts.sort(key=lambda c: c["date"], reverse=True)
        return cls(customer_id=customer_id, contacts=contacts, notes=notes, ms=ms)

    def public(self) -> dict[str, Any]:
        return asdict(self)


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


def brief(caller: Any, memory: CustomerMemory | None, *, record: str = "", today: dt.date | None = None) -> str:
    """What the conversational model gets back from ``recall_customer``: the account facts, the
    earlier contacts and what came of them, the notes, and what has worked for callers in the
    same situation. Prose it can draw on, not a form to read out."""
    tier = {"consumer": "consumer plan", "business": "business plan", "enterprise": "enterprise plan"}.get(caller.tier, caller.tier)
    contacts = memory.contacts if memory else []
    today = today or dt.date.today()
    recent = sum(1 for c in contacts if _days_ago(c["date"], today) <= 30)
    lines = [f"Caller: {caller.name}. {tier.capitalize()}, customer for {caller.tenure_months} months. "
             f"{max(caller.prior_contacts_30d, recent)} contact(s) with support in the last 30 days."]
    if contacts:
        lines.append("Earlier contacts, most recent first:")
        for c in contacts[:6]:
            lines.append(f"- {_nice_date(c['date'], today)}: {c['text']}")
    else:
        lines.append("No earlier contacts on record: this is the first time we hear from them.")
    if memory and memory.notes:
        lines.append(f"Notes: {memory.notes}")
    if record:
        lines.append(record)
    lines.append("Use this to show you know what has happened: pick up the thread, refer to one or two specifics in "
                 "your own words, and if the history makes the problem obvious confirm it in one short question "
                 "instead of asking them to explain from the start. Do not read it out as a list. Do not promise a "
                 "fix, a transfer, a credit or a callback before decide_next_step has answered.")
    return "\n".join(lines)


__all__ = ["ACTION_WORDS", "PERSONAS", "PERSONA_BY_ID", "CustomerMemory", "Persona", "brief", "contact_entry",
           "customer_entity", "seed_entries"]
