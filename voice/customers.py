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
from dataclasses import asdict, dataclass, field
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
        )

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("history")
        return d




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
                  agent_name: str = "The agent", action_words: dict[str, str] | None = None,
                  today: dt.date | None = None) -> tuple[str, dict[str, Any]]:
    """The entry the agent writes after a call, from the sealed outcome. *action_words* says
    how each action reads in a history line ("booked a specialist callback")."""
    date = _date(0, today)
    did = (action_words or {}).get(action) or action.replace("_", " ")
    text = (f"Called about {intent.replace('_', ' ')}: {summary.rstrip('.')}. {agent_name} {did}. "
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
    tier = f"{caller.tier} plan" if caller.tier and "plan" not in caller.tier else (caller.tier or "customer")
    contacts = memory.contacts if memory else []
    today = today or dt.date.today()
    recent = sum(1 for c in contacts if _days_ago(c["date"], today) <= 30)
    who = caller.name or "unknown (no account matched this call)"
    lines = [f"Caller: {who}. {tier.capitalize()}, customer for {caller.tenure_months} months. "
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
    lines.append("Use this to show you know what has happened: pick up the thread and refer to one or two specifics in "
                 "your own words. Do not read it out as a list. The history is the past; what the caller needs today "
                 "is whatever they have told you on this call. Do not promise a fix, a transfer, a credit or a "
                 "callback before decide_next_step has answered.")
    return "\n".join(lines)


__all__ = ["CustomerMemory", "Persona", "brief", "contact_entry", "customer_entity", "seed_entries"]
