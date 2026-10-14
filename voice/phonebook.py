"""Who is calling, from the number.

A phone call carries the caller's number and nothing else of ours. Before the first word,
this module answers "whose number is this?", so the agent can greet them by name with their
history in the prompt. Two sources, in order:

1. ``lookup(number)``: yours. The default body returns None. Replace it with a query to
   your CRM or billing system and return the account facts the agent works from (``name``,
   ``tier``, ``prior_contacts_30d``, ``tenure_months``, ``customer_id``); the history under
   that ``customer_id`` then comes from SenseLab as for any caller.
2. SenseLab: numbers the agent has learned itself. When a caller is registered on the page
   and then dials, or a caller nobody matched says who they are and the name matches an
   account, the number is written under ``<entity>/phonebook`` as that person, and read
   back at startup. From then on the number is them: no button, no ten-minute window.

Numbers are stored under a hash of the number, not the number, so the entity's keys do not
list who has called. The facts stored are the account facts above and nothing from the call.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from typing import Any

log = logging.getLogger(__name__)

FACT_KEYS = ("name", "tier", "prior_contacts_30d", "tenure_months", "customer_id")


def lookup(number: str) -> dict[str, Any] | None:  # noqa: ARG001 - yours to fill in
    """Your account lookup. Return the account facts for *number* (E.164, as Vapi reports
    it: ``+16825551234``), or None when it is nobody you know. Called on every inbound
    phone call before SenseLab's own book, so your system's answer wins. For example::

        row = crm.customers.find_one({"phone": number})
        if row is None:
            return None
        return {"name": row["first_name"], "tier": row["plan"], "customer_id": row["id"],
                "tenure_months": row["months_active"], "prior_contacts_30d": row["tickets_30d"]}
    """
    return None


def normalise(number: str) -> str:
    """One spelling per number: digits with a leading plus, as Vapi reports them. A number
    a caller gives out loud ("six five zero, eight six one...") comes as words or digits
    without a country code; ten digits are taken as North American (+1), which is where
    the demo's number is. Change that if yours is elsewhere."""
    text = str(number or "").lower()
    for word, digit in _DIGIT_WORDS.items():
        text = re.sub(rf"\b{word}\b", digit, text)
    digits = re.sub(r"\D", "", text)
    if len(digits) == 10:
        digits = "1" + digits
    return f"+{digits}" if digits else ""


_DIGIT_WORDS = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
                "seven": "7", "eight": "8", "nine": "9"}


def key_for(number: str) -> str:
    return "number-" + hashlib.sha256(normalise(number).encode()).hexdigest()[:20]


def facts_of(m: dict[str, Any]) -> dict[str, Any]:
    return {k: m[k] for k in FACT_KEYS if m.get(k) not in (None, "")}


class PhoneBook:
    """The numbers the agent knows, kept in SenseLab under ``<entity>/phonebook`` and held
    in memory for the call path. *client* is the SenseLab client, as the agent has it."""

    def __init__(self, client: Any, agent_id: str, entity_path: str) -> None:
        self.client, self.agent_id = client, agent_id
        self.entity = f"{entity_path}/phonebook"
        self._rows: dict[str, dict[str, Any]] = {}  # key -> facts
        self._lock = threading.Lock()

    def load(self) -> int:
        """Read the book from SenseLab (at startup). Returns how many numbers it knows."""
        try:
            entries = self.client.memory(self.agent_id).list(self.entity)
        except Exception as exc:  # noqa: BLE001 - the book starts empty and fills as calls come
            log.warning("phone book not read from SenseLab: %s", exc)
            return 0
        rows = {}
        for e in entries:
            key, value = str(getattr(e, "key", "")), getattr(e, "value", None)
            if key.startswith("number-") and isinstance(value, dict) and value.get("customer_id"):
                rows[key] = facts_of(value)
        with self._lock:
            self._rows = rows
        return len(rows)

    def who(self, number: str) -> tuple[dict[str, Any], str | None]:
        """The account facts for *number* and where they came from: ``lookup`` (your
        system), ``number`` (SenseLab's book), or nothing."""
        number = normalise(number)
        if not number:
            return {}, None
        try:
            mine = lookup(number)
        except Exception as exc:  # noqa: BLE001 - a lookup that fails is a stranger, not a dead call
            log.warning("phone lookup failed: %s", exc)
            mine = None
        if mine and mine.get("customer_id"):
            return facts_of(mine), "lookup"
        with self._lock:
            row = self._rows.get(key_for(number))
        return (dict(row), "number") if row else ({}, None)

    def remember(self, number: str, facts: dict[str, Any]) -> bool:
        """The number is this person from now on. Written to SenseLab and kept in memory;
        returns whether anything changed."""
        number = normalise(number)
        row = facts_of(facts)
        if not number or not row.get("customer_id"):
            return False
        key = key_for(number)
        with self._lock:
            if self._rows.get(key) == row:
                return False
            self._rows[key] = row
        try:
            self.client.memory(self.agent_id).write(self.entity, key, row, confidence=1.0, memory_type="fact")
        except Exception as exc:  # noqa: BLE001 - kept in memory for this run at least
            log.warning("phone book not written to SenseLab: %s", exc)
        log.info("phone book: a number is now %s (%s)", row.get("name"), row.get("customer_id"))
        return True

    def __len__(self) -> int:
        return len(self._rows)


__all__ = ["FACT_KEYS", "PhoneBook", "facts_of", "key_for", "lookup", "normalise"]
