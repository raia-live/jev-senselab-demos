"""Who is calling, from the number: your lookup first, then what the agent has learned."""

from typing import Any

import pytest

from voice import phonebook as pb


class _Store:
    def __init__(self, fail: bool = False) -> None:
        self.writes: list[tuple[str, str, dict[str, Any]]] = []
        self.fail = fail

    def memory(self, _agent_id: str) -> "_Store":
        return self

    def write(self, entity: str, key: str, value: dict[str, Any], **_kw: Any) -> None:
        if self.fail:
            raise RuntimeError("down")
        self.writes.append((entity, key, value))

    def list(self, entity: str) -> list[Any]:
        if self.fail:
            raise RuntimeError("down")
        return [type("E", (), {"key": k, "value": v})() for e, k, v in self.writes if e == entity]


def test_numbers_are_one_spelling_and_stored_under_a_hash():
    assert pb.normalise("+1 (555) 000-1234") == "+15550001234" == pb.normalise("15550001234")
    assert pb.normalise("") == "" and pb.key_for("+1 555 000 1234") == pb.key_for("+15550001234")
    assert pb.key_for("+15550001234").startswith("number-") and "5550001234" not in pb.key_for("+15550001234")


def test_the_book_learns_a_number_and_reads_it_back():
    store = _Store()
    book = pb.PhoneBook(store, "john", "acme/support")
    assert book.who("+15550001111") == ({}, None) and len(book) == 0
    facts = {"name": "Bruno", "tier": "consumer", "prior_contacts_30d": 3, "tenure_months": 39,
             "customer_id": "bruno-andrade", "history": "not stored", "profile_id": "nor this"}
    assert book.remember("+1 555 000 1111", facts) is True
    assert book.remember("+15550001111", facts) is False  # already them
    assert book.who("+15550001111") == ({"name": "Bruno", "tier": "consumer", "prior_contacts_30d": 3,
                                         "tenure_months": 39, "customer_id": "bruno-andrade"}, "number")
    entity, key, value = store.writes[0]
    assert entity == "acme/support/phonebook" and key == pb.key_for("+15550001111") and "history" not in value
    assert book.remember("+15550009999", {"name": "Nobody"}) is False  # no account id: nothing to remember
    again = pb.PhoneBook(store, "john", "acme/support")
    assert again.load() == 1 and again.who("+15550001111")[1] == "number"


def test_your_lookup_answers_first_and_a_failing_one_is_a_stranger(monkeypatch):
    store = _Store()
    book = pb.PhoneBook(store, "john", "acme/support")
    book.remember("+15550001111", {"name": "Bruno", "customer_id": "bruno-andrade"})
    monkeypatch.setattr(pb, "lookup", lambda n: {"name": "Bruno A.", "tier": "business", "customer_id": "crm-42"} if n == "+15550001111" else None)
    assert book.who("+1 (555) 000-1111") == ({"name": "Bruno A.", "tier": "business", "customer_id": "crm-42"}, "lookup")
    assert book.who("+15550002222") == ({}, None)
    monkeypatch.setattr(pb, "lookup", lambda n: 1 / 0)
    assert book.who("+15550001111")[1] == "number"  # the book still answers
    monkeypatch.setattr(pb, "lookup", lambda n: {"name": "No id"})  # a row without customer_id is no match
    assert book.who("+15550001111")[1] == "number"


def test_senselab_down_keeps_the_book_in_memory_for_the_run():
    book = pb.PhoneBook(_Store(fail=True), "john", "acme/support")
    assert book.load() == 0
    assert book.remember("+15550001111", {"name": "Bruno", "customer_id": "bruno-andrade"}) is True
    assert book.who("+15550001111")[1] == "number"


@pytest.mark.parametrize("number", ["", "   ", "abc"])
def test_no_number_is_nobody(number):
    assert pb.PhoneBook(_Store(), "john", "acme/support").who(number) == ({}, None)
