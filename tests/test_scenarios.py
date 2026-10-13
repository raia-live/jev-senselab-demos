"""Keyless checks on the scenarios: deterministic, complete, and the review never tells."""

from __future__ import annotations

import random

import pytest

from demos import NAMES, load


@pytest.mark.parametrize("name", NAMES)
def test_cases_are_deterministic(name: str) -> None:
    sc = load(name)
    a = sc.cases(seed=11, episodes=40, change_at=20, probe_at=30)
    b = sc.cases(seed=11, episodes=40, change_at=20, probe_at=30)
    assert [(c.id, c.text, c.truth) for c in a] == [(c.id, c.text, c.truth) for c in b]
    assert len({c.id for c in a}) == 40


@pytest.mark.parametrize("name", NAMES)
def test_every_wrong_action_has_a_review(name: str) -> None:
    sc = load(name)
    rng = random.Random(0)
    for case in sc.cases(seed=23, episodes=60, change_at=30, probe_at=45):
        assert case.truth in sc.actions
        for action in sc.actions:
            if action == case.truth:
                continue
            assert sc.severity_of(case, action) in ("minor_failure", "failure", "critical_failure")
            note = sc.failure_note(case, action, rng)
            assert note


@pytest.mark.parametrize("name", NAMES)
def test_failure_notes_never_name_the_right_action(name: str) -> None:
    sc = load(name)
    rng = random.Random(1)
    for case in sc.cases(seed=37, episodes=60, change_at=30, probe_at=45):
        for action in sc.actions:
            if action == case.truth:
                continue
            note = sc.failure_note(case, action, rng).lower()
            # The action identifier, and its words, must not appear in the review of a miss.
            assert case.truth not in note
            for word in case.truth.split("_"):
                if len(word) > 4 and word not in ("budget", "reach", "human", "review"):
                    assert word not in note, (name, case.truth, note)


@pytest.mark.parametrize("name", NAMES)
def test_streams_have_tacit_changed_and_probe_cases(name: str) -> None:
    sc = load(name)
    cases = sc.cases(seed=11, episodes=120, change_at=60, probe_at=90)
    assert 25 <= sum(c.is_tacit for c in cases) <= 60
    assert sum(c.is_changed for c in cases) >= 5
    assert all(c.episode >= 60 for c in cases if c.is_changed)
    assert sum(c.is_probe for c in cases) >= 4
    assert all(c.episode >= 90 and c.is_tacit for c in cases if c.is_probe)


def test_review_noise_flips_only_the_label() -> None:
    sc = load("lead_routing")
    case = sc.cases(seed=11, episodes=1, change_at=-1, probe_at=-1)[0]
    right = sc.review(case, case.truth, random.Random(0), noise=0.0)
    assert right.success and right.true_success and right.cost_usd == 0
    wrong_action = next(a for a in sc.actions if a != case.truth)
    wrong = sc.review(case, wrong_action, random.Random(0), noise=0.0)
    assert not wrong.success and not wrong.true_success and wrong.cost_usd > 0
    flipped = sc.review(case, wrong_action, random.Random(0), noise=1.0)
    assert flipped.success and not flipped.true_success and flipped.noisy and flipped.cost_usd > 0
