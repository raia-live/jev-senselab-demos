"""The scenario protocol every demo implements.

A scenario is a stream of cases a business decides every day, a written policy that is
mostly right, one tacit rule the policy gets wrong, a change half-way through the run
that moves one more rule, and a review that reports how each decision turned out.

Three properties are deliberate and shared by every demo:

* **The review never names the correct action.** It says pass or fail, how bad, what it
  cost and one line of what happened — what a real downstream team reports. The agent
  has to work out what to do differently.
* **Some reviews are wrong.** ``noise`` is the share of reviews flipped for reasons
  unrelated to the decision (a deal that was lost anyway, an appeal that was granted on
  a technicality). Real feedback is like this.
* **The tacit trigger is low-salience in the text.** The field that decides the hidden
  rule is there — in the free text as a passing mention and in the structured fields —
  but nothing points at it.

Scoring uses the hidden truth; learning uses only the review.
"""

from __future__ import annotations

import fnmatch
import hashlib
import random
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any

SEVERITIES = ("success", "minor_failure", "failure", "critical_failure")


@dataclass
class Case:
    """One case as the agent sees it, plus the hidden truth used for scoring."""

    id: str
    episode: int
    text: str
    fields: dict[str, Any]
    situation: str
    truth: str
    kind: str
    is_tacit: bool = False
    is_changed: bool = False
    is_probe: bool = False

    def visible(self) -> dict[str, Any]:
        """What goes into Jev's state: the text and the structured fields, nothing hidden."""
        return {"text": self.text, "fields": dict(self.fields)}

    def to_row(self) -> dict[str, Any]:
        return {
            "case_id": self.id,
            "situation": self.situation,
            "kind": self.kind,
            "truth": self.truth,
            "is_tacit": self.is_tacit,
            "is_changed": self.is_changed,
            "is_probe": self.is_probe,
        }


@dataclass
class Outcome:
    """What the review reported. ``success``/``severity``/``note`` may be noisy;
    ``true_success`` and ``cost_usd`` are scored against the hidden truth."""

    success: bool
    severity: str
    note: str
    true_success: bool
    cost_usd: float
    noisy: bool = False

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyClaim:
    """One written rule as a claim: on the situations matching *situations* (``fnmatch``
    patterns over the situation label), *action* is the right thing to do. The policy entry
    *key* is what an outcome then confirms or refutes."""

    key: str
    action: str
    situations: tuple[str, ...]

    def matches(self, situation: str) -> bool:
        return any(fnmatch.fnmatchcase(situation, pat) for pat in self.situations)


@dataclass
class Scenario(ABC):
    """Subclass per demo. The class attributes below are the contract."""

    name: str = ""
    title: str = ""
    case_noun: str = "case"
    actions: list[str] = field(default_factory=list)
    #: Instructions on the Jev ``Choice`` question.
    jev_instructions: str = ""
    #: One description per action, the ``criteria`` of the Jev ``Choice``.
    jev_criteria: dict[str, str] = field(default_factory=dict)
    #: The written policy as (key, text) lines. In Jev's state for the plain arms; seeded
    #: as memory entries for the SenseLab arms, so the repair loop has something to correct.
    policy: list[tuple[str, str]] = field(default_factory=list)
    #: The policy lines as claims, in order: the first whose situations match a case is the
    #: rule for it. A scenario that leaves this empty charges no policy entry with outcomes.
    policy_claims: list[PolicyClaim] = field(default_factory=list)
    #: Scenario notes appended to the judge prompt (what a failure looks like here).
    judge_rubric: str = ""
    #: Dollars a wrong decision costs, by severity.
    cost_by_severity: dict[str, float] = field(default_factory=lambda: {
        "minor_failure": 100.0, "failure": 1000.0, "critical_failure": 10000.0,
    })
    #: One-line statements for the README and report.
    tacit_rule: str = ""
    change_rule: str = ""
    expected_fix: str = ""

    # -- what a subclass implements ------------------------------------------------------

    @abstractmethod
    def make_case(self, episode: int, rng: random.Random, *, changed: bool, probe: bool) -> Case:
        """Draw one case. ``changed`` says the drift is in force; ``probe`` says to use the
        held-out phrasings for tacit cases."""

    @abstractmethod
    def severity_of(self, case: Case, action: str) -> str:
        """How bad a wrong ``action`` is on this case (never called when it is right)."""

    @abstractmethod
    def failure_note(self, case: Case, action: str, rng: random.Random) -> str:
        """What the review says about a wrong decision. Must not name the right action."""

    def success_note(self, case: Case, action: str, rng: random.Random) -> str:
        return rng.choice([
            "Handled as expected.", "No issues reported.", "Closed without escalation.",
            "Outcome as intended.",
        ])

    def noise_fail_note(self, case: Case, action: str, rng: random.Random) -> str:
        """A failure reported on a correct decision, for a reason unrelated to it."""
        return rng.choice([
            "Went badly for reasons outside the decision; flagged by the reviewing team.",
            "Reviewer marked this one a miss; no detail recorded.",
            "Escalated later by a different team; root cause unclear.",
        ])

    def noise_pass_note(self, case: Case, action: str, rng: random.Random) -> str:
        """A success reported on a wrong decision — it happened to work out."""
        return rng.choice([
            "Worked out in the end.", "No complaint received.", "Closed; nothing further.",
        ])

    # -- shared machinery ----------------------------------------------------------------

    def cases(self, *, seed: int, episodes: int, change_at: int, probe_at: int) -> list[Case]:
        """The deterministic case stream for one seed. Every arm sees the same cases."""
        rng = random.Random(f"{self.name}:{seed}:cases")
        out: list[Case] = []
        for ep in range(episodes):
            case = self.make_case(ep, rng, changed=ep >= change_at >= 0, probe=ep >= probe_at >= 0)
            if not case.id:
                case.id = f"{self.name}-{seed}-{ep:03d}"
            out.append(case)
        return out

    def review(self, case: Case, action: str, rng: random.Random, *, noise: float) -> Outcome:
        """Pass/fail, severity, cost and a note. ``noise`` flips a share of the verdicts."""
        if action not in self.actions:
            # An action outside the set is a wrong decision of the ordinary kind.
            severity = "failure"
            return Outcome(False, severity, f"Unknown action '{action}'.", False,
                           self.cost_by_severity.get(severity, 0.0))
        right = action == case.truth
        flip = rng.random() < noise
        if right:
            cost = 0.0
            if flip:
                return Outcome(False, "minor_failure", self.noise_fail_note(case, action, rng), True, cost, True)
            return Outcome(True, "success", self.success_note(case, action, rng), True, cost)
        severity = self.severity_of(case, action)
        cost = self.cost_by_severity.get(severity, 0.0)
        if flip:
            return Outcome(True, "success", self.noise_pass_note(case, action, rng), False, cost, True)
        return Outcome(False, severity, self.failure_note(case, action, rng), False, cost)

    def policy_text(self) -> str:
        return "\n".join(f"- {text}" for _, text in self.policy)

    def policy_claim(self, case: Case) -> PolicyClaim | None:
        """The written rule that decides *case*, when one does."""
        return next((c for c in self.policy_claims if c.matches(case.situation)), None)

    def candidate_actions(self) -> list[str]:
        return [f"decide:{a}" for a in self.actions]

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name, "title": self.title, "actions": list(self.actions),
            "policy": [{"key": k, "text": t} for k, t in self.policy],
            "tacit_rule": self.tacit_rule, "change_rule": self.change_rule,
            "expected_fix": self.expected_fix, "cost_by_severity": dict(self.cost_by_severity),
        }


def stable_id(*parts: Any) -> str:
    return hashlib.sha1(":".join(str(p) for p in parts).encode()).hexdigest()[:10]


__all__ = ["Case", "Outcome", "SEVERITIES", "Scenario", "stable_id"]
