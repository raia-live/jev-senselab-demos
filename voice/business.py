"""The business the voice agent answers for, read from ``voice/business.toml``.

Everything about the business is in that one file: the company and the agent's name, the
greeting, the intents callers have, the actions the agent can take, the written policy Jev
decides on, how a call's situation is named for memory, how outcomes are learned, and the
demo callers. ``voice/agent.py``, ``voice/assistant.py`` and ``voice/server.py`` read it
from here and carry nothing of Northwind themselves, so pointing ``VOICE_BUSINESS`` at
another file is all it takes to run the agent for another business.

Outcomes come in two modes. ``simulated`` is the demo's: a scenario module (the one the
batch harness runs) reviews each decision when the call ends, so the loop closes in
seconds. ``reported`` is a real business's: the agent waits for the outcome to be reported,
on the page or by ``POST /api/outcome`` from the system that knows.
"""

from __future__ import annotations

import importlib
import os
import random
import tomllib
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

from core.scenario import Case, Outcome, Scenario

from .customers import Persona

DEFAULT_PATH = Path(__file__).with_name("business.toml")
OUTCOME_MODES = ("simulated", "reported")


@dataclass(frozen=True)
class Intent:
    id: str
    label: str
    needs: tuple[str, ...] = ()
    ahead: tuple[dict[str, Any], ...] = ({},)


@dataclass(frozen=True)
class Action:
    id: str
    label: str
    jev: str
    words: str
    then: str
    # The same step for a caller the lookup did not match: there is no file to answer from,
    # so "say what the charge is" has to become "take an email and have it checked". Empty
    # means ``then`` serves both.
    then_no_file: str = ""


@dataclass(frozen=True)
class Split:
    """A number that changes what works: ``field`` below ``at`` is ``below``, else ``from_``."""

    field: str
    at: float
    below: str
    from_: str
    intents: tuple[str, ...] = ()  # empty: every intent

    def applies(self, intent: str) -> bool:
        return not self.intents or intent in self.intents

    def label(self, fields: dict[str, Any]) -> str:
        try:
            value = float(fields.get(self.field) or 0)
        except (TypeError, ValueError):
            value = 0.0
        return self.below if value < self.at else self.from_


@dataclass(frozen=True)
class CallScenario:
    """One scripted test call for ``voice.testcalls``: who calls (a persona from the file,
    or nobody we know), the script Vapi's tester follows, and the yes/no questions the
    judge answers from the transcript. Read from ``[[scenarios]]`` in the business file, so
    a business writes its own test calls next to its callers, in TOML, not Python."""

    id: str
    title: str
    script: str
    judge: tuple[tuple[str, str], ...]  # (name, what the judge checks), each a yes/no
    persona: str | None = None  # a persona id, or None for a caller we have no record of
    decisions: int = 1  # how many decide_next_step calls the server should see
    tags: tuple[str, ...] = ()

    @property
    def known(self) -> bool:
        """The opening should use the caller's name."""
        return self.persona is not None

    @classmethod
    def from_row(cls, row: dict[str, Any], *, where: str = "") -> CallScenario:
        sid = str(row.get("id") or "").strip()
        if not sid:
            raise ValueError(f"{where}: a scenario needs an id")
        script = str(row.get("script") or "").strip()
        if not script:
            raise ValueError(f"{where}: scenario {sid} needs a script")
        judge = tuple((str(j["name"]), str(j["check"])) for j in row.get("judge") or ())
        if not judge:
            raise ValueError(f"{where}: scenario {sid} needs at least one [[scenarios.judge]] entry (name, check)")
        persona = row.get("persona")
        return cls(id=sid, title=str(row.get("title") or sid), script=script, judge=judge,
                   persona=(str(persona) if persona else None), decisions=int(row.get("decisions", 1)),
                   tags=tuple(str(t) for t in row.get("tags") or ()))

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "persona": self.persona, "script": self.script,
                "judge": [{"name": n, "check": c} for n, c in self.judge], "decisions": self.decisions, "tags": list(self.tags)}


@dataclass
class ConfiguredScenario(Scenario):
    """The ``Scenario`` the live agent hands to Jev and SenseLab, built from the file. In
    simulated mode the review (and the hidden right answer) come from the simulator; in
    reported mode nothing here reviews anything."""

    simulator: Scenario | None = None

    def make_case(self, episode: int, rng: random.Random, *, changed: bool, probe: bool) -> Case:
        raise NotImplementedError("the live agent takes its cases from calls")

    def severity_of(self, case: Case, action: str) -> str:
        if self.simulator is None:
            raise NotImplementedError("outcomes are reported, not simulated")
        return self.simulator.severity_of(case, action)

    def failure_note(self, case: Case, action: str, rng: random.Random) -> str:
        if self.simulator is None:
            raise NotImplementedError("outcomes are reported, not simulated")
        return self.simulator.failure_note(case, action, rng)

    def review(self, case: Case, action: str, rng: random.Random, *, noise: float) -> Outcome:
        if self.simulator is None:
            raise NotImplementedError("outcomes are reported, not simulated")
        return self.simulator.review(case, action, rng, noise=noise)

    def classify(self, fields: dict[str, Any], *, changed: bool) -> dict[str, Any]:
        """The hidden right answer and the case flags, when a simulator knows them."""
        fn = getattr(self.simulator, "classify", None)
        if fn is None:
            return {"truth": "", "is_tacit": False, "is_changed": False}
        return dict(fn(fields, changed=changed))


@dataclass
class Business:
    path: Path
    name: str
    what: str
    agent_name: str
    greeting: str
    house_style: str
    agent_id: str
    entity_path: str
    tiers: tuple[str, ...]
    default_tier: str
    splits: tuple[Split, ...]
    intents: tuple[Intent, ...]
    field_words: dict[str, str]
    actions: tuple[Action, ...]
    policy: tuple[tuple[str, str], ...]
    jev_instructions: str
    judge_rubric: str
    outcomes_mode: str
    simulator_ref: str | None
    cost_by_severity: dict[str, float]
    change_label: str = ""  # the page's name for the simulated world's mid-run change
    personas: tuple[Persona, ...] = field(default_factory=tuple)
    scenarios: tuple[CallScenario, ...] = field(default_factory=tuple)  # the test calls

    # -- loading -----------------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path | None = None) -> Business:
        """The file at *path*, else ``$VOICE_BUSINESS``, else the demo's."""
        p = Path(path or os.environ.get("VOICE_BUSINESS") or DEFAULT_PATH)
        with p.open("rb") as fh:
            raw = tomllib.load(fh)
        return cls.from_dict(raw, path=p)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], *, path: Path = DEFAULT_PATH) -> Business:
        company = raw.get("company") or {}
        memory = raw.get("memory") or {}
        caller = raw.get("caller") or {}
        situation = raw.get("situation") or {}
        jev = raw.get("jev") or {}
        outcomes = raw.get("outcomes") or {}
        name = str(company.get("name") or "the company")
        agent_name = str(company.get("agent_name") or "the agent")
        intents = tuple(
            Intent(id=str(i["id"]), label=str(i.get("label") or i["id"].replace("_", " ")),
                   needs=tuple(str(n) for n in i.get("needs") or ()),
                   ahead=tuple(dict(a) for a in i.get("ahead") or ()) or ({},))
            for i in raw.get("intents") or ()
        )
        actions = tuple(
            Action(id=str(a["id"]), label=str(a.get("label") or a["id"].replace("_", " ")),
                   jev=str(a.get("jev") or a.get("label") or a["id"]), words=str(a.get("words") or a.get("label") or a["id"]).lower(),
                   then=str(a.get("then") or ""), then_no_file=str(a.get("then_no_file") or ""))
            for a in raw.get("actions") or ()
        )
        splits = tuple(
            Split(field=str(s["field"]), at=float(s["at"]), below=str(s["below"]), from_=str(s["from"]),
                  intents=tuple(str(x) for x in s.get("intents") or ()))
            for s in situation.get("splits") or ()
        )
        tiers = tuple(str(t) for t in caller.get("tiers") or ("consumer",))
        # VOICE_OUTCOMES overrides the file for one run: "reported" to test as yourself, where
        # what you say on the next call is the outcome, with the demo's business file as is.
        mode = str(os.environ.get("VOICE_OUTCOMES") or outcomes.get("mode") or "reported").strip().lower()
        if mode not in OUTCOME_MODES:
            raise ValueError(f"outcomes.mode must be one of {OUTCOME_MODES}, not {mode!r}")
        if not intents or not actions:
            raise ValueError(f"{path}: at least one intent and one action are needed")
        personas = tuple(Persona.from_row(p) for p in raw.get("personas") or ())
        for p in personas:
            if p.intent not in {i.id for i in intents}:
                raise ValueError(f"{path}: persona {p.id} has intent {p.intent!r}, not one of the intents")
        scenarios = tuple(CallScenario.from_row(s, where=str(path)) for s in raw.get("scenarios") or ())
        seen: set[str] = set()
        for s in scenarios:
            if s.id in seen:
                raise ValueError(f"{path}: two scenarios with id {s.id!r}")
            seen.add(s.id)
            if s.persona is not None and s.persona not in {p.id for p in personas}:
                raise ValueError(f"{path}: scenario {s.id} calls as persona {s.persona!r}, which is not in [[personas]]")
        return cls(
            path=path, name=name, what=str(company.get("what") or ""), agent_name=agent_name,
            greeting=str(company.get("greeting") or f"Hi, thanks for calling {name}, this is {agent_name}. What can I help you with?"),
            house_style=str(company.get("house_style") or ""),
            agent_id=str(memory.get("agent_id") or "jev-voice-support-agent"),
            entity_path=str(memory.get("entity_path") or "jev-demos/voice/live"),
            tiers=tiers, default_tier=str(caller.get("default_tier") or tiers[0]),
            splits=splits, intents=intents, field_words={str(k): str(v) for k, v in (raw.get("fields") or {}).items()},
            actions=actions, policy=tuple((str(p["key"]), str(p["text"])) for p in raw.get("policy") or ()),
            jev_instructions=str(jev.get("instructions") or "Choose what the agent does next on this call."),
            judge_rubric=str(jev.get("judge_rubric") or ""),
            outcomes_mode=mode, simulator_ref=(str(outcomes["simulator"]) if outcomes.get("simulator") else None),
            cost_by_severity={str(k): float(v) for k, v in (outcomes.get("cost_by_severity") or
                                                            {"minor_failure": 100, "failure": 1000, "critical_failure": 10000}).items()},
            change_label=str(outcomes.get("change_label") or ""), personas=personas, scenarios=scenarios,
        )

    # -- lookups -----------------------------------------------------------------------------

    @property
    def intent_ids(self) -> tuple[str, ...]:
        return tuple(i.id for i in self.intents)

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(a.id for a in self.actions)

    @property
    def fallback_intent(self) -> str:
        """For an intent the model did not name: the last one listed (the least common)."""
        return self.intents[-1].id

    def intent(self, intent_id: str) -> Intent | None:
        return next((i for i in self.intents if i.id == intent_id), None)

    def action(self, action_id: str) -> Action:
        a = next((a for a in self.actions if a.id == action_id), None)
        return a or Action(id=action_id, label=action_id.replace("_", " "), jev="", words=action_id.replace("_", " "), then="")

    def then_for(self, action_id: str, *, matched: bool) -> str:
        """What happens now, for a caller we matched to an account or for one we did not.
        A stranger has no file, so the words that lean on one would send the agent guessing."""
        a = self.action(action_id)
        if not matched and a.then_no_file:
            return a.then_no_file
        return a.then

    def action_labels(self) -> dict[str, str]:
        return {a.id: a.label for a in self.actions}

    def action_words(self) -> dict[str, str]:
        return {a.id: a.words for a in self.actions}

    @property
    def simulated(self) -> bool:
        return self.outcomes_mode == "simulated"

    def tier_of(self, value: Any) -> str:
        t = str(value or "")
        return t if t in self.tiers else self.default_tier

    # -- the situation ---------------------------------------------------------------------

    def situation_of(self, fields: dict[str, Any]) -> str:
        """The label a call's memory is kept under: intent, tier, then each split that applies."""
        intent = str(fields.get("intent") or self.fallback_intent)
        parts = [intent, self.tier_of(fields.get("tier"))]
        parts.extend(s.label(fields) for s in self.splits if s.applies(intent))
        return "/".join(parts)

    def candidate_fields(self, caller: Any) -> list[dict[str, Any]]:
        """Every situation a caller could be in, most common first: each intent's first
        ``ahead`` entry in the file's order, then the extra entries."""
        base = {"tier": caller.tier, "prior_contacts_30d": caller.prior_contacts_30d, "amount_usd": 0.0}
        first = [{**base, "intent": i.id, **i.ahead[0]} for i in self.intents]
        rest = [{**base, "intent": i.id, **a} for i in self.intents for a in i.ahead[1:]]
        return first + rest

    def default_fields(self, intent_id: str) -> dict[str, Any]:
        """Representative values for an intent's fields until the caller names them (the
        first ``ahead`` entry), for the record the agent recalls mid-conversation."""
        intent = self.intent(intent_id)
        return dict(intent.ahead[0]) if intent else {}

    # -- Jev and SenseLab ------------------------------------------------------------------

    @cached_property
    def scenario(self) -> ConfiguredScenario:
        simulator: Scenario | None = None
        if self.simulated:
            if not self.simulator_ref:
                raise ValueError(f"{self.path}: outcomes.mode is simulated but no simulator is named")
            simulator = load_object(self.simulator_ref)
            if set(simulator.actions) != set(self.action_ids):
                raise ValueError(f"{self.path}: the actions {sorted(self.action_ids)} must match the simulator's "
                                 f"{sorted(simulator.actions)} for its reviews to apply")
        return ConfiguredScenario(
            name=(simulator.name if simulator else self.name.lower().replace(" ", "_")),
            title=f"{self.name} support line", case_noun="call", actions=list(self.action_ids),
            jev_instructions=self.jev_instructions, jev_criteria={a.id: a.jev for a in self.actions},
            policy=list(self.policy), judge_rubric=self.judge_rubric, cost_by_severity=dict(self.cost_by_severity),
            tacit_rule=(simulator.tacit_rule if simulator else ""), change_rule=(simulator.change_rule if simulator else ""),
            expected_fix=(simulator.expected_fix if simulator else ""), simulator=simulator,
        )

    def public(self) -> dict[str, Any]:
        """What the page needs to know about the business."""
        return {
            "name": self.name, "agent_name": self.agent_name, "greeting": self.greeting,
            "tiers": list(self.tiers), "intents": [{"id": i.id, "label": i.label, "needs": list(i.needs)} for i in self.intents],
            "actions": [{"id": a.id, "label": a.label} for a in self.actions],
            "outcomes_mode": self.outcomes_mode, "change_rule": self.scenario.change_rule or None,
            "change_label": self.change_label or None,
            "personas": [p.public() for p in self.personas], "file": str(self.path),
            # The test calls, so voice.testcalls runs the business the server is running.
            "scenarios": [s.public() for s in self.scenarios],
        }


def load_object(ref: str) -> Any:
    """``package.module:attr``."""
    module, _, attr = ref.partition(":")
    if not attr:
        raise ValueError(f"simulator must be 'module:attribute', not {ref!r}")
    return getattr(importlib.import_module(module), attr)


__all__ = ["Action", "Business", "CallScenario", "ConfiguredScenario", "DEFAULT_PATH", "Intent", "Split", "load_object"]
