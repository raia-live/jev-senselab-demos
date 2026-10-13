"""The arms. Same Jev model, same question, same state layout; only the memory slot differs.

``jev``                 the written policy in state, memory slot empty.
``jev+retrieval:<cfg>`` past cases embedded locally; the k nearest rendered into the slot.
``jev+senselab``        SenseLab: ``Run.begin`` guidance plus the agent's own record for the
                        situation in the slot, ``complete`` with the reviewed outcome, and the
                        repair loop running on the platform — judge, incident, corrective entry
                        on a memory branch, Tier 1 replay, canary, ship, verify, roll back.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

from . import settings as S
from .eval_client import EvalClient
from .jev import Decision, Jev
from .loops import RepairLoop
from .scenario import Case, Outcome, Scenario
from .senselab import ReviewHistory, SenseLabClient, SenseLabEpisode

MEMORY_NOTE = (
    " The 'memory' field holds what is known about cases like this one: lessons from past "
    "outcomes, what worked or failed, the reviews that came back, recommendations. Where it "
    "bears on this case, let it take precedence over the written policy. This case's situation "
    "is named at the top of the memory; a lesson's 'When:' names the situation it is about, and "
    "a lesson about a different situation does not apply here. Read a tally like "
    "this. An action that has worked many times and failed only on its single most recent try "
    "has most likely met a wrong review: keep taking it, even if memory suggests exploring. An "
    "action that has failed on its last two or more tries has stopped working: the policy rule "
    "behind it is stale, do not take it again, and pick the action the failure reviews point "
    "to (what the customer actually did) rather than an untried one at random. A tally of one "
    "or two outcomes is weak evidence either way; the review text says what actually happened."
)


@dataclass
class LoopKnobs:
    policy: str = "auto_after_canary"
    canary_pct: int = 50
    canary_min_outcomes: int = 5
    canary_timeout_sessions: int = 60
    look_every: int = 5
    judge: bool = True
    judge_mode: str = "continuous"


@dataclass
class RunContext:
    scenario: Scenario
    jev: Jev
    run_id: str
    seed: int
    noise: float
    senselab: SenseLabClient | None = None
    eval_api: EvalClient | None = None
    knobs: LoopKnobs = field(default_factory=LoopKnobs)
    resume: bool = False

    def slug(self) -> str:
        return f"{self.scenario.name}-{self.run_id}-s{self.seed}"

    def agent_id(self, arm_slug: str) -> str:
        return f"jev-demo-{self.slug()}-{arm_slug}"

    def entity_path(self, arm_slug: str) -> str:
        return f"jev-demos/{self.scenario.name}/{self.run_id}-s{self.seed}/{arm_slug}"


def build_state(scenario: Scenario, case: Case, memory_text: str, *, policy_in_state: bool) -> dict[str, Any]:
    state: dict[str, Any] = {scenario.case_noun: case.text, "fields": dict(case.fields)}
    if policy_in_state:
        state["policy"] = [text for _, text in scenario.policy]
    state["memory"] = memory_text if memory_text.strip() else "(empty)"
    return state


class Arm:
    name = "arm"
    slug = "arm"
    policy_in_state = True

    def __init__(self, ctx: RunContext) -> None:
        self.ctx = ctx
        self.scenario = ctx.scenario

    # -- hooks ---------------------------------------------------------------------------

    def setup(self) -> dict[str, Any]:
        return {}

    def memory_text(self, episode: int, case: Case) -> str:
        return ""

    def basis(self) -> dict[str, str] | None:
        """Options for the citation question, when the memory slot has tagged entries."""
        return None

    def observe(self, episode: int, case: Case, decision: Decision, outcome: Outcome) -> dict[str, Any]:
        return {}

    def after_episode(self, episode: int) -> dict[str, Any] | None:
        return None

    def finish(self) -> dict[str, Any]:
        return {}

    # -- one episode ---------------------------------------------------------------------

    def episode(self, episode: int, case: Case) -> dict[str, Any]:
        t0 = time.perf_counter()
        memory = self.memory_text(episode, case)
        state = build_state(self.scenario, case, memory, policy_in_state=self.policy_in_state)
        decision = self.ctx.jev.decide(
            state, instructions=self.scenario.jev_instructions + MEMORY_NOTE,
            criteria=self.scenario.jev_criteria, basis=self.basis(),
        )
        # The review's noise draw depends on the case, not the arm: every arm is judged on
        # the same (occasionally wrong) feedback for the same case.
        rng = random.Random(f"{self.ctx.seed}:{case.id}:review")
        outcome = self.scenario.review(case, decision.action, rng, noise=self.ctx.noise)
        extra = self.observe(episode, case, decision, outcome)
        row: dict[str, Any] = {
            "run_id": self.ctx.run_id, "scenario": self.scenario.name, "seed": self.ctx.seed,
            "arm": self.name, "episode": episode,
            **case.to_row(), **decision.to_row(), **outcome.to_row(),
            "correct": decision.action == case.truth,
            "memory_chars": len(memory),
            "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }
        row.update(extra)
        return row


class JevArm(Arm):
    name = "jev"
    slug = "jev"


class RetrievalArm(Arm):
    policy_in_state = True

    def __init__(self, ctx: RunContext, config_name: str) -> None:
        super().__init__(ctx)
        from baselines.retrieval import CONFIGS, RetrievalStore

        self.config = CONFIGS[config_name]
        self.name = f"jev+retrieval:{config_name}"
        self.slug = f"retrieval-{config_name}"
        self.store = RetrievalStore(self.config)

    def memory_text(self, episode: int, case: Case) -> str:
        return self.store.render(episode, case)

    def observe(self, episode: int, case: Case, decision: Decision, outcome: Outcome) -> dict[str, Any]:
        self.store.add(episode, case, decision.action, outcome)
        return {}


class HistoryArm(Arm):
    """Every past case, verbatim, in the prompt — "just load the past cases into Jev".

    No embeddings, no selection: the whole history, newest first, with the action taken and
    the review. It grows by one case per decision until Jev answers ``max_tokens_exceeded``
    (about 32k input tokens, see ``core/limit.py``), at which point the decision falls back
    to the first action with confidence 0 and the row records ``jev_error``.
    """

    name = "jev+history"
    slug = "history"

    def __init__(self, ctx: RunContext) -> None:
        super().__init__(ctx)
        self.lines: list[str] = []

    def memory_text(self, episode: int, case: Case) -> str:
        if not self.lines:
            return ""
        return "Past cases, newest first:\n" + "\n".join(reversed(self.lines))

    def observe(self, episode: int, case: Case, decision: Decision, outcome: Outcome) -> dict[str, Any]:
        verdict = "PASS" if outcome.success else f"FAIL ({outcome.severity})"
        fields = ", ".join(f"{k}={v}" for k, v in case.fields.items())
        self.lines.append(f"- case {episode}: {case.text} [{fields}] -> decide:{decision.action} -> {verdict}: {outcome.note}")
        return {}


class SenseLabArm(Arm):
    name = "jev+senselab"
    slug = "senselab"
    # The written policy stays in state, as in any Jev deployment; a copy is seeded into
    # memory so the loops have the rule itself to discredit and correct.
    policy_in_state = True

    def __init__(self, ctx: RunContext) -> None:
        super().__init__(ctx)
        if ctx.senselab is None:
            raise RuntimeError("the SenseLab arm needs AMFS_API_KEY")
        if ctx.eval_api is None:
            raise RuntimeError("the SenseLab arm needs the eval API (/api/v1/eval) on your account")
        self.client = ctx.senselab
        self.agent_id = ctx.agent_id(self.slug)
        self.entity_path = ctx.entity_path(self.slug)
        self._episode: SenseLabEpisode | None = None
        self.history = ReviewHistory()
        k = ctx.knobs
        # The repair loop is the product: judge, incident, corrective entry, Tier 1 test,
        # canary, ship, recheck — all on the SenseLab side. This process only watches it.
        self.loop = RepairLoop(
            ctx.eval_api, agent_id=self.agent_id, scenario=self.scenario, policy=k.policy,
            canary_pct=k.canary_pct, canary_min_outcomes=k.canary_min_outcomes,
            canary_timeout_sessions=k.canary_timeout_sessions, look_every=k.look_every, judge=k.judge,
            judge_mode=k.judge_mode,
        )

    def setup(self) -> dict[str, Any]:
        seeded = self.client.seed_policy(self.agent_id, self.entity_path, self.scenario)
        if not seeded and not self.ctx.resume:
            # The memory lives on the SenseLab side, keyed by run id: a reused id would start
            # this arm on another run's lessons and make the curves meaningless.
            raise RuntimeError(
                f"memory for {self.entity_path} already exists on the server; pick a new --run-id "
                "or pass --resume to continue the earlier run"
            )
        info = self.loop.ready()
        info.update({"entity_path": self.entity_path, "seeded": seeded})
        return info

    def memory_text(self, episode: int, case: Case) -> str:
        self._episode = SenseLabEpisode(self.client, agent_id=self.agent_id, entity_path=self.entity_path, scenario=self.scenario)
        return self._episode.begin(case)

    def basis(self) -> dict[str, str] | None:
        assert self._episode is not None
        return self._episode.basis_criteria()

    def observe(self, episode: int, case: Case, decision: Decision, outcome: Outcome) -> dict[str, Any]:
        assert self._episode is not None
        self._episode.decided(decision)
        trace = self._episode.complete(case, decision, outcome, history=self.history)
        row = trace.to_row()
        row["memory_text"] = trace.guidance_text
        self._after_seal(episode, trace.trace_id, outcome)
        return row

    def _after_seal(self, episode: int, trace_id: str | None, outcome: Outcome) -> None:
        self.loop.after_seal(episode, trace_id, reported_success=outcome.success)

    def after_episode(self, episode: int) -> dict[str, Any] | None:
        return self.loop.look(episode)

    def finish(self) -> dict[str, Any]:
        pending = self.loop.drain()
        final = self.loop.look(-1, force=True)
        report = self.loop.report()
        report["final_look"] = final
        report["judge_pending_at_end"] = pending
        self.loop.close()
        return report


ARM_FACTORIES: dict[str, Any] = {
    "jev": lambda ctx: JevArm(ctx),
    "retrieval": lambda ctx: RetrievalArm(ctx, "text-k8"),
    "retrieval:text-k8": lambda ctx: RetrievalArm(ctx, "text-k8"),
    "retrieval:meta-k8": lambda ctx: RetrievalArm(ctx, "meta-k8"),
    "history": lambda ctx: HistoryArm(ctx),
    "senselab": lambda ctx: SenseLabArm(ctx),
}

DEFAULT_ARMS = ["jev", "retrieval:meta-k8", "senselab"]


def make_arm(spec: str, ctx: RunContext) -> Arm:
    if spec not in ARM_FACTORIES:
        raise SystemExit(f"unknown arm {spec!r}; choose from {', '.join(ARM_FACTORIES)}")
    return ARM_FACTORIES[spec](ctx)


__all__ = ["ARM_FACTORIES", "DEFAULT_ARMS", "Arm", "HistoryArm", "JevArm", "LoopKnobs", "RetrievalArm", "RunContext",
           "SenseLabArm", "build_state", "make_arm", "S"]
