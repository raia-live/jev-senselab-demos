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
from collections import Counter
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
    "this. An action that has worked many times and failed only on its last one or two tries "
    "has most likely met wrong reviews: keep taking it, even if memory suggests exploring. An "
    "action that has failed on its last three tries, or on its last two when it had worked "
    "fewer than three times, has stopped working: the policy rule behind it is stale, do not "
    "take it again, and pick the action the failure reviews point to (what the customer "
    "actually did) rather than an untried one at random. A tally of one or two outcomes is "
    "weak evidence either way; the review text says what actually happened."
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
        self.worker: int | None = None  # which of a team's workers is deciding (TeamArm sets it)
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
        self._episode = SenseLabEpisode(self.client, agent_id=self.agent_id, entity_path=self.entity_path,
                                        scenario=self.scenario, worker=self.worker)
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


def _items(value: Any) -> list[Any]:
    """The list inside an eval API collection answer (``{"items": [...]}`` or a bare list)."""
    if isinstance(value, dict):
        for k in ("items", "incidents", "behaviors"):
            if isinstance(value.get(k), list):
                return list(value[k])
        return []
    return list(value) if isinstance(value, list) else []


def merge_loop_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """One loop record for a team: the members' events, fixes, verdicts, incidents and
    behaviours side by side, in the shape ``core.report`` reads for a single agent."""
    reports = [r for r in reports if r]
    if not reports:
        return {}
    if len(reports) == 1:
        return reports[0]
    out: dict[str, Any] = {k: v for k, v in reports[0].items()
                           if k in ("policy", "canary", "judge_mode", "contract")}
    out["agent_id"] = reports[0].get("agent_id")
    out["agents"] = [{"agent_id": r.get("agent_id"), "sessions_sealed": r.get("sessions_sealed"),
                      "verdict_summary": r.get("verdict_summary"), "lifecycle": r.get("lifecycle")} for r in reports]
    out["sessions_sealed"] = sum(int(r.get("sessions_sealed") or 0) for r in reports)
    out["events"] = [dict(ev, agent=i) for i, r in enumerate(reports) for ev in (r.get("events") or [])]
    out["fixes"] = [dict(f, agent=i) for i, r in enumerate(reports) for f in (r.get("fixes") or [])]
    out["verdicts"] = [dict(v, agent=i) for i, r in enumerate(reports) for v in (r.get("verdicts") or [])]
    vs: dict[str, Any] = {"graded": 0, "failed": 0, "errors": 0, "judge_usd": 0.0, "disagreements": 0}
    cats: Counter[str] = Counter()
    for r in reports:
        s = r.get("verdict_summary") or {}
        for k in ("graded", "failed", "errors", "disagreements"):
            vs[k] += int(s.get(k) or 0)
        vs["judge_usd"] += float(s.get("judge_usd") or 0)
        cats.update({str(k): int(v) for k, v in (s.get("categories") or {}).items()})
    vs["judge_usd"] = round(vs["judge_usd"], 4)
    vs["categories"] = dict(cats.most_common(8))
    out["verdict_summary"] = vs
    life: Counter[str] = Counter()
    for r in reports:
        life.update({str(k): int(v) for k, v in (r.get("lifecycle") or {}).items()})
    out["lifecycle"] = dict(life)
    for name in ("incidents", "behaviors"):
        out[name] = {"items": [dict(it, agent=i) if isinstance(it, dict) else it
                               for i, r in enumerate(reports) for it in _items(r.get(name))]}
    out["final_look"] = [r.get("final_look") for r in reports]
    out["judge_pending_at_end"] = sum(int(r.get("judge_pending_at_end") or 0) for r in reports)
    return out


class TeamArm(Arm):
    """The same stream handled by several instances of the agent, round-robin: case 0 to
    worker 0, case 1 to worker 1, and so on.

    What differs between arms is where the memory lives. A lookup or history store is local
    to the process that built it, so each worker of those arms keeps its own and learns only
    from its own cases. The SenseLab arm is one agent whichever worker is deciding: one
    identity, one memory entity, one repair loop over every worker's outcomes. The worker
    is an attribute on the trace. A lesson one worker learns from its review is in the next
    worker's memory before the next case arrives.
    """

    def __init__(self, spec: str, ctx: RunContext, agents: int) -> None:
        super().__init__(ctx)
        if agents < 2:
            raise ValueError("a team needs at least two agents")
        self.agents = agents
        if spec == "senselab":
            self.members: list[Arm] = [ARM_FACTORIES[spec](ctx)]
        else:
            self.members = [ARM_FACTORIES[spec](ctx) for _ in range(agents)]
        lead = self.members[0]
        self.name = lead.name
        self.slug = lead.slug

    def member(self, episode: int) -> Arm:
        worker = episode % self.agents
        m = self.members[worker % len(self.members)]
        if isinstance(m, SenseLabArm):
            m.worker = worker
        return m

    def setup(self) -> dict[str, Any]:
        infos = [m.setup() for m in self.members]
        info = dict(infos[0])
        info["workers"] = self.agents
        info["stores"] = len(self.members)
        return info

    def episode(self, episode: int, case: Case) -> dict[str, Any]:
        row = self.member(episode).episode(episode, case)
        row["agent"] = episode % self.agents
        return row

    def after_episode(self, episode: int) -> dict[str, Any] | None:
        return self.member(episode).after_episode(episode)

    def finish(self) -> dict[str, Any]:
        return merge_loop_reports([m.finish() for m in self.members])


ARM_FACTORIES: dict[str, Any] = {
    "jev": lambda ctx: JevArm(ctx),
    "retrieval": lambda ctx: RetrievalArm(ctx, "text-k8"),
    "retrieval:text-k8": lambda ctx: RetrievalArm(ctx, "text-k8"),
    "retrieval:meta-k8": lambda ctx: RetrievalArm(ctx, "meta-k8"),
    "history": lambda ctx: HistoryArm(ctx),
    "senselab": lambda ctx: SenseLabArm(ctx),
}

DEFAULT_ARMS = ["jev", "retrieval:meta-k8", "senselab"]


def make_arm(spec: str, ctx: RunContext, *, agents: int = 1) -> Arm:
    if spec not in ARM_FACTORIES:
        raise SystemExit(f"unknown arm {spec!r}; choose from {', '.join(ARM_FACTORIES)}")
    if agents > 1:
        return TeamArm(spec, ctx, agents)
    return ARM_FACTORIES[spec](ctx)


__all__ = ["ARM_FACTORIES", "DEFAULT_ARMS", "Arm", "HistoryArm", "JevArm", "LoopKnobs", "RetrievalArm", "RunContext",
           "SenseLabArm", "TeamArm", "build_state", "make_arm", "merge_loop_reports", "S"]
