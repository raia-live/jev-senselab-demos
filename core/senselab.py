"""SenseLab around one Jev decision: ``Run.begin`` → Jev → ``on_tool_result`` → ``complete``.

One ``AgentMemory`` per episode is one SenseLab session. The hosted API identifies the
session from the SDK's headers, puts it in the canary or control arm of the agent's
running repair canary, serves reads from the fix's branch for the canary arm and says
which arm it drew in the ``X-AMFS-Canary`` response header. Nothing here reproduces that
routing; the adapter below only records the header so the episode row can report it.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any

import httpx
from amfs import AgentMemory, Run
from amfs.guidance import Guidance
from amfs_adapter_http import HttpAdapter
from amfs_core.lessons import lesson_key

from . import settings as S
from .jev import Decision
from .scenario import Case, Outcome, Scenario

GUIDANCE_CAP_CHARS = 3200
BEGIN_LIMIT = 16
CORRECTIONS_HEADING = "Corrections from the repair loop (a rule here stopped working; these take precedence):"
REVIEWS_KEPT = 4
# When has an action turned? One failure is one review, and some reviews are wrong. An action
# with a thin record turns on two straight failures. One that has worked PROVEN or more times
# needs three: with one review in five wrong, two wrong in a row happen one pair in 25, three
# in a row one in 125, and the cost of the extra wait is one more case per real change.
TURNED_STREAK = 2
PROVEN = 3
PROVEN_TURNED_STREAK = 3


def _slug(text: str) -> str:
    out = "".join(ch if ch.isalnum() else "-" for ch in text.lower())
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")[:48]


def strip_retry_lines(text: str) -> str:
    """Drop the within-task retry lines from the rendered priors.

    ``render_priors`` writes "Do not spend an attempt on: …" and "Try X first. Do not repeat an
    action that failed on this task." for an agent retrying one task several times. Every
    decision here is a fresh case, and one review of one past case is one data point, so the
    tallies ("Tried on this situation: …") and the recommendation stay and the imperatives go.
    """
    keep = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("Do not spend an attempt on:"):
            continue
        if s.startswith("Try ") and "Do not repeat an action that failed on this task" in s:
            continue
        keep.append(line)
    return "\n".join(keep)


@dataclass
class ReviewHistory:
    """What the reviews said about each (situation, action) this run, kept by the arm and
    written to memory as one experience entry per pair.

    ``Run.learn`` keeps one lesson per situation and the newest outcome overwrites it, so the
    rendered lesson carries one review. An agent that saw three reviews say "bought online
    while waiting for the callback" should not lose two of them. This is ordinary use of the
    memory — an experience entry, evidence-tracked like any other — not a second system.
    """

    seen: dict[tuple[str, str], list[tuple[bool, str]]] = field(default_factory=dict)
    tally: dict[tuple[str, str], list[int]] = field(default_factory=dict)  # [worked, total]

    def add(self, situation: str, action_key: str, worked: bool, note: str) -> str:
        k = (situation, action_key)
        notes = self.seen.setdefault(k, [])
        notes.append((worked, " ".join(note.split())[:160]))
        del notes[:-REVIEWS_KEPT]
        t = self.tally.setdefault(k, [0, 0])
        t[0] += int(worked)
        t[1] += 1
        return self.render(situation, action_key)

    def render(self, situation: str, action_key: str) -> str:
        k = (situation, action_key)
        worked, total = self.tally[k]
        recent = self.seen[k]
        streak = 0
        for ok, _ in reversed(recent):
            if ok:
                break
            streak += 1
        head = f"Seen on {situation} with {action_key}: worked {worked} of {total} time{'s' if total != 1 else ''}"
        if streak and total > streak:
            head += f"; the last {streak} did not"
        body = "; ".join(f"[{'worked' if ok else 'did not work'}] {n}" for ok, n in reversed(recent))
        return f"{head}. Recent reviews: {body}"

    @staticmethod
    def prefix(situation: str) -> str:
        # Two dashes between situation and action: a slug never contains two in a row, so the
        # prefix of one situation is never the prefix of another's record.
        return f"seen-{_slug(situation)}--"

    @staticmethod
    def key(situation: str, action_key: str) -> str:
        return ReviewHistory.prefix(situation) + _slug(action_key.split(":", 1)[-1])


SITUATION_HEADING = "This situation ({situation}) — what the reviews said about each action tried here:"


def situation_block(situation: str, entries: list[Any]) -> tuple[str, list[str]]:
    """Render the agent's own record for *situation*: one line per action tried, most tried
    first. *entries* are the ``seen-*`` memory entries for it, read back by key.

    Retrieval ranks by similarity to the case text and confidence, and a tally about
    "mid_market/na" does not read like a lead from Denver, so the one record that says
    "this stopped working here" can rank below rules about other situations. The agent
    knows its situation label, so it recalls the record directly — the SDK's ``read`` path,
    nothing new — and reads it first. Returns the block (or ``""``) and the full keys shown.
    """
    rows: list[tuple[int, str, str, str]] = []
    for e in entries:
        value = str(e.value)
        head, _, rest = value.partition(": ")
        # "Seen on <situation> with <action>: worked …" -> "<action>", "worked …"
        action = head.rsplit(" with ", 1)[-1] if " with " in head else head
        total = 0
        try:
            total = int(rest.split(" of ", 1)[1].split(" ", 1)[0])
        except (IndexError, ValueError):
            pass
        full = f"{e.entity_path}/{e.key}"
        rows.append((-total, action, full, f"- {full} ({action}): {rest}"))
    if not rows:
        return "", []
    rows.sort()
    keys = [full for _, _, full, _ in rows]
    lines = [line for _, _, _, line in rows]
    return SITUATION_HEADING.format(situation=situation) + "\n" + "\n".join(lines), keys


@dataclass(frozen=True)
class Tally:
    action: str
    worked: int
    total: int
    streak: int  # consecutive most-recent failures (0 when the latest review was a pass)


_TALLY_RE = re.compile(r"with (?P<action>\S+): worked (?P<worked>\d+) of (?P<total>\d+) time")
_STREAK_RE = re.compile(r"the last (?P<streak>\d+) did not")


def parse_record(entries: list[Any]) -> list[Tally]:
    """The situation's ``seen-*`` entries as tallies, most tried first."""
    out: list[Tally] = []
    for e in entries:
        value = str(e.value)
        m = _TALLY_RE.search(value)
        if not m:
            continue
        s = _STREAK_RE.search(value)
        streak = int(s.group("streak")) if s else 0
        if not s and value.partition("Recent reviews: ")[2].startswith("[did not work]"):
            streak = int(m.group("total"))  # every review so far failed
        out.append(Tally(m.group("action"), int(m.group("worked")), int(m.group("total")), streak))
    out.sort(key=lambda t: -t.total)
    return out


HOLD = "hold"
DROP = "drop"
NO_RECOMMENDATION = (
    "Recommendation: none yet. Nothing has worked here and nothing has failed twice: follow the "
    "written policy unless a failure review itself says what should have been done."
)


def turned_streak(t: Tally) -> int:
    """How many straight failures write *t*'s action off here."""
    return PROVEN_TURNED_STREAK if t.worked >= PROVEN else TURNED_STREAK


def noise_guard(mode: str | None, tallies: list[Tally]) -> tuple[str, Tally | None] | None:
    """Leave the record only when nothing in it is working.

    The SDK's ``explore`` is an exploration assignment for an agent retrying one task; here
    every decision is a fresh case and one review is one data point, and some reviews are
    wrong. Read against the situation's own record:

    - an action that has worked here and has not reached its turning streak (two straight
      failures on a thin record, three on a proven one) is *alive*; the best one is held
      (``("hold", tally)``), whatever the SDK suggested exploring past it;
    - an action that has reached it has *turned*; when nothing is alive the explore stands
      (``None``) — that is what a change looks like;
    - a record of nothing but single failures is not evidence to leave the policy on
      (``("drop", None)``): the recommendation and the untried list go, and a line says
      the policy decides unless a failure review names the right action.
    """
    if mode != "explore" or not tallies:
        return None
    alive = [t for t in tallies if t.worked >= 1 and t.streak < turned_streak(t)]
    if alive:
        return HOLD, max(alive, key=lambda t: (t.worked, -t.streak, t.total))
    if any(t.streak >= turned_streak(t) for t in tallies):
        return None
    return DROP, None


def hold_action(text: str, held: Tally | None) -> str:
    """Rewrite the explore recommendation in *text*: as an act on *held*, or, with no
    *held*, as the line that hands the decision back to the policy."""
    if held is not None:
        reason = f"{held.action} has worked {held.worked} of {held.total} time{'s' if held.total != 1 else ''} here"
        if held.streak >= 2:
            reason += (f" and failed its last {held.streak} tries; {held.streak} failures after a record of "
                       f"{held.worked} successes are more likely wrong reviews than a change (it takes "
                       f"{turned_streak(held)} in a row to call one).")
        elif held.streak:
            reason += " and failed only on its latest try; one failure after a working record is more likely a wrong review than a change."
        else:
            reason += "; the exploration suggested is past a record that is working."
        line = f"Recommendation: act -> {held.action}. {reason}"
    else:
        line = NO_RECOMMENDATION
    out = []
    for raw in text.splitlines():
        s = raw.strip()
        if s.startswith("Recommendation: explore"):
            out.append(line)
            continue
        if s.startswith(("Untried", "Not yet tried")):
            continue
        out.append(raw)
    return "\n".join(out)


def corrections_block(g: Guidance) -> tuple[str, list[str]]:
    """Render the repair loop's corrective entries (``risk-*``, ``correction-*``) that
    ``Guidance.text`` leaves out, as lines in the renderer's own shape.

    The SDK keeps corrective entries out of the rendered text so a prompt used to train a
    model never contains them (``ContextEntry.is_training_excluded``). A decision agent is
    the reader they were written for: a correction names the rule that stopped working and
    what to do instead, and a canary can only measure a fix that the sessions in its arm
    actually read. Returns the block (or ``""``) and the full keys it shows.
    """
    extra = [e for e in g.entries if e.is_training_excluded and not e.is_synthetic]
    if not extra:
        return "", []
    lines = [e.render() for e in extra]
    keys = [f"{e.entity_path}/{e.key}" for e in extra]
    return CORRECTIONS_HEADING + "\n" + "\n".join(lines), keys


PRIORS_PREFIXES = (
    "Tried on this situation:", "Tried on similar tasks here:", "No outcome recorded for this situation yet.",
    "On nearby kinds of task", "On a near-identical task here", "Not yet tried", "Recommendation:", "Untried",
    "Do not spend an attempt on:", "Try ",
)


def split_priors(text: str) -> tuple[str, str]:
    """Split the SDK text into the entries list and the priors block that follows it.

    ``Guidance.text`` renders the context entries first and the tallies and recommendation
    last. A cap that cuts from the end cuts the recommendation first; the caller puts the
    priors ahead of the list so the cap falls on the least specific lines.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip().startswith(PRIORS_PREFIXES):
            head = "\n".join(lines[:i]).rstrip()
            return head, "\n".join(lines[i:]).strip()
    return text.rstrip(), ""


def drop_policy_lines(text: str, keys: list[str]) -> tuple[str, list[str]]:
    """Remove the ``policy-*`` entries from the rendered list: the written policy is already
    in Jev's state, and the lines only crowd out what the record says."""
    policy = {k for k in keys if k.rsplit("/", 1)[-1].startswith("policy-")}
    if not policy:
        return text, keys
    dup = tuple(f"- {k} " for k in policy)
    text = "\n".join(line for line in text.splitlines() if not line.startswith(dup))
    return text, [k for k in keys if k not in policy]


def cap_lines(text: str, cap: int) -> str:
    """Cut *text* to at most *cap* characters at a line boundary, so Jev never sees half an
    entry. The lines that fall off are the lowest-ranked ones."""
    if len(text) <= cap:
        return text
    cut = text.rfind("\n", 0, cap)
    return text[: cut if cut > 0 else cap]


def tag_entries(text: str, entry_keys: list[str]) -> tuple[str, dict[str, str]]:
    """Replace each shown entry's full ``entity/path/key`` with a short ``[M#] key`` tag.

    Jev cites the tag it acted on, and the tag maps back to the entry SenseLab credits or
    charges with the outcome. Shorter lines also mean fewer input tokens per decision.
    """
    tags: dict[str, str] = {}
    # Number the tags in the order the entries appear in the text, so [M1] is the first line.
    unique = list(dict.fromkeys(entry_keys))
    ordered = sorted(unique, key=lambda k: (text.find(f"- {k} ") < 0, text.find(f"- {k} ")))
    for i, full in enumerate(ordered, start=1):
        tag = f"M{i}"
        tags[tag] = full
        short = full.rsplit("/", 1)[-1]
        text = text.replace(f"- {full} ", f"- [{tag}] {short} ", 1)
    return text, tags


@dataclass
class RoutingHeaders:
    """The last canary routing the server reported on this thread."""

    canary: str | None = None
    branch: str | None = None
    reuse: str | None = None


class _Headers(threading.local):
    def __init__(self) -> None:
        self.last = RoutingHeaders()


_HEADERS = _Headers()


def _capture(resp: httpx.Response) -> None:
    h = _HEADERS.last
    h.canary = resp.headers.get("X-AMFS-Canary", h.canary)
    h.branch = resp.headers.get("X-AMFS-Memory-Branch", h.branch)


class SenseLabClient:
    """One connection to the SenseLab API; hands out per-episode sessions."""

    def __init__(self, api_key: str, *, base_url: str = S.DEFAULT_AMFS_URL, timeout: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self._adapter = HttpAdapter(self.base_url, api_key, timeout=httpx.Timeout(timeout, connect=10.0))
        # Read the routing headers off every response without re-implementing the request loop.
        self._adapter._client.event_hooks["response"].append(_capture)  # noqa: SLF001

    def memory(self, agent_id: str) -> AgentMemory:
        """A fresh session for *agent_id*: new session id, so a new canary bucket."""
        _HEADERS.last = RoutingHeaders()
        mem = AgentMemory(agent_id=agent_id, adapter=self._adapter)
        # ``Run.begin`` retrieves with ``compact=True``, which the hosted API answers with
        # entry values cut at 120 characters. A structured lesson is a JSON object longer
        # than that, so its situation would be lost and it would render as raw JSON. Ask
        # for full values instead; the payload is a few KB.
        original = mem.retrieve

        def retrieve(*args: Any, **kwargs: Any) -> Any:
            kwargs["compact"] = False
            return original(*args, **kwargs)

        mem.retrieve = retrieve  # type: ignore[method-assign]
        return mem

    @staticmethod
    def last_routing() -> RoutingHeaders:
        return _HEADERS.last

    def seed_policy(self, agent_id: str, entity_path: str, scenario: Scenario) -> list[str]:
        """Write the written policy as memory entries, one per line, if not already there.
        Returns the keys written."""
        mem = self.memory(agent_id)
        written: list[str] = []
        for key, text in scenario.policy:
            if mem.read(entity_path, key) is None:
                mem.write(entity_path, key, text, confidence=0.9)
                written.append(key)
        return written

    def smoke(self, agent_id: str = "jev-demo-smoke") -> dict[str, Any]:
        mem = self.memory(agent_id)
        stats = mem.stats() if hasattr(mem, "stats") else {}
        return {"ok": True, "stats_keys": list(stats)[:5] if isinstance(stats, dict) else str(type(stats))}


@dataclass
class EpisodeTrace:
    """What one SenseLab-wrapped decision recorded, for the episode row."""

    guidance_text: str = ""
    guidance_strength: str = "none"
    guidance_mode: str | None = None
    guidance_next_action: str | None = None
    guidance_branch: str = "main"
    guidance_entries: list[str] = field(default_factory=list)
    guidance_error: str | None = None
    regime_shift: bool = False
    canary: str | None = None
    memory_branch: str | None = None
    trace_id: str | None = None
    commit_error: str | None = None
    memory_ms: float = 0.0
    causal_keys: list[str] = field(default_factory=list)
    corrective_entries: list[str] = field(default_factory=list)

    def to_row(self) -> dict[str, Any]:
        return {
            "memory_chars": len(self.guidance_text),
            "causal_keys": list(self.causal_keys),
            "corrective_entries": list(self.corrective_entries),
            "memory_strength": self.guidance_strength,
            "memory_mode": self.guidance_mode,
            "memory_next_action": self.guidance_next_action,
            "memory_branch": self.memory_branch or self.guidance_branch,
            "memory_entries": self.guidance_entries[:12],
            "memory_error": self.guidance_error,
            "regime_shift": self.regime_shift,
            "canary": self.canary,
            "trace_id": self.trace_id,
            "commit_error": self.commit_error,
            "memory_ms": round(self.memory_ms, 1),
        }


class SenseLabEpisode:
    """The integration pattern, in the order a customer's agent would call it."""

    def __init__(self, client: SenseLabClient, *, agent_id: str, entity_path: str, scenario: Scenario,
                 worker: int | None = None) -> None:
        self.client = client
        self.agent_id = agent_id
        self.entity_path = entity_path
        self.scenario = scenario
        self.worker = worker  # which instance of the agent is deciding, when there are several
        self.trace = EpisodeTrace()
        self.mem: AgentMemory | None = None
        self.run: Run | None = None
        self.guidance: Guidance | None = None
        self.tags: dict[str, str] = {}  # "M1" -> "entity/path/key", in rendered order
        self._lesson: dict[str, Any] | None = None  # this situation's lesson, read by key
        self.held: Tally | None = None  # the action the noise guard held this episode, if any
        self.guard: str | None = None  # HOLD, DROP or None: what the guard did this episode

    def begin(self, case: Case, *, briefing: list[Any] | None = None, entries: list[Any] | None = None) -> str:
        """Open the run; return the text for Jev's memory slot.

        ``Run.begin`` fetches the entity's briefing and then retrieves for the task. The
        briefing is per entity, not per task, and is the slow half (3 to 4 s hosted, against
        about 1 s for the retrieve), so a caller opening several runs on the same entity at
        once can fetch it once and pass it as *briefing*; likewise the entity's *entries*,
        which the situation record is read from."""
        import time

        t0 = time.perf_counter()
        self.mem = self.client.memory(self.agent_id)
        if briefing is not None:
            self.mem.briefing = lambda **_kw: briefing  # type: ignore[method-assign]
        self.run = Run(self.mem)
        g = self.run.begin(
            case.text,
            entity_path=self.entity_path,
            model=S.JEV_MODEL,
            agent_version=S.AGENT_VERSION,
            runtime=S.RUNTIME,
            candidate_actions=self.scenario.candidate_actions(),
            situation=case.situation,
            limit=BEGIN_LIMIT,
        )
        self.guidance = g
        text = strip_retry_lines(g.text or "")
        shown = list(g.entry_keys)
        own = self._situation_record(case.situation, entries=entries)
        self.held = None
        guard = noise_guard(g.mode, parse_record(own))
        context, priors = split_priors(text)
        if guard:
            self.guard, self.held = guard
            priors = hold_action(priors, self.held)
        else:
            self.guard = None
        record, record_keys = situation_block(case.situation, own)
        if record:
            # A record retrieval also ranked is shown once, in the situation block.
            dup = tuple(f"- {k} " for k in record_keys)
            context = "\n".join(line for line in context.splitlines() if not line.startswith(dup))
            shown = record_keys + [k for k in shown if k not in record_keys]
        context, shown = drop_policy_lines(context, shown)
        block, corrective = corrections_block(g)
        if block:
            shown = corrective + shown
        # Most specific first — corrections, this situation's record, the tallies and the
        # recommendation — and the general context list last, where the cap falls.
        text = "\n\n".join(part for part in (block, record, priors, context.strip()) if part)
        text, self.tags = tag_entries(text, shown)
        self.trace.guidance_text = cap_lines(text, GUIDANCE_CAP_CHARS)
        self.trace.guidance_strength = g.strength
        self.trace.guidance_mode = (f"act(held {self.held.action})" if self.held else
                                    "none(dropped)" if self.guard == DROP else g.mode)
        self.trace.guidance_next_action = g.next_action
        self.trace.guidance_branch = g.branch
        self.trace.guidance_entries = shown
        self.trace.corrective_entries = [k.rsplit("/", 1)[-1] for k in corrective]
        self.trace.guidance_error = g.error
        self.trace.regime_shift = g.regime_shift
        self.trace.memory_ms += (time.perf_counter() - t0) * 1000
        return self.trace.guidance_text

    def _situation_record(self, situation: str, *, entries: list[Any] | None = None) -> list[Any]:
        """The ``seen-*`` entries for *situation*, read back by key (one ``list`` call, unless
        the caller already has the entity's *entries*)."""
        assert self.mem is not None
        prefix = ReviewHistory.prefix(situation)
        try:
            if entries is None:
                entries = self.mem.list(self.entity_path)
        except Exception as exc:  # noqa: BLE001 - the record is a bonus; the run goes on
            self.trace.guidance_error = (self.trace.guidance_error or "") + f" record: {type(exc).__name__}"
            return []
        # The situation's own lesson, whether or not retrieval ranked it: it is what the
        # outcome confirms or refutes, and what the repair loop watches for a regime shift.
        lk = lesson_key(situation)
        for e in entries:
            if e.key == lk and isinstance(e.value, dict):
                self._lesson = {"key": f"{e.entity_path}/{e.key}", "action": e.value.get("action"),
                                "worked": e.value.get("worked")}
        return [e for e in entries if str(e.key).startswith(prefix)]

    def basis_criteria(self) -> dict[str, str] | None:
        """The citation question's options: one per shown entry, plus 'none'."""
        if not self.tags:
            return None
        crit = {tag: f"memory entry {tag}: {key.rsplit('/', 1)[-1]}" for tag, key in self.tags.items()}
        crit["none"] = "no memory entry was decisive; the written policy alone decided"
        return crit

    def causal_keys(self, case: Case, decision: Decision, outcome: Outcome) -> list[str]:
        """The entries to credit or charge with this outcome: the one Jev cited, and the
        lessons *about this situation* that claimed the action taken would (or would not)
        work. Nothing else read is blamed for a decision it did not drive — in particular not
        a lesson about another situation that happens to name the same action."""
        keys: list[str] = []
        if decision.basis and decision.basis in self.tags:
            cited = self.tags[decision.basis]
            # A ``seen-*`` entry is a record of reviews, not a claim; it is not what an outcome
            # confirms or refutes. The lessons and rules below are.
            if not cited.rsplit("/", 1)[-1].startswith("seen-"):
                keys.append(cited)
        action_key = f"decide:{decision.action}"
        lessons = [les for les in (self.guidance.lessons if self.guidance else [])
                   if les["situation"] == case.situation]
        if self._lesson is not None:
            lessons.append(self._lesson)
        for lesson in lessons:
            if lesson.get("action") != action_key:
                continue
            # A lesson that said this action works is right on a success and wrong on a
            # failure; one that said it does not work is wrong when it succeeds.
            if bool(lesson.get("worked")) or outcome.success:
                keys.append(lesson["key"])
        seen: set[str] = set()
        return [k for k in keys if not (k in seen or seen.add(k))]

    def decided(self, decision: Decision) -> None:
        """Record the Jev call and the decision it produced."""
        assert self.mem is not None and self.run is not None
        try:
            self.mem.record_llm_call(
                decision.model, decision.input_tokens, decision.output_tokens,
                provider="typesafe", latency_ms=decision.latency_ms,
            )
        except (TypeError, ValueError):
            pass
        self.run.on_tool_result(
            "decide",
            {"action": decision.action, "confidence": round(decision.confidence, 3)},
            result=f"decision recorded: {decision.action}",
            action_key=f"decide:{decision.action}",
            duration_ms=int(decision.latency_ms),
        )

    def complete(self, case: Case, decision: Decision, outcome: Outcome,
                 history: ReviewHistory | None = None, *, task_input: str | None = None) -> EpisodeTrace:
        """Write the lesson (and the review history), then seal the trace with the outcome.
        *task_input* replaces what ``begin`` was given when the run was opened ahead of the
        case (the live agent opens runs before the caller has said what they want)."""
        import time

        assert self.mem is not None and self.run is not None
        t0 = time.perf_counter()
        action_key = f"decide:{decision.action}"
        try:
            self.run.learn(case.situation, action_key, outcome.success, outcome.note)
        except Exception as exc:  # noqa: BLE001 - a lesson is a bonus; the outcome is the record
            self.trace.commit_error = f"learn: {type(exc).__name__}: {exc}"[:200]
        if history is not None:
            try:
                text = history.add(case.situation, action_key, outcome.success, outcome.note)
                self.mem.write(self.entity_path, ReviewHistory.key(case.situation, action_key), text,
                               confidence=0.85, memory_type="experience")
            except Exception as exc:  # noqa: BLE001
                self.trace.commit_error = f"history: {type(exc).__name__}: {exc}"[:200]
        causal = self.causal_keys(case, decision, outcome)
        self.trace.causal_keys = [k.rsplit("/", 1)[-1] for k in causal]
        try:
            self.run.complete(
                outcome.severity,
                outcome_ref=f"{case.id}",
                verified_by="review",
                task_input=task_input,
                evidence={"review_note": outcome.note[:300], "severity": outcome.severity},
                response_text=f"decision: {decision.action} (confidence {decision.confidence:.2f}). review: {outcome.note}",
                decision_summary=f"{case.situation}: decide:{decision.action}",
                attributes={"case_id": case.id, "kind": case.kind, "demo": self.scenario.name,
                            **({"worker": self.worker} if self.worker is not None else {})},
                causal_entry_keys=causal,
            )
            last = getattr(self.mem, "_last_trace", None)
            self.trace.trace_id = str(getattr(last, "id", "") or "") or None
        except Exception as exc:  # noqa: BLE001
            self.trace.commit_error = f"commit: {type(exc).__name__}: {exc}"[:200]
        routing = self.client.last_routing()
        self.trace.canary = routing.canary
        self.trace.memory_branch = routing.branch
        self.trace.memory_ms += (time.perf_counter() - t0) * 1000
        return self.trace


__all__ = [
    "DROP", "HOLD", "NO_RECOMMENDATION", "PROVEN", "PROVEN_TURNED_STREAK", "TURNED_STREAK", "Tally", "hold_action",
    "noise_guard", "parse_record", "turned_streak",
    "BEGIN_LIMIT", "GUIDANCE_CAP_CHARS", "EpisodeTrace", "ReviewHistory", "RoutingHeaders",
    "SenseLabClient", "SenseLabEpisode", "corrections_block", "drop_policy_lines", "situation_block", "split_priors",
    "strip_retry_lines", "tag_entries",
]
