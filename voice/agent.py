"""The live agent behind the voice demo.

Vapi runs the call (telephony, speech to text, the conversational model, text to speech).
At the decision point of the call its model calls one tool, ``decide_next_step``. This
module answers that tool call the way the batch harness answers a case: the call's facts
and the written policy go into Jev's state, SenseLab's guidance for this situation goes
into the memory slot, Jev picks the action. When the call ends, what came of the decision
is written back as the outcome of that run, and the next call in the same situation starts
from it.

What the business is (its intents, actions, policy, callers) comes from ``voice/business.toml``
through ``voice/business.py``; nothing here is Northwind's. In the demo the same
``call_handling`` scenario drives the batch harness and reviews this agent's decisions:
same actions, same policy, same situation labels, same review, so a batch run and the live
agent can share an entity path and what the batch run learned is what the live agent knows.

Latency: the server opens runs ahead of the decision so the caller never waits on a
lookup: when a call starts it begins one run per situation the caller could be in (the
account tells us tier and prior contacts; the intent is what we do not know yet) and the
tool call picks the one that matches. A run that is begun and never completed is a read;
nothing is left open on the platform.
"""

from __future__ import annotations

import logging
import random
import re
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from core import settings as S
from core.arms import MEMORY_NOTE, build_state
from core.eval_client import EvalApiError, EvalClient
from core.jev import Jev
from core.loops import RepairLoop
from core.scenario import Case, Outcome
from core.senselab import ReviewHistory, SenseLabClient, SenseLabEpisode

from . import customers as cust
from .business import Business
from .phonebook import PhoneBook

log = logging.getLogger("voice")

_REVIEW_RE = re.compile(r"\[(worked|did not work)\] (.*?)(?=; \[(?:worked|did not work)\] |$)")
_TALLY_RE = re.compile(r"with (?P<action>\S+): worked (?P<worked>\d+) of (?P<total>\d+) time")


@dataclass
class Caller:
    """What the account lookup knows before the caller says a word."""

    name: str = "Alex"
    tier: str = "consumer"
    prior_contacts_30d: int = 0
    tenure_months: int = 14
    customer_id: str = ""  # the account's id for this caller; "" when the caller is nobody we know

    @classmethod
    def from_mapping(cls, m: dict[str, Any] | None, *, tiers: tuple[str, ...] | None = None,
                     default_tier: str = "consumer") -> Caller:
        m = m or {}
        tier = str(m.get("tier") or default_tier)
        if tiers is not None and tier not in tiers:
            tier = default_tier
        return cls(
            name=str(m.get("name") or "Alex")[:40],
            tier=tier,
            prior_contacts_30d=_int(m.get("prior_contacts_30d"), 0),
            tenure_months=_int(m.get("tenure_months"), 14),
            customer_id=re.sub(r"[^a-z0-9-]", "", str(m.get("customer_id") or "").lower())[:40],
        )


def caller_line(c: Caller) -> str:
    """How the decision text describes who is calling. A caller the lookup did not match has
    no tenure and no tier we know of; the Caller defaults are for a known account."""
    if c.customer_id:
        return f"Caller on a {c.tier} account, {c.tenure_months} months with us"
    return "Caller not matched to any account (no history on file; a first contact)"


@dataclass
class Warm:
    """The entity's briefing and entries at one moment, shared by the runs a call opens."""

    briefing: list[Any]
    entries: list[Any]
    at: float


@dataclass
class LiveDecision:
    """One decision on one call, from the tool call to the outcome."""

    id: str
    call_id: str
    at: float
    caller: dict[str, Any]
    fields: dict[str, Any]
    text: str
    situation: str
    action: str
    confidence: float
    then: str  # what the agent gets across to the caller once this is the action, in its own words
    basis: str | None
    probabilities: dict[str, float]
    memory_text: str
    memory_entries: list[str]
    memory_mode: str | None
    memory_prefetched: bool
    memory_ms: float
    jev_ms: float
    jev_input_tokens: int
    jev_error: str | None = None
    # What Jev would have done on the written policy alone, with nothing from SenseLab in
    # front of it: the same call, decided twice, so the page can show the difference.
    alone_action: str | None = None
    alone_confidence: float | None = None
    alone_outcome: dict[str, Any] | None = None
    outcome: dict[str, Any] | None = None
    trace_id: str | None = None
    causal_keys: list[str] = field(default_factory=list)
    commit_error: str | None = None
    ended_reason: str | None = None
    # Which arm of a live repair canary this call drew: "canary" read the rewrite on the
    # fix's branch, "control" kept the old rule, "unrouted" when no canary was running.
    canary_arm: str | None = None
    memory_branch: str | None = None
    corrective_entries: list[str] = field(default_factory=list)  # rewritten rules in Jev's view
    # Set when this call closed an earlier step for the same caller as not working, because
    # they rang back about the same thing while it was still open.
    rang_back: str | None = None

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        d["memory_text"] = self.memory_text[:4000]
        return d


@dataclass
class _Pending:
    """A decision waiting for the call to end."""

    decision: LiveDecision
    case: Case
    episode: SenseLabEpisode
    jev: Any  # core.jev.Decision


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:80]


OWN_RECORD_NOTE = (
    " A line in the memory beginning 'This caller's own record' is this caller's own file, in their words: "
    "which steps were already taken for them and how each came out. 'Did not hold, on their word' there is a "
    "step we promised this caller that did not happen, and the policy on that applies to this call, whatever "
    "the tally says that step has done for other callers: the tally is about others, the file is about them. "
    "A step that held for them counts for taking it again. A step still open for them says nothing either way. "
    "A line 'from this call' is what this call has shown so far: a step they refused on this call, in their words, "
    "did not hold for them, the same as one that was promised and did not happen."
)

_NOTHING_ARRANGED = (" Nothing is arranged by this. If they still need something done about it, that is a request for "
                     "today: decide_next_step says what happens, and no step is promised before it has answered.")


def _truthy(v: Any) -> bool:
    """A tool argument that means yes: True, "true", "yes", "worked", 1."""
    if isinstance(v, bool):
        return v
    return str(v or "").strip().lower() in ("true", "yes", "y", "1", "worked", "it worked", "ok")


def _ago(seconds: float) -> str:
    """A span of time the way it is said on a call: "two minutes", "an hour", "three days"."""
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
    s = max(0.0, seconds)
    if s < 90:
        return "a minute"
    if s < 90 * 60:
        n = int(round(s / 60))
        return f"{words.get(n, str(n))} minutes"
    if s < 36 * 3600:
        n = max(1, int(round(s / 3600)))
        return "an hour" if n == 1 else f"{words.get(n, str(n))} hours"
    n = int(round(s / 86400))
    return f"{words.get(n, str(n))} days"


def _since(stamp: str) -> str:
    """How long ago a file entry's stamp (ISO, as customers.py writes it) was, in words."""
    minutes = cust._minutes_since(stamp)
    return _ago(minutes * 60) if minutes is not None else "some time"


def _int(v: Any, default: int) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _float(v: Any, default: float) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def rehydrate_history(entries: list[Any]) -> ReviewHistory:
    """Rebuild the review history from the ``seen-*`` entries already in memory, so a
    restarted server keeps counting where the last one stopped."""
    hist = ReviewHistory()
    for e in entries:
        key = str(getattr(e, "key", ""))
        text = getattr(e, "value", "")
        if not key.startswith("seen-") or not isinstance(text, str):
            continue
        m = _TALLY_RE.search(text)
        head = text.split(":", 1)[0]  # "Seen on <situation> with <action>"
        if not m or not head.startswith("Seen on "):
            continue
        situation = head[len("Seen on "):].rsplit(" with ", 1)[0]
        action_key = m.group("action")
        worked, total = int(m.group("worked")), int(m.group("total"))
        reviews = _REVIEW_RE.findall(text.split("Recent reviews: ", 1)[-1])
        k = (situation, action_key)
        # The rendered reviews are newest first; keep them oldest first as ``add`` does.
        hist.seen[k] = [(ok == "worked", note) for ok, note in reversed(reviews)]
        hist.tally[k] = [worked, total]
    return hist


class LiveAgent:
    """Jev decides at the decision point of a call; SenseLab remembers what came of it."""

    def __init__(self, settings: S.Settings, business: Business | None = None, *, agent_id: str | None = None,
                 entity_path: str | None = None, noise: float = 0.0, changed: bool = False,
                 repair_loop: bool = True, prefetch_workers: int = 24) -> None:
        S.require(settings)
        assert settings.typesafe_api_key and settings.amfs_api_key
        self.business = business or Business.load()
        self.scenario = self.business.scenario
        self.agent_id = agent_id or self.business.agent_id
        self.entity_path = entity_path or self.business.entity_path
        self.noise = noise
        self.changed = changed  # is the simulated world's mid-run change in force
        self.jev = Jev(settings.typesafe_api_key)
        self.client = SenseLabClient(settings.amfs_api_key, base_url=settings.amfs_url)
        self.eval_api = EvalClient(settings.amfs_api_key, base_url=settings.amfs_url) if repair_loop else None
        self.loop: RepairLoop | None = None
        self._loop_fut: Any = None  # the look in flight, off the page's request path
        self._loop_note: dict[str, Any] = {}
        self.history = ReviewHistory()
        self.decisions: list[LiveDecision] = []
        self.calls: dict[str, dict[str, Any]] = {}
        self.events: list[dict[str, Any]] = []
        self._seq = 0
        self._pending: dict[str, list[_Pending]] = {}  # call id -> decisions awaiting the outcome
        self._prefetch: dict[str, dict[str, Future[SenseLabEpisode]]] = {}
        self._slot_fields: dict[tuple[str, str], dict[str, Any]] = {}  # to open a queued slot early
        self._opening: dict[str, Future[None]] = {}  # call id -> the _open_all job, until the slots exist
        self._customer: dict[str, Future[cust.CustomerMemory | None]] = {}  # call id -> what we know of the caller
        self._lock = threading.RLock()
        self._pool = ThreadPoolExecutor(max_workers=prefetch_workers, thread_name_prefix="prefetch")
        self.ready_info: dict[str, Any] = {}
        self.begin_ms: list[float] = []
        # Vapi records every call; the recording is fetched here so the demo has the audio
        # next to the decision. ``None`` turns it off.
        self.recordings_dir: Path | None = S.REPO_ROOT / "results" / "voice"
        self.prefetch_wait_s = 12.0  # how long a decision waits for a run still being opened
        # The entity's briefing and entries, fetched once and shared by the runs opened for a
        # call (the briefing is the same for every situation, so one read serves them all).
        # Refreshed after every sealed outcome, since that is when memory changes.
        self._warm: Warm | None = None
        self._warm_future: Future[Warm] | None = None
        self.warm_max_age_s = 900.0

    # -- lifecycle -------------------------------------------------------------------------

    def start(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        seeded = self.client.seed_policy(self.agent_id, self.entity_path, self.scenario)
        entries = self._warm_up().entries
        self.history = rehydrate_history(entries)
        seeded_customers = self.seed_customers()
        self.phonebook = PhoneBook(self.client, self.agent_id, self.entity_path)
        info: dict[str, Any] = {
            "agent_id": self.agent_id, "entity_path": self.entity_path, "seeded": seeded,
            "seeded_customers": seeded_customers, "numbers_known": self.phonebook.load(),
            "entries": len(entries), "seen_pairs": len(self.history.tally),
            "lessons": sum(1 for e in entries if str(e.key).startswith(("learned-", "lesson-"))),
        }
        if self.eval_api is not None:
            self.loop = RepairLoop(self.eval_api, agent_id=self.agent_id, scenario=self.scenario,
                                   look_every=3, look_seconds=45, judge_mode="continuous")
            try:
                info["loop"] = self.loop.ready()
            except EvalApiError as exc:
                info["loop_error"] = str(exc)[:200]
                self.loop = None
        info["ms"] = round((time.perf_counter() - t0) * 1000)
        self.ready_info = info
        self._event("agent_ready", info)
        return info

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        self.jev.close()
        if self.loop is not None:
            self.loop.close()

    # -- calls -----------------------------------------------------------------------------

    def caller_from(self, m: dict[str, Any] | None) -> Caller:
        """A caller from account facts, with the tier checked against the business's."""
        return Caller.from_mapping(m, tiers=self.business.tiers, default_tier=self.business.default_tier)

    def _warm_up(self) -> Warm:
        mem = self.client.memory(self.agent_id)
        with ThreadPoolExecutor(2, thread_name_prefix="warm") as ex:
            fb = ex.submit(mem.briefing, entity_path=self.entity_path, compact=True, credit_reuse=True)
            fl = ex.submit(mem.list, self.entity_path)
            warm = Warm(briefing=list(fb.result()), entries=list(fl.result()), at=time.time())
        self._warm = warm
        return warm

    def _refresh_warm(self) -> None:
        with self._lock:
            if self._warm_future is not None and not self._warm_future.done():
                return
            self._warm_future = self._pool.submit(self._warm_up)

    def _current_warm(self, *, wait_s: float) -> Warm | None:
        """The shared briefing and entries, fresh enough to open runs on; waits for a refresh
        in flight up to *wait_s*."""
        with self._lock:
            fut, warm = self._warm_future, self._warm
        if fut is not None and not fut.done():
            try:
                return fut.result(timeout=wait_s)
            except FuturesTimeout:
                pass
            except Exception as exc:  # noqa: BLE001 - begin fetches its own
                log.warning("warm-up failed: %s", exc)
        if warm is not None and time.time() - warm.at <= self.warm_max_age_s:
            return warm
        return None

    def intro_for(self, caller: Caller) -> tuple[str, str]:
        """What the agent knows about this caller before the first word, read from SenseLab
        now: the brief (account facts and history, as prose for the prompt) and the opening
        line the caller hears, built from the same file."""
        memory: cust.CustomerMemory | None = None
        if caller.customer_id:
            try:
                memory = self._fetch_customer(caller)
            except Exception as exc:  # noqa: BLE001 - the brief is the account facts alone
                log.warning("customer memory for %s: %s", caller.customer_id, exc)
        b = self.business
        return cust.brief(caller, memory), cust.opening(caller, memory, agent_name=b.agent_name, company=b.name,
                                                        greeting=b.greeting)

    def brief_for(self, caller: Caller) -> str:
        return self.intro_for(caller)[0]

    def call_started(self, call_id: str, caller: Caller, *, call_type: str = "webCall", brief: str | None = None,
                     known_by: str | None = None) -> None:
        """A call is connecting. *brief* is the caller's history as it went into the prompt,
        when the server had it (``brief_for``); it is shown on the page as the first read.
        *known_by* says how a phone caller was matched ("page", "number"), or None."""
        with self._lock:
            self.calls[call_id] = {"id": call_id, "started": time.time(), "caller": asdict(caller),
                                   "type": call_type, "status": "in-progress", "decisions": 0,
                                   **({"customer_brief": brief} if brief else {}), **({"known_by": known_by} if known_by else {})}
        self._event("call_started", {"call_id": call_id, "caller": asdict(caller), "type": call_type, "known_by": known_by})
        if brief:
            contacts = sum(1 for line in brief.splitlines() if line.startswith("- "))
            self._event("recall", {"call_id": call_id, "customer_id": caller.customer_id, "name": caller.name,
                                   "contacts": contacts, "seeded": 0, "topic": "call start", "record": False,
                                   "ms": 0, "brief": brief[:1800]})
        if self._current_warm(wait_s=0) is None:
            self._refresh_warm()
        with self._lock:
            self._opening[call_id] = self._pool.submit(self._open_all, call_id, caller)
            self._customer[call_id] = self._pool.submit(self._fetch_customer, caller)

    def persona_named(self, name: str) -> cust.Persona | None:
        """The demo caller a name picks out: the full name, or a first name that only one
        persona has. None when nobody matches, or more than one does."""
        words = [w for w in re.sub(r"[^a-z' -]", "", str(name or "").lower()).split() if w]
        if not words:
            return None
        personas = self.business.personas
        full = [p for p in personas if p.name.lower() == " ".join(words) or p.label.lower() == " ".join(words)]
        if len(full) == 1:
            return full[0]
        first = [p for p in personas if p.name.lower().split()[0] in words]
        return first[0] if len(first) == 1 else None

    def identify(self, call_id: str, params: dict[str, Any]) -> tuple[str, Caller | None]:
        """Answer ``identify_caller``: a caller no account matched has said who they are. If
        the name picks out an account, the call becomes theirs from here: their history is
        read now and returned for the assistant to go on, the runs for their situations are
        opened, and what happens on the call is filed under them. The words back are for the
        assistant, as the brief would have been at the start."""
        with self._lock:
            call = self.calls.get(call_id)
        if call is None:
            return "No call to identify.", None
        name = str(params.get("name") or "").strip()
        persona = self.persona_named(name)
        if persona is None:
            self._event("identified", {"call_id": call_id, "name": name[:60], "matched": False})
            return (f"No account matches the name '{name[:60]}'. Carry on without a file: take their word for who "
                    "they are, and do not describe an account you cannot see."), None
        caller = self.caller_from({**persona.public(), "customer_id": persona.id})
        brief, _ = self.intro_for(caller)
        with self._lock:
            call["caller"] = asdict(caller)
            call["customer_brief"] = brief
            call["known_by"] = "name"
            self._opening[call_id] = self._pool.submit(self._open_all, call_id, caller)
            self._customer[call_id] = self._pool.submit(self._fetch_customer, caller)
        self._event("identified", {"call_id": call_id, "name": name[:60], "matched": True, "caller": asdict(caller)})
        contacts = sum(1 for line in brief.splitlines() if line.startswith("- "))
        self._event("recall", {"call_id": call_id, "customer_id": caller.customer_id, "name": caller.name,
                               "contacts": contacts, "seeded": 0, "topic": "identified by name", "record": False,
                               "ms": 0, "brief": brief[:1800]})
        return (f"Matched: {caller.name}. This is their file; from here on it is what you know about this caller.\n"
                f"{brief}"), caller

    def update_file(self, call_id: str, params: dict[str, Any]) -> str:
        """Answer ``update_file``: a detail the caller gave for their file, confirmed on the
        call. Written to their record in SenseLab now, so it is in the brief from the first
        word of the next call and there for whoever acts on this one. A phone number also
        goes in the phone book, so a call from it is them. The words back are for the
        assistant."""
        with self._lock:
            call = self.calls.get(call_id)
        if call is None:
            return "No call to put this on."
        kind = str(params.get("kind") or "other").strip().lower()
        value = " ".join(str(params.get("value") or "").split())
        note = " ".join(str(params.get("note") or "").split())
        if not value and not note:
            return "Nothing to put on file: say what the detail is."
        customer_id = str((call.get("caller") or {}).get("customer_id") or "")
        if not customer_id:
            return ("No account matched this caller, so there is no file to put it on. Keep it for this call and "
                    "use it; it cannot be kept for next time.")
        key, entry = cust.detail_entry(call_id, kind=kind, value=value, note=note)
        try:
            self.client.memory(self.agent_id).write(cust.customer_entity(self.entity_path, customer_id), key, entry,
                                                    confidence=1.0, memory_type="fact")
        except Exception as exc:  # noqa: BLE001
            log.warning("file update for %s: %s", customer_id, exc)
            return "The file could not be updated just now. Use the detail on this call and say it will be noted."
        if entry["kind"] == "phone" and getattr(self, "phonebook", None) is not None:
            self.phonebook.remember(value, dict(call["caller"]))
        what = cust.DETAIL_KINDS.get(entry["kind"], "note")
        with self._lock:
            call.setdefault("file_updates", []).append({"kind": entry["kind"], "value": value, "note": note, "t": time.time()})
            if call.get("customer_brief"):
                call["customer_brief"] += f"\nOn file now ({what}, given on this call): {value or note}."
        self._event("file_updated", {"call_id": call_id, "customer_id": customer_id, "detail": entry["kind"], "key": key,
                                     "value": value[:80], "note": note[:120]})
        closed = ""
        if entry["kind"] == "outcome":
            closed = self._close_from_caller(call_id, customer_id, worked=_truthy(params.get("worked")), words=value)
        return (f"On file: {what} {value or note}. It is on their record now and will be there next time; "
                f"tell them so in a few words, once.{closed}")

    def _close_from_caller(self, call_id: str, customer_id: str, *, worked: bool, words: str) -> str:
        """The caller has said what came of the last step we took for them. If that decision
        is still open (outcomes are reported, not simulated), their word is the outcome: the
        run is sealed with it, so what SenseLab learns from that call is what happened, not
        what was hoped. An already-sealed decision is left as it is; the note on the file
        says what the caller said, and the next decision sees it."""
        with self._lock:
            earlier = [p for cid, ps in self._pending.items() if cid != call_id for p in ps
                       if str(p.decision.caller.get("customer_id") or "") == customer_id]
            # No earlier call open: the step they are talking about may be the one taken a
            # minute ago on this call ("no, I won't be transferred, cancel it now"). Their
            # refusal is that step's outcome.
            on_this_call = not earlier
            if on_this_call:
                earlier = [p for p in self._pending.get(call_id, [])
                           if str(p.decision.caller.get("customer_id") or "") in ("", customer_id)]
            p = max(earlier, key=lambda p: p.decision.at) if earlier else None
            if p is not None:
                # Off the open list under the lock; the seal runs off the call's path (it is
                # slow, and the caller is on the line waiting for the next step).
                self._pending[p.decision.call_id].remove(p)
                if not self._pending[p.decision.call_id]:
                    del self._pending[p.decision.call_id]
        if p is None:
            return self._amend_record(call_id, customer_id, worked=worked, words=words)
        severity = "success" if worked else "failure"
        said = f"Caller said on {'this' if on_this_call else 'a later'} call: {words}"[:300]
        outcome = Outcome(success=worked, severity=severity, note=said,
                          true_success=worked, cost_usd=float(self.business.cost_by_severity.get(severity, 0.0)))
        self._pool.submit(self._seal, p, outcome, source="caller")
        did = self.business.action_words().get(p.decision.action) or p.decision.action.replace("_", " ")
        # The record Jev reads at the next decision was fetched when the call started; what
        # the caller just said goes in beside it, so the next step is weighed with it.
        self._said_on_call(call_id, (f"they said the step {'taken a moment ago on this call' if on_this_call else 'from their last call'} "
                                     f"({did}) {'held' if worked else 'did not hold'}, in their words: \"{words}\"."))
        if on_this_call:
            return (f" That closes the step taken on this call ({did}) as {'worked' if worked else 'did not work'}, on "
                    "their word; the record for callers in their situation now says so. If their problem is still "
                    "open, decide_next_step again, with what they said as the new fact: it is the only way a different "
                    "step comes back.")
        return (f" That closes the earlier step ({did}) as {'worked' if worked else 'did not work'}, on their word; "
                "the record for callers in their situation now says so." + _NOTHING_ARRANGED)

    def _rang_back(self, call_id: str, customer_id: str, intent: str) -> str:
        """A known caller is on the line about the same thing we took a step on for them
        from an earlier call, and that step is still waiting on its outcome (outcomes are
        reported, not simulated). Ringing back about it is the outcome: the step did not
        work for them. Seal those decisions on it, so the record and the judge see it
        without anyone pressing a button, and say so to the agent."""
        if not customer_id:
            return ""
        with self._lock:
            # Once per call and problem: a second decision on the same call (they refused the
            # step, the facts changed) must not mark the earlier step again, and the record
            # fetched when the call started still shows it open.
            call = self.calls.get(call_id)
            seen = call.setdefault("rang_back_for", []) if call is not None else []
            if intent in seen:
                return ""
            seen.append(intent)
            earlier = [p for cid, ps in self._pending.items() if cid != call_id for p in ps
                       if str(p.decision.caller.get("customer_id") or "") == customer_id
                       and str(p.decision.fields.get("intent") or p.decision.situation.split("/")[0]) == intent]
            # Off the open list now, under the lock, so nothing seals them twice; the seal
            # itself (the trace, the lesson, the record) is slow and runs off the call's path.
            for p in earlier:
                self._pending[p.decision.call_id].remove(p)
                if not self._pending[p.decision.call_id]:
                    del self._pending[p.decision.call_id]
        closed: list[tuple[str, str]] = []
        for p in sorted(earlier, key=lambda p: p.decision.at):
            ago = _ago(time.time() - p.decision.at)
            did = self.business.action_words().get(p.decision.action) or p.decision.action.replace("_", " ")
            outcome = Outcome(success=False, severity="failure", note=f"Rang back about the same thing {ago} later",
                              true_success=False, cost_usd=float(self.business.cost_by_severity.get("failure", 0.0)))
            self._pool.submit(self._seal, p, outcome, source="caller")
            closed.append((did, ago))
        if not closed:
            # Nothing open in this process: a step from before the server last started is on
            # their file with its outcome not in yet. The file's line gets their ring-back.
            memory = self._customer_memory(call_id, wait_s=2.0)
            about = f"Called about {intent.replace('_', ' ')}:"
            steps = [c for c in (memory.contacts if memory else []) if c.get("action") and not c.get("seeded")
                     and c.get("key") and c.get("worked") is None and str(c.get("text") or "").startswith(about)]
            if not steps:
                return ""
            last = steps[0]  # newest first
            ago = _since(str(last.get("at") or last.get("date") or ""))
            value = cust.amend_contact(last, worked=False, words=f"they rang back about it {ago} later")
            try:
                self.client.memory(self.agent_id).write(cust.customer_entity(self.entity_path, customer_id), str(last["key"]),
                                                        value, confidence=1.0, memory_type="experience")
            except Exception as exc:  # noqa: BLE001
                log.warning("amending %s for %s: %s", last["key"], customer_id, exc)
                return ""
            last.update(value)
            did = self.business.action_words().get(str(last["action"])) or str(last["action"]).replace("_", " ")
            self._event("customer_noted", {"call_id": call_id, "customer_id": customer_id, "key": last["key"],
                                           "text": value["text"], "amended": True})
            closed.append((did, ago))
        did, ago = closed[-1]
        self._said_on_call(call_id, f"they are back about the same thing, {ago} after the last step we took for them "
                                    f"({did}); that step did not hold, on that alone.")
        return (f"They are back about the same thing, {ago} after the last step we took for them ({did}). That step "
                f"is on their record as not working, on that alone.")

    def _said_on_call(self, call_id: str, line: str) -> None:
        """A line for Jev's view of this caller's record, learned on this call (their word on
        a step, a ring-back), kept on the call since the record itself was fetched at its
        start. Read back by ``decide``."""
        with self._lock:
            call = self.calls.get(call_id)
            if call is not None:
                call.setdefault("said_on_call", []).append(line)

    def _amend_record(self, call_id: str, customer_id: str, *, worked: bool, words: str) -> str:
        """No open decision to seal (it was sealed already, or by an earlier run of the
        server): the caller's word still goes on the line for the last step we took for
        them, in place of what the file said came of it. The next call reads their word."""
        memory = self._customer_memory(call_id, wait_s=2.0)
        if memory is None:
            return _NOTHING_ARRANGED
        steps = [c for c in memory.contacts if c.get("action") and not c.get("seeded") and c.get("key")]
        # The newest step still waiting on its outcome, else the newest whose line says otherwise.
        last = next((c for c in steps if c.get("worked") is None), None) or next((c for c in steps if c.get("worked") is not worked), None)
        if last is None:
            return _NOTHING_ARRANGED
        value = cust.amend_contact(last, worked=worked, words=words)
        try:
            self.client.memory(self.agent_id).write(cust.customer_entity(self.entity_path, customer_id), str(last["key"]), value,
                                                    confidence=1.0, memory_type="experience")
        except Exception as exc:  # noqa: BLE001
            log.warning("amending %s for %s: %s", last["key"], customer_id, exc)
            return _NOTHING_ARRANGED
        last.update(value)
        did = self.business.action_words().get(str(last["action"])) or str(last["action"]).replace("_", " ")
        self._event("customer_noted", {"call_id": call_id, "customer_id": customer_id, "key": last["key"], "text": value["text"],
                                       "amended": True})
        self._said_on_call(call_id, f"they said the step from their last call ({did}) {'held' if worked else 'did not hold'}, "
                                    f"in their words: \"{words}\".")
        return (f" The file's line on that step ({did}) now carries their word: it {'worked' if worked else 'did not hold'}."
                + _NOTHING_ARRANGED)

    def _open_all(self, call_id: str, caller: Caller) -> None:
        """Open a run for each situation this caller could turn out to be in. The intent is
        what the account does not know; everything else that names the situation it does."""
        t0 = time.perf_counter()
        # A refresh in flight (the previous call's seal just changed what is known) is worth
        # a short wait; past that, the last briefing will do and the retrieve is fresh anyway.
        # Short means short: this is on the path to the first decision, and a caller who gets
        # to the point quickly is waiting on it.
        warm = self._current_warm(wait_s=1.0)
        if warm is None:
            with self._lock:
                warm = self._warm
        opened: list[Future[SenseLabEpisode]] = []
        # Two lanes in order of likelihood (billing, cancellation, outage, then the rest), so
        # the common situations are ready first, while the caller is still talking.
        lanes: list[list[tuple[str, dict[str, Any], Future[SenseLabEpisode]]]] = [[], []]
        with self._lock:
            if call_id not in self.calls:
                return  # ended before the briefing came back
            slots = self._prefetch.setdefault(call_id, {})
            for i, fields in enumerate(self.business.candidate_fields(caller)):
                situation = self.business.situation_of(fields)
                if situation not in slots:
                    fut: Future[SenseLabEpisode] = Future()
                    slots[situation] = fut
                    self._slot_fields[(call_id, situation)] = fields
                    opened.append(fut)
                    lanes[i % len(lanes)].append((situation, fields, fut))
            self._opening.pop(call_id, None)
        for lane in lanes:
            if lane:
                self._pool.submit(self._open_lane, call_id, t0, lane, warm)
        self._event("memory_opening", {"call_id": call_id, "runs": len(opened), "warm": warm is not None})
        for fut in opened:
            try:
                fut.result(timeout=60)
            except Exception:  # noqa: BLE001 - reported where the run is taken
                pass
        self._event("memory_ready", {"call_id": call_id, "runs": len(opened), "ms": round((time.perf_counter() - t0) * 1000)})

    def _open_lane(self, call_id: str, t0: float, lane: list[tuple[str, dict[str, Any], Future[SenseLabEpisode]]],
                   warm: Warm | None) -> None:
        for situation, fields, fut in lane:
            if not fut.set_running_or_notify_cancel():
                continue
            try:
                fut.set_result(self._open_ahead(situation, fields, warm))
            except Exception as exc:  # noqa: BLE001 - reported where the run is taken
                fut.set_exception(exc)
                continue
            self._event("memory_run_ready", {"call_id": call_id, "situation": situation,
                                             "ms": round((time.perf_counter() - t0) * 1000)})

    def _open_ahead(self, situation: str, fields: dict[str, Any], warm: Warm | None) -> SenseLabEpisode:
        """Begin a run for *situation* before the caller has asked for anything."""
        ep = SenseLabEpisode(self.client, agent_id=self.agent_id, entity_path=self.entity_path, scenario=self.scenario)
        text = (f"Support call, {fields['intent'].replace('_', ' ')}, from a {fields['tier']} account with "
                f"{fields['prior_contacts_30d']} contact(s) with support in the last 30 days.")
        probe = Case(id="ahead", episode=0, text=text, fields=fields, situation=situation, truth="", kind=fields["intent"])
        t0 = time.perf_counter()
        self._begin(ep, probe, warm)
        self.begin_ms.append((time.perf_counter() - t0) * 1000)
        return ep

    @staticmethod
    def _begin(ep: SenseLabEpisode, case: Case, warm: Warm | None) -> str:
        if warm is None:
            return ep.begin(case)
        return ep.begin(case, briefing=warm.briefing, entries=warm.entries)

    def _take_prefetched(self, call_id: str, situation: str, *, wait_s: float) -> SenseLabEpisode | None:
        """The run opened ahead for this situation: ready, or still opening (wait for it up to
        *wait_s*; a wait is still shorter than a second begin)."""
        t0 = time.perf_counter()
        with self._lock:
            opening = self._opening.get(call_id)
        if opening is not None and not opening.done():
            # The runs are about to be opened (the briefing is still coming back): wait for the
            # slots rather than open a run of our own on top of them.
            try:
                opening.result(timeout=wait_s / 2)
            except Exception:  # noqa: BLE001 - fall through to whatever slots exist
                pass
        with self._lock:
            slots = self._prefetch.get(call_id) or {}
            fut = slots.pop(situation, None)
            fields = self._slot_fields.pop((call_id, situation), None)
        if fut is None:
            return None
        if fields is not None and not fut.running() and not fut.done():
            # Still queued behind the likelier situations: open it now rather than wait its turn.
            try:
                claimed = fut.set_running_or_notify_cancel()
            except RuntimeError:  # the lane got there first
                claimed = False
            if claimed:
                warm = self._current_warm(wait_s=0)
                try:
                    ep = self._open_ahead(situation, fields, warm)
                except Exception as exc:  # noqa: BLE001
                    fut.set_exception(exc)
                    log.warning("opening %s failed: %s", situation, exc)
                    return None
                fut.set_result(ep)
                return ep
        try:
            return fut.result(timeout=max(0.5, wait_s - (time.perf_counter() - t0)))
        except FuturesTimeout:
            log.warning("run for %s still opening after %.0fs; opening another", situation, wait_s)
            return None
        except Exception as exc:  # noqa: BLE001 - the decision falls back to a live begin
            log.warning("opening ahead for %s failed: %s", situation, exc)
            return None

    def note_speech(self, call_id: str, role: str, status: str, *, t: float | None = None) -> None:
        """Vapi's speech-update: who started or stopped talking. Timeline material. *t* is
        when it happened, if the sender knows better than now (the page does: on GPT-Live it
        derives these from the transcript stream, and the stop is known only after a pause)."""
        if call_id in self.calls:
            self._event("speech", {"call_id": call_id, "role": "assistant" if role in ("assistant", "bot") else "user",
                                   "status": "started" if status == "started" else "stopped", **({"t": t} if t else {})})

    def set_listening(self, call_id: str, on: bool = True) -> None:
        """The call's live audio is being followed (``voice/listen.py``): speech starts and
        stops come from it for both sides, so the same from the page are ignored and the
        turns reported after the fact carry words only, not lane segments."""
        if call_id in self.calls:
            with self._lock:
                self.calls[call_id]["listening"] = on
            self._event("listening", {"call_id": call_id, "on": on})

    def is_listening(self, call_id: str) -> bool:
        call = self.calls.get(call_id)
        return bool(call and call.get("listening"))

    def note_transcript(self, call_id: str, role: str, text: str, *, t: float | None = None) -> None:
        """A final transcript line from Vapi, kept on the call and announced."""
        text = " ".join(str(text or "").split())
        if not text or call_id not in self.calls:
            return
        who = "assistant" if role in ("assistant", "bot") else "user"
        with self._lock:
            self.calls[call_id].setdefault("live", []).append({"role": who, "text": text, "t": t or time.time()})
        self._event("transcript", {"call_id": call_id, "role": who, "text": text[:300], **({"t": t} if t else {})})

    def note_turns(self, call_id: str, messages: list[Any] | None) -> int:
        """The conversation so far, as Vapi's ``artifact.messages``: ``bot`` and ``user`` rows
        with epoch-ms ``time`` and ``endTime``. On GPT-Live this is the one source of the
        caller's words and of turn timing that reaches the server during a call: it comes
        with every tool call, and in full with the end-of-call report. Rows whose words are
        already on the call (relayed live by the page) are skipped; the rest are announced
        as ``turn`` events, each a line with the real start and end of the speech, so the
        timeline fills in at the right places even when the words arrive late. Returns the
        rows added."""
        if not messages or call_id not in self.calls:
            return 0
        squash = lambda s: "".join(ch for ch in s.lower() if ch.isalnum())  # noqa: E731
        new: list[dict[str, Any]] = []
        with self._lock:
            call = self.calls[call_id]
            live = call.setdefault("live", [])
            listened = bool(call.get("listening"))  # the lanes already have these from the audio
            said = {who: squash(" ".join(ln["text"] for ln in live if ln["role"] == who)) for who in ("assistant", "user")}
            for m in messages:
                if not isinstance(m, dict) or m.get("role") not in ("bot", "user"):
                    continue
                text = " ".join(str(m.get("message") or "").split())
                who = "assistant" if m["role"] == "bot" else "user"
                key = squash(text)
                if not key or key in said[who]:
                    continue
                t0 = float(m["time"]) / 1000 if m.get("time") else float(call["started"]) + float(m.get("secondsFromStart") or 0)
                t1 = float(m["endTime"]) / 1000 if m.get("endTime") else t0 + float(m.get("duration") or max(1.0, len(text) / 15))
                said[who] += key
                live.append({"role": who, "text": text, "t": t1})
                new.append({"call_id": call_id, "role": who, "text": text[:300], "t0": round(t0, 3), "t1": round(max(t1, t0 + 0.3), 3),
                            "listened": listened})
        for row in sorted(new, key=lambda r: r["t0"]):
            self._event("turn", row)
        return len(new)

    # -- the caller ------------------------------------------------------------------------

    def _seed_missing(self, mem: Any, entity: str, persona: cust.Persona) -> int:
        """Bring the account's seeded rows in line with ``business.toml``: write the rows it
        does not have, and rewrite (a new version, same key) the ones whose text changed. A
        row is known by its ``seedN`` suffix (``notes`` by its key), not its date, because the
        date in the key is computed from today and would make every row look new each
        morning. The calls written since are left alone. Returns the rows written."""
        have: dict[str, tuple[str, dict[str, Any]]] = {}  # tag -> (key the account has, its value)
        for e in mem.list(entity):
            key, value = str(getattr(e, "key", "")), getattr(e, "value", None)
            if key == "notes" or "-seed" in key:
                tag = key.rsplit("-seed", 1)[-1] if "-seed" in key else key
                have[tag] = (key, value if isinstance(value, dict) else {})
        written = 0
        for key, value in cust.seed_entries(persona):
            tag = key.rsplit("-seed", 1)[-1] if "-seed" in key else key
            if tag in have:
                old_key, old = have[tag]
                if str(old.get("text") or "") == value["text"]:
                    continue
                # The same row, reworded: a new version under the key (and date) the account has.
                key, value = old_key, {**value, "date": str(old.get("date") or value["date"])}
            mem.write(entity, key, value, confidence=1.0, memory_type="experience")
            written += 1
        return written

    def seed_twin(self, persona_id: str, customer_id: str) -> bool:
        """Seed a persona's history under another customer id: a twin for test calls, so the
        demo caller's own record is not written to by tests. True if anything was written."""
        persona = next((p for p in self.business.personas if p.id == persona_id), None)
        if persona is None:
            raise KeyError(persona_id)
        mem = self.client.memory(self.agent_id)
        return self._seed_missing(mem, cust.customer_entity(self.entity_path, customer_id), persona) > 0

    def seed_customers(self) -> list[str]:
        """Give the demo callers their seeded history. Returns the customers written to."""
        mem = self.client.memory(self.agent_id)
        seeded: list[str] = []
        for persona in self.business.personas:
            entity = cust.customer_entity(self.entity_path, persona.id)
            try:
                if self._seed_missing(mem, entity, persona):
                    seeded.append(persona.id)
            except Exception as exc:  # noqa: BLE001 - the call still works without the history
                log.warning("seeding %s: %s", persona.id, exc)
        return seeded

    def _fetch_customer(self, caller: Caller) -> cust.CustomerMemory | None:
        if not caller.customer_id:
            return None
        t0 = time.perf_counter()
        entries = self.client.memory(self.agent_id).list(cust.customer_entity(self.entity_path, caller.customer_id))
        return cust.CustomerMemory.from_entries(caller.customer_id, entries, ms=round((time.perf_counter() - t0) * 1000))

    def _customer_memory(self, call_id: str, *, wait_s: float = 6.0) -> cust.CustomerMemory | None:
        """The caller's record for this call, if it was fetched (a matched caller) and is back."""
        with self._lock:
            fut = self._customer.get(call_id)
        if fut is None:
            return None
        try:
            return fut.result(timeout=wait_s)
        except Exception as exc:  # noqa: BLE001 - the decision goes ahead without it
            log.warning("customer memory for %s: %s", call_id, exc)
            return None

    def record_words(self, call_id: str, situation: str) -> str:
        """What the decision's answer says about why: the situation's record (callers like
        this one) and, when there is one, this caller's own record with the steps."""
        with self._lock:
            own = str((self.calls.get(call_id) or {}).get("caller_record") or "")
        # The ring-back line is in the result on its own ("Before the decision"); not twice.
        own = "\n".join(line for line in own.splitlines() if "they are back about the same thing" not in line)
        return " ".join(x for x in (self.situation_record_words(situation), own) if x)

    def situation_record_words(self, situation: str) -> str:
        """The situation's record in words, for the conversational model."""
        rows = [(k[1].split(":", 1)[-1], v) for k, v in self.history.tally.items() if k[0] == situation]
        if not rows:
            return ""
        words = {k: v.lower() for k, v in self.business.action_labels().items()}
        worked = [f"{words.get(a, a)} ({w} of {t})" for a, (w, t) in rows if t and w / t >= 0.5]
        failed = [f"{words.get(a, a)} ({w} of {t})" for a, (w, t) in rows if t and w / t < 0.5]
        out = f"For callers in this situation ({situation.replace('/', ', ').replace('_', ' ')}),"
        out += f" what has worked: {', '.join(worked)}." if worked else " nothing has worked yet."
        if failed:
            out += f" What has not: {', '.join(failed)}."
        return out

    def recall(self, call_id: str, params: dict[str, Any], *, caller: Caller | None = None) -> str:
        """Answer ``recall_customer``: what the account knows about this caller, in prose."""
        with self._lock:
            call = self.calls.get(call_id)
        if call is None:
            self.call_started(call_id, caller or Caller())
            call = self.calls[call_id]
        c = self.caller_from(call["caller"])
        t0 = time.perf_counter()
        memory = self._customer_memory(call_id)
        intent = str(params.get("intent") or "")
        record = ""
        if intent in self.business.intent_ids:
            # The common values for the intent's fields until the caller names them.
            fields = {"intent": intent, "tier": c.tier, "prior_contacts_30d": c.prior_contacts_30d, "amount_usd": 0.0,
                      **self.business.default_fields(intent)}
            if _float(params.get("amount_usd"), 0.0):
                fields["amount_usd"] = _float(params.get("amount_usd"), 0.0)
            record = self.situation_record_words(self.business.situation_of(fields))
        text = cust.brief(c, memory, record=record)
        with self._lock:
            call["customer_brief"] = text
            call["customer"] = memory.public() if memory else None
        self._event("recall", {"call_id": call_id, "customer_id": c.customer_id, "name": c.name,
                               "contacts": len(memory.contacts) if memory else 0,
                               "seeded": sum(1 for x in memory.contacts if x["seeded"]) if memory else 0,
                               "topic": str(params.get("topic") or "")[:120], "record": bool(record),
                               "ms": round((time.perf_counter() - t0) * 1000), "brief": text[:1800]})
        return text

    def _note_customer(self, d: LiveDecision, outcome: Outcome | None) -> None:
        """After the seal: one line in the caller's own history, so the next call from them
        starts from what happened on this one. With *outcome* None (reported mode, the call
        just ended) the line says the outcome is not in yet; the seal rewrites it."""
        customer_id = str(d.caller.get("customer_id") or "")
        if not customer_id:
            return
        # One line per decision, not per call: a call with two steps (the first refused, a
        # second decided) keeps both, under keys that differ past the first.
        with self._lock:
            ids = [x.id for x in self.decisions if x.call_id == d.call_id]
        k = ids.index(d.id) if d.id in ids else 0
        key, value = cust.contact_entry(d.call_id if k == 0 else f"{d.call_id}-{k}",
                                        intent=str(d.fields.get("intent") or self.business.fallback_intent),
                                        summary=d.text.split(". ", 2)[-1][:240] if ". " in d.text else d.text[:240],
                                        action=d.action, outcome_note=outcome.note if outcome else "",
                                        worked=outcome.success if outcome else None,
                                        agent_name=self.business.agent_name, action_words=self.business.action_words())
        try:
            self.client.memory(self.agent_id).write(cust.customer_entity(self.entity_path, customer_id), key, value,
                                                    confidence=1.0, memory_type="experience")
        except Exception as exc:  # noqa: BLE001
            log.warning("customer note for %s: %s", customer_id, exc)
            return
        self._event("customer_noted", {"call_id": d.call_id, "customer_id": customer_id, "key": key, "text": value["text"]})

    # -- the decision ----------------------------------------------------------------------

    def decide(self, call_id: str, params: dict[str, Any], *, caller: Caller | None = None) -> LiveDecision:
        """Answer ``decide_next_step``: the action, and what to get across."""
        with self._lock:
            call = self.calls.get(call_id)
        if call is None:
            # A tool call for a call we did not see start (the server restarted, or the
            # status update was not subscribed). Register it from what the tool call carries.
            self.call_started(call_id, caller or Caller())
            call = self.calls[call_id]
        c = self.caller_from(call["caller"])
        intent = str(params.get("intent") or self.business.fallback_intent)
        if intent not in self.business.intent_ids:
            intent = self.business.fallback_intent
        # A caller the lookup did not match has no tenure and no history; the Caller defaults
        # are for a known account and must not be fed to Jev as facts about a stranger.
        tenure = c.tenure_months if c.customer_id else 0
        fields = {
            "intent": intent, "tier": c.tier, "prior_contacts_30d": c.prior_contacts_30d,
            "amount_usd": _float(params.get("amount_usd"), 0.0), "tenure_months": tenure,
            "sentiment": str(params.get("sentiment") or "calm"), "channel": "voice",
        }
        summary = " ".join(str(params.get("summary") or "").split())[:600]
        said = " ".join(str(params.get("caller_said") or "").split())[:300]
        text = (f"{caller_line(c)}, sounds {fields['sentiment']}. "
                f"{c.prior_contacts_30d} contact(s) with support in the last 30 days. "
                + (f'Caller said: "{said}" ' if said else "") + summary)
        situation = self.business.situation_of(fields)
        self._event("tool_call", {"call_id": call_id, "intent": intent, "situation": situation, "summary": summary[:160],
                                  "said": said[:160]})
        # Back about the same thing while an earlier step for them is still open: that step
        # did not work. Sealed here, before the decision, so the record Jev weighs says so.
        rang_back = self._rang_back(call_id, c.customer_id, intent)
        n = len(self.decisions)
        # In simulated mode the scenario knows the hidden right answer (for the review when
        # the call ends); in reported mode there is none, the outcome will say.
        known = self.scenario.classify(fields, changed=self.changed)
        case = Case(id=f"{call_id}:{n}", episode=n, text=text, fields=fields, situation=situation,
                    truth=str(known.get("truth") or ""), kind=intent,
                    is_tacit=bool(known.get("is_tacit")), is_changed=bool(known.get("is_changed")))

        t0 = time.perf_counter()
        ep = self._take_prefetched(call_id, situation, wait_s=self.prefetch_wait_s)
        prefetched = ep is not None
        if ep is None:
            ep = SenseLabEpisode(self.client, agent_id=self.agent_id, entity_path=self.entity_path, scenario=self.scenario)
            try:
                self._begin(ep, case, self._current_warm(wait_s=0))
            except Exception as exc:  # noqa: BLE001 - decide on the policy alone
                log.warning("begin failed on %s: %s", situation, exc)
                ep.trace.guidance_error = f"{type(exc).__name__}: {exc}"[:200]
        memory_ms = (time.perf_counter() - t0) * 1000
        memory_text = ep.trace.guidance_text
        # This caller's own record with the steps we have taken for them (SenseLab's other
        # half: the per-customer entries) goes in beside the situation's guidance, so a step
        # that has already failed this caller is weighed as such.
        own = cust.track_record(self._customer_memory(call_id, wait_s=2.0), self.business.action_words())
        # What this call itself has shown about them (their word on a step, a ring-back) is
        # not in that record yet; it goes in beside it, under the same heading, so Jev
        # weighs it the same way.
        with self._lock:
            said_on_call = list(call.get("said_on_call") or [])
        own = "\n".join([own, *(f"This caller's own record, from this call: {line}" for line in said_on_call)]).strip()
        with self._lock:
            call["caller_record"] = own
        guided = f"{memory_text}\n{own}".strip() if own else memory_text
        # The same call decided twice: Jev alone on the written policy, and Jev with what
        # SenseLab learned from earlier calls. The second one is the one the caller gets.
        alone_fut = self._pool.submit(self.jev.decide, build_state(self.scenario, case, "", policy_in_state=True),
                                      instructions=self.scenario.jev_instructions + MEMORY_NOTE,
                                      criteria=self.scenario.jev_criteria)
        state = build_state(self.scenario, case, guided, policy_in_state=True)
        decision = self.jev.decide(state, instructions=self.scenario.jev_instructions + MEMORY_NOTE + (OWN_RECORD_NOTE if own else ""),
                                   criteria=self.scenario.jev_criteria, basis=ep.basis_criteria())
        alone = None
        try:
            alone = alone_fut.result(timeout=10)
        except Exception as exc:  # noqa: BLE001 - the comparison is a bonus
            log.warning("jev alone: %s", exc)
        if ep.run is not None:
            try:
                ep.decided(decision)
            except Exception as exc:  # noqa: BLE001
                log.warning("record decision: %s", exc)
        live = LiveDecision(
            id=case.id, call_id=call_id, at=time.time(), caller=asdict(c), fields=fields, text=text,
            situation=situation, action=decision.action, confidence=round(decision.confidence, 3),
            then=self.business.then_for(decision.action, matched=bool(c.customer_id)), basis=decision.basis,
            probabilities={k: round(v, 3) for k, v in decision.probabilities.items()},
            memory_text=memory_text, memory_entries=list(ep.trace.guidance_entries[:12]),
            memory_mode=ep.trace.guidance_mode, memory_prefetched=prefetched, memory_ms=round(memory_ms),
            jev_ms=round(decision.latency_ms), jev_input_tokens=decision.input_tokens, jev_error=decision.error,
            alone_action=alone.action if alone else None,
            alone_confidence=round(alone.confidence, 3) if alone else None,
            canary_arm=ep.trace.canary, memory_branch=ep.trace.memory_branch,
            corrective_entries=list(ep.trace.corrective_entries),
            rang_back=rang_back or None,
        )
        with self._lock:
            self.decisions.append(live)
            call["decisions"] += 1
            call["last_action"] = decision.action
            self._pending.setdefault(call_id, []).append(_Pending(live, case, ep, decision))
        self._event("decision", {"call_id": call_id, "decision": live.id, "situation": situation, "action": decision.action,
                                 "confidence": live.confidence, "basis": decision.basis, "prefetched": prefetched,
                                 "memory_ms": live.memory_ms, "jev_ms": live.jev_ms,
                                 "alone_action": live.alone_action, "alone_confidence": live.alone_confidence,
                                 "canary_arm": live.canary_arm, "memory_branch": live.memory_branch,
                                 "corrective_entries": live.corrective_entries, "rang_back": live.rang_back})
        return live

    # -- the outcome -----------------------------------------------------------------------

    def call_ended(self, call_id: str, *, ended_reason: str | None = None, transcript: str | None = None,
                   artifact: dict[str, Any] | None = None, auto_review: bool = True,
                   fetch_recording: Callable[[], str | None] | None = None) -> list[LiveDecision]:
        """The call is over. Drop the runs opened for situations that did not come up and, in
        simulated mode (unless *auto_review* is off), review each decision the demo world's
        way. In reported mode the decisions wait for ``report_outcome``. *artifact* is Vapi's:
        the recording URLs and the transcript, kept with the call."""
        artifact = artifact or {}
        transcript = transcript or artifact.get("transcript")
        # The plain recording URLs are not fetchable without Vapi's credentials; the presigned
        # ones are, and they may take a few seconds to appear. *fetch_recording* gets one.
        recording = artifact.get("presignedStereoUrl") or artifact.get("presignedMonoUrl")
        has_recording = bool(recording or artifact.get("stereoRecordingUrl") or artifact.get("recordingUrl"))
        with self._lock:
            self._opening.pop(call_id, None)
            for fut in (self._prefetch.pop(call_id, None) or {}).values():
                fut.cancel()  # a queued slot is skipped by its lane; a running one finishes unused
            for k in [k for k in self._slot_fields if k[0] == call_id]:
                del self._slot_fields[k]
            self._customer.pop(call_id, None)
            call = self.calls.get(call_id)
            if call is not None:
                call["status"] = "ended"
                call["ended"] = time.time()
                call["ended_reason"] = ended_reason
                if transcript:
                    call["transcript"] = transcript[:6000]
                if recording:
                    call["recording_url"] = recording
                if artifact.get("messages"):
                    call["messages"] = [{"role": m.get("role"), "message": m.get("message") or m.get("content") or "",
                                         "secondsFromStart": m.get("secondsFromStart"),
                                         **({"name": m.get("name")} if m.get("name") else {}),
                                         **({"result": str(m.get("result"))[:300]} if m.get("result") else {})}
                                        for m in artifact["messages"] if isinstance(m, dict)][:80]
            pending = list(self._pending.get(call_id) or [])
        if has_recording and self.recordings_dir is not None:
            self._pool.submit(self._save_recording, call_id, recording, fetch_recording)
        for p in pending:
            p.decision.ended_reason = ended_reason
        self._event("call_ended", {"call_id": call_id, "ended_reason": ended_reason, "decisions": len(pending),
                                   "awaiting_outcome": not (auto_review and self.business.simulated)})
        if not pending and call is not None:
            self._note_call(call, ended_reason)  # no decision, still a call they made
        if not auto_review or not self.business.simulated:
            if pending:  # the outcome is not in yet; the call still goes on the record now
                self._pool.submit(self._note_customer, pending[-1].decision, None)
            return [p.decision for p in pending]
        return [self._review(p) for p in pending]

    def _note_call(self, call: dict[str, Any], ended_reason: str | None) -> None:
        """A call that reached no decision goes on the caller's record too, from the words
        on it, so the next call knows it happened. Decided calls are noted after the seal."""
        customer_id = str((call.get("caller") or {}).get("customer_id") or "")
        if not customer_id:
            return
        lines = [ln["text"] for ln in call.get("live") or [] if ln.get("role") == "user"]
        seconds = int(round(float(call.get("ended") or time.time()) - float(call.get("started") or 0))) if call.get("started") else 0
        key, value = cust.call_entry(call["id"], caller_lines=lines, agent_name=self.business.agent_name,
                                     ended_reason=ended_reason, seconds=seconds)
        try:
            self.client.memory(self.agent_id).write(cust.customer_entity(self.entity_path, customer_id), key, value,
                                                    confidence=1.0, memory_type="experience")
        except Exception as exc:  # noqa: BLE001
            log.warning("customer note for %s: %s", customer_id, exc)
            return
        self._event("customer_noted", {"call_id": call["id"], "customer_id": customer_id, "key": key, "text": value["text"]})

    def _save_recording(self, call_id: str, url: str | None, fetch: Callable[[], str | None] | None) -> None:
        """Download the call's recording, trying for a couple of minutes: the file lands on
        Vapi's storage shortly after the report, and a presigned URL for it comes from the
        call record."""
        assert self.recordings_dir is not None
        last_error = "no url"
        for delay in (2, 8, 20, 45, 90):
            time.sleep(delay)
            try:
                if fetch is not None:
                    url = fetch() or url
                if not url:
                    last_error = "no presigned url yet"
                    continue
                self.recordings_dir.mkdir(parents=True, exist_ok=True)
                path = self.recordings_dir / f"{_safe(call_id)}.wav"
                with httpx.stream("GET", url, timeout=120.0, follow_redirects=True) as r:
                    r.raise_for_status()
                    with path.open("wb") as fh:
                        for chunk in r.iter_bytes():
                            fh.write(chunk)
                rel = str(path.relative_to(S.REPO_ROOT))
                with self._lock:
                    call = self.calls.get(call_id)
                    if call is not None:
                        call["recording_file"] = rel
                        call["recording_url"] = url
                self._event("recording_saved", {"call_id": call_id, "file": rel, "bytes": path.stat().st_size})
                return
            except Exception as exc:  # noqa: BLE001 - try again after the next delay
                last_error = f"{type(exc).__name__}: {exc}"[:160]
        log.warning("recording for %s not saved: %s", call_id, last_error)

    def _loop_look(self, episode: int, trace_id: str | None, reported_success: bool) -> None:
        assert self.loop is not None
        try:
            self.loop.after_seal(episode, trace_id, reported_success=reported_success)
            note = self.loop.look(episode)
            if note.get("changed") or note.get("canary_active"):
                self._event("repair_loop", note)
        except Exception as exc:  # noqa: BLE001
            log.warning("repair loop: %s", exc)

    def end_call_later(self, call_id: str, **kwargs: Any) -> None:
        """``call_ended`` off the request path: the review and the seal are a few seconds of
        SenseLab calls, and the webhook (or the page) should not wait on them."""
        def run() -> None:
            try:
                self.call_ended(call_id, **kwargs)
            except Exception as exc:  # noqa: BLE001
                log.warning("call_ended %s: %s", call_id, exc)
        self._pool.submit(run)

    def _review(self, p: _Pending) -> LiveDecision:
        """What happened next, as the demo's world has it: the scenario's review of the action
        against the hidden right answer, with the configured share of wrong reviews."""
        rng = random.Random(f"live:{p.case.id}:review")
        outcome = self.scenario.review(p.case, p.decision.action, rng, noise=self.noise)
        if p.decision.alone_action:
            rng = random.Random(f"live:{p.case.id}:review")  # the same review, of the other action
            alone = self.scenario.review(p.case, p.decision.alone_action, rng, noise=self.noise)
            p.decision.alone_outcome = {"action": p.decision.alone_action, "success": alone.success,
                                        "severity": alone.severity, "cost_usd": alone.cost_usd, "note": alone.note}
        return self.report_outcome(p.decision.id, outcome, source="demo world")

    def report_outcome(self, decision_id: str, outcome: Outcome, *, source: str = "review") -> LiveDecision:
        """Seal the run with the outcome, write the lesson and the review record."""
        with self._lock:
            found = next(((cid, p) for cid, ps in self._pending.items() for p in ps if p.decision.id == decision_id),
                         None)
            if found is None:
                raise KeyError(decision_id)
            cid, p = found
            self._pending[cid].remove(p)
            if not self._pending[cid]:
                del self._pending[cid]
        return self._seal(p, outcome, source=source)

    def _seal(self, p: _Pending, outcome: Outcome, *, source: str) -> LiveDecision:
        """The slow half of ``report_outcome``, for a decision already taken off ``_pending``:
        commit the trace, write the lesson and the record, tell the page."""
        d = p.decision
        d.outcome = {**outcome.to_row(), "source": source, "at": time.time()}
        t0 = time.perf_counter()
        if p.episode.run is None:
            d.commit_error = "no run: guidance failed at begin"
        else:
            trace = p.episode.complete(p.case, p.jev, outcome, history=self.history, task_input=p.case.text)
            d.trace_id = trace.trace_id
            d.causal_keys = list(trace.causal_keys)
            d.commit_error = trace.commit_error
            if self.loop is not None:
                # The platform's side (judge, fixes, canaries) can take a while; it runs off the
                # call's path so the outcome shows the moment the trace is sealed.
                self._pool.submit(self._loop_look, p.case.episode, trace.trace_id, outcome.success)
            self._refresh_warm()  # the seal wrote the lesson and the record; the next call reads them
        self._pool.submit(self._note_customer, d, outcome)
        self._event("outcome", {"call_id": d.call_id, "decision": d.id, "situation": d.situation, "action": d.action,
                                "success": outcome.success, "severity": outcome.severity, "note": outcome.note,
                                "cost_usd": outcome.cost_usd, "trace_id": d.trace_id, "causal_keys": d.causal_keys,
                                "source": source, "seal_ms": round((time.perf_counter() - t0) * 1000),
                                "alone": d.alone_outcome})
        return d

    def pending_decisions(self, call_id: str | None = None) -> list[LiveDecision]:
        with self._lock:
            if call_id is not None:
                return [p.decision for p in self._pending.get(call_id, [])]
            return [p.decision for ps in self._pending.values() for p in ps]

    # -- the world -------------------------------------------------------------------------

    def set_changed(self, changed: bool) -> None:
        self.changed = changed
        self._event("world_changed", {"changed": changed,
                                      "note": self.scenario.change_rule if changed else "the change is reverted"})

    def memory_snapshot(self) -> dict[str, Any]:
        """What is in memory now, for the page: lessons and review records by situation."""
        mem = self.client.memory(self.agent_id)
        try:
            entries = mem.list(self.entity_path)
        except Exception as exc:  # noqa: BLE001
            return {"error": str(exc)[:200]}
        lessons, seen, policy, other = [], [], [], []
        for e in entries:
            key = str(e.key)
            row = {"key": key, "value": e.value if isinstance(e.value, (dict, str)) else str(e.value),
                   "confidence": getattr(e, "confidence", None), "version": getattr(e, "version", None)}
            if key.startswith(("learned-", "lesson-")):
                if isinstance(e.value, dict):
                    v = e.value
                    row["value"] = (f"When: {v.get('situation')}. {str(v.get('action') or '').split(':', 1)[-1]} "
                                    f"{'worked' if v.get('worked') else 'did not work'}. {v.get('text') or ''}")
                lessons.append(row)
            elif key.startswith("seen-"):
                seen.append(row)
            elif key.startswith("policy-"):
                policy.append(row)
            else:
                other.append(row)
        return {"entity_path": self.entity_path, "lessons": lessons, "seen": seen, "policy": policy, "other": other[:40]}

    def loop_snapshot(self) -> dict[str, Any]:
        """What the repair loop has: the last look's note and the fixes seen. The look itself
        runs in the pool (a due one can propose a fix or evaluate a canary, a minute of
        platform calls), so the page never waits on it."""
        if self.loop is None:
            return {"enabled": False}
        if self._loop_fut is None or self._loop_fut.done():
            self._loop_fut = self._pool.submit(self._loop_tick)
        return {"enabled": True, **self._loop_note, "fixes_seen": list(self.loop.fixes_seen.values())[-10:]}

    def _loop_tick(self) -> None:
        assert self.loop is not None
        try:
            note = self.loop.look(-1)
        except Exception as exc:  # noqa: BLE001
            self._loop_note = {"error": str(exc)[:200]}
            return
        before = (self._loop_note.get("canary_active") or {})
        self._loop_note = note
        after = note.get("canary_active") or {}

        def counts(c: dict[str, Any]) -> tuple[Any, Any, Any]:
            return (c.get("canary") or {}).get("outcomes"), (c.get("control") or {}).get("outcomes"), c.get("status")

        # A status move carries the fix in ``changed`` and the page captions it; a count move
        # on the running split carries nothing and the page just refreshes the panel.
        if note.get("changed") or (after and counts(after) != counts(before)):
            self._event("repair_loop", note)

    def state(self) -> dict[str, Any]:
        with self._lock:
            calls = sorted(self.calls.values(), key=lambda c: -c["started"])[:20]
            decisions = [d.public() for d in self.decisions[-40:]]
            events = self.events[-80:]
        begin = sorted(self.begin_ms)
        return {
            "agent_id": self.agent_id, "entity_path": self.entity_path, "changed": self.changed, "noise": self.noise,
            "dashboard": f"{S.DASHBOARD_URL}",
            "calls": calls, "decisions": decisions, "events": events,
            "begin_ms_median": (begin[len(begin) // 2] if begin else None), "begin_count": len(begin),
            "ready": self.ready_info,
        }

    def events_since(self, seq: int) -> list[dict[str, Any]]:
        """Events after *seq*, for the page's live view (it polls this a few times a second)."""
        with self._lock:
            return [e for e in self.events if e["seq"] > seq]

    def _event(self, kind: str, data: dict[str, Any]) -> None:
        """Announce *kind* to the page. The event's time is now unless *data* carries its own
        ``t`` (a speech stop known only after the pause, a turn reported after the fact)."""
        with self._lock:
            self._seq += 1
            # The event's kind is *kind*, whatever the data carries under that name (a file
            # update's detail kind used to overwrite it, and the page never saw the update).
            self.events.append({"seq": self._seq, "t": time.time(), **data, "kind": kind})
            del self.events[:-500]
        log.info("%s %s", kind, {k: v for k, v in data.items() if k not in ("memory_text",)})


def new_call_id() -> str:
    return f"local-{uuid.uuid4().hex[:12]}"


__all__ = ["Caller", "LiveAgent", "LiveDecision", "new_call_id", "rehydrate_history"]
