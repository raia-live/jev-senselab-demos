"""Scripted test calls against the live voice agent, through Vapi Simulations.

Vapi plays the caller: an AI tester that follows a script, on a real voice call to the
assistant ``voice.server`` registered. Every webhook goes through the running server, so
the call is the real thing: SenseLab is read, Jev decides, the outcome is recorded. Each
scenario is one thing a caller obviously needs on a call, and the checks are the things
that went wrong on real calls before: did the agent know the caller, did it answer a
question about the past from the file, did it decide only after the caller said why they
were calling today, was any tool answer lost, did it hang up on them.

    uv run python -m voice.testcalls                  # every scenario, as voice calls
    uv run python -m voice.testcalls --only bruno     # one of them
    uv run python -m voice.testcalls --list           # what there is
    uv run python -m voice.testcalls --report <run>   # the report for a run again

The scenarios are the ``[[scenarios]]`` of the business file the server is running (the
script the tester follows, the persona it calls as, the yes/no questions the judge answers
from the transcript), so a business writes its own test calls next to its callers and this
harness runs them unchanged. The checks that need no judge (tool answers delivered, decided
after the caller spoke, no hang-up before the caller was done, how long the agent talks in
one go) are the same for every business.

Needs the server running (``uv run python -m voice.server``) and ``VAPI_PRIVATE_KEY``. The
demo callers' records are not written to: a scenario with a persona calls as a twin
(``<persona>-sim``) seeded with the same history, so Bruno's record stays Bruno's.
Vapi bills the simulated calls like any other call, a few cents a minute each.

One difference from a real call: a simulation has the tester speak first and drops the
assistant's first message, so the opening line you hear on the phone (built by the server
from the caller's file) is not in these transcripts. John's first words here are the
model's own reply to the caller, which is also what you get when a caller talks over the
greeting; whether it knows who is calling is checked on that.

On GPT-Live, Vapi sends the server nothing before the first tool call (no status, speech or
transcript updates), so the page registers a browser call itself when the SDK connects
(``POST /api/calls/{id}/start``) and the runs open ahead from there. This harness does the
same the moment the run names a call, so the latency measured here is the latency of a
browser or phone call, not of a call the server first hears of at its first decision.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .assistant import VapiApi
from .business import Business, CallScenario

SERVER = os.environ.get("VOICE_SERVER_URL") or "http://127.0.0.1:8787"
TOOL_ERROR = "An unexpected error occurred."
DONE_WORDS = re.compile(r"\b(bye|goodbye|that'?s (all|it|everything)|nothing else|no,? (that'?s|i'?m|thank)|i'?m (good|done|all set)|all set|no thanks?|take care)\b", re.I)
INTERNAL = re.compile(r"\b(jev|senselab|sense lab|tool call|function call|language model|my prompt)\b", re.I)


# -- the tester ----------------------------------------------------------------------------

TESTER = (
    "You are a customer on a phone call with a company's support line. You have a script of\n"
    "steps below. Follow them in order, ONE step per turn: say the current step in your own\n"
    "natural words, then stop and wait for the agent to answer. Do not move to the next step\n"
    "until the agent has responded to the current one. Keep every turn to one or two short\n"
    "sentences, the way people talk on the phone. Answer any question the agent asks briefly\n"
    "and truthfully from the facts in the script; if the script does not give a fact, say you\n"
    "are not sure. Never invent account details. Do not thank the agent for things it has not\n"
    "done yet. When the script says to end the call, say goodbye and then stop talking.\n"
)


def personality(name: str) -> dict[str, Any]:
    return {"name": name, "assistant": {"model": {"provider": "openai", "model": "gpt-4.1", "temperature": 0.3,
                                                  "messages": [{"role": "system", "content": TESTER}]}}}


# -- the scenarios -------------------------------------------------------------------------

def scenarios_of(business: dict[str, Any]) -> list[CallScenario]:
    """The test calls of the business the server is running, from ``/api/config``."""
    rows = business.get("scenarios") or []
    if not rows:
        sys.exit(f"the business file has no [[scenarios]] ({business.get('file') or 'business.toml'}); "
                 "see voice/README.md, Test calls")
    return [CallScenario.from_row(r) for r in rows]


def by_id(ids: list[str] | None, scenarios: list[CallScenario]) -> list[CallScenario]:
    if not ids:
        return scenarios
    out = []
    for i in ids:
        s = next((s for s in scenarios if s.id == i), None)
        if s is None:
            sys.exit(f"no scenario {i!r}; try --list")
        out.append(s)
    return out


# -- the server side -----------------------------------------------------------------------

def server_config(server: str) -> dict[str, Any]:
    try:
        return httpx.get(f"{server}/api/config", timeout=10).json()
    except httpx.HTTPError as exc:
        sys.exit(f"the voice server is not answering at {server} ({exc}); start it with: uv run python -m voice.server")


def twin_profile(server: str, business: dict[str, Any], s: CallScenario) -> dict[str, Any]:
    """Register who this test call is from, and get the opening and the prompt values the
    server would give a real call from them. A persona calls as its twin."""
    if s.persona is None:
        body: dict[str, Any] = {"stranger": True}
    else:
        p = next((p for p in business.get("personas") or [] if p.get("id") == s.persona), None)
        if p is None:
            sys.exit(f"persona {s.persona!r} is not in the running server's business")
        twin = f"{s.persona}-sim"
        httpx.post(f"{server}/api/twins", json={"persona_id": s.persona, "customer_id": twin}, timeout=60).raise_for_status()
        body = {"name": p["name"], "tier": p.get("tier"), "prior_contacts_30d": p.get("prior_contacts_30d"),
                "tenure_months": p.get("tenure_months"), "customer_id": twin}
    r = httpx.post(f"{server}/api/profiles", json=body, timeout=60)
    r.raise_for_status()
    return r.json()


def server_events(server: str, call_id: str) -> list[dict[str, Any]]:
    try:
        ev = httpx.get(f"{server}/api/events", params={"since": 0}, timeout=10).json().get("events") or []
    except httpx.HTTPError:
        return []
    return [e for e in ev if e.get("call_id") == call_id]


# -- the run -------------------------------------------------------------------------------

def start_run(api: VapiApi, assistant_id: str, server: str, business: dict[str, Any], scenarios: list[CallScenario],
              *, chat: bool, iterations: int, profile_ids: dict[str, str] | None = None) -> dict[str, Any]:
    sims = []
    for s in scenarios:
        prof = twin_profile(server, business, s)
        if profile_ids is not None:
            profile_ids[f"jev-demo: {s.id}"] = str(prof["variable_values"].get("profile_id") or "")
        sims.append({
            "type": "simulation", "name": s.id,
            "personality": personality(f"jev-demo tester: {s.id}"),
            "scenario": {
                "name": f"jev-demo: {s.id}",
                "instructions": s.script,
                "targetOverrides": {"firstMessage": prof["first_message"], "variableValues": prof["variable_values"]},
                "evaluations": [{"structuredOutput": {"name": name, "description": desc, "schema": {"type": "boolean"}},
                                 "comparator": "=", "value": True, "required": True} for name, desc in s.judge],
            },
        })
    body = {"simulations": sims, "target": {"type": "assistant", "assistantId": assistant_id},
            "transport": {"provider": "vapi.webchat" if chat else "vapi.websocket"}, "iterations": iterations}
    return api._call("POST", "/eval/simulation/run", json=body)


def wait_for(api: VapiApi, run_id: str, *, timeout_s: int, server: str = "",
             profile_ids: dict[str, str] | None = None) -> dict[str, Any]:
    """Poll the run until it ends. With *server* and *profile_ids*, each call is registered
    with the server the moment the run names it, the way the page does for a browser call
    (``POST /api/calls/{id}/start``): on GPT-Live, Vapi sends the server nothing until the
    first tool call, so this is what lets the runs open ahead of the decision in a
    simulation too."""
    t0 = time.time()
    last = ""
    registered: set[str] = set()
    misses = 0
    while True:
        try:
            run = api._call("GET", f"/eval/simulation/run/{run_id}")
            misses = 0
        except httpx.HTTPError as e:
            # Vapi's API times out now and then; the run goes on without us, so wait and ask again.
            misses += 1
            if misses >= 10:
                raise
            print(f"  (poll failed: {e.__class__.__name__}; retrying)", flush=True)
            time.sleep(5)
            continue
        counts = run.get("itemCounts") or {}
        line = f"{run.get('status')} total={counts.get('total')} running={counts.get('running')} passed={counts.get('passed')} failed={counts.get('failed')}"
        if line != last:
            print(f"  [{int(time.time() - t0):4d}s] {line}", flush=True)
            last = line
        if run.get("status") == "ended":
            return run
        if time.time() - t0 > timeout_s:
            print("  giving up waiting; the run goes on at", run.get("url"))
            return run
        if server and profile_ids and (counts.get("running") or 0) > 0:
            try:
                items = run_items(api, run_id)
            except httpx.HTTPError:
                items = []  # next poll, three seconds on
            for it in items:
                cid = str(it.get("callId") or "")
                name = str(((it.get("metadata") or {}).get("scenario") or {}).get("name") or "")
                if cid and cid not in registered and name in profile_ids:
                    registered.add(cid)
                    r = httpx.post(f"{server}/api/calls/{cid}/start", json={"profile_id": profile_ids[name]}, timeout=10)
                    known = r.is_success and r.json().get("known")
                    print(f"  [{int(time.time() - t0):4d}s] {name}: call {cid[:8]} registered"
                          f"{' (the server already knew it)' if known else ''}", flush=True)
        time.sleep(3)


def run_items(api: VapiApi, run_id: str) -> list[dict[str, Any]]:
    items = api._call("GET", f"/eval/simulation/run/{run_id}/item")
    if isinstance(items, dict):
        items = items.get("results") or items.get("items") or []
    return list(items)


# -- listening to a call -------------------------------------------------------------------

Seg = tuple[float, float]


def speech_segments(path: str, *, step: float = 0.25, floor: float = 0.08, min_len: float = 0.5) -> tuple[list[Seg], list[Seg]]:
    """Who talked when, from the stereo recording: (caller segments, agent segments) in
    seconds. Vapi records the caller on the left channel and the assistant on the right.
    Energy per quarter second against a fraction of the channel's loudest moment; good
    enough to see a monologue, and whether the agent stops when the caller starts."""
    import array
    import math
    import wave

    with wave.open(path) as w:
        n, sr, frames = w.getnchannels(), w.getframerate(), w.getnframes()
        raw = w.readframes(frames)
    if n != 2:
        return [], []
    pcm = array.array("h")
    pcm.frombytes(raw)
    win = int(sr * step)

    def envelope(ch: int) -> list[float]:
        out = []
        for i in range(0, frames, win):
            seg = pcm[i * n + ch:min(i + win, frames) * n:n]
            if not seg:
                break
            out.append(math.sqrt(sum(x * x for x in seg) / len(seg)))
        return out

    def segments(env: list[float]) -> list[Seg]:
        thr = max(env or [0]) * floor
        out: list[Seg] = []
        on: float | None = None
        for i, v in enumerate(env):
            t = i * step
            if v > thr and on is None:
                on = t
            elif v <= thr and on is not None:
                if t - on >= min_len:
                    out.append((on, t))
                on = None
        if on is not None:
            out.append((on, len(env) * step))
        return out

    return segments(envelope(0)), segments(envelope(1))


def stretches(segs: list[Seg], *, gap: float = 1.5) -> list[Seg]:
    """Segments joined across pauses shorter than *gap*: one person's turn."""
    out: list[Seg] = []
    for a, b in segs:
        if out and a - out[-1][1] < gap:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def audio_checks(call: dict[str, Any], workdir: str) -> list[tuple[bool, str]]:
    """The two things a transcript cannot show: how long the agent talks in one go, and
    whether it stops when the caller starts talking over it."""
    url = (call.get("artifact") or {}).get("presignedStereoUrl") or call.get("stereoRecordingUrl")
    if not url:
        return [(True, "audio: no stereo recording to measure")]
    path = os.path.join(workdir, f"{call.get('id')}-stereo.wav")
    try:
        if not os.path.exists(path):
            with httpx.stream("GET", url, timeout=60) as r:
                r.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in r.iter_bytes():
                        f.write(chunk)
        caller, agent = speech_segments(path)
    except Exception as exc:  # noqa: BLE001 - the audio is a bonus; the run still reports
        return [(True, f"audio: could not measure ({exc})")]
    if not agent:
        return [(True, "audio: no agent speech found")]
    turns = stretches(agent)
    longest = max(b - a for a, b in turns)
    out = [(longest <= 30.0, f"audio: longest agent turn {longest:.0f} s ({len(turns)} turns); the limit is 30 s")]
    # Every caller utterance of two seconds or more that begins while the agent is talking:
    # how much longer did the agent keep going? Shorter ones ("oh, the extra seats", "mm-hmm")
    # are agreement or backchannel, and a speaker that finishes its sentence through them is
    # doing the right thing; a caller who keeps talking is asking it to stop.
    over: list[float] = []
    for a, b in caller:
        if b - a < 2.0:
            continue
        talking = [s for s in agent if s[0] <= a < s[1]]
        if not talking:
            continue
        end = a
        for s in agent:  # follow the agent's voice until a real pause
            if s[0] <= end + 0.6 and s[1] > end:
                end = s[1]
        over.append(end - a)
    if over:
        worst = max(over)
        out.append((worst <= 2.0, f"audio: caller talked over the agent {len(over)} time(s); the agent kept going up to {worst:.1f} s (limit 2 s)"))
    else:
        out.append((True, "audio: the caller never had to talk over the agent"))
    return out


# -- reading a call ------------------------------------------------------------------------

@dataclass
class Turn:
    role: str
    text: str
    t: float
    tool: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    tool_call_id: str = ""


def turns_of(call: dict[str, Any]) -> list[Turn]:
    out: list[Turn] = []
    for m in call.get("messages") or []:
        role = str(m.get("role") or "")
        t = float(m.get("secondsFromStart") or 0)
        if role in ("bot", "user"):
            out.append(Turn(role, str(m.get("message") or ""), t))
        elif role == "tool_calls":
            for tc in m.get("toolCalls") or []:
                fn = tc.get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}") if isinstance(fn.get("arguments"), str) else dict(fn.get("arguments") or {})
                except json.JSONDecodeError:
                    args = {"raw": fn.get("arguments")}
                out.append(Turn("tool_call", "", t, tool=str(fn.get("name") or ""), args=args, tool_call_id=str(tc.get("id") or "")))
        elif role == "tool_call_result":
            out.append(Turn("tool_result", str(m.get("result") or ""), t, tool=str(m.get("name") or ""), tool_call_id=str(m.get("toolCallId") or "")))
    return out


def our_checks(s: CallScenario, call: dict[str, Any], turns: list[Turn], events: list[dict[str, Any]], caller_name: str) -> list[tuple[bool, str]]:
    """What the transcript and the server's own events show, independent of Vapi's judge."""
    out: list[tuple[bool, str]] = []
    bots = [t for t in turns if t.role == "bot"]
    users = [t for t in turns if t.role == "user"]
    calls = [t for t in turns if t.role == "tool_call"]
    results = [t for t in turns if t.role == "tool_result"]

    # 1. The first words. (A simulation drops the server-built opening and has the tester
    # speak first, so this is the model's reply to a caller who talked first; whether it knew
    # the caller is the judge's call, from the whole conversation.)
    first = " / ".join(b.text for b in bots[:2])
    named = s.known and caller_name.lower() in first.lower()
    out.append((True, f"first words{' (used the name)' if named else ''}: {first[:110]!r}"))

    # 2. Every tool answer reached the model, and none was Vapi's "unexpected error".
    lost = [c for c in calls if c.tool_call_id and not any(r.tool_call_id == c.tool_call_id for r in results)]
    errs = [r for r in results if TOOL_ERROR in r.text or (r.tool_call_id and r.tool == r.tool_call_id)]
    out.append((not lost and not errs, f"tool answers delivered: {len(calls)} calls, {len(lost)} unanswered, {len(errs)} dropped by Vapi"))
    # On GPT-Live a hand-off the reasoner could not act on comes back as this; the one seen
    # was a goodbye that did not end the call. One that lands as the caller hangs up, with
    # nothing said after it by either side, left nobody waiting and is noted, not failed.
    empty = [r for r in results if r.tool == "delegate" and "could not produce an answer" in r.text]
    if empty:
        spoken_after = [t for t in turns if t.role in ("bot", "user") and t.t > empty[-1].t + 0.5]
        hung_up = str(call.get("endedReason") or "").startswith("customer-ended")
        if len(empty) == 1 and not spoken_after and hung_up:
            out.append((True, "1 hand-off the reasoner could not act on, as the caller hung up (nothing said after it)"))
        else:
            out.append((False, f"{len(empty)} hand-off(s) the reasoner could not act on"))
    # One tool call per turn (two at once is what loses an answer on the realtime stack).
    # GPT-Live's "delegate" is the hand-off that wraps the real tool call, not a second tool.
    ours = [c for c in calls if c.tool not in ("delegate", "endCall")]
    paired = [(a.tool, b.tool) for a, b in zip(ours, ours[1:], strict=False) if b.t - a.t < 1.0]
    out.append((not paired, f"one tool call per turn{': ' + str(paired) if paired else ''}"))
    # Our answer came back within a few seconds (the server: SenseLab run plus Jev), and the
    # caller's whole wait, hand-off included, stayed under ten.
    slow_ours, slow_wait = [], []
    for c in calls:
        r = next((r for r in results if r.tool_call_id == c.tool_call_id), None)
        if r is None:
            continue
        took = round(r.t - c.t, 1)
        if c.tool == "delegate":
            if took > 10.0:
                slow_wait.append(took)
        elif c.tool != "endCall" and took > 6.0:
            slow_ours.append((c.tool, took))
    out.append((not slow_ours, f"our tool answers within 6 s{': ' + str(slow_ours) if slow_ours else ''}"))
    waits = [round(r.t - c.t, 1) for c in calls if c.tool == "delegate" for r in results if r.tool_call_id == c.tool_call_id]
    if waits:
        out.append((not slow_wait, f"caller's wait on a hand-off within 10 s: {waits}"))

    # 3. decide_next_step came after the caller said why they were calling, with their words.
    decides = [c for c in calls if c.tool == "decide_next_step"]
    out.append((len(decides) >= s.decisions, f"decide_next_step called {len(decides)} time(s), expected {s.decisions}"))
    for d in decides:
        before = [u for u in users if u.t < d.t]
        said = str(d.args.get("caller_said") or "")
        out.append((bool(before) and bool(said.strip()), f"decided after the caller spoke ({len(before)} caller turns before it), caller_said={said[:70]!r}"))

    # 4. The close: if the assistant hung up, the caller had said they were done.
    ended = str(call.get("endedReason") or "")
    if ended.startswith("assistant"):
        last_user = users[-1].text if users else ""
        out.append((bool(DONE_WORDS.search(last_user)), f"assistant ended the call after the caller said {last_user[:60]!r}"))
    else:
        out.append((True, f"call ended by: {ended or 'unknown'}"))

    # 5. No internal names in the agent's mouth.
    leaks = [b.text for b in bots if INTERNAL.search(b.text)]
    out.append((not leaks, f"no internal names said: {leaks[0][:80]!r}" if leaks else "no internal names said"))

    # 6. The server saw it: decisions made and sealed.
    dec = [e for e in events if e.get("kind") == "decision"]
    sealed = [e for e in events if e.get("kind") == "outcome"]
    out.append((len(dec) >= s.decisions, f"server decisions: {[(e.get('situation'), e.get('action'), 'alone=' + str(e.get('alone_action'))) for e in dec]}"))
    out.append((len(sealed) >= min(1, s.decisions) or s.decisions == 0, f"outcomes sealed: {[(e.get('action'), e.get('success')) for e in sealed]}"))
    return out


def print_call(s: CallScenario, item: dict[str, Any], call: dict[str, Any], turns: list[Turn], events: list[dict[str, Any]],
               caller_name: str, workdir: str, *, agent_name: str = "AGENT") -> bool:
    print("\n" + "=" * 100)
    print(f"{s.id}: {s.title}")
    print(f"  item {item.get('id')}  status={item.get('status')}  call={item.get('callId')}  ended={call.get('endedReason')}")
    if item.get("failureReason"):
        print(f"  failure: {item['failureReason']}")
    print("-" * 100)
    who = agent_name.upper()[:6]
    for t in turns:
        if t.role == "bot":
            print(f"  [{t.t:6.1f}] {who + ':':8s}{t.text}")
        elif t.role == "user":
            print(f"  [{t.t:6.1f}] CALLER: {t.text}")
        elif t.role == "tool_call":
            print(f"  [{t.t:6.1f}]   -> {t.tool}({json.dumps(t.args)[:160]})")
        else:
            flag = "  !! DROPPED" if TOOL_ERROR in t.text else ""
            print(f"  [{t.t:6.1f}]   <- {t.tool}: {t.text[:200]}{flag}")
    print("-" * 100)
    ok_all = True
    for ev in (item.get("results") or {}).get("evaluations") or []:
        so = ev.get("structuredOutput") or {}
        name = so.get("name") or ev.get("name") or ev.get("structuredOutputId")
        passed = bool(ev.get("passed"))
        ok_all &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  judge  {name}: value={ev.get('value')!r}")
    for ok, msg in our_checks(s, call, turns, events, caller_name) + audio_checks(call, workdir):
        ok_all &= ok
        print(f"  {'PASS' if ok else 'FAIL'}  check  {msg}")
    rec = (call.get("artifact") or {}).get("recordingUrl") or call.get("recordingUrl")
    if rec:
        print(f"  recording: {rec}")
    print(f"  => {'OK' if ok_all else 'NOT OK'}")
    return ok_all


def wait_for_seal(server: str, call_id: str, want: int, *, timeout_s: float = 60.0) -> list[dict[str, Any]]:
    """The server seals a call's decisions after Vapi's end-of-call report, in the
    background; give it a minute before reading the events."""
    t0 = time.time()
    while True:
        events = server_events(server, call_id)
        sealed = sum(1 for e in events if e.get("kind") == "outcome")
        decided = sum(1 for e in events if e.get("kind") == "decision")
        if sealed >= min(want, decided) and (decided or time.time() - t0 > 10):
            return events
        if time.time() - t0 > timeout_s:
            return events
        time.sleep(3)


def fetch_call(api: VapiApi, call_id: str, attempts: int = 4) -> dict[str, Any]:
    """GET /call/{id}; a call record with a recording is a few hundred KB and Vapi has dropped the
    connection mid-body, which must not throw away a finished run's report."""
    for i in range(attempts):
        try:
            return api._call("GET", f"/call/{call_id}")
        except httpx.HTTPError as e:
            if i == attempts - 1:
                raise
            print(f"  (fetching call {call_id[:8]} failed: {e.__class__.__name__}; retrying)")
            time.sleep(2 * (i + 1))
    return {}


def report(api: VapiApi, server: str, run: dict[str, Any], scenarios: list[CallScenario], business: dict[str, Any]) -> bool:
    items = run_items(api, run["id"])
    names = {p["id"]: p["name"] for p in business.get("personas") or []}
    workdir = os.path.join(os.environ.get("TMPDIR") or "/tmp", "jev-testcalls")
    os.makedirs(workdir, exist_ok=True)
    all_ok = True
    seen = 0
    for item in items:
        snap = (item.get("metadata") or {}).get("scenario") or {}
        sid = str(snap.get("name") or "").replace("jev-demo: ", "")
        s = next((s for s in scenarios if s.id == sid), None)
        if s is None:
            continue
        seen += 1
        call_id = item.get("callId")
        call = fetch_call(api, str(call_id)) if call_id else {}
        turns = turns_of(call)
        events = wait_for_seal(server, str(call_id), s.decisions)
        caller_name = names.get(s.persona or "", "") if s.persona else ""
        all_ok &= print_call(s, item, call, turns, events, caller_name, workdir, agent_name=str(business.get("agent_name") or "agent"))
    if seen == 0:
        print("no items for these scenarios in that run")
        return False
    print("\n" + ("ALL OK" if all_ok else "SOMETHING FAILED") + f"  ({run.get('url')})")
    return all_ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated scenario ids")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--chat", action="store_true", help="text instead of voice (faster, no audio; the realtime stack needs voice)")
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--report", metavar="RUN_ID", help="print the report for an earlier run")
    ap.add_argument("--server", default=SERVER)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--business", help="list the scenarios of this business file instead of the running server's (with --list)")
    a = ap.parse_args(argv)

    if a.list:
        # No server needed: the file's scenarios (the server's own, when it is up and no
        # file is named, so what is listed is what would run).
        try:
            rows = scenarios_of(server_config(a.server)["business"]) if not a.business else []
        except SystemExit:
            rows = []
        if not rows:
            b = Business.load(a.business)
            rows = list(b.scenarios)
            print(f"from {b.path}:")
        for s in rows:
            print(f"{s.id:10s} {s.title}  [{', '.join(s.tags)}]"
                  f"{'' if s.persona else '  (a stranger)'}{'' if s.decisions == 1 else f'  ({s.decisions} decisions)'}")
        return 0

    key = os.environ.get("VAPI_PRIVATE_KEY")
    if not key:
        from dotenv import load_dotenv

        from core.settings import REPO_ROOT
        load_dotenv(REPO_ROOT / ".env")
        key = os.environ.get("VAPI_PRIVATE_KEY")
    if not key:
        sys.exit("VAPI_PRIVATE_KEY is not set")
    api = VapiApi(key)
    cfg = server_config(a.server)
    scenarios = by_id(a.only.split(",") if a.only else None, scenarios_of(cfg["business"]))

    if a.report:
        run = api._call("GET", f"/eval/simulation/run/{a.report}")
        return 0 if report(api, a.server, run, scenarios, cfg["business"]) else 1

    if not cfg.get("assistant_id"):
        sys.exit("the server has no Vapi assistant; set VAPI_PRIVATE_KEY for it and restart")
    print(f"assistant {cfg['assistant_id']} on {a.server}, entity {cfg.get('entity_path')}")
    print(f"scenarios: {', '.join(s.id for s in scenarios)}  ({'chat' if a.chat else 'voice'}, x{a.iterations})")
    profile_ids: dict[str, str] = {}
    run = start_run(api, cfg["assistant_id"], a.server, cfg["business"], scenarios, chat=a.chat, iterations=a.iterations,
                    profile_ids=profile_ids)
    print(f"run {run['id']}  {run.get('url')}")
    if run.get("message"):
        print(f"  {run['message']}")
    run = wait_for(api, run["id"], timeout_s=a.timeout, server=a.server, profile_ids=profile_ids)
    return 0 if report(api, a.server, run, scenarios, cfg["business"]) else 1


if __name__ == "__main__":
    sys.exit(main())
