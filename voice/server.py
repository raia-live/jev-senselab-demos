"""The webhook server Vapi talks to, plus a small page to make calls from the browser.

    uv run uvicorn voice.server:app --port 8787

Routes
  POST /vapi/webhook      Vapi server messages: status-update, speech-update, transcript, tool-calls, end-of-call-report,
                          assistant-request (phone numbers: who is calling comes from the number, voice/phonebook.py),
                          transfer-destination-request.
  GET  /                  the demo page (browser calls through the Vapi Web SDK).
  GET  /api/config        what the page needs: the public key and the assistant id.
  GET  /api/state         calls, decisions, events; polled by the page.
  GET  /api/memory        the agent's memory for this entity, grouped.
  GET  /api/loop          the repair loop's view: fixes, canary, verdict counts.
  POST /api/profiles      register the caller the page is about to be (account facts).
  POST /api/world         {"changed": true|false}: flip the mid-run change in the demo's world.
  POST /api/calls         (text mode) start a call without Vapi; returns the call id.
  POST /api/calls/{id}/end  (text mode) end it; reviews the decisions unless auto_review=false.
  POST /api/outcome       report an outcome by hand: {"decision_id", "success", "severity", "note"}.

Environment (``.env``): TYPESAFE_API_KEY, AMFS_API_KEY, and for Vapi VAPI_PUBLIC_KEY (the page),
VAPI_PRIVATE_KEY (creating the assistant), optional VAPI_ASSISTANT_ID, VAPI_WEBHOOK_SECRET,
VOICE_PUBLIC_URL (the tunnel URL Vapi can reach), VOICE_BUSINESS (the business file; default
``voice/business.toml``), VOICE_AGENT_ID, VOICE_ENTITY_PATH, VOICE_NOISE, VOICE_OUTCOMES (``reported`` or
``simulated``, overriding the business file for this run).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from core import settings as S
from core.scenario import SEVERITIES, Outcome

from . import assistant as va
from . import listen as li
from .agent import Caller, LiveAgent, new_call_id
from .business import Business

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("voice.server")

WEB_DIR = Path(__file__).parent / "web"


class Config:
    def __init__(self) -> None:
        self.settings = S.load()
        self.public_key = (os.environ.get("VAPI_PUBLIC_KEY") or "").strip() or None
        self.private_key = (os.environ.get("VAPI_PRIVATE_KEY") or "").strip() or None
        self.assistant_id = (os.environ.get("VAPI_ASSISTANT_ID") or "").strip() or None
        self.webhook_secret = (os.environ.get("VAPI_WEBHOOK_SECRET") or "").strip() or None
        self.public_url = (os.environ.get("VOICE_PUBLIC_URL") or "").strip().rstrip("/") or None
        self.business = Business.load()
        self.agent_id = (os.environ.get("VOICE_AGENT_ID") or self.business.agent_id).strip()
        self.entity_path = (os.environ.get("VOICE_ENTITY_PATH") or self.business.entity_path).strip()
        self.noise = float(os.environ.get("VOICE_NOISE") or 0.0)
        self.repair_loop = (os.environ.get("VOICE_REPAIR_LOOP") or "1").strip() not in ("0", "false", "no")
        self.stack = (os.environ.get("VOICE_STACK") or va.DEFAULT_STACK).strip()
        # On the realtime stack a tool result that comes back while the model's turn is still
        # open is dropped by the platform ("An unexpected error occurred."), and the model
        # never hears it. Both tool answers on one real call came back in under 400 ms and
        # were lost; the ones that took 1.5 s and 4.6 s on an earlier call landed. So on that
        # stack a tool answer is held until this many ms have passed since the call came in.
        # GPT-Live's reasoner waits for the answer like any tool caller; no floor there.
        default_floor = "1500" if self.stack == "realtime" else "0"
        self.tool_floor_ms = int(os.environ.get("VOICE_TOOL_MIN_MS") or default_floor)
        # Follow each call's live audio for who is speaking (voice/listen.py). VOICE_LISTEN=0 turns it off.
        self.listen = (os.environ.get("VOICE_LISTEN") or "1").strip() not in ("0", "false", "no")
        self.phone_numbers: list[str] = []


cfg = Config()
app = FastAPI(title="Jev + SenseLab voice demo", docs_url=None, redoc_url=None)
agent: LiveAgent | None = None
profiles: dict[str, dict[str, Any]] = {}  # profile id -> caller facts registered by the page
listeners: dict[str, asyncio.Task[None]] = {}  # call id -> the task following its live audio
looked_up: set[str] = set()  # calls whose record was fetched for the listen URL (once each)


def start_listening(call_id: str, call: dict[str, Any] | None) -> bool:
    """Follow the call's live audio for speech starts and stops, if Vapi told us where it
    streams (``call.monitor.listenUrl``, on the call record). Webhook messages carry the
    record; when the one in hand has no monitor (the page knows only the call's id), the
    record is fetched once, off the request's path. On GPT-Live this is the only live word
    on who is speaking; on the other stacks Vapi's ``speech-update`` says the same and the
    audio is a second opinion with better timing."""
    if not cfg.listen or call_id in listeners:
        return False
    url = ((call or {}).get("monitor") or {}).get("listenUrl")
    if not url:
        if cfg.private_key and call_id not in looked_up:
            looked_up.add(call_id)
            asyncio.get_running_loop().create_task(_listen_after_lookup(call_id))
        return False
    ag = _agent()

    def on_speech(role: str, status: str, at: float) -> None:
        ag.note_speech(call_id, role, status, t=at)

    task = asyncio.get_running_loop().create_task(li.listen(call_id, str(url), on_speech, on_open=lambda: ag.set_listening(call_id)))
    listeners[call_id] = task
    task.add_done_callback(lambda _t: listeners.pop(call_id, None))
    return True


LOOKUP_AFTER_S = (0.3, 1.5, 4.0)  # the record gains monitor.listenUrl once the call is in progress


async def _listen_after_lookup(call_id: str) -> None:
    for wait in LOOKUP_AFTER_S:
        await asyncio.sleep(wait)
        if call_id not in looked_up:  # the call ended meanwhile
            return
        try:
            call = await run_in_threadpool(va.fetch_call, cfg.private_key or "", call_id)
        except Exception as exc:  # noqa: BLE001 - the page still has the SDK's stream
            log.info("no listen stream for %s: %s", call_id, exc)
            return
        if call and (call.get("monitor") or {}).get("listenUrl"):
            start_listening(call_id, call)
            return
    log.info("no listen stream for %s: the call record has no monitor.listenUrl", call_id)


def stop_listening(call_id: str) -> None:
    looked_up.discard(call_id)
    task = listeners.pop(call_id, None)
    if task is not None and not task.done():
        task.cancel()


@app.on_event("startup")
def _startup() -> None:
    global agent
    agent = LiveAgent(cfg.settings, cfg.business, agent_id=cfg.agent_id, entity_path=cfg.entity_path, noise=cfg.noise,
                      repair_loop=cfg.repair_loop)
    log.info("business: %s (%s), %s", cfg.business.name, cfg.business.path, cfg.business.outcomes_mode)
    info = agent.start()
    log.info("agent ready: %s", json.dumps({k: v for k, v in info.items() if k != "loop"}))
    if cfg.private_key and cfg.public_url:
        try:
            a = va.upsert_assistant(cfg.private_key, business=cfg.business, server_url=f"{cfg.public_url}/vapi/webhook",
                                    secret=cfg.webhook_secret)
            cfg.assistant_id = a["id"]
            log.info("assistant %s (%s) points at %s/vapi/webhook", a["id"], a.get("name"), cfg.public_url)
            api = va.VapiApi(cfg.private_key)
            try:
                # Our numbers answer through this server (assistant-request), so each call
                # gets the caller's facts and history in its prompt. Point them here.
                cfg.phone_numbers = api.route_numbers(a["id"], server_url=f"{cfg.public_url}/vapi/webhook",
                                                      secret=cfg.webhook_secret)
            finally:
                api.close()
            if cfg.phone_numbers:
                log.info("phone numbers answering through this server: %s", cfg.phone_numbers)
        except Exception as exc:  # noqa: BLE001 - the page says so; text mode still works
            log.warning("could not upsert the Vapi assistant: %s", exc)
    elif cfg.assistant_id:
        log.info("using assistant %s from VAPI_ASSISTANT_ID", cfg.assistant_id)
    else:
        log.info("no VOICE_PUBLIC_URL: browser calls are off, text mode (POST /api/calls) works")


@app.on_event("shutdown")
def _shutdown() -> None:
    if agent is not None:
        agent.close()


def _agent() -> LiveAgent:
    if agent is None:
        raise HTTPException(503, "agent not ready")
    return agent


# -- Vapi --------------------------------------------------------------------------------


OUR_KEYS = ("name", "tier", "prior_contacts_30d", "tenure_months", "profile_id", "customer_id")


def profile_from_call(call: dict[str, Any]) -> dict[str, Any]:
    """The caller's account facts, and their history if it was prepared. A browser call
    carries them in the variable values the page passed to ``vapi.start``; a phone call
    carries the carrier's values and nothing of ours, so it gets the profile most recently
    registered on the page ("use this caller"), if that was within ten minutes."""
    raw = dict((call.get("assistantOverrides") or {}).get("variableValues") or {})
    raw.update(call.get("metadata") or {})
    values = {k: raw[k] for k in (*OUR_KEYS, "history") if raw.get(k) not in (None, "")}
    pid = str(values.get("profile_id") or "")
    if pid and pid in profiles:
        values = {**profiles[pid], **values}
    if not values and profiles:
        last = max(profiles.values(), key=lambda p: p["registered"])
        if time.time() - last["registered"] < 600:
            values = dict(last)
    return values


def caller_from_call(call: dict[str, Any]) -> Caller:
    return _agent().caller_from(profile_from_call(call))


# -- who is calling, from the number ----------------------------------------------------------
def caller_number(call: dict[str, Any]) -> str:
    """The caller's number on an inbound phone call, as Vapi reports it."""
    return str(((call or {}).get("customer") or {}).get("number") or "").strip()


def phone_profile(call: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Who an inbound phone call is from, and how we know. A caller registered on the page
    in the last ten minutes wins (that is how the demo switches callers) and teaches the
    phone book the number; otherwise the book answers (``voice/phonebook.py``: your lookup
    first, then the numbers the agent has learned); otherwise nobody."""
    prof = profile_from_call(call)
    number = caller_number(call)
    book = _agent().phonebook
    if prof:
        if prof.get("name") and number:
            book.remember(number, prof)
        return prof, ("page" if prof.get("name") else None)
    if number:
        return book.who(number)
    return {}, None


NO_ACCOUNT = ("No account matched this caller. There is no name, plan, tenure or history on file, so state none; "
              "whatever is known about them comes from this call.")


def variable_values(profile: dict[str, Any]) -> dict[str, str]:
    """What the prompt's ``{{...}}`` are filled with: the account facts (``{{account}}``, one
    sentence) and the history. A caller we could not match has no name, and gets the
    no-account sentence instead of the defaults a ``Caller`` carries, so the model has
    nothing to mistake for facts about them."""
    out = {k: str(v) for k, v in profile.items() if k in (*OUR_KEYS, "history") and k != "profile_id"}
    if not out.get("name") or out["name"] == "unknown":
        out["name"] = "unknown"
        out["account"] = NO_ACCOUNT
        out["history"] = ""
    else:
        out["account"] = (f"name {out['name']}, {out.get('tier', 'consumer')} plan, {out.get('prior_contacts_30d', '0')} "
                          f"contact(s) with support in the last 30 days, {out.get('tenure_months', '?')} months with us")
    return out


def tool_result(d: Any, business: Business, *, record: str = "", stack: str = "live") -> str:
    """What comes back from ``decide_next_step``: the decision, what happens now, and why
    (the situation's record, so the agent can say what has and has not worked for people
    like this caller). The decision itself is not up for discussion, and the result says
    so. It does not end the call: the close is the model's, and the prompt has it wait for
    the caller. On GPT-Live the reader is the reasoner, which hands facts to the speaker, so
    the result is plain facts; on the one-model stacks the reader is the voice itself, and
    the result also says how to deliver it."""
    label = business.action(d.action).label
    # They rang back about the same thing: the earlier step is already marked as not
    # working, and the agent says so before the new step, instead of pretending it held.
    back = getattr(d, "rang_back", None) or ""
    if stack == "live":
        why = f" Why: {record}" if record else ""
        rang = (f" Before the decision: {back} Own it in one line (the earlier step did not happen for them, and it "
                f"is on their file as such) before saying what happens now.") if back else ""
        return (f"Decided: {label.lower()} ({d.action}). What happens now: {d.then}{why}{rang} Instructions for the "
                f"assistant, not words for the caller: this is the one next step on this call, offer nothing else; "
                f"any detail taken from the caller (a number, a time) is repeated back to them once.")
    why = f" Why, in plain words and without the numbers: {record}" if record else ""
    rang = (f" First: {back} Say in one line that you can see it did not happen for them and it is on their file as "
            f"such, then the new step.") if back else ""
    return (f"Next step: {d.action}. Decided: {label.lower()}. This is what happens on this call; do not offer, "
            f"hint at or agree to anything else. {d.then}{why}{rang} Say it the way you would to a friend, in a sentence "
            f"or two, with the reason, then let them respond. Confirm any detail you take from them by repeating it "
            f"back. Do not end the call until they have said they are done.")


_TOOL_RANK = {"identify_caller": 0, "update_file": 1, "recall_customer": 2, "decide_next_step": 3}


def tool_order(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The order to run one turn's tool calls in, whatever order the model listed them: who
    is calling, then what they said goes on file, then the decision. A refusal put on file
    ("no, don't transfer me") has to close the step it refuses before the new decision opens
    the next one, or it would close the wrong step. Stable within a rank."""
    def rank(tc: dict[str, Any]) -> int:
        fn = tc.get("function") or {}
        return _TOOL_RANK.get(str(fn.get("name") or tc.get("name") or ""), 9)
    return sorted(tool_calls, key=rank)


def tool_reply(tc: dict[str, Any], d: Any, business: Business, *, record: str = "", stack: str = "live") -> dict[str, Any]:
    """One entry of ``results``. No scripted line: the model tells the caller in its own words."""
    return {"toolCallId": tc.get("id"), "name": "decide_next_step", "result": tool_result(d, business, record=record, stack=stack)}


@app.post("/vapi/webhook")
async def vapi_webhook(request: Request) -> JSONResponse:
    if cfg.webhook_secret and request.headers.get("x-vapi-secret") != cfg.webhook_secret:
        raise HTTPException(401, "bad x-vapi-secret")
    body = await request.json()
    msg = body.get("message") or body
    typ = str(msg.get("type") or "")
    call = msg.get("call") or {}
    call_id = str(call.get("id") or msg.get("callId") or "unknown")
    ag = _agent()

    if typ == "status-update":
        status = str(msg.get("status") or "")
        if status == "in-progress" and call_id not in ag.calls:
            prof = profile_from_call(call)
            ag.call_started(call_id, ag.caller_from(prof), call_type=str(call.get("type") or "webCall"),
                            brief=str(prof.get("history") or "") or None)
        if status == "in-progress":
            start_listening(call_id, call)
        return JSONResponse({})

    if typ == "assistant-request":
        if not cfg.assistant_id:
            return JSONResponse({"error": "no assistant configured on this server"})
        # An inbound phone call. The number is routed through this server so the caller's
        # account facts and history can go into the prompt (a phone call carries none of
        # ours). Pre-open the runs now too, before the caller has spoken.
        prof, known_by = phone_profile(call)
        # Nobody registered on the page and a number we have not met: a stranger, greeted
        # as one (no name, no history). If they say who they are, identify_caller matches it.
        caller = ag.caller_from(prof) if prof.get("name") else Caller(name="")
        if prof.get("history") and prof.get("first_message"):
            history, first = str(prof["history"]), str(prof["first_message"])
        else:
            history, first = await run_in_threadpool(ag.intro_for, caller)
        ag.call_started(call_id, caller, call_type=str(call.get("type") or "inboundPhoneCall"), brief=history,
                        known_by=known_by)
        start_listening(call_id, call)
        return JSONResponse({"assistantId": cfg.assistant_id,
                             "assistantOverrides": {"firstMessage": first,
                                                    "variableValues": variable_values({**asdict(caller), "history": history})}})

    if typ == "tool-calls":
        t0 = time.monotonic()
        if call_id in ag.calls:
            start_listening(call_id, call)  # a call the server first hears of here
        # The conversation so far rides along with the tool call. On GPT-Live it is the only
        # live record of the caller's words that reaches the server (see note_turns).
        ag.note_turns(call_id, (msg.get("artifact") or {}).get("messages"))
        results = []
        for tc in tool_order(msg.get("toolCallList") or []):
            fn = tc.get("function") or {}
            name = str(fn.get("name") or tc.get("name") or "")
            args = fn.get("arguments") if fn else tc.get("parameters")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"summary": args}
            if name == "decide_next_step":
                d = ag.decide(call_id, dict(args or {}), caller=caller_from_call(call))
                results.append(tool_reply(tc, d, ag.business, record=ag.record_words(call_id, d.situation), stack=cfg.stack))
            elif name == "recall_customer":
                text = ag.recall(call_id, dict(args or {}), caller=caller_from_call(call))
                results.append({"toolCallId": tc.get("id"), "name": name, "result": text})
            elif name == "identify_caller":
                text, who = await run_in_threadpool(ag.identify, call_id, dict(args or {}))
                if who is not None and caller_number(call):
                    ag.phonebook.remember(caller_number(call), asdict(who))
                results.append({"toolCallId": tc.get("id"), "name": name, "result": text})
            elif name == "update_file":
                text = await run_in_threadpool(ag.update_file, call_id, dict(args or {}))
                results.append({"toolCallId": tc.get("id"), "name": name, "result": text})
            else:
                results.append({"toolCallId": tc.get("id"), "name": name, "error": f"unknown tool {name}"})
        wait = cfg.tool_floor_ms / 1000 - (time.monotonic() - t0)
        if wait > 0:
            await asyncio.sleep(wait)
        return JSONResponse({"results": results})

    if typ == "speech-update":
        ag.note_speech(call_id, str(msg.get("role") or ""), str(msg.get("status") or ""))
        return JSONResponse({})

    if typ == "transcript":
        if str(msg.get("transcriptType") or "final") == "final":
            ag.note_transcript(call_id, str(msg.get("role") or ""), str(msg.get("transcript") or ""))
        return JSONResponse({})

    if typ == "end-of-call-report":
        artifact = dict(msg.get("artifact") or {})
        for k in ("recordingUrl", "stereoRecordingUrl", "transcript"):
            if msg.get(k) and not artifact.get(k):
                artifact[k] = msg[k]
        ag.note_turns(call_id, artifact.get("messages"))  # whatever the call did not show live
        stop_listening(call_id)
        ag.end_call_later(call_id, ended_reason=str(msg.get("endedReason") or ""), artifact=artifact,
                          fetch_recording=(lambda cid=call_id: va.presigned_recording(cfg.private_key, cid)) if cfg.private_key else None)
        return JSONResponse({})

    if typ == "transfer-destination-request":
        # The demo has no human queue to send a call to; the assistant ends the call instead.
        return JSONResponse({"error": "no transfer destination in this demo"})

    return JSONResponse({})


# -- the page and its API ------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (WEB_DIR / "index.html").read_text()


@app.get("/recordings/{name}")
def recording(name: str) -> FileResponse:
    """A saved call recording (``results/voice/<call_id>.wav``)."""
    ag = _agent()
    if ag.recordings_dir is None or "/" in name or name.startswith("."):
        raise HTTPException(404)
    path = ag.recordings_dir / name
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


@app.get("/api/config")
def api_config() -> dict[str, Any]:
    return {
        "public_key": cfg.public_key, "assistant_id": cfg.assistant_id, "agent_id": cfg.agent_id,
        "entity_path": cfg.entity_path, "dashboard": S.DASHBOARD_URL, "browser_calls": bool(cfg.public_key and cfg.assistant_id),
        "noise": cfg.noise, "changed": _agent().changed, "phone_numbers": cfg.phone_numbers,
        "business": cfg.business.public(), "stack": cfg.stack,
    }


@app.get("/api/state")
def api_state() -> dict[str, Any]:
    return _agent().state()


@app.get("/api/events")
def api_events(since: int = 0) -> dict[str, Any]:
    """Events after *since* (their ``seq``), for the live view."""
    events = _agent().events_since(since)
    return {"events": events, "seq": events[-1]["seq"] if events else since}


@app.get("/api/memory")
def api_memory() -> dict[str, Any]:
    return _agent().memory_snapshot()


@app.get("/api/loop")
def api_loop() -> dict[str, Any]:
    return _agent().loop_snapshot()


@app.post("/api/profiles")
async def api_profiles(request: Request) -> dict[str, Any]:
    """Register who the next call is from. Reads their history from SenseLab now, so it is
    in the prompt from the first word: the page passes ``variable_values`` to ``vapi.start``,
    and a phone call within ten minutes gets the same values from ``assistant-request``."""
    body = await request.json()
    ag = _agent()
    pid = uuid.uuid4().hex[:10]
    # ``{"stranger": true}``: a caller we have no record of, greeted as one (the test calls use it).
    c = Caller(name="") if body.get("stranger") else ag.caller_from(body)
    history, first = await run_in_threadpool(ag.intro_for, c)
    profiles[pid] = {"profile_id": pid, "registered": time.time(), **asdict(c), "history": history, "first_message": first}
    for old in [k for k, v in profiles.items() if time.time() - v["registered"] > 3600]:
        del profiles[old]
    return {"profile_id": pid, "first_message": first,
            "variable_values": {"profile_id": pid, **variable_values(profiles[pid])}}


@app.post("/api/twins")
async def api_twins(request: Request) -> dict[str, Any]:
    """Seed a demo caller's history under another customer id, for test calls
    (``voice/testcalls.py``): ``{"persona_id": "bruno-andrade", "customer_id": "bruno-sim-1"}``.
    The demo caller's own record stays untouched by what the tests do."""
    body = await request.json()
    ag = _agent()
    try:
        seeded = await run_in_threadpool(ag.seed_twin, str(body.get("persona_id") or ""), str(body.get("customer_id") or ""))
    except KeyError as exc:
        raise HTTPException(404, f"no persona {exc}") from exc
    return {"customer_id": body.get("customer_id"), "seeded_now": seeded}


@app.post("/api/world")
async def api_world(request: Request) -> dict[str, Any]:
    body = await request.json()
    _agent().set_changed(bool(body.get("changed")))
    return {"changed": _agent().changed}


@app.post("/api/calls")
async def api_calls(request: Request) -> dict[str, Any]:
    """Text mode: a call without Vapi. The simulator and the page's text box use it."""
    body = await request.json()
    call_id = str(body.get("call_id") or new_call_id())
    _agent().call_started(call_id, _agent().caller_from(body.get("caller")), call_type="text")
    return {"call_id": call_id}


@app.post("/api/calls/{call_id}/start")
async def api_call_start(call_id: str, request: Request) -> dict[str, Any]:
    """The page says a browser call has connected (``{"profile_id": ...}``). On the classic
    stacks Vapi's ``status-update`` says the same a moment later and this is a no-op; on
    GPT-Live that message never comes (nor do speech and transcript updates: only tool calls
    and the end-of-call report reach the server), so without this the server would first
    hear of the call at its first tool call, and the runs opened ahead would open then."""
    body = await request.json() if await request.body() else {}
    ag = _agent()
    known = call_id in ag.calls
    if not known:
        prof = profiles.get(str(body.get("profile_id") or ""), {})
        ag.call_started(call_id, ag.caller_from(prof) if prof.get("name") else Caller(name=""), call_type="webCall",
                        brief=str(prof.get("history") or "") or None)
    start_listening(call_id, None)  # the page knows only the id; the record says where the audio streams
    return {"call_id": call_id, "known": known}


@app.post("/api/calls/{call_id}/notes")
async def api_call_notes(call_id: str, request: Request) -> dict[str, Any]:
    """A ``transcript`` or ``speech-update`` message relayed by the page from the browser
    SDK, in Vapi's own shape, with an optional ``t`` (epoch seconds, the server's clock) for
    when it happened. The page relays only on GPT-Live, where the server gets none of them
    from Vapi. What Vapi's browser SDK sends on GPT-Live is the agent's transcript as one
    growing string, a few words at a time, with a single final at the end of the call; the
    page turns that into speech starts and stops and one line per utterance before relaying,
    so what arrives here is the same shape as on the other stacks."""
    msg = await request.json()
    ag = _agent()
    typ = str(msg.get("type") or "")
    t = float(msg["t"]) if isinstance(msg.get("t"), (int, float)) else None
    if typ == "speech-update":
        if ag.is_listening(call_id):
            return {"ignored": "the call's audio says who is speaking"}
        ag.note_speech(call_id, str(msg.get("role") or ""), str(msg.get("status") or ""), t=t)
    elif typ == "transcript" and str(msg.get("transcriptType") or "final") == "final":
        ag.note_transcript(call_id, str(msg.get("role") or ""), str(msg.get("transcript") or ""), t=t)
    return {}


@app.post("/api/calls/{call_id}/decide")
async def api_decide(call_id: str, request: Request) -> dict[str, Any]:
    body = await request.json()
    ag = _agent()
    d = ag.decide(call_id, dict(body), caller=ag.caller_from(body.get("caller")))
    return {"decision": d.public(), "result": tool_result(d, ag.business)}


@app.post("/api/calls/{call_id}/end")
async def api_end(call_id: str, request: Request) -> dict[str, Any]:
    body = await request.json() if await request.body() else {}
    ag = _agent()
    pending = [d.public() for d in ag.pending_decisions(call_id)]
    ag.end_call_later(call_id, ended_reason=str(body.get("ended_reason") or "text-mode"),
                      auto_review=bool(body.get("auto_review", True)))
    # In simulated mode the outcomes land a few seconds later; poll /api/state for them (the
    # page does). In reported mode they wait for POST /api/outcome.
    return {"decisions": pending, "awaiting_outcome": not ag.business.simulated}


@app.post("/api/outcome")
async def api_outcome(request: Request) -> dict[str, Any]:
    body = await request.json()
    severity = str(body.get("severity") or ("success" if body.get("success") else "failure"))
    if severity not in SEVERITIES:
        raise HTTPException(400, f"severity must be one of {SEVERITIES}")
    success = severity == "success"
    ag = _agent()
    outcome = Outcome(success=success, severity=severity, note=str(body.get("note") or "")[:300],
                      true_success=success, cost_usd=float(ag.business.cost_by_severity.get(severity, 0.0)))
    try:
        d = ag.report_outcome(str(body.get("decision_id")), outcome, source=str(body.get("source") or "human"))
    except KeyError:
        raise HTTPException(404, "no pending decision with that id") from None
    return {"decision": d.public()}
