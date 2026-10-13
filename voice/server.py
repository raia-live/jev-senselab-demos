"""The webhook server Vapi talks to, plus a small page to make calls from the browser.

    uv run uvicorn voice.server:app --port 8787

Routes
  POST /vapi/webhook      Vapi server messages: status-update, speech-update, transcript, tool-calls, end-of-call-report,
                          assistant-request (phone numbers), transfer-destination-request.
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
VOICE_PUBLIC_URL (the tunnel URL Vapi can reach), VOICE_AGENT_ID, VOICE_ENTITY_PATH, VOICE_NOISE.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from core import settings as S
from core.scenario import SEVERITIES, Outcome

from . import assistant as va
from .agent import DEFAULT_AGENT_ID, DEFAULT_ENTITY_PATH, Caller, LiveAgent, new_call_id
from .customers import PERSONAS

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
        self.agent_id = (os.environ.get("VOICE_AGENT_ID") or DEFAULT_AGENT_ID).strip()
        self.entity_path = (os.environ.get("VOICE_ENTITY_PATH") or DEFAULT_ENTITY_PATH).strip()
        self.noise = float(os.environ.get("VOICE_NOISE") or 0.0)
        self.repair_loop = (os.environ.get("VOICE_REPAIR_LOOP") or "1").strip() not in ("0", "false", "no")
        self.phone_numbers: list[str] = []


cfg = Config()
app = FastAPI(title="Jev + SenseLab voice demo", docs_url=None, redoc_url=None)
agent: LiveAgent | None = None
profiles: dict[str, dict[str, Any]] = {}  # profile id -> caller facts registered by the page


@app.on_event("startup")
def _startup() -> None:
    global agent
    agent = LiveAgent(cfg.settings, agent_id=cfg.agent_id, entity_path=cfg.entity_path, noise=cfg.noise,
                      repair_loop=cfg.repair_loop)
    info = agent.start()
    log.info("agent ready: %s", json.dumps({k: v for k, v in info.items() if k != "loop"}))
    if cfg.private_key and cfg.public_url:
        try:
            a = va.upsert_assistant(cfg.private_key, server_url=f"{cfg.public_url}/vapi/webhook",
                                    secret=cfg.webhook_secret)
            cfg.assistant_id = a["id"]
            log.info("assistant %s (%s) points at %s/vapi/webhook", a["id"], a.get("name"), cfg.public_url)
            api = va.VapiApi(cfg.private_key)
            try:
                cfg.phone_numbers = [str(n.get("number")) for n in api.list_phone_numbers()
                                     if n.get("assistantId") == a["id"] and n.get("number")]
            finally:
                api.close()
            if cfg.phone_numbers:
                log.info("phone numbers on the assistant: %s", cfg.phone_numbers)
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


def caller_from_call(call: dict[str, Any]) -> Caller:
    """The caller's account facts. A browser call carries them in the variable values the
    page passed to ``vapi.start``; a phone call carries the carrier's values and nothing of
    ours, so it gets the profile most recently registered on the page ("Use this caller for
    the next phone call"), if that was within ten minutes."""
    raw = dict((call.get("assistantOverrides") or {}).get("variableValues") or {})
    raw.update(call.get("metadata") or {})
    values = {k: raw[k] for k in OUR_KEYS if raw.get(k) not in (None, "")}
    pid = str(values.get("profile_id") or "")
    if pid and pid in profiles:
        values = {**profiles[pid], **values}
    if not values and profiles:
        last = max(profiles.values(), key=lambda p: p["registered"])
        if time.time() - last["registered"] < 600:
            values = dict(last)
    return Caller.from_mapping(values)


AFTER = {
    "resolve_in_bot": ("Now handle it on this call in two or three sentences, answer any questions, and when there is "
                       "nothing else, say goodbye and end the call."),
    "transfer_to_agent": ("There is no human queue in this demo, so tell the caller a colleague will pick it up from "
                          "here, answer any last question, thank them for calling, and end the call."),
    "schedule_callback": ("Now confirm the number on file and that it will be within one business day. If they ask why "
                          "this will work when earlier attempts did not, answer from what you know of their history. "
                          "When there is nothing else, say goodbye and end the call."),
    "issue_credit": "Now confirm the amount, answer any questions, and when there is nothing else, say goodbye and end the call.",
}


def tool_result(d: Any) -> str:
    """What goes into the conversation history for the model."""
    return f'Next step: {d.action}. The caller has just been told: "{d.say}" {AFTER[d.action]}'


def tool_reply(tc: dict[str, Any], d: Any) -> dict[str, Any]:
    """One entry of ``results``: the record for the model, and the line Vapi speaks to the
    caller right away, word for word, so the decision is what the caller hears."""
    return {
        "toolCallId": tc.get("id"), "name": "decide_next_step", "result": tool_result(d),
        "message": {"type": "request-complete", "role": "assistant", "content": d.say},
    }


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
            ag.call_started(call_id, caller_from_call(call), call_type=str(call.get("type") or "webCall"))
        return JSONResponse({})

    if typ == "assistant-request":
        if not cfg.assistant_id:
            return JSONResponse({"error": "no assistant configured on this server"})
        # An inbound phone call: pre-open the runs now, before the caller has spoken.
        ag.call_started(call_id, caller_from_call(call), call_type=str(call.get("type") or "inboundPhoneCall"))
        return JSONResponse({"assistantId": cfg.assistant_id})

    if typ == "tool-calls":
        results = []
        for tc in msg.get("toolCallList") or []:
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
                results.append(tool_reply(tc, d))
            elif name == "recall_customer":
                text = ag.recall(call_id, dict(args or {}), caller=caller_from_call(call))
                results.append({"toolCallId": tc.get("id"), "name": name, "result": text})
            else:
                results.append({"toolCallId": tc.get("id"), "name": name, "error": f"unknown tool {name}"})
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
        "personas": [p.public() for p in PERSONAS],
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
    body = await request.json()
    pid = uuid.uuid4().hex[:10]
    c = Caller.from_mapping(body)
    profiles[pid] = {"profile_id": pid, "registered": time.time(), "name": c.name, "tier": c.tier,
                     "prior_contacts_30d": c.prior_contacts_30d, "tenure_months": c.tenure_months,
                     "customer_id": c.customer_id}
    for old in [k for k, v in profiles.items() if time.time() - v["registered"] > 3600]:
        del profiles[old]
    return {"profile_id": pid, "variable_values": {k: str(v) for k, v in profiles[pid].items() if k != "registered"}}


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
    _agent().call_started(call_id, Caller.from_mapping(body.get("caller")), call_type="text")
    return {"call_id": call_id}


@app.post("/api/calls/{call_id}/decide")
async def api_decide(call_id: str, request: Request) -> dict[str, Any]:
    body = await request.json()
    d = _agent().decide(call_id, dict(body), caller=Caller.from_mapping(body.get("caller")))
    return {"decision": d.public(), "result": tool_result(d)}


@app.post("/api/calls/{call_id}/end")
async def api_end(call_id: str, request: Request) -> dict[str, Any]:
    body = await request.json() if await request.body() else {}
    ag = _agent()
    pending = [d.public() for d in ag.pending_decisions(call_id)]
    ag.end_call_later(call_id, ended_reason=str(body.get("ended_reason") or "text-mode"),
                      auto_review=bool(body.get("auto_review", True)))
    # The outcomes land a few seconds later; poll /api/state for them (the page does).
    return {"decisions": pending}


@app.post("/api/outcome")
async def api_outcome(request: Request) -> dict[str, Any]:
    body = await request.json()
    severity = str(body.get("severity") or ("success" if body.get("success") else "failure"))
    if severity not in SEVERITIES:
        raise HTTPException(400, f"severity must be one of {SEVERITIES}")
    success = severity == "success"
    ag = _agent()
    outcome = Outcome(success=success, severity=severity, note=str(body.get("note") or "")[:300],
                      true_success=success, cost_usd=float(ag.scenario.cost_by_severity.get(severity, 0.0)))
    try:
        d = ag.report_outcome(str(body.get("decision_id")), outcome, source=str(body.get("source") or "human"))
    except KeyError:
        raise HTTPException(404, "no pending decision with that id") from None
    return {"decision": d.public()}
