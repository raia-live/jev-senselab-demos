"""The Vapi assistant: created or updated by name through the Vapi API.

    uv run python -m voice.assistant --server-url https://<your-tunnel>/vapi/webhook

Vapi runs the conversation. The prompt, the greeting and the tools are built from the
business file (``voice/business.toml`` or ``$VOICE_BUSINESS``), so the same code answers for
any business. Its two tools are answered by ``voice/server.py``: ``recall_customer`` with
what SenseLab remembers of this caller (earlier contacts and what came of them) so the agent
can pick the thread up, and ``decide_next_step`` with Jev's decision. Nothing the caller hears
is scripted: the model says the decision in its own words, and the tools carry no canned
lines, only a "still with you" if an answer takes long.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

import httpx

from .business import Business

VAPI_URL = "https://api.vapi.ai"
ASSISTANT_NAME = "Jev support agent (SenseLab demo)"


def _intent_lines(b: Business) -> str:
    return "; ".join(f"{i.label} ({i.id})" for i in b.intents)


def _needs_lines(b: Business) -> str:
    parts = []
    for i in b.intents:
        if i.needs:
            words = " and ".join(b.field_words.get(n, n.replace("_", " ")) for n in i.needs)
            parts.append(f"for {i.label.split(' (')[0]}, {words}")
    return ("The one more thing you need before you decide: " + "; ".join(parts) +
            ". If the history already has it, confirm it rather than ask.") if parts else ""


def system_prompt(b: Business) -> str:
    """Who the agent is and how a call goes, for this business. Written for a person on the
    phone, not a procedure; the few hard rules (recall first, decide before promising, say
    nothing of the machinery) are in it."""
    where = f"{b.name}, {b.what}" if b.what else b.name
    style = f"\n\nHouse style: {b.house_style.strip()}" if b.house_style.strip() else ""
    return f"""You are {b.agent_name}, on the phones at {where}. You are talking to a customer right now, out loud, on a live call.

Who you are on the phone: warm, unhurried, direct. You sound like someone who has done this a long time and still likes \
people. Short sentences. Contractions. One thought at a time. You react before you act ("Oh no. Twice?"), you let \
people finish, and you do not fill silence with corporate noise. A little humour is fine, never at the caller's expense. \
No lists, no "I understand your frustration", no "absolutely", no "great question". Their name once early and perhaps \
once near the end, not every turn. Never the same filler twice in one call. Numbers and dates the way people say them \
("forty-nine dollars", "the twenty-third"). If you did not catch something, say so and ask again; do not guess.{style}

What the account already tells you: name {{{{name}}}}, {{{{tier}}}} plan, {{{{prior_contacts_30d}}}} contact(s) with support in \
the last 30 days, {{{{tenure_months}}}} months with us. Do not recite these; use them.

How a call goes:
- The greeting has been said. Listen. As soon as you know roughly what they are calling about, even vaguely ("my bill, \
again"), say one short natural thing that fits what they said, then call recall_customer with what you heard. Do not \
make them tell the whole story first.
- recall_customer gives you their history: earlier contacts, what was tried, what came of it, and what tends to work for \
people in their situation. Talk like someone who has read the file: pick up the thread, mention one or two specifics in \
your own words, and if the history makes the problem obvious, check it in one short question ("Is this the same \
forty-nine dollar charge?") rather than asking them to start from the beginning. If something we did let them down, say \
so plainly and apologise once, like a person, not a policy. If there is no history, say so lightly and ask what is going on.
- Then talk it through. Answer what they ask from what you know. One question at a time. If they are upset, let them be \
upset for a moment before you move on. What they can be calling about: {_intent_lines(b)}. {_needs_lines(b)}
- When you know what it is (and the amount, where one is needed), call decide_next_step. Until it answers, do not \
promise, book, transfer, credit or resolve anything, and do not hint at which way it will go. Its answer is what happens \
on this call. Put it in your own words, warmly, in a sentence or two, and let them react.
- If they push back ("why would it work this time?"), answer from their actual history: what was tried, what came of it, \
why this is different. Hold the course kindly; do not re-decide because they are unhappy. Call decide_next_step again \
only if a different problem comes up or new facts change the picture.
- Before you go, ask if there is anything else. Then say goodbye the way you would to a regular, and use endCall.

Never mention tools, systems, policies, models, memory, Jev or SenseLab. If asked whether you are a person, be honest \
in a few words and carry on.
"""


def recall_tool(b: Business) -> dict[str, Any]:
    return {
        "type": "function",
        "async": False,
        "function": {
            "name": "recall_customer",
            "description": ("What we know about this caller: earlier contacts, what was tried, what came of it, and what "
                            "has worked for callers in their situation. Call it as soon as the caller says what they are "
                            "calling about, before you respond to the substance."),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "What the caller said they are calling about, in a few words."},
                    "intent": {"type": "string", "enum": list(b.intent_ids), "description": "Your best guess at the intent, if you have one."},
                    **{name: {"type": "number", "description": f"{words[0].upper()}{words[1:]}, if they named one."}
                       for name, words in b.field_words.items()},
                },
                "required": ["topic"],
            },
        },
        "messages": [
            # No scripted opener: the model says something that fits, then the tool runs. A
            # word only if the lookup is slow, and if it fails.
            {"type": "request-response-delayed", "content": "Still with you, just pulling that up.", "timingMilliseconds": 2500},
            {"type": "request-failed", "content": "I can't get to your account history right now, so tell me what's going on."},
        ],
    }


def decide_tool(b: Business) -> dict[str, Any]:
    labels = ", ".join(a.label.lower() for a in b.actions)
    return {
        "type": "function",
        "async": False,
        "function": {
            "name": "decide_next_step",
            "description": (f"Decide what to do next on this call once the caller's need is clear: {labels}. "
                            "Call it before promising any of those."),
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string", "enum": list(b.intent_ids), "description": "What the caller needs."},
                    "summary": {"type": "string",
                                "description": "One or two sentences on what the caller said, in their words where possible, "
                                               "including anything about earlier contacts or what they were promised."},
                    **{name: {"type": "number", "description": f"{words[0].upper()}{words[1:]}. 0 if there is none."}
                       for name, words in b.field_words.items()},
                    "sentiment": {"type": "string", "enum": ["calm", "frustrated", "angry"], "description": "How the caller sounds."},
                },
                "required": ["intent", "summary"],
            },
        },
        "messages": [
            {"type": "request-response-delayed", "content": "Bear with me one moment.", "timingMilliseconds": 3500},
            {"type": "request-failed", "content": "I'm having trouble reaching our systems. Let me get a colleague to help you."},
        ],
    }


def assistant_config(b: Business, *, server_url: str, secret: str | None, voice_id: str, voice_provider: str) -> dict[str, Any]:
    voice: dict[str, Any] = {"provider": voice_provider, "voiceId": voice_id}
    if voice_provider == "11labs":
        # Multilingual v2 is ElevenLabs' most natural model; it costs a few hundred ms over
        # turbo, which the prefetch leaves room for. Lower stability lets the delivery vary;
        # style adds the small inflections that make it sound like a person.
        voice.update({"model": "eleven_multilingual_v2", "stability": 0.4, "similarityBoost": 0.8,
                      "style": 0.35, "useSpeakerBoost": True})
    cfg: dict[str, Any] = {
        "name": ASSISTANT_NAME,
        "firstMessage": b.greeting,
        "firstMessageMode": "assistant-speaks-first",
        "model": {
            "provider": "openai", "model": "gpt-4.1", "temperature": 0.7,
            "messages": [{"role": "system", "content": system_prompt(b)}],
            "tools": [recall_tool(b), decide_tool(b), {"type": "endCall"}],
        },
        "voice": voice,
        "transcriber": {"provider": "deepgram", "model": "nova-3", "language": "en"},
        # The conversational settings. Backchannels ("mm-hm") while the caller talks; a short
        # wait plus smart endpointing so a pause mid-thought is not taken for the end of the
        # turn; a couple of words from the caller interrupt, a cough does not; a word if the
        # line goes quiet.
        "backchannelingEnabled": True,
        "startSpeakingPlan": {"waitSeconds": 0.5, "smartEndpointingPlan": {"provider": "livekit"}},
        "stopSpeakingPlan": {"numWords": 2, "voiceSeconds": 0.25, "backoffSeconds": 1.0},
        "messagePlan": {"idleMessages": ["Take your time.", "Still here whenever you're ready."],
                        "idleTimeoutSeconds": 10, "idleMessageMaxSpokenCount": 2},
        "server": {"url": server_url, "timeoutSeconds": 20, **({"secret": secret} if secret else {})},
        "serverMessages": ["status-update", "speech-update", "transcript", "tool-calls", "end-of-call-report"],
        "clientMessages": ["transcript", "tool-calls", "tool-calls-result", "status-update", "speech-update"],
        # Every call is recorded (stereo: caller one channel, the agent the other); the
        # server keeps the file next to the decision so the demo has the audio.
        "artifactPlan": {"recordingEnabled": True, "recordingFormat": "wav;l16"},
        "maxDurationSeconds": 600,
        "backgroundSound": "off",
    }
    return cfg


class VapiApi:
    def __init__(self, private_key: str) -> None:
        self._c = httpx.Client(base_url=VAPI_URL, headers={"Authorization": f"Bearer {private_key}"}, timeout=30.0)

    def _call(self, method: str, path: str, **kw: Any) -> Any:
        r = self._c.request(method, path, **kw)
        if r.status_code >= 400:
            raise RuntimeError(f"Vapi {method} {path}: {r.status_code} {r.text[:300]}")
        return r.json() if r.content else None

    def list_assistants(self) -> list[dict[str, Any]]:
        return list(self._call("GET", "/assistant"))

    def create_assistant(self, cfg: dict[str, Any]) -> dict[str, Any]:
        return self._call("POST", "/assistant", json=cfg)

    def update_assistant(self, assistant_id: str, cfg: dict[str, Any]) -> dict[str, Any]:
        return self._call("PATCH", f"/assistant/{assistant_id}", json=cfg)

    def list_phone_numbers(self) -> list[dict[str, Any]]:
        return list(self._call("GET", "/phone-number"))

    def create_free_number(self, assistant_id: str, *, area_code: str = "415", name: str = "SenseLab demo") -> dict[str, Any]:
        """A free Vapi US number, pointed at the assistant (inbound calls go to it directly).
        When the area code is out, Vapi's error names ones that are not; the first is taken."""
        body = {"provider": "vapi", "numberDesiredAreaCode": area_code, "assistantId": assistant_id, "name": name}
        try:
            return self._call("POST", "/phone-number", json=body)
        except RuntimeError as exc:
            m = re.search(r"Try one of ([\d, ]+)", str(exc))
            if not m:
                raise
            body["numberDesiredAreaCode"] = m.group(1).split(",")[0].strip()
            return self._call("POST", "/phone-number", json=body)

    def close(self) -> None:
        self._c.close()


def presigned_recording(private_key: str, call_id: str) -> str | None:
    """A URL for the call's stereo recording that can be fetched (the plain ``recordingUrl``
    cannot be, without Vapi's storage credentials). ``None`` until Vapi has it."""
    api = VapiApi(private_key)
    try:
        call = api._call("GET", f"/call/{call_id}")  # noqa: SLF001
    finally:
        api.close()
    art = (call or {}).get("artifact") or {}
    return art.get("presignedStereoUrl") or art.get("presignedMonoUrl") or None


def upsert_assistant(private_key: str, *, business: Business | None = None, server_url: str, secret: str | None = None,
                     voice_id: str | None = None, voice_provider: str | None = None) -> dict[str, Any]:
    """Create the assistant, or update the one with this name. Returns the assistant."""
    business = business or Business.load()
    voice_id = voice_id or os.environ.get("VAPI_VOICE_ID") or "matilda"
    voice_provider = voice_provider or os.environ.get("VAPI_VOICE_PROVIDER") or "11labs"
    cfg = assistant_config(business, server_url=server_url, secret=secret, voice_id=voice_id, voice_provider=voice_provider)
    api = VapiApi(private_key)
    try:
        existing = next((a for a in api.list_assistants() if a.get("name") == ASSISTANT_NAME), None)
        if existing:
            # A fixed goodbye from an earlier version would be spoken on top of the model's
            # own; a PATCH only merges, so clear it explicitly.
            return api.update_assistant(existing["id"], {**cfg, "endCallMessage": None})
        return api.create_assistant(cfg)
    finally:
        api.close()


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    from core.settings import REPO_ROOT

    load_dotenv(REPO_ROOT / ".env")
    p = argparse.ArgumentParser(description="create or update the Vapi assistant for the voice demo")
    p.add_argument("--server-url", help="the public URL of this server's /vapi/webhook (default: $VOICE_PUBLIC_URL + /vapi/webhook)")
    p.add_argument("--voice", help="voice id (default: $VAPI_VOICE_ID or 'matilda')")
    p.add_argument("--voice-provider", help="voice provider (default: $VAPI_VOICE_PROVIDER or '11labs')")
    p.add_argument("--number", action="store_true", help="also get a free US phone number for the assistant, if it has none")
    p.add_argument("--area-code", default="415")
    p.add_argument("--business", help="the business file (default: $VOICE_BUSINESS or voice/business.toml)")
    p.add_argument("--show-prompt", action="store_true", help="print the system prompt and exit")
    args = p.parse_args(argv)
    business = Business.load(args.business)
    if args.show_prompt:
        print(system_prompt(business))
        return 0
    key = os.environ.get("VAPI_PRIVATE_KEY")
    if not key:
        raise SystemExit("VAPI_PRIVATE_KEY is not set (put it in .env)")
    server_url = args.server_url or (os.environ.get("VOICE_PUBLIC_URL", "").rstrip("/") + "/vapi/webhook")
    if server_url == "/vapi/webhook":
        raise SystemExit("pass --server-url or set VOICE_PUBLIC_URL")
    a = upsert_assistant(key, business=business, server_url=server_url, secret=os.environ.get("VAPI_WEBHOOK_SECRET") or None,
                         voice_id=args.voice, voice_provider=args.voice_provider)
    out: dict[str, Any] = {"assistant_id": a["id"], "name": a.get("name"), "business": business.name,
                           "server_url": server_url, "voice": a.get("voice"), "model": (a.get("model") or {}).get("model")}
    if args.number:
        api = VapiApi(key)
        try:
            numbers = [n for n in api.list_phone_numbers() if n.get("assistantId") == a["id"]]
            if not numbers:
                numbers = [api.create_free_number(a["id"], area_code=args.area_code)]
            out["phone_numbers"] = [n.get("number") for n in numbers]
        finally:
            api.close()
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
