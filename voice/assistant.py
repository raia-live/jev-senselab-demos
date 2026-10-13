"""The Vapi assistant: created or updated by name through the Vapi API.

    uv run python -m voice.assistant --server-url https://<your-tunnel>/vapi/webhook

Vapi runs the conversation. Its two tools are answered by ``voice/server.py``:
``recall_customer`` with what SenseLab remembers of this caller (earlier contacts and what came
of them) so Riley can pick the thread up, and ``decide_next_step`` with Jev's decision. The
system prompt makes the model recall first, talk it through, call decide before it resolves,
transfers, books or credits anything, and follow what comes back.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

import httpx

VAPI_URL = "https://api.vapi.ai"
ASSISTANT_NAME = "Jev support agent (SenseLab demo)"
COMPANY = "Northwind Cloud"

SYSTEM_PROMPT = f"""You are Riley, a support agent at {COMPANY}, a small software company, on a live phone call.

Account facts for this caller, from the account lookup: name {{{{name}}}}, plan tier {{{{tier}}}}, \
{{{{prior_contacts_30d}}}} contact(s) with support in the last 30 days, {{{{tenure_months}}}} months as a customer.

How you sound: like a person on the phone, not a script. Short sentences. Contractions. One idea \
at a time. React to what you hear before you act on it ("Oh no, twice? Sorry about that."). Use the \
caller's name once, early. No lists, no corporate phrases, no "I understand your frustration". \
Never say you are an AI unless asked.

How to run the call:
1. The greeting has been said. As soon as the caller says what they are calling about, even vaguely \
("my bill, again"), call recall_customer with what you heard. Say a few words while it runs ("One sec, \
let me pull up your account."). Do not ask them to explain the whole thing first.
2. recall_customer tells you what we know about this person: earlier contacts, what was tried, what \
came of it, and what has worked for callers in their situation. Use it. If the history makes the \
problem obvious, confirm it in one short question ("Is this the same forty-nine dollar charge?") \
instead of asking them to start from the beginning. Mention one or two specifics in your own words; \
if something we did failed them, say so plainly and apologise once. If there is no history, say so \
lightly and ask what is going on.
3. Talk it through. Answer their questions from what you know. Ask one short question at a time. \
For a billing dispute you need the amount in dollars; if the history already has it, confirm it \
rather than asking.
4. When you know the intent (billing dispute, cancellation, outage, upgrade or plan change, how-to) \
and, for a billing dispute, the amount, call decide_next_step. Do not promise, transfer, book, credit \
or resolve anything before it has answered. Its line is spoken to the caller for you; do not repeat it.
5. Carry on as the result says. If the caller pushes back or asks why this will work, answer in plain \
words from their history: what was tried, what came of it, why this is different. Hold the course \
kindly. Call decide_next_step again only if they bring up a different problem or new facts that change \
the picture.
6. Before you end: ask if there is anything else. Then say goodbye and use endCall.
7. Never mention tools, policies, systems, memory, Jev or SenseLab. No lists, no headings.
"""

FIRST_MESSAGE = f"Hi, thanks for calling {COMPANY}, this is Riley. What can I do for you today?"

RECALL_TOOL: dict[str, Any] = {
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
                "intent": {"type": "string", "enum": ["billing_dispute", "cancellation", "outage", "upgrade", "how_to"],
                           "description": "Your best guess at the intent, if you have one."},
                "amount_usd": {"type": "number", "description": "The amount in dollars, if they named one."},
            },
            "required": ["topic"],
        },
    },
    "messages": [
        {"type": "request-start", "content": "One sec, let me pull up your account.", "blocking": False},
        {"type": "request-failed", "content": "I can't get to your account history right now, so tell me what's going on."},
    ],
}

DECIDE_TOOL: dict[str, Any] = {
    "type": "function",
    "async": False,
    "function": {
        "name": "decide_next_step",
        "description": ("Decide what to do next on this call once the caller's need is clear: handle it here, "
                        "transfer to a human, book a specialist callback, or apply a credit. Call it before "
                        "promising any of those."),
        "parameters": {
            "type": "object",
            "properties": {
                "intent": {"type": "string", "enum": ["billing_dispute", "cancellation", "outage", "upgrade", "how_to"],
                           "description": "What the caller needs."},
                "summary": {"type": "string",
                            "description": "One or two sentences on what the caller said, in their words where possible, "
                                           "including anything about earlier contacts or what they were promised."},
                "amount_usd": {"type": "number", "description": "The disputed amount in dollars, for a billing dispute. 0 otherwise."},
                "sentiment": {"type": "string", "enum": ["calm", "frustrated", "angry"], "description": "How the caller sounds."},
            },
            "required": ["intent", "summary"],
        },
    },
    "messages": [
        {"type": "request-start", "content": "Let me check what we can do here.", "blocking": False},
        {"type": "request-failed", "content": "I'm having trouble reaching our systems. Let me get a colleague to help you."},
        {"type": "request-response-delayed", "content": "Bear with me one moment.", "timingMilliseconds": 4000},
    ],
}


def assistant_config(*, server_url: str, secret: str | None, voice_id: str, voice_provider: str) -> dict[str, Any]:
    voice: dict[str, Any] = {"provider": voice_provider, "voiceId": voice_id}
    if voice_provider == "11labs":
        # Multilingual v2 is ElevenLabs' most natural model; it costs a few hundred ms over
        # turbo, which the prefetch leaves room for. Lower stability lets the delivery vary;
        # style adds the small inflections that make it sound like a person.
        voice.update({"model": "eleven_multilingual_v2", "stability": 0.4, "similarityBoost": 0.8,
                      "style": 0.35, "useSpeakerBoost": True})
    cfg: dict[str, Any] = {
        "name": ASSISTANT_NAME,
        "firstMessage": FIRST_MESSAGE,
        "firstMessageMode": "assistant-speaks-first",
        "model": {
            "provider": "openai", "model": "gpt-4.1", "temperature": 0.3,
            "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
            "tools": [RECALL_TOOL, DECIDE_TOOL, {"type": "endCall"}],
        },
        "voice": voice,
        "transcriber": {"provider": "deepgram", "model": "nova-3", "language": "en"},
        "server": {"url": server_url, "timeoutSeconds": 20, **({"secret": secret} if secret else {})},
        "serverMessages": ["status-update", "speech-update", "transcript", "tool-calls", "end-of-call-report"],
        "clientMessages": ["transcript", "tool-calls", "tool-calls-result", "status-update", "speech-update"],
        # Every call is recorded (stereo: caller one channel, Riley the other); the server
        # keeps the file next to the decision so the demo has the audio.
        "artifactPlan": {"recordingEnabled": True, "recordingFormat": "wav;l16"},
        "endCallMessage": f"Thanks for calling {COMPANY}. Goodbye.",
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


def upsert_assistant(private_key: str, *, server_url: str, secret: str | None = None,
                     voice_id: str | None = None, voice_provider: str | None = None) -> dict[str, Any]:
    """Create the assistant, or update the one with this name. Returns the assistant."""
    voice_id = voice_id or os.environ.get("VAPI_VOICE_ID") or "matilda"
    voice_provider = voice_provider or os.environ.get("VAPI_VOICE_PROVIDER") or "11labs"
    cfg = assistant_config(server_url=server_url, secret=secret, voice_id=voice_id, voice_provider=voice_provider)
    api = VapiApi(private_key)
    try:
        existing = next((a for a in api.list_assistants() if a.get("name") == ASSISTANT_NAME), None)
        if existing:
            return api.update_assistant(existing["id"], cfg)
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
    args = p.parse_args(argv)
    key = os.environ.get("VAPI_PRIVATE_KEY")
    if not key:
        raise SystemExit("VAPI_PRIVATE_KEY is not set (put it in .env)")
    server_url = args.server_url or (os.environ.get("VOICE_PUBLIC_URL", "").rstrip("/") + "/vapi/webhook")
    if server_url == "/vapi/webhook":
        raise SystemExit("pass --server-url or set VOICE_PUBLIC_URL")
    a = upsert_assistant(key, server_url=server_url, secret=os.environ.get("VAPI_WEBHOOK_SECRET") or None,
                         voice_id=args.voice, voice_provider=args.voice_provider)
    out: dict[str, Any] = {"assistant_id": a["id"], "name": a.get("name"), "server_url": server_url,
                           "voice": a.get("voice"), "model": (a.get("model") or {}).get("model")}
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
