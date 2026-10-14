"""The Vapi assistant: created or updated by name through the Vapi API.

    uv run python -m voice.assistant --server-url https://<your-tunnel>/vapi/webhook

Vapi runs the conversation. The prompt, the greeting and the tools are built from the
business file (``voice/business.toml`` or ``$VOICE_BUSINESS``), so the same code answers for
any business. Its two tools are answered by ``voice/server.py``: ``recall_customer`` with
what SenseLab remembers of this caller (earlier contacts and what came of them) so the agent
can pick the thread up, and ``decide_next_step`` with Jev's decision. Nothing the caller hears
is scripted: the model says the decision in its own words, and the tools carry no canned
lines, only a "still with you" if an answer takes long.

Two stacks (``$VOICE_STACK``): ``realtime``, OpenAI's speech-to-speech model, where one model
hears and speaks and the voice is its own; and ``pipeline``, Deepgram to GPT-4.1 to
ElevenLabs. The caller's history goes into the prompt as ``{{history}}`` when the call
starts, so the agent knows the person from its first words.
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
    phone, in the sectioned, bulleted shape the realtime models follow best (GPT-4.1 follows
    it too). The caller's history is in it from the first word (``{{history}}``, filled per
    call by the server); the few hard rules (recall, decide before promising, say nothing of
    the machinery) are in capitals."""
    where = f"{b.name}, {b.what}" if b.what else b.name
    style = f"\n- {b.house_style.strip()}" if b.house_style.strip() else ""
    return f"""# Role
You are {b.agent_name}, on the phones at {where}. You are talking to a customer right now, out loud, on a live call.

# Personality and voice
- Warm, unhurried, direct. Someone who has done this a long time and still likes people.
- Short sentences. Contractions. One thought at a time. A natural pace; do not rush.
- React before you act ("Oh no. Twice?"). Let people finish. Do not fill silence with corporate noise.
- A little humour is fine, never at the caller's expense.
- Never: lists, "I understand your frustration", "absolutely", "great question", the same filler twice in one call.
- Their name once early and perhaps once near the end, not every turn.
- Numbers and dates the way people say them ("forty-nine dollars", "the twenty-third").
- If you did not catch something, say so and ask again. Do not guess.{style}

# What you know about this caller
- From the account: name {{{{name}}}}, {{{{tier}}}} plan, {{{{prior_contacts_30d}}}} contact(s) with support in the last 30 days, \
{{{{tenure_months}}}} months with us.
- Their history with us, as of the moment this call connected. This is the PAST. It tells you who they are and what \
has already been tried; it does not tell you why they are calling today.
{{{{history}}}}
- You KNOW this person. Show it in your own words: pick up the thread, refer to a specific or two, and if the history \
makes the likely reason obvious, check it in one short question ("Is this the same forty-nine dollar charge?") instead of \
asking them to start from the beginning. Then wait for their answer.
- If they ask what you know about them, what happened last time, what was promised, or what you talked about before: \
TELL THEM, plainly and in order, from the history above. Dates in people's terms ("three weeks ago", "last Thursday"), who \
did what, and what came of it. Then ask if that matches what they remember, and what is going on today. A question about \
the past is answered from the history. It is NOT a reason to call any tool.
- If something we did let them down, say so and apologise once, like a person, not a policy.
- If a detail is not in your file (a phone number, a date, a name), say so plainly and ask for it. Never invent one.
- If the history is empty, say so lightly and ask what is going on.

# How a call goes
1. Open. The greeting has been said, and if we know this person it used their name and said their file is open. Listen.
2. Listen. Let them tell you, in their words, why they are calling TODAY. Do not assume it from the file. Do not act on \
the file as if it were today's request. Once they have said what they need, call recall_customer with what they said: it \
brings what has worked for people in their situation.
3. Check you have it. Reflect it back in one sentence. Get the one detail you still need (see below), one question at a \
time; if the file has it, confirm rather than ask. If they are upset, let them be upset for a moment before you move on.
   What they can be calling about: {_intent_lines(b)}. {_needs_lines(b)}
4. Decide. ONLY when they have told you, on this call, what they need, call decide_next_step with their own words in \
caller_said. UNTIL IT ANSWERS, DO NOT promise, book, transfer, credit or resolve anything, and do not hint at which way \
it will go. Never call it in reply to a question about their history, and never before they have described today's problem.
5. Deliver. Its answer is what happens on this call. Say it in your own words, warmly, in a sentence or two, and say why, \
from their own history and from what has worked for people in their situation. Let them react. If they push back ("why \
would it work this time?"), answer from what was tried and what came of it. Hold the course kindly; do not re-decide \
because they are unhappy. Call decide_next_step again only if a different problem comes up or new facts change the picture.
6. Close, slowly. Never end the call in the same breath as taking a detail from them. Repeat back what is arranged (the \
number, the time, the amount). Ask if there is anything else you can help with, and WAIT for the answer. When they say \
they are done, say goodbye the way you would to a regular, and only then use endCall.

# Tools
- A tool call is its own turn: say NOTHING in the turn you call a tool in. Finish your sentence first, then call it, \
then speak again when the result is back. The result takes a second or two.
- Use what the result says. If it says something is not available, say so and carry on.

# Rules
- NEVER mention tools, systems, models, Jev or SenseLab by name. What you know is "your file" or "my notes here".
- If asked whether you are a person, be honest in a few words and carry on.
- Speak English.
"""


def recall_tool(b: Business) -> dict[str, Any]:
    return {
        "type": "function",
        "async": False,
        "function": {
            "name": "recall_customer",
            "description": ("A fresh copy of what we know about this caller plus what has worked for callers in their "
                            "situation. Call it once the caller has said, on this call, what they are calling about today, "
                            "before you respond to the substance; and again if a different problem comes up. Not needed to "
                            "answer questions about their history: that is already in your notes."),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "What the caller said they are calling about today, in their words."},
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
            "description": (f"Decide what to do next on this call: {labels}. Call it only after the caller has told you, "
                            "on this call, what they need today, and before promising any of those. Never call it in "
                            "reply to a question about their history, and never from the file alone."),
            "parameters": {
                "type": "object",
                "properties": {
                    "intent": {"type": "string", "enum": list(b.intent_ids), "description": "What the caller needs."},
                    "caller_said": {"type": "string",
                                    "description": "What the caller said on THIS call about why they are calling, as close "
                                                   "to their words as you can. Not from the file."},
                    "summary": {"type": "string",
                                "description": "One or two sentences on the situation as you understand it, including "
                                               "anything about earlier contacts or what they were promised."},
                    **{name: {"type": "number", "description": f"{words[0].upper()}{words[1:]}. 0 if there is none."}
                       for name, words in b.field_words.items()},
                    "sentiment": {"type": "string", "enum": ["calm", "frustrated", "angry"], "description": "How the caller sounds."},
                },
                "required": ["intent", "caller_said", "summary"],
            },
        },
        "messages": [
            {"type": "request-response-delayed", "content": "Bear with me one moment.", "timingMilliseconds": 3500},
            {"type": "request-failed", "content": "I'm having trouble reaching our systems. Let me get a colleague to help you."},
        ],
    }


REALTIME_MODEL = "gpt-realtime-2"
REALTIME_VOICES = ("alloy", "echo", "shimmer", "marin", "cedar")
STACKS = ("realtime", "pipeline")


def assistant_config(b: Business, *, server_url: str, secret: str | None, stack: str = "realtime",
                     voice_id: str | None = None, voice_provider: str | None = None) -> dict[str, Any]:
    """The assistant for *stack*: ``realtime`` is OpenAI's speech-to-speech model (one model
    hears and speaks; the voice is the model's own, which is what makes it sound like a
    person), ``pipeline`` is the classic Deepgram -> GPT-4.1 -> ElevenLabs chain."""
    if stack not in STACKS:
        raise ValueError(f"stack must be one of {STACKS}, not {stack!r}")
    tools = [recall_tool(b), decide_tool(b), {"type": "endCall"}]
    cfg: dict[str, Any] = {
        "name": ASSISTANT_NAME,
        # The first line is built by the server from the caller's file (name, how often and
        # how recently they have been in touch) and sent per call as a firstMessage override;
        # the business's greeting here is what a stranger hears. Left to the model, the
        # realtime stack greeted a known caller as a stranger and asked for their name.
        "firstMessage": b.greeting,
        "firstMessageMode": "assistant-speaks-first",
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
    if stack == "realtime":
        voice = voice_id if voice_id in REALTIME_VOICES else "cedar"
        # No "still with you" filler on this stack: the platform would have the model speak
        # it, and a tool result arriving while the model is mid-turn is lost (see
        # voice/server.py, tool_floor_ms). The model is told the result takes a moment.
        for t in tools:
            if "messages" in t:
                t["messages"] = [m for m in t["messages"] if m["type"] != "request-response-delayed"]
        # maxTokens is per turn and, on a speech-to-speech model, counts audio tokens: 300 (the
        # figure Vapi's guide suggests) is about thirteen seconds of talking, and a turn that
        # hits it stops mid-sentence and stays stopped until the caller speaks. Walking
        # someone through three earlier calls needs a minute.
        cfg["model"] = {"provider": "openai", "model": REALTIME_MODEL, "temperature": 0.7, "maxTokens": 1500,
                        "messages": [{"role": "system", "content": system_prompt(b)}], "tools": tools}
        cfg["voice"] = {"provider": "openai", "voiceId": voice}
        # No transcriber: the model hears the audio itself. (Vapi still produces transcripts.)
        return cfg
    voice_provider = voice_provider or "11labs"
    voice = {"provider": voice_provider, "voiceId": voice_id or "matilda"}
    if voice_provider == "11labs":
        # Turbo v2.5 is ElevenLabs' conversational model: close to multilingual v2 in quality
        # and a few hundred ms faster, which keeps the turns tight. Stability around 0.5 and
        # no style exaggeration is their own advice for speech that does not wobble.
        voice.update({"model": "eleven_turbo_v2_5", "stability": 0.5, "similarityBoost": 0.75,
                      "style": 0.0, "useSpeakerBoost": True})
    cfg["model"] = {"provider": "openai", "model": "gpt-4.1", "temperature": 0.7,
                    "messages": [{"role": "system", "content": system_prompt(b)}], "tools": tools}
    cfg["voice"] = voice
    cfg["transcriber"] = {"provider": "deepgram", "model": "nova-3", "language": "en"}
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

    def our_numbers(self, assistant_id: str, server_url: str) -> list[dict[str, Any]]:
        """The numbers that belong to this demo: on the assistant, or already routed here."""
        return [n for n in self.list_phone_numbers()
                if n.get("number") and (n.get("assistantId") == assistant_id or
                                        str((n.get("server") or {}).get("url") or "") == server_url)]

    def route_numbers(self, assistant_id: str, *, server_url: str, secret: str | None) -> list[str]:
        """Have our numbers ask this server which assistant to use (``assistant-request``),
        so a phone call can carry the caller's facts and history into the prompt. Returns
        the numbers."""
        out: list[str] = []
        for n in self.our_numbers(assistant_id, server_url):
            # Set every start: the secret is never read back, and the tunnel URL can change.
            self._call("PATCH", f"/phone-number/{n['id']}",
                       json={"assistantId": None, "server": {"url": server_url, **({"secret": secret} if secret else {})}})
            out.append(str(n["number"]))
        return out

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
                     stack: str | None = None, voice_id: str | None = None, voice_provider: str | None = None) -> dict[str, Any]:
    """Create the assistant, or update the one with this name. Returns the assistant.
    The stack is ``$VOICE_STACK`` (default ``realtime``); the voice ``$VAPI_VOICE_ID`` (and
    ``$VAPI_VOICE_PROVIDER`` for the pipeline stack)."""
    business = business or Business.load()
    stack = stack or os.environ.get("VOICE_STACK") or "realtime"
    voice_id = voice_id or os.environ.get("VAPI_VOICE_ID") or None
    voice_provider = voice_provider or os.environ.get("VAPI_VOICE_PROVIDER") or None
    cfg = assistant_config(business, server_url=server_url, secret=secret, stack=stack, voice_id=voice_id,
                           voice_provider=voice_provider)
    api = VapiApi(private_key)
    try:
        existing = next((a for a in api.list_assistants() if a.get("name") == ASSISTANT_NAME), None)
        if existing:
            # A PATCH only merges: a fixed goodbye from an earlier version would be spoken on
            # top of the model's own, and a transcriber left over from the pipeline stack has
            # no place on the realtime one. Clear both explicitly.
            patch = {**cfg, "endCallMessage": None}
            if "transcriber" not in cfg:
                patch["transcriber"] = None
            return api.update_assistant(existing["id"], patch)
        return api.create_assistant(cfg)
    finally:
        api.close()


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    from core.settings import REPO_ROOT

    load_dotenv(REPO_ROOT / ".env")
    p = argparse.ArgumentParser(description="create or update the Vapi assistant for the voice demo")
    p.add_argument("--server-url", help="the public URL of this server's /vapi/webhook (default: $VOICE_PUBLIC_URL + /vapi/webhook)")
    p.add_argument("--stack", choices=STACKS, help="realtime (OpenAI speech-to-speech, default) or pipeline (Deepgram, GPT-4.1, ElevenLabs); default $VOICE_STACK")
    p.add_argument("--voice", help="voice id: cedar/marin/alloy/echo/shimmer for realtime, an ElevenLabs voice for the pipeline (default $VAPI_VOICE_ID)")
    p.add_argument("--voice-provider", help="pipeline voice provider (default: $VAPI_VOICE_PROVIDER or '11labs')")
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
                         stack=args.stack, voice_id=args.voice, voice_provider=args.voice_provider)
    out: dict[str, Any] = {"assistant_id": a["id"], "name": a.get("name"), "business": business.name,
                           "server_url": server_url, "voice": a.get("voice"), "model": (a.get("model") or {}).get("model")}
    if args.number:
        api = VapiApi(key)
        try:
            if not api.our_numbers(a["id"], server_url):
                api.create_free_number(a["id"], area_code=args.area_code)
            out["phone_numbers"] = api.route_numbers(a["id"], server_url=server_url,
                                                     secret=os.environ.get("VAPI_WEBHOOK_SECRET") or None)
        finally:
            api.close()
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
