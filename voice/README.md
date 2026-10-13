# A support line that learns from its calls

Call a support bot from your browser or your phone. At the point in the call where it has to
choose what to do (sort it out itself, hand you to a human, book a specialist callback, apply a
credit), it asks Jev. Jev gets the written policy and what SenseLab learned from earlier calls
in this situation. When the call ends, what came of the decision goes back to SenseLab, and the
next caller in the same situation gets a bot that knows. That loop, outcome to lesson to the
next decision, is the recursive self-improvement (RSI) the page is built around.

Every call is decided twice: Jev alone on the written policy, and Jev with SenseLab's guidance.
The caller gets the second. The page shows both, and when the outcome comes in it shows what
the first would have cost. That is the comparison the demo exists for.

SenseLab also remembers the caller. Each person who rings has their own history in memory
(earlier contacts, what was tried, what came of it), read at the start of the call and written
to when the call's outcome is known. So Riley can say "is this the same forty-nine dollar
charge?" instead of asking someone to explain a problem for the fourth time, and can answer
"why would it work this time?" with what actually happened the last three times. Call twice as
the same person and the second call starts from the first.

Vapi runs the call: the phone or browser audio, speech to text, the conversational model (GPT-4.1),
text to speech (ElevenLabs if you connected it, Vapi's voices otherwise). This folder is the
server Vapi talks to, the page you call from, and a simulator for when you have no microphone.

## What you need

- The two keys from the main README (`TYPESAFE_API_KEY`, `AMFS_API_KEY`).
- A [Vapi](https://vapi.ai) account. Its public key goes in `VAPI_PUBLIC_KEY` (the browser uses
  it to start calls), its private key in `VAPI_PRIVATE_KEY` (the server uses it to create the
  assistant). Optionally connect ElevenLabs to your Vapi account for the voice.
- A tunnel so Vapi can reach your laptop: `ngrok` works (`brew install ngrok`).

```
uv sync --extra voice
```

## Run it

```
ngrok http 8787                               # terminal 1: copy the https URL it prints
VOICE_PUBLIC_URL=https://<that-url> uv run uvicorn voice.server:app --port 8787   # terminal 2
open http://127.0.0.1:8787
```

On startup the server seeds the written policy into memory (if it is not there), creates or
updates the Vapi assistant "Jev support agent (SenseLab demo)" pointed at your tunnel, and
starts the repair loop watcher for the agent. Then open the page, pick who is calling, press
Call, and talk.

Give the line some history first. A support line that has taken no calls has nothing to have
learned, so before a demo send it the scenario's callers:

```
uv run python -m voice.simulate --calls 30 --no-change
```

Thirty calls take about five minutes and leave SenseLab with a record for every situation,
including the one the written policy gets wrong. You can watch them land on the page.

Then pick Dana (consumer, third call about the same $49) and have a conversation, not a line.
"Details" has a script with five turns; the first is just "Hi. I'm calling about my bill.
Again." Riley looks Dana up, finds the duplicate charge, the credit finance reversed and the
callback that never came, and asks whether this is the same charge rather than making her
explain. Push back when the decision comes ("why would it work this time?") and Riley answers
from the history. The written policy says a dispute under $200 is resolved in the bot; Jev
alone follows it, or hands the frustrated caller to a human. SenseLab's record says both of
those lost this caller before and a specialist callback kept her. Jev with SenseLab books the
callback. When you hang up, the world confirms it, the page tells you what Jev alone would
have cost, and a line about this call is added to Dana's history.

Then flip "retention offer live" and call as Priya. The policy still says transfer. The
first call transfers, the queue loses the caller, the review says so, and the next call keeps
them in the bot.

The five callers come with a seeded history (`voice/customers.py`), because a first demo call
needs something to know; those entries are marked as seeded. Everything after them is written
by the agent from what happened on a call. The history is kept in SenseLab at
`<entity>/customers/<id>`, one entry per contact.

## What you see on the page

The page is a live graph of the call: the caller, Riley (Vapi), Jev and SenseLab as boxes,
with the traffic between them animated as it happens: the call connecting, SenseLab opening
a run per possible situation, Riley asking Jev, Jev's answer going back, the line being
spoken, the call ending, the outcome arriving and being sealed. One line under the graph
narrates each step. Below it a timeline shows the same call second by second, one lane per
party: who was speaking when, the moment Riley looked the caller up, the moment Riley asked
Jev, Jev's 200 ms, SenseLab opening the runs and sealing the outcome. "Details" opens a side
panel with the caller's script, what SenseLab knows about this caller (the text Riley got
back), Jev's read of the four actions, the exact text SenseLab put in front of Jev, the
lessons drawn so far and the record by situation. Earlier calls are chips along the bottom; clicking one replays its graph and
timeline, with the recording when there is one.

Every call is recorded by Vapi (stereo: you on one channel, Riley on the other). The server
pulls the recording into `results/voice/<call_id>.wav` and the page plays it under the
decision, with the turn-by-turn transcript. For a demo video, screen-record the page while
you make the call, and put the stereo recording under it afterwards.

"What the agent remembers" lists the lessons and review records in memory, and "Repair loop"
shows the platform's side: traces judged, fixes proposed, canaries.

## Outcomes

A real deployment learns the outcome days later (the caller rang back, the survey came in,
the account closed). The demo cannot wait, so when a call ends the demo's world reviews the
decision the same way the batch scenario does: against the hidden right answer for that
caller, with a note in the words a review would use. The world's rules are in
`demos/call_handling/scenario.py`: repeat billing callers need a specialist callback; once
the retention offer is live, consumer cancellations are kept by the bot. Nobody told the
agent either rule. The page marks these outcomes "demo world".

To report outcomes yourself instead (your own CRM, a survey, a person), end the call with
`POST /api/calls/{id}/end` and `{"auto_review": false}`, then send
`POST /api/outcome` with `{"decision_id", "severity", "note"}` when you know. The decision
waits in `pending` until then. Vapi calls always get the automatic review.

## No microphone

The page has a text box ("No microphone? Type the call instead"): the same decision, the same
outcome, no audio. And the simulator sends the scenario's stream of callers to the running
server in the exact shape Vapi sends tool calls:

```
uv run python -m voice.simulate --calls 40
```

Forty calls take a few minutes and leave the live agent with forty outcomes behind it. A browser
call after that starts from what they taught.

## A phone number

```
uv run python -m voice.assistant --server-url https://<tunnel>/vapi/webhook --number
```

gets a free Vapi US number pointed at the assistant. Inbound calls hit the same server. A
phone call carries no account facts (there is no caller-id lookup behind the number in this
demo), so pick the caller on the page and press "use this caller"; the next call to the
number within ten minutes is that person, history and all. Without that, a phone caller is a
consumer with no prior contacts and nothing on record.

## Latency

`Run.begin` is two reads of the SenseLab API: the entity's briefing (3 to 4 s on the hosted
account, the same for every situation) and a retrieve for the task (1 to 3 s, and the hosted
account serves them roughly one at a time). A caller will not wait for that at the decision
point, so the server opens runs ahead of it: when the call connects it begins one run per
situation the caller could be in (the account tells us tier and prior contacts; the intent
is the only thing we do not know yet), sharing one briefing across them, and the tool call
takes the one that matches. The runs open in order of likelihood, two at a time: small
billing disputes and cancellations first (ready 2 to 4 s after the call connects), outages
and large disputes next, the rest last. A decision that arrives before its run has started
opens it on the spot rather than waiting its turn. The decision itself takes Jev's 200 ms.
The briefing is refreshed after every sealed outcome, which is when memory changes. A run
opened and not used is one read; nothing stays open on the platform. The timeline shows when
the situation the caller turned out to be in was ready, and when all of them were. The
review and the seal run after the call, off the webhook's path.

## Files

```
voice/agent.py       Jev + SenseLab per call: open ahead, recall the caller, decide, seal with the outcome, repair loop
voice/customers.py   the five demo callers, their seeded history, and the per-caller memory the agent writes
voice/server.py      FastAPI: /vapi/webhook, the page, /api/*
voice/assistant.py   create/update the Vapi assistant (prompt, tools, voice, server URL); --number
voice/simulate.py    the scenario's callers, sent as Vapi tool calls
voice/web/index.html the page
demos/call_handling/ the scenario both the batch harness and the live agent use
```
