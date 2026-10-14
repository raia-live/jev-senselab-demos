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
(earlier contacts, what was tried, what came of it). It is read before the call connects and
goes into the prompt, so John knows who is on the line from the first word, and it is written
to when the call's outcome is known. The first line is built from that file: a known caller is
greeted by name and told their history is in hand ("I can see you've been in touch three times
this past month, most recently five days ago, so no need to start from the beginning. What's
going on today?"), and presumes nothing about why they are calling. Ask him "what do you know
about me?" or "what did we talk about last time?" and he tells you, in order, from the file. When
the decision comes he says why, from your own history and from what has and has not worked
for people in your situation, and "why would it work this time?" gets what actually happened
the last three times. Call twice as the same person and the second call starts from the first.

Vapi runs the call: the phone or browser audio and the speech. By default the speech is
OpenAI's realtime model (`gpt-realtime-2`), which listens and talks directly, with no
transcription step in between; that is what makes it sound like a person on a line rather
than text read aloud ("The voice", below). This folder is the server Vapi talks to, the page
you call from, and a simulator for when you have no microphone.

Nothing the caller hears is scripted. John is given a character and a few rules (look the
caller up before answering the substance, ask Jev before promising anything, never mention
the machinery), and talks. Jev's decision comes back to the model as what happens next and
what to get across, and John puts it in his own words. The business itself (Northwind Cloud,
its intents, actions, policy and demo callers) is one file, `voice/business.toml`; swap it
for yours and the same agent answers for your company ("Make it your business", below).

## What you need

- The two keys from the main README (`TYPESAFE_API_KEY`, `AMFS_API_KEY`).
- A [Vapi](https://vapi.ai) account. Its public key goes in `VAPI_PUBLIC_KEY` (the browser uses
  it to start calls), its private key in `VAPI_PRIVATE_KEY` (the server uses it to create the
  assistant). The default voice stack needs nothing else; for the ElevenLabs stack, connect
  ElevenLabs to your Vapi account.
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

Then pick Bruno (consumer, third call about the same $49) and have a conversation, not a line.
John speaks first, and he already has the file: he greets Bruno by name, says he can see the
three calls this month, and asks what is going on today. Try "what do you know about me?"
or "what did we talk about last time?" and he tells you: the duplicate charge, the credit
finance reversed, the callback that never came. Then tell him why you are calling today
("Details" has a script with five turns; the first is just "I'm calling about my bill.
Again."). Push back when the decision comes ("why would it work this time?") and John answers
from the history. The written policy says a dispute under $200 is resolved in the bot; Jev
alone follows it, or hands the frustrated caller to a human. SenseLab's record says both of
those lost this caller before and a specialist callback kept him. Jev with SenseLab books the
callback. When you hang up, the world confirms it, the page tells you what Jev alone would
have cost, and a line about this call is added to Bruno's history.

Then flip "retention offer live" and call as Priya. The policy still says transfer. The
first call transfers, the queue loses the caller, the review says so, and the next call keeps
them in the bot.

The five callers come with a seeded history (`[[personas]]` in `voice/business.toml`), because
a first demo call needs something to know; those entries are marked as seeded. Everything after
them is written by the agent from what happened on a call. The history is kept in SenseLab at
`<entity>/customers/<id>`, one entry per contact.

For a clean take, start on an empty memory: `VOICE_ENTITY_PATH=jev-demos/call_handling/take2`
(the caller histories live under the entity, so they are re-seeded there too), then run the
simulator for thirty calls before you record.

## What you see on the page

The page is a live graph of the call: the caller, John (Vapi), Jev and SenseLab as boxes,
with the traffic between them animated as it happens: the call connecting, SenseLab opening
a run per possible situation, John asking Jev, Jev's answer going back, John telling the
caller, the call ending, the outcome arriving and being sealed. One line under the graph
narrates each step. Below it a timeline shows the same call second by second, one lane per
party: who was speaking when, the moment John looked the caller up, the moment John asked
Jev, Jev's 200 ms, SenseLab opening the runs and sealing the outcome. "Details" opens a side
panel with the caller's script, what SenseLab knows about this caller (the text John got
back), Jev's read of the four actions, the exact text SenseLab put in front of Jev, the
lessons drawn so far and the record by situation. Earlier calls are chips along the bottom; clicking one replays its graph and
timeline, with the recording when there is one.

Every call is recorded by Vapi (stereo: you on one channel, John on the other). The server
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

To report outcomes yourself instead (your own CRM, a survey, a person), set
`outcomes.mode = "reported"` in the business file. Then nothing is assumed when a call ends:
the decision waits, "Details" shows an "It worked / It did not" control for it, and
`POST /api/outcome` with `{"decision_id", "severity", "note"}` does the same from code. The
lesson is written the moment the outcome lands, however long that takes. In simulated mode a
single call can still be held back with `POST /api/calls/{id}/end` and `{"auto_review": false}`.

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

gets a free Vapi US number and routes it through this server: when the number rings, Vapi
asks the server who should answer, and the server replies with the assistant and the caller's
account facts and history, so a phone call starts with as much known as a browser call. A
phone call carries no account facts of its own (there is no caller-id lookup behind the
number in this demo), so pick the caller on the page and press "use this caller"; the page
reads their history and the next call to the number within ten minutes is that person.
Without that, a phone caller is a consumer with no prior contacts and nothing on record. The
server re-routes any number it owns on every start, so a new tunnel URL needs nothing more
than a restart.

Phone audio is narrowband (8 kHz; the same line quality as any phone call), so the voice is
at its best in a browser call, where the audio is wideband. If you are recording a demo,
record a browser call.

## How the conversation is shaped

The system prompt (`voice/assistant.py`, `system_prompt`) is a character and the shape of a
call, not a flowchart: who John is on the phone, how he reacts before he acts, what he never
says, what he knows about the person calling (their account facts and their history, filled
in per call), and six beats a call goes through. The opening is the one line the server
writes (`voice/customers.py`, `opening`), from the file; left to the realtime model it
greeted a known caller as a stranger and asked for their name. From the caller's first words
the model is on its own. It listens for why they are calling today, and the prompt is blunt that the file is the past
and not today's request: a question about history is answered from the file and is never a
reason to call a tool. He checks he has it and gets the one detail he needs. He decides,
through Jev, only once the caller has said on this call what they need; the tool requires
the caller's own words. He delivers the decision with the reason. And he closes slowly:
repeats back what was arranged, asks if there is anything else, waits, and only hangs up
after the caller says they are done. Those last two came out of a real call where the agent
decided from the file before the caller had stated a problem and hung up the moment it had a
phone number.

The only scripted audio is a "still with you" if a lookup takes more than a couple of
seconds. The rest is Vapi's conversational settings: backchannels while the caller is
talking, a short wait plus smart endpointing so a pause mid-sentence is not taken for the
end of a turn, two words from the caller interrupt John but a cough does not, and a word if
the line goes quiet. The model runs at temperature 0.7 so two calls do not sound alike. All
of it is in `assistant_config`.

John looks the caller up again when they turn out to be calling about something else, so
the situation record is for the right situation. Questions about their own history he
answers from the prompt, where it already is.

## The voice

Two stacks, chosen with `VOICE_STACK`:

- `realtime` (default): OpenAI's `gpt-realtime-2` hears the caller and speaks back
  directly. There is no speech-to-text step and no text-to-speech step, so it carries tone,
  pace and hesitation the way a person does, reacts mid-sentence, and does not read
  punctuation aloud. This is the same kind of model behind ChatGPT's voice mode. Voices:
  `marin`, `cedar` (the default), `alloy`, `echo`, `shimmer`; set `VAPI_VOICE_ID`.
- `pipeline`: Deepgram transcribes, GPT-4.1 writes, ElevenLabs (`eleven_turbo_v2_5`)
  speaks. More control over each stage and any ElevenLabs voice (`VAPI_VOICE_ID`, with
  `VAPI_VOICE_PROVIDER=11labs`), at the cost of a text hop in the middle. Needs ElevenLabs
  connected to your Vapi account.

Jev's decision works the same on both: a tool call from the model, answered by this server.
One thing the realtime stack needs that the pipeline does not: a tool answer that reaches
Vapi while the model's turn is still open is dropped (Vapi logs "An unexpected error
occurred." against the tool, and the model never hears it). On one real call both answers
came back in under 400 ms and both were lost; the agent went quiet and the caller hung up.
Answers that took 1.5 s and 4.6 s on an earlier call landed. So on that stack the server
holds every tool answer until 1.5 s after the request came in (`VOICE_TOOL_MIN_MS`), the
prompt tells the model a tool call is a silent turn, and the tools carry no spoken "still
with you" filler, which would open a turn of its own.

## Make it your business

Everything about Northwind is in `voice/business.toml`. Copy it and point the server at
your copy:

```
cp voice/business.toml my-business.toml
VOICE_BUSINESS=my-business.toml VOICE_PUBLIC_URL=https://<tunnel> uv run uvicorn voice.server:app --port 8787
```

The server rebuilds the Vapi assistant from the file on start (`python -m voice.assistant
--show-prompt` prints the prompt it will use). What to change:

- `[company]`: the name, what you do, the agent's name, the greeting, and a line or two of
  house style if your people talk a particular way.
- `[[intents]]`: what people call about. For each, a label in plain words and, where the
  agent needs a number before it can decide (an amount, a count), `needs` naming it and
  `ahead` with the values to open memory for while the caller is still talking. List them
  by how common they are; the first ones are ready first.
- `[[actions]]`: what the agent can do at the decision point. `jev` is what Jev decides
  between; `then` is what the agent gets across once it is chosen (in its own words, so
  write it as facts, not a line); `words` is how it reads in a caller's history.
- `[[policy]]`: your written rules, one per line. Jev follows them unless memory says a
  rule has stopped working for a situation.
- `[situation]`: how calls are grouped for learning. Intent and tier by default; add a split
  where a number changes what works (the demo splits billing disputes by repeat callers and
  by amount).
- `[outcomes]`: `mode = "reported"` for a real business (see Outcomes), with the dollar cost
  of a miss by how bad it was. Keep `simulated` only if you also write a scenario module
  for the batch harness with the same actions, as `demos/call_handling/scenario.py` does.
- `[memory]`: the agent id and entity path in SenseLab. A new entity path is an empty memory.
- `[[personas]]`: callers to try, each with the account facts, a seeded history and a
  script. Delete the demo's, or write your own.

What stays the same: the account facts a call carries (name, tier, prior contacts in 30
days, tenure), which come from the page or from `POST /api/profiles`; wire your own lookup
into `caller_from_call` in `voice/server.py` if you have one. The tiers are yours to name.

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
voice/business.toml  the business: company, intents, actions, policy, situations, outcomes, demo callers
voice/business.py    reads it; builds the scenario Jev and SenseLab see, names situations, opens runs ahead
voice/agent.py       Jev + SenseLab per call: open ahead, recall the caller, decide, seal with the outcome, repair loop
voice/customers.py   the per-caller memory: seeded history, the brief the agent gets, the note it writes after a call
voice/server.py      FastAPI: /vapi/webhook, the page, /api/*
voice/assistant.py   create/update the Vapi assistant from the business file (prompt, tools, voice stack, server URL); --number
voice/simulate.py    the scenario's callers, sent as Vapi tool calls
voice/web/index.html the page
demos/call_handling/ the demo's scenario: the batch harness runs it, and it reviews the live agent's decisions
```
