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
GPT-Live (`gpt-live-1`), OpenAI's full-duplex model as Vapi runs it: a speaker that hears
and talks and keeps listening while it talks, and a reasoner it hands work to, which calls
our tools and passes the answer back while the conversation carries on. No transcription
step in between; that is what makes it sound like a person on a line and not like text read
aloud ("The voice", below). This folder is the server Vapi talks to, the page you call from,
a simulator for when you have no microphone, and a set of scripted test calls that Vapi
plays as the caller.

Nothing the caller hears is scripted. John is given a character and one rule (he never
decides what happens about a caller's problem; the backend does, and he asks it before he
promises anything), and talks. Jev's decision comes back as what happens next, what to get
across and why, and John puts it in his own words. The business itself (Northwind Cloud,
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
`<entity>/customers/<id>`, one entry per contact. Every call goes on it: a decided call is
noted after the seal, with what was done and what came of it; a call that reached no decision
(the caller asked something and hung up) is noted at hang-up, in their words. So John can
answer "what was my last call about?" even when the last call was that question.

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
account facts and history, so a phone call starts with as much known as a browser call. The
server re-routes any number it owns on every start, so a new tunnel URL needs nothing more
than a restart.

A phone call carries the caller's number and nothing else of ours, so who is calling comes
from the number (`voice/phonebook.py`). Three answers, in order:

- Your own lookup. `lookup(number)` in `voice/phonebook.py` returns None until you fill it
  in; put your CRM or billing query there and return the account facts (`name`, `tier`,
  `prior_contacts_30d`, `tenure_months`, `customer_id`). The history under that
  `customer_id` then comes from SenseLab as for any caller. This answer wins.
- What the agent has learned. The first time a number calls while a caller is registered on
  the page, or a caller nobody matched says who they are and the name matches an account,
  the number is written to SenseLab under `<entity>/phonebook` as that person (keyed by a
  hash of the number) and read back at startup. From then on the number is them: no button,
  no window.
- Nobody. John greets a stranger and asks who he is speaking with. If the caller gives a
  name that matches an account, John has their file from that moment and the number is
  remembered.

For the demo, pick the caller on the page and press "use this caller" before the first call
from a phone; the next call to the number within ten minutes is that person, and after that
the number is. A caller registered on the page always wins over the book, which is how you
switch personas from one phone.

Phone audio is narrowband (8 kHz; the same line quality as any phone call), so the voice is
at its best in a browser call, where the audio is wideband. If you are recording a demo,
record a browser call.

## How the conversation is shaped

On GPT-Live there are two prompts (`voice/assistant.py`, `speaker_prompt` and
`reasoner_prompt`). The speaker's is a character and the shape of a call, not a flowchart:
who John is on the phone, how he reacts before he acts, what he never says, what he knows
about the person calling (their account facts and their history, filled in per call), and
six beats a call goes through. It opens with the one rule, because without it the model
answered callers from their file and never asked: John never decides what happens about a
caller's problem. Every request goes to the backend before he answers it, even when the
file has notes on it; the file is what was said before, the backend says what happens now.
What he handles himself is who they are, their history, one clarifying question, pushback
on a decision already made, and small talk. The reasoner's prompt is shorter: when the
speaker hands over, work out whether the caller is done (then end the call and nothing
else), otherwise call `decide_next_step` once per problem and hand back what was decided,
what happens now, and the reason, in plain words the speaker can say. The reasoner has no
lookup of its own; questions about the account go back to the speaker's file.

The opening is the one line the server writes (`voice/customers.py`, `opening`), from the
file; left to the model it greeted a known caller as a stranger and asked for their name.
From the caller's first words the model is on its own. It listens for why they are calling
today, and the prompt is blunt that the file is the past and not today's request: a question
about history is answered from the file and is never a reason to hand over. He checks he
has it and gets the one detail he needs. He decides, through Jev, only once the caller has
said on this call what they need; the tool requires the caller's own words. He delivers the
decision with the reason, and "why would it work this time?" gets what has and has not
worked for callers in their situation, with the counts. And he closes slowly: repeats back
what was arranged, asks if there is anything else, waits, and hangs up only after the caller
says they are done. Saying goodbye does not end a GPT-Live call; the reasoner has to call
`endCall`, and a silence hook hangs up if the caller has gone.

Most of those rules came out of test calls ("Test calls", below). The agent explained a
charge it should have asked about, promised to "check" something with no tool to check
with, told a stranger their charge was "a setup fee" from a file that did not exist, kept a
call open after the caller said goodbye, ended one on "okay, that's fine", and could not
name the account manager because five test calls the same day had pushed the account facts
out of the brief. Later runs added a stranger told the charge was "an extra seat added last
month" (the step text said "say what the charge is, from the file", and there was no file;
each action now has a second text for a caller nobody matched, and the server picks it),
"what email do you have for me?" answered with "let me check" and twenty seconds of
silence (not on file means say so and ask), and a speaker that read the reasoner's
instructions out loud ("the file either explains it or it doesn't"), so the reasoner now
writes instructions, not lines, and the step text has no conditionals in it. Each has a
line in the prompt or a fix in the code, and a test.

John looks the caller up again when they turn out to be calling about something else, so
the situation record is for the right situation. Questions about their own history he
answers from the prompt, where it already is.

## The voice

Three stacks, chosen with `VOICE_STACK`:

- `live` (default): GPT-Live (`gpt-live-1`), as Vapi runs it. A speaker hears the caller
  and talks back directly, and keeps listening while it talks, so it can be interrupted
  mid-word and can carry on a conversation while a reasoner (`gpt-5.6-terra`, low reasoning
  effort) works a tool call for it. No speech-to-text step and no text-to-speech step, so it
  carries tone, pace and hesitation the way a person does. Voices: `cedar` (the default),
  `marin`, `alloy`, `echo`, `shimmer` and the newer set (`quartz`, `ripple`, `vesper`,
  `willow`, `stone`, `gleam`, `meridian`, `bossa`, `tempo`, `beacon`, `delta`, `cinder`,
  `ash`, `ballad`, `coral`, `sage`, `verse`); set `VAPI_VOICE_ID`.
- `realtime`: the earlier speech-to-speech model (`gpt-realtime-2`), one prompt, turn by
  turn. Voices: `marin`, `cedar`, `alloy`, `echo`, `shimmer`.
- `pipeline`: Deepgram transcribes, GPT-4.1 writes, ElevenLabs (`eleven_turbo_v2_5`)
  speaks. More control over each stage and any ElevenLabs voice (`VAPI_VOICE_ID`, with
  `VAPI_VOICE_PROVIDER=11labs`), at the cost of a text hop in the middle. Needs ElevenLabs
  connected to your Vapi account.

Jev's decision works the same on all three: a tool call, answered by this server. Two
things GPT-Live does differently, both found on real calls and visible in Vapi's call logs:

- Vapi sends the server nothing live but the tool calls: no status update when the call
  starts, no speech or transcript messages (a `gpt-realtime-2` call sends well over a
  hundred), and asking for `conversation-update` or `model-output` changes nothing. A phone
  call is unaffected, because the number asks the server who should answer. A browser call
  would be unknown to the server until its first decision, so the runs could not open ahead
  of it. The page therefore registers the call itself the moment the Vapi SDK connects (`POST /api/calls/{id}/start` with the
  chosen profile). The SDK's `call-start` event does not fire on GPT-Live either; the page
  treats the call as live from the moment `start()` returns.
- The conversation reaches the page two ways, because Vapi gives it two ways. The browser
  SDK gets the agent's transcript as one growing string, a few words at a time while it
  speaks, and a single final at the end of the call; the page turns that into words on
  screen as they are said, a speaking segment on the timeline, and one line per utterance
  (a pause of 1.2 s ends one), and relays those to `POST /api/calls/{id}/notes`. The
  caller's words come to the server inside each tool call and the end-of-call report
  (`artifact.messages`), so the caller's lines appear at each decision and at hang-up. On
  a phone call that second path is the only one for words.
- Who is speaking, right now, is the part GPT-Live keeps to itself. On the other two
  stacks Vapi streams every call live (`monitor.listenUrl` on the call record, a WebSocket
  of raw PCM with the caller on one channel and the agent on the other) and
  `voice/listen.py` follows it: loudness per channel every 100 ms, a turn starts after
  200 ms of sound and ends after 700 ms of quiet, with an adaptive noise floor so a phone
  line does not read as a voice. Those starts and stops light the two boxes and draw the
  timeline's lanes with the timing of the audio itself. On GPT-Live, Vapi refuses that
  stream (close code 1008, "This action is not supported on the full duplex harness", as
  of October 2026), and sends no speech updates either. So on a GPT-Live browser call the
  page does it: the agent's turns from the SDK's transcript stream, the caller's from the
  page's own microphone, with the same loudness rule. On a GPT-Live phone call there is no
  live signal at all; the boxes and lanes fill in at each decision and at hang-up, from the
  turns Vapi sends with them. For a phone demo where the live flow matters, run
  `VOICE_STACK=realtime`. `VOICE_LISTEN=0` turns the audio stream off on any stack.
- The speaker has no tools. It delegates to the reasoner, and anything the reasoner does
  not do, does not happen: a goodbye without `endCall` leaves the line open. The reasoner's
  prompt checks for a goodbye before anything else, and a `customer.speech.timeout` hook
  (30 s of silence) says goodbye and ends the call as a last resort.

On the `realtime` stack a tool answer that reaches Vapi while the model's turn is still open
is dropped, so the server holds every tool answer until 1.5 s after the request came in
(`VOICE_TOOL_MIN_MS`, 0 on the other stacks). On every stack phone audio is narrowband, so
for a recording use a browser call.

## Test calls

`voice/testcalls.py` runs scripted calls against the live agent through Vapi Simulations:
Vapi plays the caller, an AI tester following a script, on a real voice call, and every
webhook goes through the running server, so SenseLab is read, Jev decides and the outcome is
sealed exactly as on a call from you. The scenarios are `[[scenarios]]` in the business
file, next to the callers, so the harness runs whatever business the server is running.
Northwind's five are each one thing a caller obviously needs: Bruno's fourth call about the
same $49 (does John know him, recount the history, give a reason when pushed back on, read
the number back, close slowly), a caller nobody has a record of (is John honest about being
automated, does he invent nothing), Priya changing topic mid-call, Tomas's enterprise outage
(a name and a time), and Marcus's plain question answered from the file (and only from it:
one run had John call his $29 "a storage overage" when the file said three seats, so the
judge now fails any charge the file does not name).

A scenario is the script the tester follows, who it calls as, and the yes/no questions the
judge answers from the transcript, every one of which must be a yes:

```toml
[[scenarios]]
id = "ana"
title = "Ana moves her cleaning"
persona = "ana-reyes"        # a [[personas]] id; leave it out for a caller you have no record of
decisions = 1                # how many decisions the server should make on the call (default 1)
tags = ["known"]
script = """
You are Ana. Facts: you have a cleaning next Tuesday at 3 and need it moved to Thursday. Steps:
1. After the greeting, say: 'Hi, I need to move my cleaning to Thursday.'
2. Whatever the agent proposes, say: 'That works. Thanks, bye.'
"""
[[scenarios.judge]]
name = "knew_the_booking"
check = "The agent referred to the existing Tuesday booking without being told about it."
[[scenarios.judge]]
name = "proper_close"
check = "The agent said goodbye only after the caller said she was done."
```

Write the script as facts plus numbered steps, one step per turn, with the words in
quotes; the tester says each in its own words and waits for the agent before the next.
Give it the facts the agent might ask for (a number, an email), and tell it what to say
when the agent asks whether there is anything else, or it will be left without a line and
start improvising. The judge entries are the things that went wrong on your real calls; a
check that can be answered yes or no from the transcript alone is one the judge grades
well. On top of them, every scenario gets the same checks on the call record and the
server's events: every tool answer delivered, one tool call per turn, our answers within
6 s, the caller's wait on a hand-off within 10 s, the decision made only after the caller
said why they were calling, no internal names said aloud, the call ended by the right side,
the outcome sealed, and from the stereo recording, no agent turn over 30 s and no talking
on through a caller who speaks up.

```
uv run python -m voice.testcalls                  # every scenario, as voice calls, about six minutes
uv run python -m voice.testcalls --only bruno     # one of them (comma-separated for several)
uv run python -m voice.testcalls --list           # what the running server's business has
uv run python -m voice.testcalls --report <run>   # the report for an earlier run
```

It needs the server running and `VAPI_PRIVATE_KEY`. The callers' records are not written
to: a scenario with a persona calls as a twin (`<persona>-sim`) seeded with the same
history. Vapi bills the simulated calls like any other call, a few cents a minute each. One
difference from a real call: the tester speaks first and the assistant's opening line is
dropped, so John's first words in these transcripts are his reply to the caller, which is
also what a caller who talks over the greeting gets. Run them after every change to the
prompts or the business file: GPT-Live is a beta and its speaker is not deterministic, so
a prompt that passed once is not a prompt that passes, and the transcripts are where the
bad turns show up.

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
  write it as facts, not a line, and without conditionals: the speaker reads those out);
  `then_no_file` is the same step for a caller nobody matched, who has no file to answer
  from (leave it out and `then` serves both); `words` is how it reads in a caller's history.
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
- `[[scenarios]]`: the test calls (see Test calls), each a script, who it calls as, and the
  judge's yes/no checks. Write one for every kind of call you expect; the harness runs
  whatever is here.

What stays the same: the account facts a call carries (name, tier, prior contacts in 30
days, tenure, an account id). On a browser call they come from the page; on a phone call
from the number, through `lookup(number)` in `voice/phonebook.py`, which is where your CRM
goes (see A phone number). The tiers are yours to name.

## Latency

The server opens the SenseLab runs ahead of the decision, so the caller never waits on a
lookup: when the call connects it begins one run per situation the caller could be in (the
account tells us tier and prior contacts; the intent is the only thing we do not know yet),
sharing one briefing across them, and the tool call takes the one that matches. The runs
open in order of likelihood, the common situations first, and a decision that arrives before
its run has started opens it on the spot. By the time the caller has said why they are
calling, the guidance is waiting and the decision is Jev's 200 ms. The briefing is refreshed
after every sealed outcome, which is when memory changes. A run opened and not used costs
nothing more than the read; nothing stays open on the platform. The timeline shows when the
situation the caller turned out to be in was ready, and when all of them were. The review
and the seal run after the call, off the webhook's path.

## Files

```
voice/business.toml  the business: company, intents, actions, policy, situations, outcomes, demo callers, test calls
voice/business.py    reads it; builds the scenario Jev and SenseLab see, names situations, opens runs ahead
voice/agent.py       Jev + SenseLab per call: open ahead, recall the caller, decide, seal with the outcome, repair loop
voice/customers.py   the per-caller memory: seeded history, the brief the agent gets, the note it writes after a call
voice/phonebook.py   who is calling, from the number: your lookup (fill it in), then the numbers the agent has learned
voice/server.py      FastAPI: /vapi/webhook, the page, /api/*
voice/listen.py      who is speaking, from the call's live audio stream (realtime and pipeline stacks)
voice/assistant.py   create/update the Vapi assistant from the business file (prompt, tools, voice stack, server URL); --number
voice/simulate.py    the scenario's callers, sent as Vapi tool calls
voice/testcalls.py   runs the business file's [[scenarios]] as voice calls through Vapi Simulations, judge plus our own checks
voice/web/index.html the page
demos/call_handling/ the demo's scenario: the batch harness runs it, and it reviews the live agent's decisions
```
