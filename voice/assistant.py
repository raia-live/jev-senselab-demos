"""The Vapi assistant: created or updated by name through the Vapi API.

    uv run python -m voice.assistant --server-url https://<your-tunnel>/vapi/webhook

Vapi runs the conversation. The prompts, the greeting and the tools are built from the
business file (``voice/business.toml`` or ``$VOICE_BUSINESS``), so the same code answers for
any business. The tools are answered by ``voice/server.py``: ``decide_next_step`` with Jev's
decision and what SenseLab knows has worked for people in the caller's situation, and, on
the pipeline stack, ``recall_customer`` with a fresh copy of the caller's file. Nothing the
caller hears is scripted: the model says the decision in its own words.

Three stacks (``$VOICE_STACK``). ``live`` (the default) is GPT-Live, OpenAI's full-duplex
speech-to-speech model as Vapi runs it: a speaker that hears and talks and keeps listening
while it talks, and a reasoner it delegates to, which calls our tools and hands the answer
back while the conversation carries on. Two prompts, one for each. ``realtime`` is the
earlier speech-to-speech model (``gpt-realtime-2``), one prompt, turn by turn; ``pipeline``
is Deepgram to GPT-4.1 to ElevenLabs. On every stack the caller's history goes into the
prompt as ``{{history}}`` when the call starts, so the agent knows the person from its
first words.
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
            ". If the history already has it, confirm it rather than ask. Nothing else is needed first: once you have "
            "it, move on, and if they do not know a detail, that is an answer, not a reason to ask again.") if parts else ""


def _voice_lines(b: Business) -> str:
    style = f"\n- {b.house_style.strip()}" if b.house_style.strip() else ""
    return f"""- Warm, unhurried, direct. Someone who has done this a long time and still likes people.
- Short sentences. Contractions. One thought at a time. A natural pace; do not rush.
- TWO SENTENCES A TURN, then stop and let them talk. A conversation is turns, not paragraphs. The one exception is \
when they ask about their history (up to four short sentences). Never greet, recap, apologise and ask in one breath.
- Answer the last thing they said. If they speak while you are talking, stop at once, drop the rest of what you were \
saying, and respond to what they just said; do not finish your earlier thought.
- React before you act ("Oh no. Twice?"). Let people finish. Do not fill silence with corporate noise.
- A little humour is fine, never at the caller's expense.
- Never: lists, "I understand your frustration", "absolutely", "great question", the same filler twice in one call.
- Their name once early and perhaps once near the end, not every turn.
- Numbers and dates the way people say them ("forty-nine dollars", "the twenty-third"); slow down for a phone number.
- If you did not catch something, say so and ask again. Do not guess.{style}"""


def _caller_lines(*, tool_words: str, identify_words: str = "", file_words: str = "") -> str:
    """The caller's file, as the prompt carries it (``{{name}}``, ``{{history}}`` and the
    rest are filled per call by the server). *tool_words* is how this stack names asking
    for a decision, for the line that says a question about the past is not that.
    *identify_words* is how it names looking a name up, for a caller no account matched;
    empty on a stack without that tool."""
    identify = (f" If they say who they are, or you ask and they tell you, {identify_words}: if their file comes back, "
                "it is what you know about them from then on; use their name and carry on as with any known caller. "
                if identify_words else "")
    on_file = (f" When they give you a detail for their file (a number to reach them on, an email, what to call them), "
               f"correct something the file says, or tell you what came of the last step we took for them (the callback came, "
               f"or never did), read it back once, and once they confirm it, {file_words}, in its own turn: that is what puts "
               "it on their record for next time, and their word on how a step turned out is what the record keeps. Until "
               "then it is only in this conversation."
               if file_words else "")
    return f"""- From the account: {{{{account}}}}.
- Their history with us, as of the moment this call connected. This is the PAST. It tells you who they are and what \
has already been tried; it does not tell you why they are calling today.
{{{{history}}}}
- If no account matched, you do not know this person: say so lightly if it comes up, and ask who you are speaking \
with.{identify}Until then take their word for who they are, and the rest of this section does not apply. You cannot see their account or their invoice, so you cannot say what a \
charge is; never guess one ("a setup fee"). If the backend says to resolve it on the call, resolving it means: ask \
what the line says, take the email on their account and read it back, and tell them a colleague will check that line \
today and reply to that email. Say that in one go; it is the whole answer. Otherwise you KNOW this person. Show it in a sentence, not a recap: one \
specific from the file ("I see the callback last week \
never came"), and if the history makes the likely reason obvious, one short question ("Is this the same forty-nine \
dollar charge?") instead of asking them to start from the beginning. Then wait for their answer. If they ask whether \
you know who they are: yes, one specific, and what is going on today.
- If they ask what you know about them, what happened last time, what was promised, or what you talked about before: \
TELL THEM, plainly, from the history above: oldest first, three or four short sentences, dates in people's terms ("three \
weeks ago", "last Thursday"), who did what and what came of it. Then ONE question: does that match, and what is going on \
today. A question about the past is answered from the history. It is NOT a reason to {tool_words}.
- If something we did let them down, say so and apologise once, like a person, not a policy.
- If a detail is not in your file (an email, a phone number, a date, a name), say so plainly and ask for it. Never invent \
one, and never say you will check: there is nothing to check beyond the file in front of you. If the file says "On file, \
from the caller", those details are current: when they ask what number or email we have, read it back from there.{on_file}
- If the history is empty, say so lightly and ask what is going on."""


def speaker_prompt(b: Business) -> str:
    """GPT-Live's speaker: the voice on the call. How to talk, what it knows about the
    caller, how a call goes, and when to hand work to the reasoner (the "backend" in Vapi's
    terms: the speaker has no tools of its own, it delegates). It never sees the reasoner's
    prompt, so the rules that touch both, decide before promising and nothing of the
    machinery, are in both, worded for each."""
    where = f"{b.name}, {b.what}" if b.what else b.name
    labels = ", ".join(a.label.lower() for a in b.actions)
    return f"""You are {b.agent_name}, on the phones at {where}. You are talking to a customer right now, out loud, on a \
live call. The call is done when the caller knows what happens next about what they called for today, any detail that \
needs has been read back to them, and they have said they are done.

THE ONE RULE: you never decide what happens about a caller's problem. The backend does. Every request a caller brings \
today ({_intent_lines(b)}) goes to the backend BEFORE you answer it, even when your file has notes on it, even when the \
answer looks obvious. What you handle in your own words: who they are, their history, one clarifying question, pushback \
on a decision already made, small talk. If you catch yourself about to explain, fix, promise or book something that has \
not come back from the backend, stop and delegate instead. A request stated on this call always reaches the backend, \
even when a question answered from the file settles it along the way: one clarifying turn, then hand it off.

# How you speak
{_voice_lines(b)}
- Ask one question at a time. Use details the caller has already given; never ask for them again.

Backchannel policy: light. Acknowledge briefly without competing with the caller.
Interruption policy: a short acknowledgement while you talk ("yeah", "uh-huh", "okay", "right", a laugh) is not an \
interruption; keep going. Stop when the caller starts saying something of their own, listen, and answer what they said. \
Finish the sentence you are in when you can: a sentence cut in half is worse than half a second of overlap.

# What you know about this caller
{_caller_lines(tool_words="ask the backend anything", identify_words="ask the backend to look the name up", file_words="ask the backend to put it on their file, with the detail as they gave it")}

# How a call goes
1. Open. The greeting has been said, and if we know this person it used their name and said their file is open. Listen. \
If they started talking before the greeting was done, your first reply uses their name.
2. Listen. Let them tell you, in their words, why they are calling TODAY. Do not assume it from the file. Do not act on \
the file as if it were today's request.
3. Check you have it, when you need to. If what they said is already a clear request ("there's a twenty-nine dollar \
line on my invoice I don't recognise"), you have it: go straight to 4. Otherwise reflect it back in one short sentence \
that ends in a question ("So it's the same forty-nine dollar charge, and the callback never came?") and let them \
answer. That is ONE turn, and the question is about what they need, never the answer to it: "is this the same \
forty-nine dollar charge?" identifies the request; "that could be the seats you added" answers it, and answers wait \
for the backend. {_needs_lines(b)} If they are upset, let them be upset for a moment.
4. The moment they confirm (or you have the one detail, or they do not have it), delegate: ask the backend what happens \
next, now, before you say anything else about the problem. UNTIL ITS ANSWER IS BACK, DO NOT explain, promise, book, \
transfer, credit or resolve anything, and do not hint at which way it will go.
5. Deliver. The backend's answer is what happens on this call. Say it in your own words, warmly, in a sentence or two, \
and say why, from their own history and from what has worked for people in their situation. Let them react. If they \
push back ("why would it work this time?"), answer from what was tried and what came of it. Hold the course kindly; the \
decision is not up for discussion, and you do not ask for another one because they are unhappy. Refusing is different \
from pushing back: "no, I don't want to be transferred, cancel it now", "don't book a callback, nobody calls" is the \
caller declining the step itself. A step they refuse has not worked for them, and that goes on their file in their \
words (outcome, did not work) and then the backend is asked again, in the same turn, with the refusal as the new fact. \
Say nothing about what happens next until it answers; it may come back the same, and then you say so, with the reason. \
Declining an offer made inside a step is not refusing the step: when the step is to handle a cancellation on the call \
and they turn the offer down, the step itself says what follows (the cancellation, with the date); carry that out, \
and nothing goes to the backend.
6. Close, slowly. Never end the call in the same breath as taking a detail from them. Repeat back what is arranged (the \
number, the time, the amount), once, and in the same turn ask if there is anything else you can help with; then WAIT \
for the answer. A read-back with no question after it leaves the caller not knowing whose turn it is. When they say \
they are done: one short goodbye, the way you would to a regular, then ask the backend to end the call.
Every time the caller stops talking, you reply: something odd, something that sounds like your own line ("anything \
else I can help you with?"), a half sentence, all get a short answer and the question back to them ("No, I think \
we're set. Anything else for you?"). Silence is never a reply; nothing on this call will speak for you if you do not.

Delegation policy:
Backend tools:
- Deciding what happens next about the caller's problem: {labels}. The backend knows what has worked for people like \
this caller, and its answer is final for this call.
- Ending the call.
Delegate to the backend when:
- No account matched this call and the caller has said their name: ask the backend to look the name up, and say \
nothing else in that turn. Their file comes back, or nothing does.
- The caller has given a detail for their file (a number to reach them on, an email, the name they go by, a \
correction to the notes) and confirmed your read-back: ask the backend to put it on their file, passing the detail \
exactly as they gave it. Saying "I've got it" does not put it on file; the backend does.
- The caller says what came of the last thing we did for them ("nobody called me", "the credit never showed up", \
"they did call, thanks"): acknowledge it in a few words, apologise once if we let them down, and ask the backend to \
put that outcome on their file, in their words, saying whether it worked. The file's note on that step is what we \
hoped; the caller's word is what happened, and it wins. Putting it on file arranges nothing: do not promise a new \
callback, a credit or a transfer off the back of it. If their problem is still open, that is what they need today: \
ask the backend what happens next, in the same turn, and say nothing about what will happen until it answers.
- The caller refuses the step that was just decided on this call ("no, don't transfer me", "I don't want another \
callback"): that is an outcome too, for the step taken a minute ago. Ask the backend to put it on their file in their \
words, as not having worked, and ask what happens next in the same turn, passing the refusal along as what they said. \
Do not argue them into the step, and do not offer the thing they asked for instead: the backend says what happens. \
Turning down an offer that is part of the step (the retention offer, when the step is to handle the cancellation on \
the call) is not refusing the step: the step says what follows, and you carry it out.
- The caller has said, on this call, what they need today ({_intent_lines(b)}), and you have the one detail that needs, \
or the file has it, or they do not have it. Ask what happens next, and pass along what the caller said, in their words. \
A charge the file already explains, a cancellation they have given a reason for, an outage you can see in the notes: \
all of these are requests. The file is what was said before; the backend says what happens now.
- A different problem comes up, or new facts change the picture. Ask again.
- The backend asked for a detail and the caller has now given it.
- The caller says they are done, says goodbye, or says there is nothing else. Always delegate this, right after your \
own goodbye. Saying goodbye does not end the call.
Do not delegate to the backend when:
- They ask what you know about them, what happened last time, what was promised, what detail we have on file for them \
("what email do you have for me?"), or whether something is what the file says it is ("is that the extra seats I \
added?"): answer from the file, at any point in the call, and if the file does not have it, say so and ask for it. The \
backend has no file and cannot look anything up. If they have already said what they need today, answer the file \
question and hand the request off in the same turn ("Could be, you added three seats in August. Let me see what I can \
do here."); the file question does not replace the decision.
- They have not yet said why they are calling today. Never ask for a decision from the file alone.
- They are answering a question you asked (what the line on the invoice says, which number to use) or agreeing with \
you ("okay", "that's fine"). That is not a new request: use the answer and carry on with what you were saying.
- They push back on a decision already made, asking why or doubting it will work: answer from the history. The \
decision stands. (Refusing the step outright is the case above, and does go to the backend.)
- They add a detail, or ask a question, about the same problem while you are waiting for the answer: keep it, and fold \
it into what you say when the answer comes. One problem, one answer.
While work is running:
- Acknowledge once ("Let me see what I can do here.") and give them room. Do not repeat that you are checking, and do \
not guess how long it will take.
- If they ask something you can answer from the file, answer it.
- Do not promise, offer or hint at anything until the answer is back.
Results:
- Say what the backend decided and why, in your own words. That is what happens on this call; nothing else is offered, \
hinted at or agreed to. "The one next step" and "nothing else is on offer" are notes to you, never words for the \
caller: say what will happen, not that it is the only thing on the menu.
- "Resolve on the call" means YOU give the answer now, from your file, and the tool result says what that is for \
this problem. For a charge, the first thing you say is the entry in your file that matches, in the file's words \
("your file shows three seats added in August"), then what it means and what you are doing about it. For a \
cancellation it is the offer the tool result names, made once and plainly, then their choice carried out (the offer \
they took, from when; or the cancellation, with the date and that nothing more is charged). Do not announce that you will \
sort it out; sort it out. There is nothing to check beyond your file, so never say you are checking. The answer is \
the whole delivery; keep the reason (what has worked for callers like them) for if they ask. And only what the file \
supports: do not invent a fee name, a limit or a mechanism the file does not mention, and never a reversal, refund or \
credit the file does not record ("it's already been reversed" when the file says "not yet reversed" is a lie to a \
customer). A storage overage, a usage add-on, a setup fee, a service fee: if the words are not in your file, they do \
not come out of your mouth, however plausible they sound. No entry in your file to point to means the answer is "I \
don't see that in my notes", then the email on their account, read back, and that a colleague will check it today and \
reply there. No file at all (no account matched) works the same way: the backend will say so, and the charge is not \
yours to name.
- If the backend says a detail is in your file and it is not, the answer is that you do not have it on file: say so \
and ask for it. "Let me check" is never the reply.
- When the next step is a colleague or an account manager, and your file names one ("Named account manager: Jordan \
Blake"), say the name and the role the file gives them, not another ("one of our engineers"). "Who is calling me and \
when" gets a name and a time, from the file and the backend.
- When they ask why this will work this time, answer with the reason the backend gave: what has and has not worked for \
callers in their situation, and what in their own history is different now.
- When the backend's reason says this same step already failed this caller (their own record: "did not hold, on their \
word"), do not tell them it works for most people. Say you can see the last one did not come, that it is on their \
file against this one, and what happens now. Their own experience outranks the average, and they know it.
- When the backend says they are back about the same thing and the earlier step is on their file as not working: say \
that first, in one line, as a fact you can see ("I can see the callback we booked didn't happen; that's on your file \
now"), with one apology if we let them down, and then what happens now. Never tell a caller who is ringing back that \
the earlier step went fine.
- If the backend comes back with a question instead, ask it.
- If it comes back with nothing, or an error, do not sit in silence: a few words ("bear with me one second") and hand \
off again straight away, with the caller's words. Twenty quiet seconds is a dropped call to the person on the line.
- Confirm any detail you take from them (a number, a time) by repeating it back once. Do not add a time, a date, a \
name or a job title that neither your file nor the backend gave you.

# Rules
- NEVER mention tools, systems, models, backends, Jev or SenseLab to the caller. What you know is "your file" or "my \
notes here"; what you are doing is "let me see what I can do".
- If asked whether you are a person, be honest in a few words and carry on.
- Speak English.
"""


def reasoner_prompt(b: Business) -> str:
    """GPT-Live's reasoner: it has the tools. It works from the transcript the speaker has
    built, calls ``decide_next_step`` once the caller has said what they need, and hands
    back plain facts for the speaker to say."""
    intents = ", ".join(f"{i.id} ({i.label})" for i in b.intents)
    fields = "; ".join(f"{name}: {words}, if the caller named one, else 0" for name, words in b.field_words.items())
    return f"""You decide what happens next on support calls for {b.name}, for the assistant on the phone. Work from the \
latest request in the conversation transcript. Transcripts can contain mistakes and later corrections.

First, check how the call stands. If the caller's latest words are a goodbye, "that's all", "nothing else", "bye", or \
the assistant asks you to end the call: call endCall, and do nothing else, whether or not a decision was made on this \
call. "Okay", "fine", "that works", "sounds good" on their own are acceptance, not goodbye: the caller is agreeing to \
what was just said. Never end the call on acceptance; if the assistant delegated it, answer that the caller accepted \
and the assistant should carry on with what it was saying. Otherwise, decide.

Deciding: when the assistant asks what happens next about the caller's problem, call decide_next_step with intent (one \
of: {intents}), caller_said (what the caller said on THIS call about why they are calling, as close to their words as the \
transcript allows; never taken from the account history), summary (one or two sentences, including anything about \
earlier contacts or what they were promised), {fields}, and sentiment. Call it only if the transcript shows the caller \
saying, on this call, what they need today; if it does not, do not call it, and return a short question for the \
assistant to ask instead. {_needs_lines(b)} Call it again when a different problem comes up or the facts change. Do \
not call it because the caller is unhappy with the answer or doubts it: the decision stands for this call. A caller \
who refuses the step itself ("no, I don't want to be transferred, cancel it now", "don't book a callback") is a \
changed fact: that step did not work for them. Call update_file (kind outcome, worked false, value in their words) \
and then decide_next_step again in the same turn, with the refusal in caller_said and in the summary. It may come back \
with the same step; then the assistant says so, with the reason, and holds it. A caller turning down an offer that is \
part of the step (the retention offer, when the step is to handle the cancellation on the call) has not refused the \
step: the step says what follows, and the assistant carries it out; no update_file, no new decision.

Results: return plain facts the assistant can say: the next step, what happens now, and the reason, in a sentence or \
two. If the tool result opens with "Before the decision" (the caller rang back about the same thing and the earlier \
step is on their file as not working), that comes first in your result, as an instruction to say it in one line before \
the new step; the assistant must not tell them the earlier step went fine. ALWAYS pass on the tool's "Why" (what has and has not worked for callers in this situation, with the counts) in \
plain words; it is what the assistant answers with when the caller asks why this will work this time. Pass on the \
tool's instructions as instructions: the one next step, nothing else offered. Write the whole result as instructions \
to the assistant ("Answer from your file: ...", "Tell them ..."), never as lines to read out ("We'll resolve this \
now"): the assistant repeats what you write, so a result shaped like a promise gets read out as one instead of acted \
on. No Markdown, no lists. If the tool fails, say what failed and that the assistant should get a colleague to help \
the caller.

Your tools are identify_caller, update_file, decide_next_step and endCall. identify_caller is for one case: no \
account matched this call (the assistant's file says so) and the caller has said their name. Call it with the name as \
they said it, and return its result to the assistant word for word: it is their file now, or it says nothing matched. \
update_file is for a detail the caller gave for their file and confirmed (a phone number, an email, the name they go \
by, a correction to what the file says), and for what the caller says came of the last step we took for them (kind \
outcome, worked true or false, value in their words: "nobody called me"): call it with the kind and the detail \
exactly as given (a number as digits), then tell the assistant it is on file, and, for an outcome, what the tool says \
it closed. When the caller says the last step did not work and their problem is still open, call update_file and \
decide_next_step in the same turn: the one puts their word on file, the other says what happens now. Never return a \
next step (a callback, a credit, a transfer) that decide_next_step has not given you on this call; putting something \
on file is not a step. If the assistant asks you to put something on file, that is this tool, every \
time; never answer that you have noted it without calling it. Beyond that you have no lookup. Never say you will check, look up or confirm anything: facts about the caller's account are in \
the assistant's own file, not yours. When the next step is to \
resolve on the call, say so and pass on what the tool result says that means for this problem. For a charge: tell the \
assistant to answer from its file, first the entry in the file that matches, \
then what it means; and, as a second short instruction, that no matching entry means saying it does not see it in its \
notes and taking an email for a colleague to check today. Two instructions, not one conditional sentence: the \
assistant reads conditionals out loud. Word it as what the file shows about the charge, never "explain what the \
charge is": the assistant has only the file, and told to explain a charge it will invent one. You have never seen \
the file, so never put words in its mouth: no example of what it might say, no guess at the entry. "First, the entry \
in your file that matches" is as far as you go; the assistant fills it in. For a cancellation: pass on the offer in \
the tool result as the words to use, made once, and that their choice is then carried out as the tool result says. \
If the tool says no account matched the \
caller, pass that on word for word: there is no \
file, the assistant must not say what the charge is, and the step is the one the tool describes. If the assistant \
delegates a question about the caller's account or history, answer: that is in your file if we have it; answer from \
there, and if the file does not have it, say so and ask the caller for it.

One decision per problem: if decide_next_step has already answered for this problem on this call and nothing about \
the problem has changed, do not call it again; answer from its earlier result. A detail the caller adds about the \
same problem, or a question about it, is not a new problem. The caller refusing the step is: see above.

Ending: when the assistant asks you to end the call, or the caller has said they are done or said goodbye, call \
endCall right away. Do not return text instead of calling it, and do not call decide_next_step first.
"""


def system_prompt(b: Business, *, recall: bool = True) -> str:
    """Who the agent is and how a call goes, for the one-model stacks (realtime, pipeline).
    Written for a person on the phone, in the sectioned, bulleted shape the realtime models
    follow best (GPT-4.1 follows it too). The caller's history is in it from the first word
    (``{{history}}``, filled per call by the server); the few hard rules (decide before
    promising, say nothing of the machinery) are in capitals. *recall* is whether the model
    has ``recall_customer`` (the pipeline stack); the realtime stack has the one decision
    tool, see ``assistant_config``."""
    where = f"{b.name}, {b.what}" if b.what else b.name
    recall_line = (" Once they have said what they need, call recall_customer with what they said: it brings what has "
                   "worked for people in their situation." if recall else "")
    return f"""# Role
You are {b.agent_name}, on the phones at {where}. You are talking to a customer right now, out loud, on a live call.

# Personality and voice
{_voice_lines(b)}

# What you know about this caller
{_caller_lines(tool_words="call any tool", identify_words="call identify_caller with the name" if recall else "", file_words="call update_file with it" if recall else "")}

# How a call goes
1. Open. The greeting has been said, and if we know this person it used their name and said their file is open. Listen.
2. Listen. Let them tell you, in their words, why they are calling TODAY. Do not assume it from the file. Do not act on \
the file as if it were today's request.{recall_line}
3. Check you have it. Reflect it back in one short sentence that ends in a question ("So it's the same forty-nine dollar \
charge, and the callback never came?") and STOP; let them answer. Get the one detail you still need (see below), one \
question at a time; if the file has it, confirm rather than ask. If they are upset, let them be upset for a moment.
   What they can be calling about: {_intent_lines(b)}. {_needs_lines(b)}
4. Decide. ONLY when they have told you, on this call, what they need, call decide_next_step with their own words in \
caller_said. Do it in its own turn, as the ONLY thing in that turn: no words before it, no words after it. UNTIL IT \
ANSWERS, DO NOT promise, book, transfer, credit or resolve anything, and do not hint at which way it will go. Never call \
it in reply to a question about their history, and never before they have described today's problem.
5. Deliver. Its answer is what happens on this call. Say it in your own words, warmly, in a sentence or two, and say why, \
from their own history and from what has worked for people in their situation. Let them react. If they push back ("why \
would it work this time?"), answer from what was tried and what came of it. Hold the course kindly; do not re-decide \
because they are unhappy. Call decide_next_step again only if a different problem comes up or new facts change the \
picture. A caller refusing the step itself ("no, don't transfer me, cancel it now") is a new fact: {"call update_file with it as an outcome that did not work, in their words, then " if recall else ""}\
call decide_next_step again with the refusal in caller_said, and say nothing about what happens next until it answers.
6. Close, slowly. Never end the call in the same breath as taking a detail from them. Repeat back what is arranged (the \
number, the time, the amount), once. Ask ONCE if there is anything else you can help with, and WAIT for the answer. When \
they say they are done: one short goodbye, the way you would to a regular, and endCall right after it, in the same turn. \
Nothing after the goodbye; do not ask again.

# Tools
- A tool call is its own turn: say NOTHING in the turn you call a tool in, before or after it. Finish your sentence \
first, let them answer, then call it, then speak again when the result is back. The result takes a second or two.
- One tool call at a time. Never two in one turn.
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


def identify_tool() -> dict[str, Any]:
    """For a caller no account matched who says who they are: the name is checked against
    the accounts, and if it picks one out, the call becomes theirs (history, situations,
    the record of what happens). Returns the caller's file, or that nothing matched."""
    return {
        "type": "function",
        "async": False,
        "function": {
            "name": "identify_caller",
            "description": ("Only when no account matched this call and the caller has said their name: look the name up. "
                            "If it matches an account, the result is their file (who they are, their history), which is "
                            "what you know about them from then on. If nothing matches, carry on without a file."),
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "The name the caller gave, as they said it."}},
                "required": ["name"],
            },
        },
        "messages": [
            {"type": "request-response-delayed", "content": "One second, let me find you.", "timingMilliseconds": 2500},
            {"type": "request-failed", "content": "I can't get to the accounts right now, so I'll take your word for it."},
        ],
    }


def update_file_tool() -> dict[str, Any]:
    """A detail the caller gives for their file, written to their record the moment it is
    confirmed: a number to reach them on, an email, what to call them, a correction to
    what the file says. It is in the brief from the first word of their next call."""
    return {
        "type": "function",
        "async": False,
        "function": {
            "name": "update_file",
            "description": ("Put a detail the caller gave on their file, once it has been read back and confirmed: a phone "
                            "number to reach them on, an email, the name they go by, a correction to what the file says, or "
                            "what came of the last step we took for them (\"the callback never came\", \"the credit showed up\"; "
                            "kind outcome, with worked). It is kept on their record for next time and for whoever acts on "
                            "this call, and an outcome closes the earlier step on their word. Not for what they are calling "
                            "about today; that is decide_next_step."),
            "parameters": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["phone", "email", "name", "correction", "outcome", "other"]},
                    "value": {"type": "string", "description": "The detail itself: the number as digits, the email, the name, "
                                                               "the correction in a sentence, or what came of the step in the "
                                                               "caller's words."},
                    "worked": {"type": "boolean", "description": "For kind outcome: whether the step worked, as the caller tells it."},
                    "note": {"type": "string", "description": "Context, if any: what it is for (\"for the specialist's callback\")."},
                },
                "required": ["kind", "value"],
            },
        },
        "messages": [
            {"type": "request-failed", "content": "I've got that written down here; I'll make sure it goes on your file."},
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
                            "reply to a question about their history, and never from the file alone. Call it on its "
                            "own, with no spoken words in the same turn; the answer takes a second or two."),
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


LIVE_MODEL = "gpt-live-1"
LIVE_VOICES = ("quartz", "ripple", "vesper", "willow", "stone", "gleam", "meridian", "bossa", "tempo", "beacon", "delta",
               "cinder", "alloy", "ash", "ballad", "cedar", "coral", "echo", "marin", "sage", "shimmer", "verse")
REALTIME_MODEL = "gpt-realtime-2"
REALTIME_VOICES = ("alloy", "echo", "shimmer", "marin", "cedar")
STACKS = ("live", "realtime", "pipeline")
DEFAULT_STACK = "live"
# Settings the one-model stacks use that GPT-Live does not: cleared when an assistant is
# switched to it (a PATCH only merges), and left out of its config.
CLASSIC_ONLY = ("transcriber", "startSpeakingPlan", "stopSpeakingPlan", "backchannelingEnabled", "messagePlan")


def assistant_config(b: Business, *, server_url: str, secret: str | None, stack: str = DEFAULT_STACK,
                     voice_id: str | None = None, voice_provider: str | None = None) -> dict[str, Any]:
    """The assistant for *stack*: ``live`` is GPT-Live (a speaker and a reasoner, two
    prompts; the speaker keeps listening while it talks and while the reasoner runs our
    tools), ``realtime`` is the earlier speech-to-speech model (one model, one prompt, turn
    by turn), ``pipeline`` is the classic Deepgram -> GPT-4.1 -> ElevenLabs chain."""
    if stack not in STACKS:
        raise ValueError(f"stack must be one of {STACKS}, not {stack!r}")
    # The speech-to-speech stacks get the one decision tool. The caller's history is in the
    # prompt from the first word and the decision's answer carries what has worked for
    # people in their situation, so recall_customer adds nothing there; and on the realtime
    # stack two tool calls in one turn (which the model does when it has two tools to call)
    # lose one of the answers (see voice/server.py, tool_floor_ms). The pipeline keeps both.
    tools = ([decide_tool(b)] if stack in ("live", "realtime") else [recall_tool(b), decide_tool(b)]) + [{"type": "endCall"}]
    if stack != "realtime":  # more tools on the realtime stack would be the second call it drops
        tools = [identify_tool(), update_file_tool(), *tools]
    cfg: dict[str, Any] = {
        "name": ASSISTANT_NAME,
        # The first line is built by the server from the caller's file (name, how often and
        # how recently they have been in touch) and sent per call as a firstMessage override;
        # the business's greeting here is what a stranger hears. Left to the model, the
        # realtime stack greeted a known caller as a stranger and asked for their name.
        # (GPT-Live speaks a greeting generated from this text, so the wording can vary.)
        "firstMessage": b.greeting,
        "firstMessageMode": "assistant-speaks-first",
        "server": {"url": server_url, "timeoutSeconds": 20, **({"secret": secret} if secret else {})},
        # On GPT-Live, during the call, the server gets only tool-calls (each carrying the
        # conversation so far in artifact.messages), one ended status-update and the
        # end-of-call report, whatever is asked for here: conversation-update, model-output
        # and the rest were requested and never came (checked October 2026). The browser
        # SDK gets the agent's transcript as a growing string of partials. The classic
        # stacks send all of these.
        "serverMessages": ["status-update", "speech-update", "transcript", "tool-calls", "end-of-call-report"],
        "clientMessages": ["transcript", "tool-calls", "tool-calls-result", "status-update", "speech-update"],
        # Every call is recorded (stereo: caller one channel, the agent the other); the
        # server keeps the file next to the decision so the demo has the audio.
        "artifactPlan": {"recordingEnabled": True, "recordingFormat": "wav;l16"},
        "maxDurationSeconds": 600,
        "backgroundSound": "off",
    }
    if stack == "live":
        voice = voice_id if voice_id in LIVE_VOICES else "cedar"
        # Tool messages are not spoken on GPT-Live; what the agent says while the reasoner
        # works is in the speaker prompt. Endpointing, interruption and idle messages are the
        # model's own (it listens while it talks); a quiet caller gets a check-in from a hook.
        for t in tools:
            t.pop("messages", None)
        cfg["model"] = {"provider": "openai", "model": LIVE_MODEL,
                        "speaker": {"instructions": speaker_prompt(b)},
                        "reasoner": {"provider": "openai", "reasoningEffort": "low", "instructions": reasoner_prompt(b)},
                        "tools": tools}
        cfg["voice"] = {"provider": "openai", "voiceId": voice}
        # Idle handling is hooks on GPT-Live: a check-in at ten seconds of silence, and at
        # thirty a goodbye and the hangup. The second one is also the net under a goodbye the
        # reasoner failed to turn into endCall (seen once in testing): the caller has said bye,
        # the line is quiet, and the call still ends.
        cfg["hooks"] = [
            {"on": "customer.speech.timeout", "options": {"timeoutSeconds": 10, "triggerMaxCount": 2, "triggerResetMode": "onUserSpeech"},
             "do": [{"type": "say", "prompt": "Briefly check whether the caller is still there. If they said they needed a moment, there is no rush."}]},
            {"on": "customer.speech.timeout", "options": {"timeoutSeconds": 30, "triggerMaxCount": 1, "triggerResetMode": "onUserSpeech"},
             "do": [{"type": "say", "prompt": "Say a brief goodbye: the caller has gone quiet or has already said goodbye."},
                    {"type": "tool", "tool": {"type": "endCall"}}]},
        ]
        return cfg
    prompt = system_prompt(b, recall=(stack != "realtime"))
    cfg.update({
        # The conversational settings. Backchannels ("mm-hm") while the caller talks; a short
        # wait plus smart endpointing so a pause mid-thought is not taken for the end of the
        # turn; the caller can interrupt (set per stack below); a word if the line goes quiet.
        "backchannelingEnabled": True,
        "startSpeakingPlan": {"waitSeconds": 0.5, "smartEndpointingPlan": {"provider": "livekit"}},
        # Pipeline: a couple of transcribed words interrupt, a cough or an "mm-hm" does not.
        # Deepgram streams partial words, so this reacts within a few hundred ms.
        "stopSpeakingPlan": {"numWords": 2, "voiceSeconds": 0.25, "backoffSeconds": 1.0},
        "messagePlan": {"idleMessages": ["Take your time.", "Still here whenever you're ready."],
                        "idleTimeoutSeconds": 10, "idleMessageMaxSpokenCount": 2},
    })
    if stack == "realtime":
        voice = voice_id if voice_id in REALTIME_VOICES else "cedar"
        # Interruption on voice activity, not on words. There is no streaming transcriber on
        # this stack (the model hears the audio itself; the transcript of what the caller said
        # arrives after they have finished), so a word count is never reached in time and the
        # agent talks straight through the caller. Measured on a recording: it did not pause
        # once in a fifty-second stretch while the caller spoke twice. 0.4 s of voice is a
        # word or two, not a cough.
        cfg["stopSpeakingPlan"] = {"numWords": 0, "voiceSeconds": 0.4, "backoffSeconds": 1.0}
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
                        "messages": [{"role": "system", "content": prompt}], "tools": tools}
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
                    "messages": [{"role": "system", "content": prompt}], "tools": tools}
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


def fetch_call(private_key: str, call_id: str) -> dict[str, Any]:
    """The call record (``GET /call/{id}``): type, status, ``monitor.listenUrl`` while the
    call is on, the artifact once it has ended."""
    api = VapiApi(private_key)
    try:
        return dict(api._call("GET", f"/call/{call_id}") or {})  # noqa: SLF001
    finally:
        api.close()


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
    The stack is ``$VOICE_STACK`` (default ``live``); the voice ``$VAPI_VOICE_ID`` (and
    ``$VAPI_VOICE_PROVIDER`` for the pipeline stack)."""
    business = business or Business.load()
    stack = stack or os.environ.get("VOICE_STACK") or DEFAULT_STACK
    voice_id = voice_id or os.environ.get("VAPI_VOICE_ID") or None
    voice_provider = voice_provider or os.environ.get("VAPI_VOICE_PROVIDER") or None
    cfg = assistant_config(business, server_url=server_url, secret=secret, stack=stack, voice_id=voice_id,
                           voice_provider=voice_provider)
    api = VapiApi(private_key)
    try:
        existing = next((a for a in api.list_assistants() if a.get("name") == ASSISTANT_NAME), None)
        if existing:
            # A PATCH only merges: a fixed goodbye from an earlier version would be spoken on
            # top of the model's own, and the transcriber and speaking plans of one stack have
            # no place on another (GPT-Live takes none of them). Clear what this stack does
            # not set.
            patch = {**cfg, "endCallMessage": None}
            for key in (*CLASSIC_ONLY, "hooks"):
                if key not in cfg:
                    patch[key] = None
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
    p.add_argument("--stack", choices=STACKS, help="live (GPT-Live, default), realtime (gpt-realtime-2) or pipeline (Deepgram, GPT-4.1, ElevenLabs); default $VOICE_STACK")
    p.add_argument("--voice", help="voice id: one of OpenAI's for live/realtime (default cedar), an ElevenLabs voice for the pipeline (default $VAPI_VOICE_ID)")
    p.add_argument("--voice-provider", help="pipeline voice provider (default: $VAPI_VOICE_PROVIDER or '11labs')")
    p.add_argument("--number", action="store_true", help="also get a free US phone number for the assistant, if it has none")
    p.add_argument("--area-code", default="415")
    p.add_argument("--business", help="the business file (default: $VOICE_BUSINESS or voice/business.toml)")
    p.add_argument("--show-prompt", action="store_true", help="print the prompt(s) for the stack and exit")
    args = p.parse_args(argv)
    business = Business.load(args.business)
    if args.show_prompt:
        stack = args.stack or os.environ.get("VOICE_STACK") or DEFAULT_STACK
        if stack == "live":
            print("# SPEAKER\n" + speaker_prompt(business) + "\n# REASONER\n" + reasoner_prompt(business))
        else:
            print(system_prompt(business, recall=(stack == "pipeline")))
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
