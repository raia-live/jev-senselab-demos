# Jev + SenseLab demos

Four business decisions made by [Jev](https://typesafe.ai) (TypeSafe AI's decision model),
getting better over time because [SenseLab](https://amfs.sense-lab.ai) is running its
self-improvement loop around it. Clone it, add two keys, run one command, open the report.

```bash
git clone https://github.com/raia-live/jev-senselab-demos && cd jev-senselab-demos
cp .env.example .env            # TYPESAFE_API_KEY and AMFS_API_KEY, that's it
uv sync --extra baseline        # (or: pip install -e . ; the extra is only for the retrieval arm)
uv run python -m core.runner --scenario lead_routing --quick     # ~8 min
open results/*/lead_routing/report.html
```

No SenseLab account yet? Sign up at https://amfs.sense-lab.ai, create an API key, paste it
into `.env`. The dashboard there shows every trace, incident, fix and canary the run produces
while it runs.

## The result

Same Jev model (`jev-1.13.0`), same 120 leads, same written policy, same reviews. The only
thing that changes between rows is what Jev gets told about the situation.

| arm | wrong decisions | dollars lost | accuracy | tacit rule | after the change | new wording | recovered in |
|---|---:|---:|---:|---:|---:|---:|---:|
| `jev` (policy only) | 59 | $142,900 | 51% | 0% | 0% | 0% | never |
| `jev+retrieval:meta-k8` (8 past cases) | 7 | $12,900 | 94% | 95% | 80% | 100% | 7 episodes |
| `jev+senselab` | 6 | $10,400 | 95% | 98% | 80% | 100% | 7 episodes |

This is [`results-sample/lead_routing`](results-sample/lead_routing). You can rebuild it
with `--replay results-sample` without any keys, and every number here can be recomputed
from the per-episode rows in that folder. Wrong decisions and dollars are scored against the
hidden truth; the arms only ever saw the review.

What the rows say:

- Jev with SenseLab lost $10,400. Jev on the policy alone lost $142,900 (93% more), and Jev
  with the best lookup we could build lost $12,900 (19% more). 6 wrong decisions against 59
  and 7.
- There is a rule nobody wrote down (EMEA mid-market deals only close through a partner),
  and 42 of the 120 leads hit it. SenseLab got it wrong once, the very first time, then
  never again. That's the 98%. The lookup got it wrong twice (episodes 0 and 13).
- The business changes at episode 60. Three leads were lost before memory switched the
  action (60, 66, 67), and the lookup lost the same three. SenseLab waits for a working
  action to fail three times in a row before it drops it, on purpose; the 400-episode runs
  below say why. The lookup flipped its action 3x as often on the way (7% churn vs 2%).
- The wording of the leads changes at episode 90. Neither memory arm cared, because the
  lessons are keyed to the situation and the words don't matter.
- The memory Jev gets does not grow. The slot is capped at 3,200 characters, and everything
  Jev received per decision (case, policy, memory) averaged 3.1k over the run. It fills up by
  about episode 30 and is the same size at episode 120 as it would be at 12,000, because a
  tally is one line no matter how many outcomes it counts. Paste the history in instead and
  Jev refuses the call at case 276 (more on that below).

## Where the gap grows

The table above is one agent, 120 leads, 10% review noise. Both memory arms sit close to the
floor there. About six of those misses are the price of learning from outcomes at all,
whatever does the learning: first contact with the tacit rule, the three losses it takes to
be sure the business changed, a flipped review landing on a one-case record. So the gap at
120 is one decision. Here's what happens when you change the things a real deployment
changes.

### Three instances, one stream

`--agents 3` hands the same 120 leads round-robin to three instances of the agent, the way a
deployment runs several workers behind one queue. A lookup store is local to the worker that
built it, so each of the three learns only from its own cases. SenseLab's agent is one agent
whichever worker is deciding: one memory, one repair loop, the worker recorded on the trace.

| arm | wrong decisions | dollars lost | after the change | wrong per worker | workers that paid the EMEA rule's first miss |
|---|---:|---:|---:|---|---|
| `jev` | 59 | $142,900 | 0% | 22 / 21 / 16 | 3 of 3 |
| `jev+retrieval:meta-k8` | 13 | $30,200 | 47% | 3 / 6 / 4 | 3 of 3 |
| `jev+senselab` | 6 | $10,400 | 80% | 5 / 1 / 0 | 1 of 3 |

The lookup paid for the EMEA rule three times (episodes 0, 1 and 2, one per worker), paid for
the change three times too (60, 66, 67), then kept paying for it (76, 80, 83, 85, 113) because
each worker's store had only a third of the evidence, and lost 2.3x the dollars it lost with
one worker. SenseLab made the same six misses as the single agent and no others: the first
failure anywhere teaches everyone, so workers 1 and 2 walked into their first EMEA lead
already knowing. Worker 2 got nothing wrong all run; worker 1's one miss is episode 67, the
third lead at the change, which falls on its turn. One judge graded the 120 traces and one
repair loop ran, the same as for a single agent.

Worth saying out loud: a lookup that three workers share is a memory service, with everything
that comes with running one. That's the thing you'd be building. Run:
[`results-sample/lead_routing-team3`](results-sample/lead_routing-team3).

### 400 episodes, and noisier reviews

Same stream, 400 leads instead of 120, change at 200, new wording from 300. Then the same
again with one review in five wrong instead of one in ten.

| run | arm | wrong decisions | dollars lost | after the change | recovered in |
|---|---|---:|---:|---:|---:|
| 400, noise 0.1 | `jev` | 185 | $444,100 | 0% | never |
| | `jev+retrieval:meta-k8` | 7 | $12,900 | 90% | 41 episodes |
| | `jev+senselab` | 6 | $10,400 | 92% | 6 episodes |
| 400, noise 0.2 | `jev` | 187 | $444,500 | 0% | never |
| | `jev+retrieval:meta-k8` | 11 | $22,900 | 85% | 5 episodes |
| | `jev+senselab` | 7 | $10,600 | 92% | 6 episodes |

At noise 0.1 the two learners are one lead apart over 400 episodes, and it is worth being
plain about where. Up to the change they miss the same three: the first EMEA mid-market lead
(0), then two small ones (27, 36) after a wrong review at 16 marked a correct disqualify as
a miss and both went looking for a better action until 53. At the change both lose three
(200, 205, 206) before switching. SenseLab's record then shows `route_to_ae` failing three
times in a row and the next such lead goes self-serve. The lookup loses one more at 241:
its eight nearest cases are still mostly the old, confirmed `route_to_ae` outcomes, and it
takes 41 episodes for the new ones to outnumber them.

With more wrong reviews the lookup picks up misses late in the run (351, 373, 393), all on
the changed situation: a wrong review on `self_serve` plus the old `route_to_ae` wins still
among its eight neighbours, and it drifts back to the action the business left behind.
SenseLab's seven are three first exposures, one bad pair of reviews on a thin record at 49,
and three leads at the change, the same three as at noise 0.1. That third one is
deliberate. An action that has worked three or more times is only written off after three
straight failures, because two wrong reviews in a row happen one pair in 25 at this noise
level and a real change costs one more lead to confirm. Before that rule, the same run was
behind the lookup at episode 140 (8 wrong to 5), every extra miss a working action dropped
after two bad reviews. The rule is `turned_streak` in `core/senselab.py`.

Runs: [`results-sample/lead_routing-400`](results-sample/lead_routing-400) and
[`results-sample/lead_routing-400-noise20`](results-sample/lead_routing-400-noise20).

## What you're looking at

Jev is good at the decision you ask it to make. What it can't do on its own is find out
later whether the decision was right, or change its mind when the business changes, or
notice that a rule which used to work has quietly stopped working. That's the part SenseLab
does.

| arm | what Jev sees besides the case and the policy |
|---|---|
| `jev` | nothing, the policy alone |
| `jev+history` | every past case, verbatim, with its action and review ("just load the past cases") |
| `jev+retrieval:meta-k8` | the 8 most similar past cases in the same situation, with their reviews (a lookup) |
| `jev+senselab` | what every past outcome taught about this situation (tallies, the last reviews, evidence status, a recommendation), with the repair loop running between decisions: judge, incident, corrective entry, test, canary, ship |

`jev+senselab` is the whole SenseLab loop. There's no "memory without the loop" arm because
that isn't a thing we sell.

Every demo has the same three acts, so you can see each mechanism do its job:

1. A tacit rule the written policy misses. About a third of the stream is a case the policy
   gets wrong. Nobody wrote the rule down, so it can only be learned from outcomes.
2. The business changes at episode 60. A rule that was right becomes wrong, and the system
   has to notice from outcomes alone that something validated has stopped working.
3. New wording at episode 90. Same cases, phrased differently. Anything that learned the
   words instead of the lesson breaks here.

Reviews never contain the right answer, and 10% of them are flipped at random, because real
reviewers are late, terse, and sometimes wrong.

## The demos

| folder | decision | the tacit rule | the change |
|---|---|---|---|
| [`demos/lead_routing`](demos/lead_routing) | route an inbound lead | EMEA mid-market closes only via partners | NA mid-market goes self-serve |
| [`demos/ap_invoices`](demos/ap_invoices) | handle an invoice exception | recurring services have no PO; pay on receipt | audit: large PO-matched invoices need sign-off |
| [`demos/marketing`](demos/marketing) | adjust an ad set's budget | fresh paid-social must be held, not scaled | privacy change flatters retargeting CPA |
| [`demos/moderation`](demos/moderation) | act on a reported post | news footage is age-gated, not removed | EU medical misinformation must be removed |
| [`demos/call_handling`](demos/call_handling) | what a support bot does next on a call | repeat billing callers need a specialist callback | a retention offer lets the bot keep cancellations |

The last one is also live: [`voice/`](voice/) puts the same decision behind a phone bot you
can call from your browser or a real number (Vapi runs the call; Jev decides; the outcome
goes back into memory when you hang up). Every call is decided twice, Jev alone and Jev with
SenseLab, so the page can show the difference, and SenseLab remembers each caller between
calls. The business it answers for is one file (`voice/business.toml`): swap in your company,
intents, actions and policy and the same agent runs your line. See
[voice/README.md](voice/README.md).

```bash
uv run python -m core.runner --scenario ap_invoices        # full run: 120 episodes, ~35 min
uv run python -m core.runner --scenario marketing --quick  # 40 episodes, ~8 min
uv run python -m core.runner --scenario lead_routing --agents 3   # the same stream handled by three instances
uv run python -m core.runner --scenario lead_routing --episodes 400 --change-at 200 --probe-at 300   # a longer run
uv run python -m core.runner --all                         # every demo
uv run python -m core.runner --replay results-sample       # rebuild the committed reports, no keys needed
```

## How the loop makes Jev better

Jev makes the decision. Everything below happens around it, between decisions. Nothing about
Jev changes and nothing gets retrained.

### Memory is keyed to the situation

Each decision opens a SenseLab run with the case and a situation label you can observe
(`mid_market/emea`). What comes back is what past outcomes taught about that situation:
which action worked, which didn't, how many times, what the last few reviews said. It's
ranked by evidence and comes with a recommendation. Jev reads it in its state and cites the
entry it relied on. In the committed run the tacit rule was in memory after its first
failure and held for the other 41 cases.

### Outcomes credit what actually drove the decision

When the review arrives, the run is sealed with the outcome and the entry Jev cited. A rule
that led to a failure loses confidence. One that keeps working gains it. A rule that just
happened to sit in the same prompt isn't touched. Confidence decays unless outcomes keep
confirming it, so stale knowledge fades instead of piling up.

### Leave the record only when nothing in it is working

Reviews are imperfect, so before Jev sees the memory the harness reads SenseLab's
recommendation against the situation's own record. An action that has worked here and
hasn't failed twice in a row gets held, whatever one bad review says. When nothing is working
and something has failed twice, the record has turned and the exploration stands; that's
what a change looks like. A record of single failures goes back to the written policy. This
is `noise_guard` in `core/senselab.py`, a few dozen lines with one constant. It fired 15
times in the committed run (9 holds, 6 hand-backs) and the integration template has it.

### Noticing a rule stopped working

A long-validated entry that starts failing is a regime shift, and the briefing says so. Act
two turns on this. The policy says "NA mid-market goes to an AE", the lesson said so too and
was validated for 60 episodes, then it loses two deals in a row and memory moves to what the
reviews were pointing at.

### The repair loop runs between decisions, behind a gate

The platform grades every decision with a judge as it happens, clusters the failures into
behaviours, opens an incident, and a repair agent drafts a corrective entry (a note naming
the rule that stopped working and what to do instead) on a memory branch. The draft is
replayed against the runs that failed (Tier 1). Only if it would have fixed them does it go
to half of live sessions (the canary) to be compared with control, and only if it holds does
it ship to `main`. In the committed run the judge graded 119 of the 120 decisions for $0.78,
opened three incidents, and both of the repair agent's drafts were stopped at Tier 1 because
they wouldn't have fixed the failed runs. A draft the gate rejects never touches a live
decision, which is what you want from a loop that edits memory on its own. Every step is an event on the dashboard
and in the report.

### What this means for a Jev deployment

The decision model stays pinned and auditable. What changes over time is the memory around
it, and every change has a cause you can look up (an outcome, a judge verdict, a canary
result) and reverse.

## Jev's input limit, and why the memory has to live outside it

Jev takes about 32k input tokens per call (32,768 in our probe; roughly 140,000 characters
of plain text, fewer for field-heavy text). Past that the API returns
`400 max_tokens_exceeded`. `python -m core.limit` measures it against your key. That limit
decides how you can build a memory for Jev:

- Put every past case in the prompt (`jev+history`) and it grows by one case per decision.
  In [`results-sample/lead_routing-history-wall`](results-sample/lead_routing-history-wall)
  Jev refused it from case 276 on, and it never decided again.
- A top-k lookup (`jev+retrieval:meta-k8`) stays under the limit by only ever showing 8
  cases, which is also all it knows at any decision. After a year it's still deciding on 8
  of thousands, one wrong review among them is 12% of its evidence, and remembering more
  means paying about 400 more characters of Jev's input on every call, forever.
- SenseLab keeps the history outside Jev and sends a capped summary (3,200 characters here;
  case, policy and memory together averaged 3.1k per decision in the committed run, the same
  size at the end as at episode 30). "Worked 20 of 24, the last 2 did not" is the whole
  history of that action in one line. One wrong review among 24 is 4% of the evidence. And
  the history is where the judge and the repair loop do their work, on all of it, between
  decisions.

Each report has a "What fits in Jev's input" chart with the three lines against the limit.

## Use it on your own decision

[`templates/your_decision/decide.py`](templates/your_decision) is the whole integration in
one file. Edit three blocks (your actions, the state you already send Jev, how you find out
whether a decision worked) and run it.

## What a run costs

Jev: 900 to 2,000 input tokens per decision depending on the arm. A 120-episode run of the
default three arms is 360 calls.

SenseLab: the memory calls are part of your plan. The judge and repair agent spend from your
account's evaluation budget (the cap is yours to set; the dashboard shows it). Judging the
committed 120-episode run cost $0.78 and the three-instance run $0.75 (`metrics.json` has the
figure); the repair agent's proposals and their replay tests come on top. `meta.json` records
the account's budget before and after each run.

Time: `--quick` is about 8 minutes. A full demo is about 35 minutes, including the loop's
drain at the end. The Jev-only and retrieval arms finish in under two minutes and run in
parallel with the SenseLab arm.

Each run gets a fresh agent id and entity path (`jev-demo-<scenario>-<run-id>-s<seed>-<arm>`),
so runs never contaminate each other. `--resume --run-id <id>` continues a finished run on
the memory it learned. With `--agents N` the cases go round-robin to N instances of the
agent: the lookup and history arms get one store per instance, the SenseLab arm stays one
agent with one memory and one repair loop, and each row records which instance decided.

## Layout

```
core/        runner, Jev wrapper, SenseLab episode, repair-loop watcher, report builder, limit probe
demos/       one folder per decision: scenario.py (cases, policy, reviews) + README
voice/       the live voice agent: Vapi runs the call, Jev decides, SenseLab remembers (see voice/README.md)
baselines/   the retrieval lookup baseline
templates/   the integration template for your own decision
results-sample/   the committed runs this README quotes; CI rebuilds them with --replay
```

## FAQ

"We could just put past cases in the prompt." Two arms do exactly that, built as well as we
could build them. `jev+history` pastes all of them and gets refused by Jev at case 276.
`jev+retrieval:meta-k8` picks the 8 nearest with local embeddings, recency weighting and the
scenario's own situation label as a filter, which is more than most deployments have. The
table at the top is the comparison: 7 wrong and $12,900 against 6 and $10,400 with one
agent, 13 and $30,200 against 6 and $10,400 with three, 3x the churn. And a lookup has no idea whether a past review was noise or
signal, nothing that notices a validated pattern has started failing, no corrected rule to
fall back on, and nothing running when the agent isn't.

Does SenseLab change Jev? No. The Jev model is pinned (`jev-1.13.0`). SenseLab changes what
Jev gets told about the situation, and records why.

Where do I see the loop? On the dashboard at https://amfs.sense-lab.ai under the run's agent
(traces, incidents, fixes with their test and canary results), and in the report's event list.

Is the review the right answer? Never. Reviews say what happened ("closed-lost after 32 days
in the AE queue"), not what should have been done. The correct action is hidden from every
arm and only used to score the report.

Why one seed? It's a demo. One real run, committed with its per-episode rows so you can
check every number. `--seeds 11,12,13` runs more if you want them.
