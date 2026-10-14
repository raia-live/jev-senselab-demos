# Inbound lead routing

Jev reads an inbound lead (free text plus a few structured fields) and picks one of five
actions: `route_to_ae`, `sdr_nurture`, `self_serve`, `partner`, `disqualify`. A reviewer
reports later how the routing turned out.

```bash
uv run python -m core.runner --scenario lead_routing            # the full example (~25 min)
uv run python -m core.runner --scenario lead_routing --quick    # a short look (~8 min)
```

Open `results/<run-id>/lead_routing/report.html` when it finishes.

## What the written policy says

| rule | action |
|---|---|
| Enterprise (1000+) with buying intent and a corporate email | account executive |
| Mid-market (100–999) with buying intent | account executive |
| SMB (under 100) | self-serve |
| Personal email address | disqualify |
| Content download / webinar, no demo or pricing ask | SDR nurture |

## What the policy misses

EMEA mid-market leads close only through a regional partner. The AE queue loses them:
time zones, local procurement, invoicing in the buyer's language. The region is a passing
mention in the lead (a city, a timezone, a currency) and a `country` field; nothing in the
policy points at it. About 30% of the stream is this case, so a Jev that follows the
policy alone is wrong about a third of the time and never finds out why from the policy.

Half-way through the run the business changes. A pricing update means NA mid-market
leads now buy self-serve while they wait for the AE callback; routing them to an AE loses
the deal. The rule that was right for 60 episodes becomes wrong.

From episode 90 the same cases arrive in new wording (held-out phrasings). A system
that memorised the text of past cases, rather than what the outcome taught, falls over here.

## What the review tells the agent

Only the outcome and a one-line note, never the right answer. "Closed-lost after 32 days in
the AE queue; buyer signed with a regional reseller" is a failure note; nothing says
`partner`. About 10% of reviews are flipped at random, because real reviewers are.

## What to look for in the report

The committed run is [`results-sample/lead_routing`](../../results-sample/lead_routing)
(seed 11, 120 episodes); the numbers below are from it.

- The EMEA rule is learned from one failure. `jev+senselab` routed the first EMEA
  mid-market lead to an AE, read the review, and routed the other 41 to a partner: 98% on
  the tacit cases. Plain `jev` follows the policy and gets all 42 wrong. The lookup needed
  two misses (95%).
- The change at 60 costs two leads, then memory moves. The record for `mid_market/na`
  shows `route_to_ae` failing twice in a row with reviews that say the prospect bought on
  the pricing page; the third such lead goes to `self_serve`. 87% on the changed cases and
  recovered in 6 episodes, against 80% and 7 for the lookup, which lost a third lead first.
- New wording at 90 changes nothing for either memory arm (100%), because a lesson keyed
  to the situation does not care how the lead was worded. Plain `jev` is at 0% here too.
- The guard, in action. Rows whose `memory_mode` is `act(held …)` are decisions where one
  bad review had SenseLab suggesting exploration and the harness held the working action
  instead (10 times); `none(dropped)` is a thin record handed back to the policy (6 times).
- What fits in Jev's input. The second chart: the history arm from the wall run climbs
  into Jev's 32k-token limit and is refused at case 276; the lookup and SenseLab stay flat.
- The repair loop. The judge graded 119 of the SenseLab arm's 120 decisions as they
  happened (one trace upload hit a rate limit), grouped the flagged ones into behaviours and
  opened incidents; both of the repair agent's drafts were stopped at the Tier 1 replay test
  because they would not have fixed the failed runs. The events are listed with their episode numbers.

Bottom line for this demo: 6 wrong decisions and $10,400 lost, against 7 and $12,900 for
the best lookup we could build and 59 and $142,900 for Jev on the policy alone. With three
workers on the stream the lookup goes to 13 and $30,200; SenseLab stays at 6.
