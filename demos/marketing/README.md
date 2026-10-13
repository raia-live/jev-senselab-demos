# Paid-media budget optimisation

Jev reads an ad set's daily snapshot (channel, audience, days live, spend, CPA against
target, CTR trend, frequency) and picks one of five actions: `scale_budget`, `hold`,
`cut_budget`, `pause`, `refresh_creative`. The next week's performance is the review.

```bash
uv run python -m core.runner --scenario marketing
uv run python -m core.runner --scenario marketing --quick
```

## What the written policy says

| rule | action |
|---|---|
| CPA at or under target, CTR stable or rising | scale the budget |
| CPA between 1.0x and 1.5x target | hold |
| CPA above 1.5x target after 7+ days live | cut the budget |
| CPA above 2x target | pause |
| Frequency above 4 with falling CTR | refresh the creative |

## What the policy misses

**A paid-social prospecting ad set under 7 days old must be held, not scaled, whatever its
CPA.** A budget jump during the platform's learning phase resets it and CPA doubles the
following week. The policy's first rule says to scale exactly these. Thirty percent of the
stream.

Half-way through, a browser privacy change makes display retargeting's reported CPA
flatter than reality (non-incremental conversions). Retargeting ad sets under target now
need their budget cut, not scaled.

From episode 90 the fresh paid-social ad sets arrive with new wording in the launch note.

## What the review tells the agent

Next week's numbers in one line, "CPA doubled the week after the change; spend up 80% for
the same conversions", never the action that would have been right. 10% of reviews are
flipped at random.

## What to look for in the report

The tacit rule learned from outcomes, the response to the change at 60 (incident →
corrective entry → test → canary → ship, listed under the charts), and the held-out
phrasing block at 90.
