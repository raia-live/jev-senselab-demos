# Content moderation with appeals

Jev reads a reported post (text, media description, account age, region, the reporter's
reason) and picks one of five actions: `keep`, `remove`, `restrict_reach`, `age_gate`,
`escalate_human`. The review is what happened next: an appeal upheld or denied, a regulator's
notice, a press complaint, or nothing.

```bash
uv run python -m core.runner --scenario moderation
uv run python -m core.runner --scenario moderation --quick
```

## What the written policy says

| rule | action |
|---|---|
| Graphic violence or gore | remove |
| Nudity | remove; educational or medical context: age-gate |
| Harassment naming or tagging a person | remove; no identifiable target: restrict reach |
| Repetitive promotion from accounts under 30 days | remove |
| Medical misinformation | restrict reach; if it tells people to stop treatment: remove |
| Hate speech targeting a protected group | remove |

## What the policy misses

Graphic footage posted as news reporting must be age-gated, not removed. Removals are
overturned on appeal and draw public censorship complaints. The policy's first rule says to
remove it. Thirty percent of the stream.

Half-way through, a new regional law means medical misinformation from EU accounts must
be removed rather than reach-restricted; leaving it up draws a regulator's notice.

From episode 90 the news-footage posts arrive with new wording.

## What the review tells the agent

"Appeal upheld; the post was reinstated and the removal was covered by two outlets", never
the action that would have been right. 10% of reviews are flipped at random.

## What to look for in the report

The tacit rule learned from outcomes, the response to the change at 60 (incident →
corrective entry → test → canary → ship, listed under the charts), and the held-out
phrasing block at 90.
