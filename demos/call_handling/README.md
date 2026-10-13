# Support call handling

Jev reads what a support caller said (free text plus the account's fields) and picks what the
bot does next: `resolve_in_bot`, `transfer_to_agent`, `schedule_callback`, `issue_credit`. A
review reports later how it turned out: a callback about the same issue, a cancellation, a
caller who hung up in a queue, a credit finance reversed, or nothing, which is the good one.

This scenario drives both the batch harness and the live voice agent in [`voice/`](../../voice):
same actions, same policy, same situation labels, same review. A live call and a batch case
with the same facts share memory.

```bash
uv run python -m core.runner --scenario call_handling            # the full example
uv run python -m core.runner --scenario call_handling --quick    # a short look
```

## What the written policy says

| rule | action |
|---|---|
| Billing questions and disputes under $200 | resolve in the bot |
| Billing disputes of $200 or more | transfer to a billing agent |
| Cancellation requests | transfer to the retention team |
| Outage reports, upgrades, how-to | resolve in the bot |
| Enterprise accounts, anything but a how-to | transfer to the enterprise desk |

## What the policy misses

A caller who has already contacted support twice in the last 30 days about a billing dispute
is not helped by another bot resolution. They call back again, and then they cancel. A
scheduled callback from a billing specialist closes it. The repeat shows up as a passing
mention in what the caller says ("this is the third time I'm calling") and a
`prior_contacts_30d` field; nothing in the policy points at it. About 30% of the stream is
this caller.

Half-way through the run a self-serve retention offer goes live in the bot. Consumer
cancellations the bot used to transfer are now kept by the offer, and the retention queue
it used to send them to runs a forty-minute wait that callers do not sit through. The rule
that was right for 60 calls becomes wrong, and more people call to cancel for a while after
the launch is announced.

From episode 90 the repeat callers say it differently ("the agent I spoke to on the 12th
promised a refund"), without the words the earlier calls used.

## What the review tells the agent

Only the outcome and a one-line note, never the right answer. "Called again after 4 days,
same dispute; cancelled on that call" is a failure note; nothing says `schedule_callback`.
Transferring a repeat caller is a minor failure (a human fixes it, in half an hour of queue
time), resolving them in the bot is the expensive one. Resolving an enterprise account in the
bot is the critical one.

## The costs

| what the review reports | cost |
|---|---:|
| agent time on a call the bot could have closed, or a credit reversed | $35 |
| the caller rang back and cancelled, or hung up in a queue | $600 |
| an enterprise account handled by the bot | $5,000 |
