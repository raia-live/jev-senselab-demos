# AP invoice exceptions

Jev reads an incoming invoice (vendor, amount, PO status, a short note from AP) and picks one
of five actions: `auto_approve`, `route_to_approver`, `hold_for_po_match`,
`request_vendor_correction`, `escalate_fraud_review`. AP reports later whether the invoice
was paid cleanly, paid late, bounced, or turned out to be fraud.

```bash
uv run python -m core.runner --scenario ap_invoices
uv run python -m core.runner --scenario ap_invoices --quick
```

## What the written policy says

| rule | action |
|---|---|
| Matches an open PO within 2% | auto-approve |
| No PO, over $5,000 | budget approver |
| No PO, at or under $5,000, existing vendor | auto-approve |
| Exceeds the PO by more than 2% | hold for PO match |
| New vendor, or bank details changed | fraud review |
| Missing tax ID / invoice number / legal entity | request a corrected invoice |

## What the policy misses

**Recurring service invoices over $5k (utilities, telecom, leases, subscriptions) carry no
PO and must be paid on receipt.** Sending them to the approver queue makes them late, and
late means service interruptions and reconnection fees. The policy's "no PO over $5k →
approver" rule catches exactly these. Thirty percent of the stream.

Half-way through, an audit finding changes the rules: standard-tier vendors' PO-matched
invoices over $25k now need approver sign-off. Auto-approving them, which the policy says
to do, becomes an audit exception.

From episode 90 the recurring invoices arrive with new wording in the AP note.

## What the review tells the agent

Outcome plus a note such as "budget owner asked why a utility bill was in their queue; paid
after the due date", never the action that would have been right. 10% of reviews are
flipped at random.

## What to look for in the report

The same three things as in every demo: the tacit rule learned from outcomes, the response
to the change at 60 (incident → corrective entry → test → canary → ship, listed under the
charts), and the held-out phrasing block at 90.
