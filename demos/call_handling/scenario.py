"""Support call handling: what the voice agent does at the decision point of a call.

Actions: ``resolve_in_bot``, ``transfer_to_agent``, ``schedule_callback``, ``issue_credit``.

Written policy: billing disputes under $200 and how-to, outage and upgrade calls are resolved
by the bot; disputes of $200 or more, cancellations and enterprise accounts go to a human.

Tacit rule the policy misses: a caller who has already contacted support twice in the last
30 days about a billing dispute is not helped by another bot resolution. They call back
again and cancel. A scheduled callback from a billing specialist closes it. The repeat is
a passing mention in what the caller says ("third time I'm calling") and a
``prior_contacts_30d`` field; nothing points at it.

Change half-way: a self-serve retention offer goes live in the bot. Consumer cancellations
the bot used to have to transfer are now kept by the bot's offer, and the retention queue
it used to transfer them to runs a 40-minute wait that callers do not sit through.

The same scenario drives the batch harness (``python -m core.runner --scenario
call_handling``) and the live Vapi agent in ``voice/``: the live server builds the same
state from the call and uses the same situation label, so what the batch run learned is
what the live agent knows.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from core.scenario import Case, PolicyClaim, Scenario

TIERS = ["consumer", "business", "enterprise"]

KIND_WEIGHTS = [
    ("billing_repeat", 0.30),      # the tacit rule
    ("cancel_consumer", 0.20),     # the change
    ("billing_first_small", 0.12),
    ("billing_large", 0.08),
    ("outage", 0.10),
    ("upgrade", 0.08),
    ("how_to", 0.07),
    ("enterprise_other", 0.05),
]

OPENERS = {
    "billing_dispute": [
        "There's a charge of ${amount} on my latest invoice that I don't recognise.",
        "I was billed ${amount} for something I cancelled; I want it fixed.",
        "My bill went up by ${amount} this month and nobody told me why.",
        "You charged me ${amount} twice for the same period.",
    ],
    "cancellation": [
        "I want to cancel my subscription, effective today.",
        "Please cancel my account; I'm not using it enough to justify the price.",
        "I'd like to close my account. How do I do that?",
    ],
    "outage": [
        "Nothing is loading on my end, is the service down?",
        "I've been getting errors for the last hour; is there an outage?",
    ],
    "upgrade": [
        "I want to move to the next plan up; can you do that on this call?",
        "How do I add more seats to my plan?",
    ],
    "how_to": [
        "How do I export my data to a spreadsheet?",
        "Where do I change the email address on the account?",
        "Can you walk me through setting up two-factor authentication?",
    ],
}

# How a repeat caller mentions the repeat during the first ninety episodes...
REPEAT_CUES = [
    "This is the third time I'm calling about this.",
    "I already called twice about this and it's still not fixed.",
    "I've been on the phone with you people twice this month about the same thing.",
    "Second call this week; the last person said it was sorted.",
]
# ...and the held-out ways after the probe episode.
PROBE_REPEAT_CUES = [
    "I have a ticket number from last week and another one from the week before.",
    "The agent I spoke to on the {day} promised a fix; here we are again.",
    "We've been through this already, more than once; please check my history.",
    "I was told last time to wait a billing cycle. I did. Same charge.",
]
DAYS = ["3rd", "9th", "12th", "17th", "21st"]

SENTIMENT = ["calm", "frustrated", "frustrated", "angry"]


def _pick_weighted(rng: random.Random, pairs: list[tuple[str, float]]) -> str:
    r = rng.random() * sum(w for _, w in pairs)
    acc = 0.0
    for name, w in pairs:
        acc += w
        if r <= acc:
            return name
    return pairs[-1][0]


def situation_of(fields: dict[str, Any]) -> str:
    """The situation label for a call, from the fields the live server also has. Shared
    with ``voice/`` so a live call and a batch case with the same facts share memory."""
    intent = str(fields.get("intent"))
    tier = str(fields.get("tier"))
    if intent == "billing_dispute":
        repeat = "repeat" if int(fields.get("prior_contacts_30d") or 0) >= 2 else "first"
        size = "large" if float(fields.get("amount_usd") or 0) >= 200 else "small"
        return f"billing_dispute/{tier}/{repeat}/{size}"
    return f"{intent}/{tier}"


def truth_of(fields: dict[str, Any], *, changed: bool) -> str:
    """The hidden right action. Used for scoring; never shown to any arm."""
    intent = str(fields.get("intent"))
    tier = str(fields.get("tier"))
    if tier == "enterprise" and intent != "how_to":
        return "transfer_to_agent"
    if intent == "billing_dispute":
        if float(fields.get("amount_usd") or 0) >= 200:
            return "transfer_to_agent"
        if int(fields.get("prior_contacts_30d") or 0) >= 2:
            return "schedule_callback"
        return "resolve_in_bot"
    if intent == "cancellation":
        if tier == "consumer" and changed:
            return "resolve_in_bot"
        return "transfer_to_agent"
    return "resolve_in_bot"


@dataclass
class CallHandling(Scenario):
    name: str = "call_handling"
    title: str = "Support call handling"
    case_noun: str = "call"
    actions: list[str] = field(default_factory=lambda: ["resolve_in_bot", "transfer_to_agent", "schedule_callback", "issue_credit"])
    jev_instructions: str = "Choose what the support agent does next on this call."
    jev_criteria: dict[str, str] = field(default_factory=lambda: {
        "resolve_in_bot": "Handle it on this call: explain, adjust, apply the standard offer or walk the caller through it.",
        "transfer_to_agent": "Hand the call to a human agent now.",
        "schedule_callback": "Book a callback from a specialist within one business day and end the call.",
        "issue_credit": "Apply an account credit for the disputed amount on this call.",
    })
    policy: list[tuple[str, str]] = field(default_factory=lambda: [
        ("policy-enterprise", "Enterprise accounts: transfer to the enterprise desk for anything other than a how-to question."),
        ("policy-billing-small", "Billing questions and disputes under $200: resolve in the bot. Explain the charge and adjust it if it is clearly wrong."),
        ("policy-billing-large", "Billing disputes of $200 or more: transfer to a billing agent."),
        ("policy-cancellation", "Cancellation requests: transfer to the retention team."),
        ("policy-outage", "Outage and service-down reports: resolve in the bot with the status page information and the current ETA."),
        ("policy-upgrade-howto", "Upgrades, plan changes and how-to questions: resolve in the bot."),
    ])
    policy_claims: list[PolicyClaim] = field(default_factory=lambda: [
        PolicyClaim("policy-enterprise", "transfer_to_agent",
                    ("billing_dispute/enterprise/*", "cancellation/enterprise", "outage/enterprise", "upgrade/enterprise")),
        PolicyClaim("policy-billing-small", "resolve_in_bot", ("billing_dispute/*/*/small",)),
        PolicyClaim("policy-billing-large", "transfer_to_agent", ("billing_dispute/*/*/large",)),
        PolicyClaim("policy-cancellation", "transfer_to_agent", ("cancellation/*",)),
        PolicyClaim("policy-outage", "resolve_in_bot", ("outage/*",)),
        PolicyClaim("policy-upgrade-howto", "resolve_in_bot", ("upgrade/*", "how_to/*")),
    ])
    judge_rubric: str = (
        "A decision failed when the review reports a callback about the same issue, a cancellation, "
        "a caller who hung up in a transfer queue, a credit reversed by finance, or agent time spent on "
        "a call the bot could have closed. Name the failure by the intent, the tier, whether the caller "
        "was a repeat caller, and the action taken, for example 'repeat billing dispute resolved in bot; "
        "called back and cancelled' or 'consumer cancellation transferred; hung up in queue'. Keep the "
        "phrase identical across runs that fail the same way."
    )
    cost_by_severity: dict[str, float] = field(default_factory=lambda: {
        "minor_failure": 35.0, "failure": 600.0, "critical_failure": 5000.0,
    })
    tacit_rule: str = ("Repeat callers (two or more contacts in 30 days) about a billing dispute are not helped by another bot "
                       "resolution; they call back and cancel. A specialist callback closes it.")
    change_rule: str = ("After the self-serve retention offer goes live, consumer cancellations are kept by the bot; transferring "
                        "them to the retention queue loses them.")
    expected_fix: str = "Billing disputes from repeat callers: schedule a specialist callback instead of resolving in the bot."

    # -- cases ---------------------------------------------------------------------------

    def classify(self, fields: dict[str, Any], *, changed: bool) -> dict[str, Any]:
        """The hidden right answer and the case flags for a live call's fields, so the
        voice agent's decisions are reviewed the way the batch harness reviews a case."""
        intent, tier = str(fields.get("intent")), str(fields.get("tier"))
        prior, amount = int(fields.get("prior_contacts_30d") or 0), float(fields.get("amount_usd") or 0)
        return {
            "truth": truth_of(fields, changed=changed),
            "is_tacit": intent == "billing_dispute" and prior >= 2 and amount < 200 and tier != "enterprise",
            "is_changed": intent == "cancellation" and tier == "consumer" and changed,
        }

    def make_case(self, episode: int, rng: random.Random, *, changed: bool, probe: bool) -> Case:
        weights = KIND_WEIGHTS
        if changed:
            # The retention offer launch is announced; more people call to cancel for a while.
            weights = [(k, 0.30 if k == "cancel_consumer" else w) for k, w in KIND_WEIGHTS]
        kind = _pick_weighted(rng, weights)
        tier = "consumer"
        prior = rng.choice([0, 0, 0, 1])
        amount = 0.0
        if kind == "billing_repeat":
            intent = "billing_dispute"
            tier = rng.choice(["consumer", "consumer", "business"])
            prior = rng.choice([2, 2, 3])
            amount = float(rng.choice([19, 29, 49, 84, 120, 149, 180]))
        elif kind == "cancel_consumer":
            intent = "cancellation"
            tier = "consumer"
        elif kind == "billing_first_small":
            intent = "billing_dispute"
            tier = rng.choice(["consumer", "business"])
            prior = rng.choice([0, 0, 1])
            amount = float(rng.choice([12, 29, 49, 84, 120, 160]))
        elif kind == "billing_large":
            intent = "billing_dispute"
            tier = rng.choice(["consumer", "business", "business"])
            prior = rng.choice([0, 1, 2])
            amount = float(rng.choice([240, 390, 620, 1150]))
        elif kind == "outage":
            intent = "outage"
            tier = rng.choice(["consumer", "business"])
        elif kind == "upgrade":
            intent = "upgrade"
            tier = rng.choice(["consumer", "business"])
        elif kind == "how_to":
            intent = "how_to"
            tier = rng.choice(TIERS)
        else:  # enterprise_other
            intent = rng.choice(["billing_dispute", "cancellation", "outage", "upgrade"])
            tier = "enterprise"
            if intent == "billing_dispute":
                amount = float(rng.choice([180, 420, 980, 2400]))
                prior = rng.choice([0, 1, 2])
        fields = {
            "intent": intent, "tier": tier, "prior_contacts_30d": prior, "amount_usd": amount,
            "tenure_months": rng.randint(2, 60), "sentiment": rng.choice(SENTIMENT), "channel": "voice",
        }
        truth = truth_of(fields, changed=changed)
        is_tacit = kind == "billing_repeat"
        is_changed = kind == "cancel_consumer" and changed
        use_probe = probe and is_tacit
        text = self._text(rng, fields, probe=use_probe)
        return Case(
            id="", episode=episode, text=text, fields=fields, situation=situation_of(fields), truth=truth,
            kind=kind, is_tacit=is_tacit, is_changed=is_changed, is_probe=use_probe,
        )

    def _text(self, rng: random.Random, fields: dict[str, Any], *, probe: bool) -> str:
        intent = fields["intent"]
        opener = rng.choice(OPENERS[intent]).format(amount=int(fields["amount_usd"]) if fields["amount_usd"] else "")
        parts = [f"Caller on a {fields['tier']} account, {fields['tenure_months']} months with us, sounds {fields['sentiment']}.",
                 f'Caller: "{opener}"']
        if fields["prior_contacts_30d"] >= 2 and intent == "billing_dispute":
            cue = rng.choice(PROBE_REPEAT_CUES).format(day=rng.choice(DAYS)) if probe else rng.choice(REPEAT_CUES)
            parts.append(f'Caller adds: "{cue}"')
        elif fields["prior_contacts_30d"] == 1 and rng.random() < 0.5:
            parts.append('Caller adds: "I did mention this to someone a few weeks back."')
        return " ".join(parts)

    # -- review --------------------------------------------------------------------------

    def severity_of(self, case: Case, action: str) -> str:
        truth, f = case.truth, case.fields
        if f["tier"] == "enterprise" and truth == "transfer_to_agent" and action in ("resolve_in_bot", "issue_credit"):
            return "critical_failure"
        if truth == "schedule_callback":
            # the repeat billing caller
            return "minor_failure" if action in ("transfer_to_agent", "issue_credit") else "failure"
        if truth == "transfer_to_agent":
            return "minor_failure" if action == "schedule_callback" else "failure"
        # truth == resolve_in_bot
        if case.is_changed:
            return "failure" if action == "transfer_to_agent" else "minor_failure"
        return "failure" if action == "issue_credit" else "minor_failure"

    def failure_note(self, case: Case, action: str, rng: random.Random) -> str:
        truth, f = case.truth, case.fields
        days = rng.randint(1, 6)
        if truth == "schedule_callback":
            if action == "resolve_in_bot":
                return rng.choice([
                    f"CSAT 1/5. Called back {days} days later about the same charge, then cancelled.",
                    f"Called again after {days} days, same dispute; cancelled on that call.",
                    "Post-call survey: 'nobody fixes anything'. Account closed within the week.",
                ])
            if action == "transfer_to_agent":
                return f"Human closed it in {rng.randint(14, 32)} minutes; CSAT 4/5. Half an hour of queue time on a billing lookup."
            return "Credit applied; finance reversed it as unsupported. Caller phoned again the following week."
        if truth == "transfer_to_agent":
            if f["tier"] == "enterprise":
                if action in ("resolve_in_bot", "issue_credit"):
                    return "Enterprise account manager escalated: their contract requires a named person on every call. Renewal at risk."
                return "Enterprise caller was told to wait a day; the account manager complained the same afternoon."
            if f["intent"] == "cancellation":
                if action == "resolve_in_bot":
                    return rng.choice([
                        "Bot read the retention script; caller cancelled online an hour later.",
                        "Caller said the offer was not what they asked for and cancelled anyway. CSAT 2/5.",
                    ])
                if action == "schedule_callback":
                    return "Callback booked; caller had cancelled online before it happened."
                return "Credit applied; caller still cancelled the next day. Credit written off."
            # large billing dispute
            if action == "resolve_in_bot":
                return f"Caller disputed ${int(f['amount_usd'])} again with their bank; chargeback filed."
            if action == "schedule_callback":
                return "Caller did not wait for the callback; filed a chargeback the same day."
            return f"${int(f['amount_usd'])} credit applied without review; finance flagged it."
        # truth == resolve_in_bot
        if case.is_changed:
            if action == "transfer_to_agent":
                return rng.choice([
                    f"Hung up after {rng.randint(22, 41)} minutes in the retention queue; cancelled online that evening.",
                    "Transferred to retention; abandoned the queue and cancelled from the app.",
                ])
            if action == "schedule_callback":
                return "Callback booked; the caller had already cancelled online by then."
            return "Credit applied; caller cancelled anyway. Credit written off."
        if action == "transfer_to_agent":
            return f"Agent spent {rng.randint(8, 19)} minutes on a question the bot answers; caller fine."
        if action == "schedule_callback":
            return "Caller annoyed at waiting a day for something simple; CSAT 2/5."
        return f"${int(f['amount_usd']) if f['amount_usd'] else 25} credit applied with no dispute on record; finance reversed it."

    def success_note(self, case: Case, action: str, rng: random.Random) -> str:
        if action == "resolve_in_bot":
            return rng.choice(["Resolved on the call; CSAT 5/5; no further contact in 30 days.",
                               "Closed in the bot; no callback in 30 days."])
        if action == "transfer_to_agent":
            return f"Agent closed it in {rng.randint(6, 18)} minutes; CSAT 4/5."
        if action == "schedule_callback":
            return "Specialist called back next morning; CSAT 5/5; no further contact in 30 days."
        return "Credit stood; caller satisfied; no further contact."

    def noise_fail_note(self, case: Case, action: str, rng: random.Random) -> str:
        return rng.choice([
            "CSAT 1/5 left on the survey; the transcript shows the issue was handled.",
            "Caller phoned back about an unrelated charge; flagged as a repeat by the dashboard.",
            "Marked a miss by QA; no reason recorded.",
        ])


SCENARIO = CallHandling()

__all__ = ["SCENARIO", "CallHandling", "situation_of", "truth_of"]
