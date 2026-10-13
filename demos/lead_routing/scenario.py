"""Inbound lead routing.

Actions: ``route_to_ae``, ``sdr_nurture``, ``self_serve``, ``partner``, ``disqualify``.

Written policy: enterprise and mid-market with buying intent go to an account executive,
SMB goes self-serve, personal emails are disqualified, low-intent leads are nurtured.

Tacit rule the policy misses: **EMEA mid-market closes only through a regional partner**.
The AE queue loses them — time zones, local procurement, invoicing in the buyer's
language. Region is a passing mention in the lead text (a city, a timezone, a currency)
and a ``country`` field; nothing points at it.

Change half-way: a pricing update moves **NA mid-market to self-serve**. Leads that an AE
used to close now buy on the pricing page while waiting for a callback.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from core.scenario import Case, Scenario

REGION = {
    "DE": "emea", "FR": "emea", "NL": "emea", "ES": "emea", "IT": "emea", "SE": "emea", "PL": "emea", "GB": "emea",
    "US": "na", "CA": "na",
    "SG": "apac", "AU": "apac", "JP": "apac",
}

CITIES = {
    "DE": [("Munich", "CET", "€"), ("Berlin", "CET", "€"), ("Hamburg", "CET", "€")],
    "FR": [("Lyon", "CET", "€"), ("Paris", "CET", "€")],
    "NL": [("Rotterdam", "CET", "€"), ("Utrecht", "CET", "€")],
    "ES": [("Madrid", "CET", "€"), ("Barcelona", "CET", "€")],
    "IT": [("Milan", "CET", "€")],
    "SE": [("Stockholm", "CET", "SEK")],
    "PL": [("Warsaw", "CET", "PLN")],
    "GB": [("Manchester", "GMT", "£"), ("Leeds", "GMT", "£")],
    "US": [("Austin", "CST", "$"), ("Denver", "MST", "$"), ("Raleigh", "EST", "$"), ("Portland", "PST", "$"), ("Columbus", "EST", "$")],
    "CA": [("Toronto", "EST", "CAD"), ("Calgary", "MST", "CAD")],
    "SG": [("Singapore", "SGT", "SGD")],
    "AU": [("Sydney", "AEST", "AUD"), ("Melbourne", "AEST", "AUD")],
    "JP": [("Osaka", "JST", "¥")],
}

INDUSTRIES = ["logistics", "healthcare", "fintech", "manufacturing", "retail", "legal services", "education",
              "construction", "media", "insurance", "hospitality", "energy"]

FIRST = ["Anna", "Tomas", "Priya", "Marcus", "Elena", "Jonas", "Sofia", "Daniel", "Mei", "Lukas", "Hannah", "Omar"]
LAST = ["Keller", "Novak", "Rao", "Lindqvist", "Moreau", "Schmidt", "Bianchi", "Carter", "Tanaka", "Walsh", "Fischer", "Diaz"]
ROLES = ["Head of Operations", "VP Sales", "IT Director", "COO", "Procurement Lead", "RevOps Manager", "CTO"]

INTENT_LINES = {
    "demo_request": [
        "We'd like a demo for our {team} team. Budget is approved for this quarter.",
        "Looking to see the product in action — can someone walk us through it this week?",
        "Requesting a demo; we are replacing our current tool by end of quarter.",
    ],
    "pricing": [
        "Can you send pricing for roughly {seats} seats? We need a quote to take to our {role_short}.",
        "What would {seats} seats cost on an annual contract? Evaluating two vendors.",
    ],
    "trial": [
        "Started a trial yesterday and want to get the rest of the team on it.",
    ],
    "content_download": [
        "Downloaded your benchmark report; interested in the methodology.",
        "Signed up for the webinar replay. Not evaluating yet, just reading up.",
        "Grabbed the ebook for a presentation I'm putting together.",
    ],
}

# Held-out phrasings for the EMEA mid-market case after the probe episode: the same region,
# signalled by words the first ninety episodes never used.
PROBE_EMEA_CUES = [
    "We need EU data residency and a GDPR-compliant DPA before we can proceed.",
    "Please quote excluding VAT; our purchasing entity is registered in {country_name}.",
    "Our head office is in {city}; a call between 09:00 and 17:00 local time works best.",
    "Invoices must be in {currency} with a 30-day payment term under our local terms.",
]
COUNTRY_NAME = {"DE": "Germany", "FR": "France", "NL": "the Netherlands", "ES": "Spain", "IT": "Italy", "SE": "Sweden",
                "PL": "Poland", "GB": "the UK"}

TEAMS = ["ops", "sales", "support", "finance", "field"]

KIND_WEIGHTS = [
    ("mm_emea_intent", 0.32),
    ("mm_na_intent", 0.20),
    ("mm_apac_intent", 0.06),
    ("ent_intent", 0.12),
    ("smb_intent", 0.12),
    ("personal", 0.08),
    ("low_intent", 0.10),
]


def _pick_weighted(rng: random.Random, pairs: list[tuple[str, float]]) -> str:
    r = rng.random() * sum(w for _, w in pairs)
    acc = 0.0
    for name, w in pairs:
        acc += w
        if r <= acc:
            return name
    return pairs[-1][0]


def _bucket(size: int) -> str:
    if size >= 1000:
        return "enterprise"
    if size >= 100:
        return "mid_market"
    return "smb"


@dataclass
class LeadRouting(Scenario):
    name: str = "lead_routing"
    title: str = "Inbound lead routing"
    case_noun: str = "lead"
    actions: list[str] = field(default_factory=lambda: ["route_to_ae", "sdr_nurture", "self_serve", "partner", "disqualify"])
    jev_instructions: str = "Choose the routing action for this inbound lead."
    jev_criteria: dict[str, str] = field(default_factory=lambda: {
        "route_to_ae": "Assign to an account executive for a sales-led cycle.",
        "sdr_nurture": "Hand to an SDR nurture sequence; the lead is not ready to buy.",
        "self_serve": "Send to self-serve signup and the pricing page; no sales involvement.",
        "partner": "Hand to a regional channel partner or reseller who owns the relationship.",
        "disqualify": "Not a fit; close the lead.",
    })
    policy: list[tuple[str, str]] = field(default_factory=lambda: [
        ("policy-enterprise", "Enterprise leads (1000+ employees) with a corporate email and buying intent: route to an account executive."),
        ("policy-mid-market", "Mid-market leads (100-999 employees) with buying intent (demo, pricing or trial): route to an account executive."),
        ("policy-smb", "SMB leads (under 100 employees): send to self-serve signup."),
        ("policy-personal-email", "Leads from personal email addresses (gmail, yahoo, outlook, etc.): disqualify."),
        ("policy-low-intent", "Content downloads and webinar signups with no demo, pricing or trial request: SDR nurture sequence."),
    ])
    judge_rubric: str = (
        "A decision failed when the review reports a lost deal, a churned prospect, wasted sales time "
        "or a mis-qualified lead. Name the failure by the lead's segment and the action taken, using "
        "the size bucket and the region (from the country field or the city/timezone/currency in the "
        "text), for example 'mid-market EMEA lead routed to AE and lost' or 'NA mid-market lead routed "
        "to AE; bought self-serve'. Keep the phrase identical across runs that fail the same way."
    )
    cost_by_severity: dict[str, float] = field(default_factory=lambda: {
        "minor_failure": 200.0, "failure": 2500.0, "critical_failure": 15000.0,
    })
    tacit_rule: str = "EMEA mid-market leads close only through a regional partner; the AE queue loses them."
    change_rule: str = "After the pricing update, NA mid-market leads buy self-serve; routing them to an AE loses the deal."
    expected_fix: str = "Mid-market leads in EMEA: hand to a regional partner, not an account executive."

    # -- cases ---------------------------------------------------------------------------

    def make_case(self, episode: int, rng: random.Random, *, changed: bool, probe: bool) -> Case:
        kind = _pick_weighted(rng, KIND_WEIGHTS)
        country: str
        size: int
        intent: str
        email_type = "corporate"
        if kind == "mm_emea_intent":
            country = rng.choice(["DE", "FR", "NL", "ES", "IT", "SE", "PL", "GB"])
            size = rng.randint(120, 950)
            intent = rng.choice(["demo_request", "pricing", "pricing", "trial"])
        elif kind == "mm_na_intent":
            country = rng.choice(["US", "US", "US", "CA"])
            size = rng.randint(120, 950)
            intent = rng.choice(["demo_request", "pricing", "trial"])
        elif kind == "mm_apac_intent":
            country = rng.choice(["SG", "AU", "JP"])
            size = rng.randint(120, 950)
            intent = rng.choice(["demo_request", "pricing"])
        elif kind == "ent_intent":
            country = rng.choice(["US", "US", "DE", "GB", "CA", "FR"])
            size = rng.randint(1100, 12000)
            intent = rng.choice(["demo_request", "pricing"])
        elif kind == "smb_intent":
            country = rng.choice(["US", "US", "DE", "GB", "NL", "CA", "AU"])
            size = rng.randint(4, 95)
            intent = rng.choice(["demo_request", "pricing", "trial"])
        elif kind == "personal":
            country = rng.choice(["US", "DE", "GB", "FR", "CA"])
            size = rng.choice([rng.randint(5, 90), rng.randint(120, 900), rng.randint(1100, 5000)])
            intent = rng.choice(["demo_request", "pricing", "trial", "content_download"])
            email_type = "personal"
        else:  # low_intent
            country = rng.choice(["US", "US", "DE", "GB", "FR", "CA", "AU"])
            size = rng.choice([rng.randint(120, 900), rng.randint(1100, 5000)])
            intent = "content_download"

        bucket = _bucket(size)
        region = REGION[country]
        truth = self._truth(kind, bucket, region, email_type, intent, changed)
        is_tacit = kind == "mm_emea_intent"
        is_changed = kind == "mm_na_intent" and changed
        use_probe = probe and is_tacit
        text = self._text(rng, country, size, intent, email_type, probe=use_probe)
        if email_type == "personal":
            situation = "personal-email"
        elif intent == "content_download":
            situation = f"low-intent/{bucket}"
        else:
            situation = f"{bucket}/{region}"
        fields = {
            "company_size": size, "size_bucket": bucket, "email_type": email_type,
            "intent": intent, "country": country, "industry": rng.choice(INDUSTRIES),
        }
        return Case(
            id="", episode=episode, text=text, fields=fields, situation=situation, truth=truth,
            kind=kind, is_tacit=is_tacit, is_changed=is_changed, is_probe=use_probe,
        )

    @staticmethod
    def _truth(kind: str, bucket: str, region: str, email_type: str, intent: str, changed: bool) -> str:
        if email_type == "personal":
            return "disqualify"
        if intent == "content_download":
            return "sdr_nurture"
        if bucket == "smb":
            return "self_serve"
        if bucket == "enterprise":
            return "route_to_ae"
        # mid-market with intent
        if region == "emea":
            return "partner"
        if region == "na" and changed:
            return "self_serve"
        return "route_to_ae"

    def _text(self, rng: random.Random, country: str, size: int, intent: str, email_type: str, *, probe: bool) -> str:
        city, tz, cur = rng.choice(CITIES[country])
        first, last = rng.choice(FIRST), rng.choice(LAST)
        role = rng.choice(ROLES)
        industry = rng.choice(INDUSTRIES)
        domain = f"{last.lower()}-{rng.choice(['group', 'systems', 'labs', 'co', 'partners'])}.{rng.choice(['com', 'io', 'net'])}"
        if email_type == "personal":
            domain = rng.choice(["gmail.com", "yahoo.com", "outlook.com", "icloud.com"])
        email = f"{first.lower()}.{last.lower()}@{domain}"
        size_line = rng.choice([
            f"We are a {size}-person {industry} company",
            f"We're a {industry} business with about {size} employees",
            f"{industry.capitalize()} firm, roughly {size} staff",
        ])
        intent_line = rng.choice(INTENT_LINES[intent]).format(
            team=rng.choice(TEAMS), seats=max(5, size // rng.choice([3, 4, 6])), role_short=rng.choice(["CFO", "board", "COO"]),
        )
        if probe:
            cue = rng.choice(PROBE_EMEA_CUES).format(
                country_name=COUNTRY_NAME.get(country, country), city=city, currency=cur,
            )
            place = ""
        else:
            cue = ""
            place = rng.choice([
                f" based in {city}", f" headquartered in {city}", f" ({city}, {tz})",
                f"; we work {tz} hours", f" — budget is in {cur}",
            ])
        parts = [f"{size_line}{place}.", intent_line]
        if cue:
            parts.append(cue)
        parts.append(f"— {first} {last}, {role} ({email})")
        return " ".join(parts)

    # -- review --------------------------------------------------------------------------

    def severity_of(self, case: Case, action: str) -> str:
        truth = case.truth
        bucket = case.fields["size_bucket"]
        if truth == "route_to_ae" and bucket == "enterprise" and action in ("self_serve", "disqualify"):
            return "critical_failure"
        if truth == "partner" and action == "disqualify":
            return "critical_failure"
        if truth in ("partner", "route_to_ae"):
            return "failure"
        if truth == "self_serve":
            # SMB routed to sales wastes time; a post-change NA mid-market lead routed to an AE loses the deal.
            return "failure" if case.is_changed else "minor_failure"
        if truth == "sdr_nurture":
            return "failure" if action == "disqualify" else "minor_failure"
        return "minor_failure"  # disqualify truth: time wasted on a non-lead

    def failure_note(self, case: Case, action: str, rng: random.Random) -> str:
        truth = case.truth
        days = rng.randint(18, 61)
        if truth == "partner":
            if action == "route_to_ae":
                return rng.choice([
                    f"Closed-lost after {days} days in the AE queue; buyer signed with a regional reseller.",
                    f"AE could not land a meeting across time zones; prospect went quiet after {days} days.",
                    "Prospect told the AE they purchase through a local vendor with in-country invoicing; closed-lost.",
                    f"Deal stalled {days} days on contract terms in the buyer's jurisdiction; lost to a local competitor.",
                ])
            if action == "self_serve":
                return "Prospect did not complete signup; said they need a local contract and an invoice in their currency."
            if action == "sdr_nurture":
                return f"Prospect bought from a competitor with local presence while in the nurture sequence ({days} days)."
            return "Qualified mid-market buyer disqualified; the account later appeared in a competitor's case study."
        if truth == "route_to_ae":
            if case.fields["size_bucket"] == "enterprise":
                if action in ("self_serve", "disqualify"):
                    return "Enterprise prospect dropped at trial; procurement required a sales process and security review."
                if action == "partner":
                    return "Enterprise prospect declined to work through a partner; wanted a direct vendor relationship."
                return f"Enterprise prospect sat in nurture for {days} days and chose another vendor."
            # mid-market NA/APAC before the change
            if action == "self_serve":
                return "Mid-market prospect abandoned checkout; asked for a sales conversation that never happened."
            if action == "partner":
                return "Prospect was confused by the partner handoff and re-submitted the form; slow start, deal lost."
            if action == "sdr_nurture":
                return f"Prospect was ready to buy; nurture sequence ran {days} days and they went elsewhere."
            return "Qualified buyer disqualified by mistake."
        if truth == "self_serve":
            if case.is_changed:
                return rng.choice([
                    f"Prospect bought on the pricing page before the AE called back; {rng.randint(2, 5)} AE hours logged.",
                    "Prospect said the new plan was all they needed and bought online; the AE cycle added a week and no value.",
                ])
            return rng.choice([
                f"AE spent {rng.randint(2, 4)} hours on a {max(5, case.fields['company_size'] // 3)}-seat deal that closed at list price anyway.",
                "Small team; the sales cycle slowed down a signup that would have happened on its own.",
            ])
        if truth == "sdr_nurture":
            if action == "disqualify":
                return "Lead disqualified; later converted via a colleague's referral — pipeline missed."
            return "Not evaluating yet; asked to be contacted next year. AE time wasted."
        # truth == disqualify
        return rng.choice([
            "Student project, no budget; 40 minutes of sales time spent.",
            "Personal side project; could not identify a company or budget.",
        ])

    def success_note(self, case: Case, action: str, rng: random.Random) -> str:
        if action == "route_to_ae":
            return f"Closed-won in {rng.randint(21, 70)} days."
        if action == "partner":
            return f"Partner closed the deal in {rng.randint(14, 45)} days."
        if action == "self_serve":
            return "Signed up and activated within a week."
        if action == "sdr_nurture":
            return "Engaged with the sequence; booked a call later in the quarter."
        return "Correctly disqualified; no further action."


SCENARIO = LeadRouting()

__all__ = ["SCENARIO", "LeadRouting"]
