"""Accounts-payable invoice exceptions.

Actions: ``auto_approve``, ``route_to_approver``, ``hold_for_po_match``,
``request_vendor_correction``, ``escalate_fraud_review``.

Written policy: PO-matched invoices within tolerance are approved automatically; invoices
with no PO over $5k go to a budget approver; small no-PO invoices from known vendors are
approved; PO mismatches are held; a new vendor changing bank details is a fraud review.

Tacit rule the policy misses: **recurring service invoices (utilities, telecom, leases,
SaaS subscriptions) over $5k never carry a PO and must be paid on receipt**. Routing them
to an approver makes them late: late fees, service interruptions, lost early-payment
discounts. The only signals are the vendor's name, a line item like "monthly service
charge" and the ``invoice_type`` field.

Change half-way: an audit finding adds a control — **standard-tier vendors' PO-matched
invoices over $25k now need approver sign-off** even when the match is clean.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from core.scenario import Case, Scenario

RECURRING_VENDORS = [
    ("Northgrid Power & Light", "utilities", "monthly electricity service charge"),
    ("Metro Fibre Telecom", "telecom", "monthly circuit and voice service"),
    ("Harbourview Properties", "lease", "monthly office lease — suite 400"),
    ("Cloudline Infrastructure", "saas", "monthly platform subscription — enterprise plan"),
    ("Civic Water Authority", "utilities", "quarterly water and sewer service"),
    ("Stackwise Software", "saas", "annual licence renewal, 240 seats"),
    ("Parkside Facilities Group", "lease", "monthly parking and facilities charge"),
]
ONEOFF_VENDORS = [
    ("Beacon Consulting Partners", "consulting", "project discovery phase — milestone 1"),
    ("Redline Print & Signage", "marketing", "trade show booth graphics"),
    ("Orbit Logistics", "freight", "inbound freight, 3 pallets"),
    ("Tenfold Staffing", "staffing", "contract staffing — 160 hours"),
    ("Summit Office Supply", "supplies", "office supplies, Q3 restock"),
    ("Granite Legal LLP", "legal", "contract review — 14.5 hours"),
    ("Kestrel Industrial", "equipment", "replacement compressor unit"),
    ("Vantage Media Buying", "marketing", "paid media, September flight"),
]
NEW_VENDORS = [
    ("Apex Global Trading Ltd", "equipment"), ("BlueOak Services LLC", "consulting"),
    ("Nordic Tech Supply AB", "equipment"), ("Primeline Solutions Inc", "staffing"),
]
APPROVERS = ["R. Okafor (Facilities)", "L. Haddad (IT)", "M. Sørensen (Marketing)", "J. Park (Operations)"]
CURRENCIES = ["USD", "USD", "USD", "EUR", "GBP"]

# Held-out phrasings for the recurring-service case after the probe episode: the same kind
# of invoice, described in words the first ninety episodes never used.
PROBE_RECURRING_LINES = [
    "Statement period {period}; account number ending {acct}. Service address on file.",
    "Usage-based charges for the billing cycle ending {period}; meter and circuit IDs attached.",
    "Scheduled instalment under master services agreement MSA-{acct}; no change to terms.",
    "Standing charge plus consumption for {period}, per the tariff on file.",
]

KIND_WEIGHTS = [
    ("recurring_nopo_over5k", 0.30),   # tacit: pay on receipt
    ("po_match_std_large", 0.18),      # changes at the audit finding
    ("po_match_small", 0.14),
    ("nopo_small_existing", 0.10),
    ("po_mismatch", 0.12),
    ("new_vendor_bank_change", 0.08),
    ("nopo_large_oneoff", 0.08),
]


def _pick_weighted(rng: random.Random, pairs: list[tuple[str, float]]) -> str:
    r = rng.random() * sum(w for _, w in pairs)
    acc = 0.0
    for name, w in pairs:
        acc += w
        if r <= acc:
            return name
    return pairs[-1][0]


def _amount_band(amount: float) -> str:
    if amount > 25000:
        return "over-25k"
    if amount > 5000:
        return "5k-25k"
    return "under-5k"


@dataclass
class ApInvoices(Scenario):
    name: str = "ap_invoices"
    title: str = "AP invoice exceptions"
    case_noun: str = "invoice"
    actions: list[str] = field(default_factory=lambda: [
        "auto_approve", "route_to_approver", "hold_for_po_match", "request_vendor_correction", "escalate_fraud_review",
    ])
    jev_instructions: str = "Choose how accounts payable should handle this invoice."
    jev_criteria: dict[str, str] = field(default_factory=lambda: {
        "auto_approve": "Approve for payment now, without further sign-off.",
        "route_to_approver": "Send to the budget owner for approval before payment.",
        "hold_for_po_match": "Hold until the purchase order is matched or amended.",
        "request_vendor_correction": "Return to the vendor for a corrected invoice.",
        "escalate_fraud_review": "Escalate to the fraud review team; do not pay.",
    })
    policy: list[tuple[str, str]] = field(default_factory=lambda: [
        ("policy-po-match", "Invoices that match an open purchase order within 2% tolerance: approve automatically."),
        ("policy-no-po-large", "Invoices with no purchase order above $5,000: route to the budget approver."),
        ("policy-no-po-small", "Invoices with no purchase order at or under $5,000 from an existing vendor: approve automatically."),
        ("policy-po-mismatch", "Invoices that exceed the purchase order by more than 2%: hold for PO match."),
        ("policy-bank-change", "A new vendor, or a vendor whose bank details changed on this invoice: escalate to fraud review."),
        ("policy-defective", "Invoices missing a tax ID, invoice number or legal entity name: request a corrected invoice."),
    ])
    judge_rubric: str = (
        "A decision failed when the review reports a late fee, a service interruption, a lost discount, a "
        "duplicate or fraudulent payment, an audit exception or an unnecessary delay. Name the failure by the "
        "invoice's kind and the action taken, using the invoice_type, po_status, amount band and vendor_tier "
        "fields, for example 'recurring service invoice over $5k with no PO routed to approver; paid late' or "
        "'standard-tier PO-matched invoice over $25k auto-approved; audit exception'. Keep the phrase identical "
        "across runs that fail the same way."
    )
    cost_by_severity: dict[str, float] = field(default_factory=lambda: {
        "minor_failure": 150.0, "failure": 1800.0, "critical_failure": 25000.0,
    })
    tacit_rule: str = ("Recurring service invoices (utilities, telecom, leases, subscriptions) over $5k carry no PO and "
                       "must be paid on receipt; the approver queue makes them late.")
    change_rule: str = ("After the audit finding, standard-tier vendors' PO-matched invoices over $25k need approver "
                        "sign-off; auto-approving them is an audit exception.")
    expected_fix: str = "Recurring service invoices with no PO: approve on receipt, do not route to an approver."

    # -- cases ---------------------------------------------------------------------------

    def make_case(self, episode: int, rng: random.Random, *, changed: bool, probe: bool) -> Case:
        kind = _pick_weighted(rng, KIND_WEIGHTS)
        tier = "standard"
        po_status = "none"
        invoice_type = "one_off"
        bank_change = False
        defective = False
        category = ""
        if kind == "recurring_nopo_over5k":
            vendor, category, line = rng.choice(RECURRING_VENDORS)
            amount = round(rng.uniform(5200, 48000), 2)
            invoice_type = "recurring"
            tier = rng.choice(["strategic", "standard", "standard"])
        elif kind == "po_match_std_large":
            vendor, category, line = rng.choice(ONEOFF_VENDORS)
            amount = round(rng.uniform(26000, 140000), 2)
            po_status = "match"
            tier = "standard"
        elif kind == "po_match_small":
            vendor, category, line = rng.choice(ONEOFF_VENDORS)
            amount = round(rng.uniform(400, 24000), 2)
            po_status = "match"
            tier = rng.choice(["strategic", "standard"])
        elif kind == "nopo_small_existing":
            vendor, category, line = rng.choice(ONEOFF_VENDORS)
            amount = round(rng.uniform(120, 4900), 2)
        elif kind == "po_mismatch":
            vendor, category, line = rng.choice(ONEOFF_VENDORS)
            amount = round(rng.uniform(2000, 60000), 2)
            po_status = "over"
        elif kind == "new_vendor_bank_change":
            vendor, category = rng.choice(NEW_VENDORS)
            line = rng.choice(["initial order — equipment and setup", "consulting retainer — month 1", "deposit on order"])
            amount = round(rng.uniform(3000, 90000), 2)
            tier = "new"
            bank_change = True
            po_status = rng.choice(["none", "match"])
        else:  # nopo_large_oneoff
            vendor, category, line = rng.choice(ONEOFF_VENDORS)
            amount = round(rng.uniform(5200, 70000), 2)
            defective = rng.random() < 0.3

        band = _amount_band(amount)
        truth = self._truth(kind, tier, band, defective, changed)
        is_tacit = kind == "recurring_nopo_over5k"
        is_changed = kind == "po_match_std_large" and changed
        use_probe = probe and is_tacit
        currency = rng.choice(CURRENCIES)
        text = self._text(rng, vendor, line, amount, currency, po_status, invoice_type, bank_change, defective, probe=use_probe)
        if bank_change:
            situation = "bank-change/new-vendor"
        elif defective:
            situation = "defective/no-po"
        elif po_status == "over":
            situation = f"po-over/{band}"
        elif po_status == "match":
            situation = f"po-match/{tier}/{band}"
        else:
            situation = f"no-po/{invoice_type}/{band}"
        fields = {
            "vendor": vendor, "vendor_tier": tier, "amount": amount, "currency": currency,
            "po_status": po_status, "invoice_type": invoice_type, "category": category,
            "bank_details_changed": bank_change, "fields_missing": defective, "amount_band": band,
        }
        return Case(
            id="", episode=episode, text=text, fields=fields, situation=situation, truth=truth,
            kind=kind, is_tacit=is_tacit, is_changed=is_changed, is_probe=use_probe,
        )

    @staticmethod
    def _truth(kind: str, tier: str, band: str, defective: bool, changed: bool) -> str:
        if kind == "new_vendor_bank_change":
            return "escalate_fraud_review"
        if kind == "po_mismatch":
            return "hold_for_po_match"
        if kind == "nopo_large_oneoff":
            return "request_vendor_correction" if defective else "route_to_approver"
        if kind == "nopo_small_existing":
            return "auto_approve"
        if kind == "recurring_nopo_over5k":
            return "auto_approve"
        if kind == "po_match_std_large":
            return "route_to_approver" if changed else "auto_approve"
        return "auto_approve"  # po_match_small

    def _text(self, rng: random.Random, vendor: str, line: str, amount: float, currency: str, po_status: str,
              invoice_type: str, bank_change: bool, defective: bool, *, probe: bool) -> str:
        inv_no = f"INV-{rng.randint(10000, 99999)}"
        period = rng.choice(["1–30 Sep", "Aug 2026", "Q3 2026", "15 Aug – 14 Sep", "Sep 2026"])
        parts = [f"{vendor} — {inv_no}, {currency} {amount:,.2f}, due in {rng.choice([10, 14, 30, 30, 45])} days."]
        if probe:
            parts.append(rng.choice(PROBE_RECURRING_LINES).format(period=period, acct=rng.randint(1000, 9999)))
        elif invoice_type == "recurring":
            parts.append(rng.choice([
                f"Line item: {line} ({period}).",
                f"{line.capitalize()}, {period}. Same amount as the previous {rng.randint(3, 11)} invoices, give or take.",
                f"{line.capitalize()} for {period}; terms {rng.choice(['2/10 net 30', 'net 30', 'due on receipt'])}.",
            ]))
        else:
            parts.append(f"Line item: {line}.")
        if po_status == "match":
            parts.append(f"References PO-{rng.randint(4000, 9999)}; quantities and unit prices match.")
        elif po_status == "over":
            parts.append(f"References PO-{rng.randint(4000, 9999)}; invoice exceeds the PO by {rng.randint(4, 23)}%.")
        else:
            parts.append(rng.choice(["No PO referenced.", "PO field blank.", "No purchase order on the invoice."]))
        if bank_change:
            parts.append(rng.choice([
                "Remittance instructions differ from the vendor master; new IBAN provided in the email body.",
                "First invoice from this vendor; bank details supplied in a PDF attachment.",
            ]))
        if defective:
            parts.append(rng.choice(["Tax ID missing.", "No invoice number printed.", "Legal entity name does not match our records."]))
        return " ".join(parts)

    # -- review --------------------------------------------------------------------------

    def severity_of(self, case: Case, action: str) -> str:
        truth = case.truth
        if truth == "escalate_fraud_review" and action == "auto_approve":
            return "critical_failure"
        if truth == "escalate_fraud_review":
            return "failure"
        if truth == "hold_for_po_match":
            return "failure" if action == "auto_approve" else "minor_failure"
        if truth == "auto_approve":
            if case.is_tacit:
                # Recurring service held or routed: late fee, lost discount, interruption.
                return "failure"
            return "minor_failure"  # a clean invoice delayed
        if truth == "route_to_approver":
            if case.is_changed and action == "auto_approve":
                return "failure"  # audit exception
            return "failure" if action == "auto_approve" else "minor_failure"
        # request_vendor_correction
        return "failure" if action == "auto_approve" else "minor_failure"

    def failure_note(self, case: Case, action: str, rng: random.Random) -> str:
        truth = case.truth
        days = rng.randint(6, 24)
        if truth == "auto_approve" and case.is_tacit:
            return rng.choice([
                f"Paid {days} days late; the provider added a late-payment charge and a disconnection notice.",
                f"Sat {days} days in a queue; the budget owner asked why a fixed monthly service charge needed sign-off.",
                "Early-payment discount missed while the invoice waited; vendor flagged the account as past due.",
                f"Service suspended for non-payment on day {days}; restored after an urgent manual payment.",
                "Second reminder from the vendor; this charge is the same every month under the existing agreement.",
            ])
        if truth == "auto_approve":
            return rng.choice([
                f"Clean invoice delayed {days} days; vendor relationship manager complained about the slow cycle.",
                "Bounced back from the queue: nothing left to sign off, the PO already covered it.",
            ])
        if truth == "route_to_approver":
            if case.is_changed and action == "auto_approve":
                return rng.choice([
                    "Internal audit flagged the payment as an exception under the new controls; finance documented an override.",
                    "Payment released without the sign-off the revised control framework now expects for this vendor tier; audit exception logged.",
                ])
            if action == "auto_approve":
                return "Paid without budget-owner review; the department had no budget line for it and the charge was disputed."
            return f"Invoice sat {days} days in the wrong queue; the budget owner never saw it."
        if truth == "hold_for_po_match":
            if action == "auto_approve":
                return "Overpaid against the PO; recovery from the vendor took two months."
            return "Approver could not act on it without the PO amendment; bounced back after a week."
        if truth == "escalate_fraud_review":
            if action == "auto_approve":
                return "Payment went to an account controlled by a third party; the vendor had never changed their bank details."
            return "Suspicious remittance change went through a normal queue; caught late by the bank's own check."
        # request_vendor_correction
        if action == "auto_approve":
            return "Paid an invoice with no tax ID; the tax team had to reverse and re-book it."
        return f"Defective invoice circulated for {days} days before anyone asked for a reissue."

    def success_note(self, case: Case, action: str, rng: random.Random) -> str:
        if action == "auto_approve":
            return rng.choice(["Paid on time; no follow-up.", "Paid within terms; discount captured." if case.is_tacit else "Paid within terms."])
        if action == "route_to_approver":
            return f"Approved by {rng.choice(APPROVERS)} in {rng.randint(1, 4)} days; paid within terms."
        if action == "hold_for_po_match":
            return "PO amended and matched; paid the following week."
        if action == "request_vendor_correction":
            return "Vendor reissued a corrected invoice within three days."
        return "Fraud team confirmed a compromised vendor mailbox; payment blocked."


SCENARIO = ApInvoices()

__all__ = ["SCENARIO", "ApInvoices"]
