"""Paid-media budget optimisation.

Each case is a weekly snapshot of one ad set; the decision is what to do with its budget.
Actions: ``scale_budget``, ``hold``, ``cut_budget``, ``pause``, ``refresh_creative``.

Written policy: scale what beats its CPA target, hold what is near target, cut what is
1.5× over, pause what is 2× over, refresh creative when frequency is high and CTR falls.

Tacit rule the policy misses: **a paid-social prospecting ad set under a week old must be
held, not scaled, however good its CPA looks**. A budget jump resets the platform's
learning phase; CPA doubles the following week. The only signal is the launch date in the
text and a ``days_live`` field.

Change half-way: a browser privacy change drains the display retargeting pools. **Display
retargeting with a CPA under target now needs its budget cut**: the reported CPA is flattered
by conversions that would have happened anyway, and the incrementality holdout shows it.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from core.scenario import Case, Scenario

PRODUCTS = ["the spring collection", "the Pro plan", "the EV charger", "the mobile app", "the home insurance quote flow",
            "the meal-kit trial", "the B2B webinar", "the credit card offer"]
REGIONS = ["US", "UK", "DE", "FR", "CA", "AU"]
LAUNCH_LINES_FRESH = [
    "Launched {days} days ago.", "Went live on {weekday}.", "Set up {days} days ago from the Q4 brief.",
    "Flighting since {weekday}.", "Live for {days} days.",
]
LAUNCH_LINES_ESTABLISHED = [
    "Launched {days} days ago.", "Live for {days} days.", "Running since the start of the quarter ({days} days).",
    "In market {days} days; no structural changes this month.",
]
# Held-out phrasings for a fresh paid-social ad set after the probe episode.
PROBE_FRESH_LINES = [
    "Duplicated from the winning ad set earlier this week.", "Cleared ad review on {weekday}.",
    "First full week of data is not in yet.", "Budget has been untouched since the {weekday} start.",
]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

KIND_WEIGHTS = [
    ("fresh_social_prospecting_under", 0.30),  # tacit: hold
    ("display_retargeting_under", 0.18),        # changes with the privacy update
    ("search_under_stable", 0.12),
    ("near_target", 0.12),
    ("over_1_5", 0.10),
    ("over_2", 0.08),
    ("high_freq_falling", 0.10),
]


def _pick_weighted(rng: random.Random, pairs: list[tuple[str, float]]) -> str:
    r = rng.random() * sum(w for _, w in pairs)
    acc = 0.0
    for name, w in pairs:
        acc += w
        if r <= acc:
            return name
    return pairs[-1][0]


def _cpa_band(ratio: float) -> str:
    if ratio <= 1.0:
        return "under-target"
    if ratio <= 1.5:
        return "near-target"
    if ratio <= 2.0:
        return "over-1.5x"
    return "over-2x"


@dataclass
class Marketing(Scenario):
    name: str = "marketing"
    title: str = "Paid-media budget optimisation"
    case_noun: str = "ad_set"
    actions: list[str] = field(default_factory=lambda: ["scale_budget", "hold", "cut_budget", "pause", "refresh_creative"])
    jev_instructions: str = "Choose this week's budget action for the ad set."
    jev_criteria: dict[str, str] = field(default_factory=lambda: {
        "scale_budget": "Increase the daily budget by 20-30%.",
        "hold": "Leave budget and creative unchanged this week.",
        "cut_budget": "Reduce the daily budget by 30-50%.",
        "pause": "Pause the ad set.",
        "refresh_creative": "Keep the budget; swap in new creative.",
    })
    policy: list[tuple[str, str]] = field(default_factory=lambda: [
        ("policy-scale", "CPA at or under target with CTR stable or rising: scale the budget."),
        ("policy-hold", "CPA between 1.0x and 1.5x target: hold."),
        ("policy-cut", "CPA above 1.5x target after 7+ days live: cut the budget."),
        ("policy-pause", "CPA above 2x target: pause."),
        ("policy-refresh", "Frequency above 4 with falling CTR: refresh the creative."),
    ])
    judge_rubric: str = (
        "A decision failed when the review reports a CPA blow-up after a budget change, wasted spend shown by "
        "an incrementality holdout, a paused winner, or a stale ad set left running. Name the failure by the ad "
        "set's channel, audience, CPA band and age (from the days_live field or the launch line in the text) and "
        "the action taken, for example 'fresh paid-social prospecting ad set under target scaled; CPA doubled' or "
        "'display retargeting under target scaled; holdout showed no lift'. Keep the phrase identical across runs "
        "that fail the same way."
    )
    cost_by_severity: dict[str, float] = field(default_factory=lambda: {
        "minor_failure": 300.0, "failure": 3000.0, "critical_failure": 12000.0,
    })
    tacit_rule: str = ("A paid-social prospecting ad set under 7 days old must be held, not scaled, whatever its CPA: "
                       "a budget jump resets learning and CPA doubles.")
    change_rule: str = ("After the browser privacy change, display retargeting with CPA under target must have its budget "
                        "cut: the reported CPA is flattered by non-incremental conversions.")
    expected_fix: str = "Paid-social prospecting ad sets live under 7 days: hold the budget, do not scale."

    # -- cases ---------------------------------------------------------------------------

    def make_case(self, episode: int, rng: random.Random, *, changed: bool, probe: bool) -> Case:
        kind = _pick_weighted(rng, KIND_WEIGHTS)
        ctr_trend = rng.choice(["stable", "rising"])
        frequency = round(rng.uniform(1.2, 3.2), 1)
        if kind == "fresh_social_prospecting_under":
            channel, audience = "paid_social", "prospecting"
            days_live = rng.randint(2, 6)
            ratio = round(rng.uniform(0.55, 0.98), 2)
        elif kind == "display_retargeting_under":
            channel, audience = "display", "retargeting"
            days_live = rng.randint(14, 90)
            ratio = round(rng.uniform(0.5, 0.95), 2)
            frequency = round(rng.uniform(2.5, 3.9), 1)
        elif kind == "search_under_stable":
            channel, audience = "search", "prospecting"
            days_live = rng.randint(10, 120)
            ratio = round(rng.uniform(0.6, 0.98), 2)
        elif kind == "near_target":
            channel = rng.choice(["paid_social", "search", "video"])
            audience = rng.choice(["prospecting", "retargeting"])
            days_live = rng.randint(8, 90)
            ratio = round(rng.uniform(1.02, 1.48), 2)
        elif kind == "over_1_5":
            channel = rng.choice(["paid_social", "search", "video", "display"])
            audience = rng.choice(["prospecting", "retargeting"])
            days_live = rng.randint(8, 60)
            ratio = round(rng.uniform(1.52, 1.98), 2)
            ctr_trend = rng.choice(["stable", "falling"])
        elif kind == "over_2":
            channel = rng.choice(["paid_social", "video", "display"])
            audience = rng.choice(["prospecting", "retargeting"])
            days_live = rng.randint(5, 60)
            ratio = round(rng.uniform(2.05, 3.6), 2)
            ctr_trend = rng.choice(["stable", "falling"])
        else:  # high_freq_falling
            channel = rng.choice(["paid_social", "video"])
            audience = rng.choice(["prospecting", "retargeting"])
            days_live = rng.randint(14, 90)
            ratio = round(rng.uniform(0.9, 1.4), 2)
            frequency = round(rng.uniform(4.2, 6.8), 1)
            ctr_trend = "falling"

        band = _cpa_band(ratio)
        truth = self._truth(kind, changed)
        is_tacit = kind == "fresh_social_prospecting_under"
        is_changed = kind == "display_retargeting_under" and changed
        use_probe = probe and is_tacit
        spend = round(rng.uniform(900, 18000), -1)
        text = self._text(rng, channel, audience, days_live, ratio, ctr_trend, frequency, spend, probe=use_probe)
        age = "fresh" if days_live < 7 else "established"
        if frequency > 4 and ctr_trend == "falling":
            situation = f"{channel}/high-frequency"
        else:
            situation = f"{channel}/{audience}/{band}/{age}"
        fields = {
            "channel": channel, "audience": audience, "days_live": days_live, "cpa_vs_target": ratio,
            "cpa_band": band, "ctr_trend": ctr_trend, "frequency": frequency, "spend_7d": spend,
            "region": rng.choice(REGIONS),
        }
        return Case(
            id="", episode=episode, text=text, fields=fields, situation=situation, truth=truth,
            kind=kind, is_tacit=is_tacit, is_changed=is_changed, is_probe=use_probe,
        )

    @staticmethod
    def _truth(kind: str, changed: bool) -> str:
        return {
            "fresh_social_prospecting_under": "hold",
            "display_retargeting_under": "cut_budget" if changed else "scale_budget",
            "search_under_stable": "scale_budget",
            "near_target": "hold",
            "over_1_5": "cut_budget",
            "over_2": "pause",
            "high_freq_falling": "refresh_creative",
        }[kind]

    def _text(self, rng: random.Random, channel: str, audience: str, days_live: int, ratio: float, ctr_trend: str,
              frequency: float, spend: float, *, probe: bool) -> str:
        product = rng.choice(PRODUCTS)
        label = {"paid_social": "Paid social", "search": "Search", "display": "Display", "video": "Video"}[channel]
        head = f"{label} {audience} ad set for {product}: ${spend:,.0f} spent over the last 7 days, CPA at {ratio:.2f}x target."
        trend = {"stable": "CTR steady", "rising": "CTR up week on week", "falling": "CTR falling"}[ctr_trend]
        detail = f"{trend}; frequency {frequency}."
        if probe:
            launch = rng.choice(PROBE_FRESH_LINES).format(weekday=rng.choice(WEEKDAYS))
        else:
            lines = LAUNCH_LINES_FRESH if days_live < 7 else LAUNCH_LINES_ESTABLISHED
            launch = rng.choice(lines).format(days=days_live, weekday=rng.choice(WEEKDAYS))
        return " ".join([head, detail, launch])

    # -- review --------------------------------------------------------------------------

    def severity_of(self, case: Case, action: str) -> str:
        truth = case.truth
        if truth == "hold" and case.is_tacit:
            return "critical_failure" if action == "pause" else ("failure" if action == "scale_budget" else "minor_failure")
        if truth == "scale_budget":
            return "failure" if action == "pause" else "minor_failure"
        if truth == "cut_budget":
            if case.is_changed:
                return "failure" if action == "scale_budget" else "minor_failure"
            return "failure" if action == "scale_budget" else "minor_failure"
        if truth == "pause":
            return "failure" if action == "scale_budget" else "minor_failure"
        if truth == "refresh_creative":
            return "failure" if action == "scale_budget" else "minor_failure"
        return "minor_failure"  # near-target hold

    def failure_note(self, case: Case, action: str, rng: random.Random) -> str:
        truth = case.truth
        if truth == "hold" and case.is_tacit:
            if action == "scale_budget":
                return rng.choice([
                    f"CPA went to {round(case.fields['cpa_vs_target'] * rng.uniform(1.9, 2.6), 2)}x target the week after the budget change; delivery became erratic.",
                    "Platform reported the ad set re-entered learning after the change; conversions dropped for six days.",
                    "Performance collapsed following the budget move on an ad set that had barely started delivering.",
                ])
            if action == "pause":
                return "Paused a new ad set that was outperforming target; relaunch cost a week and the early data."
            if action == "cut_budget":
                return "Cut a brand-new ad set under target; it never gathered enough data to optimise."
            return "Swapped creative on an ad set a few days old; the comparison week was lost."
        if truth == "scale_budget":
            if action == "pause":
                return "Paused the best-performing ad set in the account; the weekly lead target was missed."
            return "Left spend on the table: this ad set had headroom at a CPA well under target."
        if truth == "cut_budget":
            if case.is_changed:
                return rng.choice([
                    "Incrementality holdout showed the retargeting spend bought conversions that would have happened anyway.",
                    "Lift test came back flat for this audience; reported CPA no longer reflects incremental cost.",
                    "Finance asked why retargeting spend went up while the holdout group converted at the same rate.",
                ])
            return "Spend continued at 1.5x target CPA for another week with no improvement."
        if truth == "pause":
            return "Burned budget at more than twice the target CPA for another week."
        if truth == "refresh_creative":
            return "Audience fatigue: frequency kept climbing and CTR fell further; nothing in the ad set changed for another week."
        return "Unnecessary change on an ad set tracking near target; the week's data is harder to read."

    def success_note(self, case: Case, action: str, rng: random.Random) -> str:
        return {
            "scale_budget": f"Scaled; CPA held at {round(case.fields['cpa_vs_target'] * rng.uniform(0.95, 1.1), 2)}x target.",
            "hold": "Held; performance tracked as expected.",
            "cut_budget": "Cut; weekly CPA improved and spend came back under plan.",
            "pause": "Paused; budget reallocated to better ad sets.",
            "refresh_creative": f"New creative lifted CTR by {rng.randint(12, 40)}% in the first week.",
        }[action]


SCENARIO = Marketing()

__all__ = ["SCENARIO", "Marketing"]
