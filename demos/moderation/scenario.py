"""Content moderation with appeals.

Each case is a reported post; the decision is what to do with it. The review is what
happened next: an appeal upheld or denied, harm reports, a regulator's letter.
Actions: ``keep``, ``remove``, ``restrict_reach``, ``age_gate``, ``escalate_human``.

Written policy: graphic violence is removed; nudity is removed unless educational or
medical, then age-gated; harassment of a named person is removed, general harassment
has its reach restricted; spam from new accounts is removed; medical misinformation has
its reach restricted, or is removed when it gives harmful advice; hate speech is removed.

Tacit rule the policy misses: **graphic footage posted as news reporting (conflict, protests,
disasters) must be age-gated, not removed**. Removals are overturned on appeal, the
newsroom complains publicly and the platform is accused of censorship. The only signals are
the ``context`` field and newsroom language in the post.

Change half-way: a new regional law takes effect. **Medical misinformation from EU accounts
must now be removed**, not reach-restricted; leaving it up draws a regulator's notice.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from core.scenario import Case, Scenario

NEWS_SOURCES = ["a regional broadcaster", "a wire-service stringer", "an independent journalist", "a public radio station",
                "a documentary crew", "a local newspaper's video desk"]
NEWS_EVENTS = ["the aftermath of the shelling in the eastern district", "clashes at the port protest", "the flood rescue",
               "the apartment fire", "the aftermath of the highway pile-up", "the stadium crush"]
NEWS_LINES = [
    "Footage from {event}, posted by {source}. Caption gives the date, place and a content warning.",
    "{source_cap} uploaded raw video of {event}; the caption names the city and the crew on the ground.",
    "Clip shows {event}; posted by {source} with a byline and a note that faces are blurred.",
]
# Held-out phrasings for the news case after the probe episode.
PROBE_NEWS_LINES = [
    "Eyewitness video of {event}, verified by the newsroom's standards desk before publishing.",
    "Material from {event} filed from the scene; the outlet's editor is tagged in the post.",
    "Segment on {event} as broadcast, with the station's lower-third still visible.",
]
PERSONAL_GRAPHIC = [
    "User's own video of a street fight filmed from a balcony, captioned with laughing emoji.",
    "Dashcam clip of a serious crash, reposted for shock value with no context.",
    "Clip of an animal being hurt, shared by an account that posts similar material daily.",
]
NUDITY_EDU = [
    "Breast self-examination tutorial from a cancer charity; shows how to check for lumps.",
    "Anatomy illustration in a nursing student's study post.",
    "Birth photography from a midwifery account, with the parents' permission noted.",
]
HARASSMENT_NAMED = [
    "Post names a classmate, calls them slurs and tells followers where they work.",
    "Thread tagging a local councillor with repeated threats to 'find' her.",
    "Repost of a colleague's photo with her full name and insults about her appearance.",
]
HARASSMENT_GENERAL = [
    "Rant calling a whole fandom 'subhuman losers'; no individual is named.",
    "Post mocking everyone who uses the city's bike lanes with crude insults.",
]
SPAM = [
    "Account created 2 days ago posting the same crypto giveaway link in forty comments.",
    "New account replying 'DM for cheap followers' under popular posts.",
]
MEDICAL_CLAIMS = [
    "Post claims a vitamin protocol reverses type 2 diabetes in six weeks, citing a 'suppressed study'.",
    "Video claiming 5G towers weaken the immune system, with a map of local masts.",
    "Thread saying a common sweetener causes cancer, linking to a blog with no sources.",
    "Post claiming a detox tea clears microplastics from the blood; 20k shares.",
    "Infographic saying the flu shot gives you the flu, shared by a wellness account.",
]
MEDICAL_HARMFUL = [
    "Post claims a vitamin protocol cures type 1 diabetes and tells readers to stop insulin.",
    "Video telling parents to treat measles with essential oils instead of seeing a doctor.",
]
HATE = [
    "Post calling for members of a religious minority to be 'cleared out' of the neighbourhood.",
    "Meme dehumanising an ethnic group with vermin imagery.",
]
EU = ["DE", "FR", "IT", "ES", "NL", "IE"]
NON_EU = ["US", "BR", "IN", "GB", "CA", "AU"]

KIND_WEIGHTS = [
    ("graphic_news", 0.30),          # tacit: age-gate
    ("medical_misinfo_eu", 0.18),    # changes with the new law
    ("graphic_personal", 0.10),
    ("nudity_edu", 0.08),
    ("harassment_named", 0.10),
    ("harassment_general", 0.08),
    ("spam_new", 0.08),
    ("hate", 0.08),
]


def _pick_weighted(rng: random.Random, pairs: list[tuple[str, float]]) -> str:
    r = rng.random() * sum(w for _, w in pairs)
    acc = 0.0
    for name, w in pairs:
        acc += w
        if r <= acc:
            return name
    return pairs[-1][0]


@dataclass
class Moderation(Scenario):
    name: str = "moderation"
    title: str = "Content moderation with appeals"
    case_noun: str = "report"
    actions: list[str] = field(default_factory=lambda: ["keep", "remove", "restrict_reach", "age_gate", "escalate_human"])
    jev_instructions: str = "Choose the moderation action for this reported post."
    jev_criteria: dict[str, str] = field(default_factory=lambda: {
        "keep": "Leave the post up with no change.",
        "remove": "Take the post down and notify the author.",
        "restrict_reach": "Leave it up but exclude it from recommendations and search.",
        "age_gate": "Leave it up behind an interstitial and an 18+ gate.",
        "escalate_human": "Send to the senior human review queue; no action yet.",
    })
    policy: list[tuple[str, str]] = field(default_factory=lambda: [
        ("policy-graphic", "Graphic violence or gore: remove."),
        ("policy-nudity", "Nudity: remove, unless the context is educational or medical, then age-gate."),
        ("policy-harassment", "Harassment that names or tags a person: remove. Harassment with no identifiable target: restrict reach."),
        ("policy-spam", "Repetitive promotional posting from accounts under 30 days old: remove."),
        ("policy-medical", "Medical misinformation: restrict reach. If it tells people to stop or avoid treatment, remove."),
        ("policy-hate", "Hate speech targeting a protected group: remove."),
    ])
    judge_rubric: str = (
        "A decision failed when the review reports an appeal upheld against the action, a public complaint from a "
        "newsroom, user harm reports on content left up, or a regulator's notice. Name the failure by the report "
        "reason, the context field, the account's region and the action taken, for example 'graphic news footage "
        "removed; appeal upheld' or 'EU medical misinformation reach-restricted; regulator notice'. Keep the phrase "
        "identical across runs that fail the same way."
    )
    cost_by_severity: dict[str, float] = field(default_factory=lambda: {
        "minor_failure": 100.0, "failure": 1500.0, "critical_failure": 20000.0,
    })
    tacit_rule: str = ("Graphic footage posted as news reporting must be age-gated, not removed; removals are overturned on "
                       "appeal and draw public censorship complaints.")
    change_rule: str = ("After the new regional law, medical misinformation from EU accounts must be removed, not "
                        "reach-restricted; leaving it up draws a regulator's notice.")
    expected_fix: str = "Graphic content in a news or documentary context: age-gate instead of removing."

    # -- cases ---------------------------------------------------------------------------

    def make_case(self, episode: int, rng: random.Random, *, changed: bool, probe: bool) -> Case:
        kind = _pick_weighted(rng, KIND_WEIGHTS)
        account_age = rng.randint(200, 4000)
        strikes = rng.choice([0, 0, 0, 1])
        region = rng.choice(NON_EU + EU)
        context = "personal"
        harmful_advice = False
        if kind == "graphic_news":
            reason, context = "graphic_violence", rng.choice(["news", "news", "documentary"])
            source, event = rng.choice(NEWS_SOURCES), rng.choice(NEWS_EVENTS)
            line = rng.choice(PROBE_NEWS_LINES if probe else NEWS_LINES)
            text = line.format(event=event, source=source, source_cap=source[0].upper() + source[1:])
        elif kind == "medical_misinfo_eu":
            reason, region = "medical_misinformation", rng.choice(EU)
            harmful_advice = rng.random() < 0.2
            text = rng.choice(MEDICAL_HARMFUL if harmful_advice else MEDICAL_CLAIMS)
        elif kind == "graphic_personal":
            reason, text = "graphic_violence", rng.choice(PERSONAL_GRAPHIC)
        elif kind == "nudity_edu":
            reason, context, text = "nudity", rng.choice(["educational", "medical"]), rng.choice(NUDITY_EDU)
        elif kind == "harassment_named":
            reason, text = "harassment", rng.choice(HARASSMENT_NAMED)
            strikes = rng.choice([0, 1, 2])
        elif kind == "harassment_general":
            reason, text = "harassment", rng.choice(HARASSMENT_GENERAL)
        elif kind == "spam_new":
            reason, text, account_age = "spam", rng.choice(SPAM), rng.randint(1, 20)
        else:
            reason, text = "hate_speech", rng.choice(HATE)
            strikes = rng.choice([0, 1, 2, 3])

        truth = self._truth(kind, harmful_advice, changed)
        is_tacit = kind == "graphic_news"
        # Harmful-advice posts were already removals; only the plain claims change.
        is_changed = kind == "medical_misinfo_eu" and changed and not harmful_advice
        use_probe = probe and is_tacit
        in_eu = region in EU
        if kind == "medical_misinfo_eu":
            situation = f"medical/{'eu' if in_eu else 'non-eu'}/{'harmful-advice' if harmful_advice else 'claim'}"
        elif reason == "harassment":
            situation = f"harassment/{'named' if kind == 'harassment_named' else 'general'}"
        elif reason == "graphic_violence":
            situation = f"graphic/{context}"
        elif reason == "nudity":
            situation = f"nudity/{context}"
        else:
            situation = reason.replace("_", "-")
        text = f"{text} Report reason: {reason.replace('_', ' ')}. Reported {rng.randint(1, 40)} times in {rng.randint(1, 48)}h."
        fields = {
            "report_reason": reason, "context": context, "account_age_days": account_age,
            "prior_strikes": strikes, "region": region, "eu_account": in_eu,
            "content_type": rng.choice(["video", "image", "text"]) if reason != "graphic_violence" else "video",
        }
        return Case(
            id="", episode=episode, text=text, fields=fields, situation=situation, truth=truth,
            kind=kind, is_tacit=is_tacit, is_changed=is_changed, is_probe=use_probe,
        )

    @staticmethod
    def _truth(kind: str, harmful_advice: bool, changed: bool) -> str:
        if kind == "graphic_news":
            return "age_gate"
        if kind == "medical_misinfo_eu":
            if changed or harmful_advice:
                return "remove"
            return "restrict_reach"
        if kind == "nudity_edu":
            return "age_gate"
        if kind == "harassment_general":
            return "restrict_reach"
        return "remove"  # graphic_personal, harassment_named, spam_new, hate

    # -- review --------------------------------------------------------------------------

    def severity_of(self, case: Case, action: str) -> str:
        truth = case.truth
        if truth == "remove":
            if case.kind in ("hate", "harassment_named", "graphic_personal") and action == "keep":
                return "critical_failure"
            if case.is_changed and action in ("keep", "restrict_reach", "age_gate"):
                return "failure"  # regulator's notice
            return "failure" if action == "keep" else "minor_failure"
        if truth == "age_gate":
            if case.is_tacit:
                return "failure" if action == "remove" else ("critical_failure" if action == "keep" else "minor_failure")
            return "failure" if action == "remove" else "minor_failure"
        if truth == "restrict_reach":
            return "failure" if action == "remove" else "minor_failure"
        return "minor_failure"

    def failure_note(self, case: Case, action: str, rng: random.Random) -> str:
        truth = case.truth
        hours = rng.randint(6, 72)
        if truth == "age_gate" and case.is_tacit:
            if action == "remove":
                return rng.choice([
                    f"Appeal upheld after {hours}h: the appeals board reinstated the post as newsworthy. The outlet published a piece about the takedown.",
                    "Takedown reversed on appeal; a press-freedom group cited it in a letter to the platform.",
                    f"Reinstated by the appeals board after {hours}h; the newsroom's editor complained publicly about censorship.",
                ])
            if action == "keep":
                return "Left fully open: autoplayed to minors; harm reports and a safety complaint followed."
            if action == "restrict_reach":
                return "Newsroom complained that its reporting was suppressed in search and recommendations; appeal upheld."
            return f"Sat in the senior queue for {hours}h while the story was live; the outlet asked publicly why."
        if truth == "age_gate":
            if action == "remove":
                return "Appeal upheld: educational content wrongly removed; the charity posted about it."
            return "Left open with no gate; shown to minors and reported."
        if truth == "remove":
            if case.is_changed and action != "remove":
                return rng.choice([
                    "Regulator's notice received: the post stayed available to users in the member state beyond the statutory window.",
                    "Flagged by the national authority under the new law; the platform had to report why it remained visible.",
                ])
            if action == "keep":
                return "Post stayed up; harm reports and press attention followed." if case.kind != "spam_new" else "Spam spread to dozens more threads before anyone acted."
            if action == "restrict_reach":
                return "Still visible to followers; the targeted person filed a complaint and the post was eventually taken down by hand."
            if action == "age_gate":
                return "An age gate did nothing here; the post was reported again within hours."
            return f"Sat in the senior queue for {hours}h while reports accumulated."
        if truth == "restrict_reach":
            if action == "remove":
                return "Appeal upheld: no individual targeted; removal reversed and a strike withdrawn."
            if action == "keep":
                return "Kept spreading through recommendations; complaint volume doubled."
            return "Over-handled a low-severity post; reviewer time wasted."
        return "Unnecessary action."

    def success_note(self, case: Case, action: str, rng: random.Random) -> str:
        if action == "age_gate":
            return "Stayed up behind the gate; no appeal, no further reports."
        if action == "remove":
            return "Removed; no appeal filed." if rng.random() < 0.7 else "Removed; appeal denied."
        if action == "restrict_reach":
            return "Reach restricted; report volume fell to zero within a day."
        if action == "keep":
            return "Kept; no further reports."
        return "Senior reviewer confirmed the call."


SCENARIO = Moderation()

__all__ = ["SCENARIO", "Moderation"]
