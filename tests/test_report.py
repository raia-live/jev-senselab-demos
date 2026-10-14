"""The report builder works on a tiny synthetic results directory, without any keys."""

from __future__ import annotations

import json
from pathlib import Path

from core import report
from core.senselab import (
    CORRECTIONS_HEADING,
    DROP,
    HOLD,
    NO_RECOMMENDATION,
    ReviewHistory,
    cap_lines,
    corrections_block,
    drop_policy_lines,
    hold_action,
    noise_guard,
    parse_record,
    situation_block,
    split_priors,
    strip_retry_lines,
    tag_entries,
    turned_streak,
)


def _rows(arm: str, seed: int, correct: list[bool]) -> list[dict]:
    rows = []
    for ep, ok in enumerate(correct):
        rows.append({
            "run_id": "t", "scenario": "lead_routing", "seed": seed, "arm": arm, "episode": ep,
            "situation": "mid_market/emea", "truth": "partner", "kind": "mm_emea_intent",
            "is_tacit": True, "is_changed": ep >= 6, "is_probe": ep >= 9,
            "action": "partner" if ok else "route_to_ae", "correct": ok, "confidence": 0.9,
            "success": ok, "severity": "success" if ok else "failure", "cost_usd": 0 if ok else 2500,
            "noisy": False, "true_success": ok, "note": "", "jev_input_tokens": 800, "state_bytes": 1500,
        })
    return rows


def test_build_scenario_and_index(tmp_path: Path) -> None:
    run = tmp_path / "run"
    sdir = run / "lead_routing"
    sdir.mkdir(parents=True)
    rows = _rows("jev", 11, [False] * 12) + _rows("jev+senselab", 11, [False, False, True, True, True, True, True, True, True, True, True, True])
    (sdir / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (sdir / "meta.json").write_text(json.dumps({
        "run_id": "t", "episodes": 12, "seeds": [11], "change_at": 6, "probe_at": 9, "noise": 0.1, "jev_model": "jev-x",
        "scenario": {"name": "lead_routing", "title": "Inbound lead routing", "tacit_rule": "t", "change_rule": "c",
                     "expected_fix": "f", "policy": [{"key": "p", "text": "policy line"}]},
    }))
    (sdir / "loop-senselab-s11.json").write_text(json.dumps({
        "events": [{"kind": "fix_status", "episode": 4, "fix": {"id": "abcdef123456", "status": "proposed", "lever": "memory", "title": "EMEA → partner"}},
                   {"kind": "canary_start", "episode": 6, "fix_id": "abcdef123456"},
                   {"kind": "fix_status", "episode": 10, "fix": {"id": "abcdef123456", "status": "memory_shipped", "lever": "memory", "title": "EMEA → partner"}}],
        "fixes": [{"id": "abcdef123456", "status": "memory_shipped", "lever": "memory", "title": "EMEA → partner",
                   "test": {"status": "passed"}, "canary": {"status": "passed", "canary": {"graded": 5, "read": 6}, "control": {"graded": 5}}}],
        "verdict_summary": {"graded": 12, "failed": 2},
    }))
    summary = report.build_scenario(sdir, quiet=True)
    assert summary is not None
    m = summary["metrics"]
    assert m["jev"]["wrong"] == 12 and m["jev+senselab"]["wrong"] == 2
    assert m["jev+senselab"]["recover_cycle1"] == 1
    assert m["jev+senselab"]["tacit_accuracy"] > m["jev"]["tacit_accuracy"]
    assert (sdir / "report.html").exists() and (sdir / "metrics.json").exists()
    html = (sdir / "report.html").read_text()
    assert "memory_shipped" in html and "Inbound lead routing" in html
    report.build_all(run, quiet=True)
    assert (run / "index.html").exists()


def test_team_metrics_and_section(tmp_path: Path) -> None:
    from core.arms import merge_loop_reports

    run = tmp_path / "run"
    sdir = run / "lead_routing"
    sdir.mkdir(parents=True)
    # Three agents, round-robin. The lookup pays the first miss once per agent (eps 0, 1, 2);
    # the shared memory pays it once (ep 0) and agent 1 and 2 already know.
    lookup = _rows("jev+retrieval:meta-k8", 11, [False, False, False] + [True] * 9)
    shared = _rows("jev+senselab", 11, [False] + [True] * 11)
    for r in lookup + shared:
        r["agent"] = r["episode"] % 3
    (sdir / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in lookup + shared) + "\n")
    (sdir / "meta.json").write_text(json.dumps({
        "run_id": "t", "episodes": 12, "seeds": [11], "agents": 3, "change_at": -1, "probe_at": -1, "noise": 0.1,
        "jev_model": "jev-x", "scenario": {"name": "lead_routing", "title": "Inbound lead routing", "tacit_rule": "t",
                                           "change_rule": "c", "expected_fix": "f", "policy": []},
    }))
    merged = merge_loop_reports([
        {"agent_id": "a0", "sessions_sealed": 4, "events": [{"kind": "x", "episode": 0}], "fixes": [],
         "verdict_summary": {"graded": 4, "failed": 1, "judge_usd": 0.01, "categories": {"c1": 1}}, "incidents": {"items": [{"id": 1}]}},
        {"agent_id": "a1", "sessions_sealed": 4, "events": [], "fixes": [{"id": "f1"}],
         "verdict_summary": {"graded": 4, "failed": 0, "judge_usd": 0.02, "categories": {}}, "incidents": []},
        {"agent_id": "a2", "sessions_sealed": 4, "events": [], "fixes": [], "verdict_summary": {"graded": 4, "failed": 1, "judge_usd": 0.01,
                                                                                                "categories": {"c1": 1}}},
    ])
    assert merged["sessions_sealed"] == 12 and merged["verdict_summary"]["graded"] == 12
    assert merged["verdict_summary"]["failed"] == 2 and merged["verdict_summary"]["categories"] == {"c1": 2}
    assert merged["events"][0]["agent"] == 0 and merged["fixes"][0]["agent"] == 1
    assert len(merged["incidents"]["items"]) == 1
    (sdir / "loop-senselab-s11.json").write_text(json.dumps(merged))

    summary = report.build_scenario(sdir, quiet=True)
    assert summary is not None
    t = summary["metrics"]["jev+retrieval:meta-k8"]["team"]
    assert t["agents"] == 3 and t["tacit_first_miss_agents"] == 3 and t["wrong_per_agent"] == {0: 1, 1: 1, 2: 1}
    t = summary["metrics"]["jev+senselab"]["team"]
    assert t["tacit_first_miss_agents"] == 1 and t["wrong_per_agent"] == {0: 1, 1: 0, 2: 0}
    html = (sdir / "report.html").read_text()
    assert "3 instances, one stream" in html and "3 of 3" in html and "1 of 3" in html
    assert "cumulative wrong decisions" in html
    assert summary["lifecycle"]["verdicts"]["11"]["graded"] == 12


def test_tag_entries_numbers_in_rendered_order() -> None:
    text = "Memory context:\n- a/b/key-two (high): second\n- a/b/key-one (high): first\n"
    tagged, tags = tag_entries(text, ["a/b/key-one", "a/b/key-two"])
    assert tags == {"M1": "a/b/key-two", "M2": "a/b/key-one"}
    assert "- [M1] key-two (high)" in tagged and "- [M2] key-one (high)" in tagged


def test_corrections_block_shows_what_the_sdk_text_omits() -> None:
    from amfs.guidance import Guidance
    from amfs_core.render import ContextEntry

    entries = [
        ContextEntry("a/b", "policy-x", "the rule", confidence=0.9),
        ContextEntry("a/b", "risk-x-out-of-scope", "Risk: the rule was applied out of scope.", confidence=0.7),
        ContextEntry("a/b", "lesson-contrast-c1", {"avoid": ["a/b/policy-x"]}, confidence=0.6),
    ]
    g = Guidance(text="Memory context:\n- a/b/policy-x (high, untested): the rule", strength="thin",
                 guidance_id="g1", entries=entries)
    assert g.entry_keys == ["a/b/policy-x"]  # the SDK leaves both excluded keys out
    block, keys = corrections_block(g)
    assert keys == ["a/b/risk-x-out-of-scope", "a/b/policy-x"]  # corrective yes, synthetic contrast no
    assert block.startswith(CORRECTIONS_HEADING) and "- a/b/risk-x-out-of-scope (" in block
    # A policy entry that says what the written policy says is Jev's state already and stays
    # out; one the repair loop rewrote (a new version, on a branch or on main) is a correction.
    block, keys = corrections_block(g, frozenset({"the rule"}))
    assert keys == ["a/b/risk-x-out-of-scope"] and "policy-x" not in block
    block, keys = corrections_block(g, frozenset({"the rule, as written"}))
    assert keys == ["a/b/risk-x-out-of-scope", "a/b/policy-x"] and "- a/b/policy-x (high, untested): the rule" in block
    tagged, tags = tag_entries(block + "\n\n" + g.text, keys + g.entry_keys)
    assert tags == {"M1": "a/b/risk-x-out-of-scope", "M2": "a/b/policy-x"}
    assert cap_lines("line one\nline two\nline three", 15) == "line one"
    assert cap_lines("short", 15) == "short"


def test_retry_lines_go_and_tallies_stay() -> None:
    text = ("Tried on this situation: decide:a 4/6, lost last 2; decide:b 0/1\n"
            "Recommendation: act -> decide:c. decide:c is the only action that has worked (1/2)\n"
            "Do not spend an attempt on: decide:a (4/6, superseded here); decide:b (0/1) — these failed on this situation.\n"
            "Try decide:c first. Do not repeat an action that failed on this task.")
    out = strip_retry_lines(text)
    assert out.startswith("Tried on this situation") and "Recommendation: act" in out
    assert "Do not spend" not in out and "Do not repeat" not in out


def test_review_history_keeps_recent_reviews_and_tally() -> None:
    h = ReviewHistory()
    h.add("mid_market/na", "decide:route_to_ae", True, "Closed-won in 21 days.")
    h.add("mid_market/na", "decide:route_to_ae", False, "Bought online while waiting for the callback.")
    text = h.add("mid_market/na", "decide:route_to_ae", False, "The AE cycle added a week and no value.")
    assert text.startswith("Seen on mid_market/na with decide:route_to_ae: worked 1 of 3 times; the last 2 did not.")
    assert text.index("[did not work] The AE cycle") < text.index("[worked] Closed-won")  # newest first
    assert ReviewHistory.key("mid_market/na", "decide:route_to_ae") == "seen-mid-market-na--route-to-ae"
    assert not ReviewHistory.key("graphic/news-extra", "decide:keep").startswith(ReviewHistory.prefix("graphic/news"))
    for i in range(10):
        h.add("smb/na", "decide:self_serve", True, f"note {i}")
    assert len(h.seen[("smb/na", "decide:self_serve")]) == 4 and h.tally[("smb/na", "decide:self_serve")] == [10, 10]


def test_noise_guard_leaves_the_record_only_when_nothing_in_it_works() -> None:
    from types import SimpleNamespace as E

    def seen(situation: str, action: str, text: str) -> E:
        return E(entity_path="jev-demos/x/y", key=f"seen-{situation}--{action}", value=text)

    # lr5 ep 27-70: one flipped review made disqualify 0/1 and the explore cycled through every
    # other action. Nothing has worked and nothing has failed twice: the policy decides.
    thin = [seen("personal-email", "disqualify", "Seen on personal-email with decide:disqualify: worked 0 of 1 time. "
                 "Recent reviews: [did not work] Reviewer marked this one a miss; no detail recorded."),
            seen("personal-email", "route-to-ae", "Seen on personal-email with decide:route_to_ae: worked 0 of 1 time. "
                 "Recent reviews: [did not work] Student project, no budget; 40 minutes of sales time spent.")]
    assert noise_guard("explore", parse_record(thin)) == (DROP, None)
    sdk = "Tried on this situation: decide:disqualify 0/1; decide:route_to_ae 0/1\nRecommendation: explore. tried and failed here.\nUntried: decide:partner, decide:self_serve"
    out = hold_action(sdk, None)
    assert out == "Tried on this situation: decide:disqualify 0/1; decide:route_to_ae 0/1\n" + NO_RECOMMENDATION
    # lr5 ep 76: route_to_ae turned (4 straight failures) but self_serve is alive at 1/2 — hold it.
    change = [seen("mid-market-na", "route-to-ae", "Seen on mid_market/na with decide:route_to_ae: worked 5 of 9 times; the last 4 did not. Recent reviews: x"),
              seen("mid-market-na", "self-serve", "Seen on mid_market/na with decide:self_serve: worked 1 of 2 times; the last 1 did not. Recent reviews: x")]
    kind, held = noise_guard("explore", parse_record(change))
    assert kind == HOLD and held.action == "decide:self_serve" and (held.worked, held.total, held.streak) == (1, 2, 1)
    out = hold_action("Recommendation: explore. regime shift suspected.\nUntried: decide:partner", held)
    assert out == ("Recommendation: act -> decide:self_serve. decide:self_serve has worked 1 of 2 times here and failed only on "
                   "its latest try; one failure after a working record is more likely a wrong review than a change.")
    # A proven action that failed its last two is held: two wrong reviews in a row happen.
    doubted = [seen("mid-market-na", "route-to-ae", "Seen on mid_market/na with decide:route_to_ae: worked 20 of 22 times; the last 2 did not. Recent reviews: x")]
    kind, held = noise_guard("explore", parse_record(doubted))
    assert kind == HOLD and held.action == "decide:route_to_ae" and turned_streak(held) == 3
    assert "failed its last 2 tries; 2 failures after a record of 20 successes" in hold_action("Recommendation: explore. x", held)
    # Three straight failures on it, and it has turned: the explore stands. That is a change.
    turned = [seen("mid-market-na", "route-to-ae", "Seen on mid_market/na with decide:route_to_ae: worked 20 of 23 times; the last 3 did not. Recent reviews: x")]
    assert noise_guard("explore", parse_record(turned)) is None
    # A thin record turns on two: one success does not buy the benefit of the doubt.
    thin_turned = [seen("mid-market-emea", "partner", "Seen on mid_market/emea with decide:partner: worked 1 of 3 times; the last 2 did not. Recent reviews: x")]
    assert turned_streak(parse_record(thin_turned)[0]) == 2 and noise_guard("explore", parse_record(thin_turned)) is None
    # A proven action with one recent miss is held; an act is never touched; no record, nothing to do.
    proven = [seen("personal-email", "disqualify", "Seen on personal-email with decide:disqualify: worked 5 of 6 times; the last 1 did not. Recent reviews: x")]
    assert noise_guard("explore", parse_record(proven))[1].action == "decide:disqualify"
    assert noise_guard("act", parse_record(proven)) is None and noise_guard("explore", []) is None
    # Every review failed so far and there were two of them: that is a turn, not noise.
    allbad = [seen("smb-na", "self-serve", "Seen on smb/na with decide:self_serve: worked 0 of 2 times. Recent reviews: [did not work] a; [did not work] b")]
    assert parse_record(allbad)[0].streak == 2 and noise_guard("explore", parse_record(allbad)) is None


def test_split_priors_and_drop_policy_lines() -> None:
    ep = "jev-demos/x/y"
    sdk = (f"Memory context:\n- {ep}/learned-a (high): When: a. decide:x worked.\n- {ep}/policy-smb (high): SMB: self-serve.\n"
           f"- {ep}/seen-smb-na--self-serve (high): Seen on smb/na …\n\nTried on this situation: decide:x 3/3\nRecommendation: act -> decide:x. why")
    context, priors = split_priors(sdk)
    assert priors == "Tried on this situation: decide:x 3/3\nRecommendation: act -> decide:x. why"
    assert context.endswith("Seen on smb/na …") and "Tried" not in context
    keys = [f"{ep}/learned-a", f"{ep}/policy-smb", f"{ep}/seen-smb-na--self-serve"]
    context, kept = drop_policy_lines(context, keys)
    assert kept == [f"{ep}/learned-a", f"{ep}/seen-smb-na--self-serve"] and "policy-smb" not in context
    assert split_priors("Memory context:\n- a b") == ("Memory context:\n- a b", "")
    assert drop_policy_lines("x", ["jev-demos/x/y/learned-a"]) == ("x", ["jev-demos/x/y/learned-a"])


def test_situation_block_orders_by_tries_and_tags() -> None:
    from types import SimpleNamespace as E

    ep = "jev-demos/x/y"
    entries = [
        E(entity_path=ep, key="seen-mid-market-na--self-serve",
          value="Seen on mid_market/na with decide:self_serve: worked 1 of 1 time. Recent reviews: [worked] Bought online."),
        E(entity_path=ep, key="seen-mid-market-na--route-to-ae",
          value="Seen on mid_market/na with decide:route_to_ae: worked 9 of 12 times; the last 3 did not. Recent reviews: [did not work] Bought online while waiting."),
    ]
    block, keys = situation_block("mid_market/na", entries)
    assert keys == [f"{ep}/seen-mid-market-na--route-to-ae", f"{ep}/seen-mid-market-na--self-serve"]
    tagged, tags = tag_entries(block, keys)
    assert "- [M1] seen-mid-market-na--route-to-ae (decide:route_to_ae): worked 9 of 12 times" in tagged
    assert "- [M2] seen-mid-market-na--self-serve (decide:self_serve): worked 1 of 1 time" in tagged
    assert situation_block("smb/na", []) == ("", [])


def test_tag_entries_dedupes_keys() -> None:
    text = "A:\n- a/b/k1 (x): one\n\nB:\n- a/b/k2 (y): two\n"
    tagged, tags = tag_entries(text, ["a/b/k1", "a/b/k2", "a/b/k1"])
    assert tags == {"M1": "a/b/k1", "M2": "a/b/k2"} and tagged.count("[M1]") == 1
