"""Reports: a console table and a self-contained HTML page per scenario, plus an index.

Everything is computed from ``episodes.jsonl`` and the ``loop-*.json`` files, so
``python -m core.runner --replay results-sample/`` rebuilds the pages without any keys.
"""

from __future__ import annotations

import html
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from .limit import JEV_MAX_INPUT_TOKENS, JEV_MAX_STATE_CHARS
from .senselab import GUIDANCE_CAP_CHARS

JEV_USD_PER_INPUT_MTOK = 0.042  # TypeSafe list price for input tokens at the time of writing
WINDOW = 10
ARM_ORDER = ["jev", "jev+history", "jev+retrieval:text-k8", "jev+retrieval:meta-k8", "jev+senselab"]
ARM_COLORS = {
    "jev": "#9aa0a6", "jev+history": "#5f6368", "jev+retrieval:text-k8": "#f4b400", "jev+retrieval:meta-k8": "#e37400",
    "jev+senselab": "#0f9d58",
}


def load_rows(sdir: Path) -> list[dict[str, Any]]:
    path = sdir / "episodes.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def load_loops(sdir: Path) -> dict[tuple[str, int], dict[str, Any]]:
    out: dict[tuple[str, int], dict[str, Any]] = {}
    for p in sorted(sdir.glob("loop-*-s*.json")):
        stem = p.stem  # loop-<slug>-s<seed>[-resume]
        if stem.endswith("-resume"):
            continue
        try:
            slug, seed = stem[len("loop-"):].rsplit("-s", 1)
            out[(slug, int(seed))] = json.loads(p.read_text())
        except (ValueError, json.JSONDecodeError):
            continue
    return out


def _acc(rows: list[dict[str, Any]]) -> float | None:
    return (sum(1 for r in rows if r.get("correct")) / len(rows)) if rows else None


def _recover_episode(rows: list[dict[str, Any]], *, start: int) -> int | None:
    """First episode from which the arm gets at least 4 of the next 5 of these cases right."""
    seq = [r for r in sorted(rows, key=lambda r: r["episode"]) if r["episode"] >= start]
    for i in range(len(seq)):
        window = seq[i:i + 5]
        if len(window) < 5:
            break
        if sum(1 for r in window if r.get("correct")) >= 4:
            return int(seq[i]["episode"]) - start
    return None


def _churn(rows: list[dict[str, Any]]) -> float | None:
    """Share of consecutive same-truth cases on which the action flipped."""
    seq = sorted(rows, key=lambda r: r["episode"])
    flips = pairs = 0
    for a, b in zip(seq, seq[1:], strict=False):
        if a.get("truth") == b.get("truth"):
            pairs += 1
            if a.get("action") != b.get("action"):
                flips += 1
    return (flips / pairs) if pairs else None


def seed_metrics(rows: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    change_at = int(meta.get("change_at", -1))
    main = [r for r in rows if not r.get("resume")]
    tacit = [r for r in main if r.get("is_tacit")]
    changed = [r for r in main if r.get("is_changed")]
    probe = [r for r in main if r.get("is_probe")]
    pre = [r for r in main if change_at < 0 or r["episode"] < change_at]
    post = [r for r in main if change_at >= 0 and r["episode"] >= change_at]
    conf_wrong = [r for r in main if not r.get("correct") and float(r.get("confidence") or 0) >= 0.8]
    in_tok = [int(r.get("jev_input_tokens") or 0) for r in main]
    state = [int(r.get("state_bytes") or 0) for r in main]
    return {
        "n": len(main),
        "wrong": sum(1 for r in main if not r.get("correct")),
        "dollars": round(sum(float(r.get("cost_usd") or 0) for r in main), 2),
        "accuracy": _acc(main),
        "accuracy_pre": _acc(pre),
        "accuracy_post": _acc(post),
        "tacit_accuracy": _acc(tacit),
        "changed_accuracy": _acc(changed),
        "probe_accuracy": _acc(probe),
        "confidently_wrong": (len(conf_wrong) / len(main)) if main else None,
        "churn_tacit": _churn(tacit),
        "recover_cycle1": _recover_episode(tacit, start=0),
        "recover_cycle2": _recover_episode(changed, start=change_at) if change_at >= 0 and changed else None,
        "mean_state_bytes": (statistics.fmean(state) if state else None),
        "mean_input_tokens": (statistics.fmean(in_tok) if in_tok else None),
        "jev_usd": round(sum(in_tok) / 1e6 * JEV_USD_PER_INPUT_MTOK, 4),
        "errors": sum(1 for r in main if r.get("error") or r.get("jev_error")),
    }


def _mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return statistics.fmean(vals) if vals else None


def _range(values: list[float | None]) -> tuple[float, float] | None:
    vals = [v for v in values if v is not None]
    return (min(vals), max(vals)) if vals else None


def aggregate(rows: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    by_arm: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_arm[r["arm"]][int(r["seed"])].append(r)
    out: dict[str, Any] = {}
    for arm, seeds in by_arm.items():
        per_seed = {seed: seed_metrics(rs, meta) for seed, rs in seeds.items()}
        keys = next(iter(per_seed.values())).keys()
        agg: dict[str, Any] = {"seeds": sorted(per_seed), "per_seed": per_seed}
        for k in keys:
            vals = [m[k] for m in per_seed.values()]
            agg[k] = _mean(vals)
            agg[f"{k}_range"] = _range(vals)
        agg["recover_cycle1_unrecovered"] = sum(1 for m in per_seed.values() if m["recover_cycle1"] is None)
        agg["recover_cycle2_unrecovered"] = sum(1 for m in per_seed.values() if m["recover_cycle2"] is None and m["changed_accuracy"] is not None)
        out[arm] = agg
    return out


def curves(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Rolling accuracy and cumulative dollars per arm, averaged over seeds, by episode."""
    by_arm: dict[str, dict[int, dict[int, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r.get("resume"):
            continue
        by_arm[r["arm"]][int(r["seed"])][int(r["episode"])] = r
    out: dict[str, Any] = {}
    for arm, seeds in by_arm.items():
        episodes = sorted({ep for s in seeds.values() for ep in s})
        acc_series: list[float | None] = []
        usd_series: list[float | None] = []
        for ep in episodes:
            accs, usds = [], []
            for s in seeds.values():
                if ep not in s:
                    continue
                window = [s[e] for e in range(max(0, ep - WINDOW + 1), ep + 1) if e in s]
                accs.append(sum(1 for r in window if r.get("correct")) / len(window))
                usds.append(sum(float(s[e].get("cost_usd") or 0) for e in s if e <= ep))
            acc_series.append(statistics.fmean(accs) if accs else None)
            usd_series.append(statistics.fmean(usds) if usds else None)
        out[arm] = {"episodes": episodes, "rolling_accuracy": acc_series, "cumulative_usd": usd_series}
    return out


def footprint(rows: list[dict[str, Any]], meta: dict[str, Any], metrics: dict[str, Any],
              history_rows: list[dict[str, Any]] | None = None) -> dict[str, Any] | None:
    """Characters sent to Jev per decision as history grows, for three ways of remembering:
    every past case in the prompt, a top-k lookup, and SenseLab — against Jev's input limit.

    The per-case size is taken from the run itself: the ``jev`` arm's state is the case plus
    the policy, so a past case (with its action and review) is that minus the policy. The
    lookup and SenseLab lines are the measured means for those arms.
    """
    jev_rows = sorted((r for r in rows if r["arm"] == "jev" and r.get("state_bytes")), key=lambda r: (r["seed"], r["episode"]))
    if not jev_rows:
        return None
    seed = jev_rows[0]["seed"]
    hist_rows = sorted((r for r in rows if r["arm"] == "jev+history" and r["seed"] == seed), key=lambda r: r["episode"])
    if not hist_rows and history_rows:
        # Measured in a sibling run of the same scenario (the history arm needs a few hundred
        # episodes to reach the limit; the main run does not have to carry them).
        hist_rows = sorted(history_rows, key=lambda r: r["episode"])
    measured = None
    if hist_rows:
        # The history arm actually ran: plot what it sent and where Jev started refusing.
        all_hist = [{"x": r["episode"], "y": r["state_bytes"]} for r in hist_rows]
        err = next((r for r in hist_rows if r.get("jev_error")), None)
        crossing = err["episode"] if err else len(hist_rows)
        measured = err["episode"] if err else None
        limit = err["state_bytes"] if err else JEV_MAX_STATE_CHARS
        horizon = hist_rows[-1]["episode"]
    else:
        policy_chars = len(json.dumps([p["text"] for p in meta["scenario"].get("policy", [])]))
        per_case = [max(80, r["state_bytes"] - policy_chars - len('"memory": "(empty)"') + len(r.get("note") or "") + 24) for r in jev_rows]
        mean_case = statistics.fmean(per_case)
        base = statistics.fmean(r["state_bytes"] for r in jev_rows)
        crossing = int((JEV_MAX_STATE_CHARS - base) / mean_case) + 1
        limit = JEV_MAX_STATE_CHARS
        horizon = max(crossing + crossing // 5, len(per_case))
        all_hist, total = [], 0.0
        for n in range(horizon + 1):
            if n:
                total += per_case[n - 1] if n - 1 < len(per_case) else mean_case
            all_hist.append({"x": n, "y": round(base + total)})
    flat = lambda value: [{"x": 0, "y": round(value)}, {"x": horizon, "y": round(value)}]  # noqa: E731
    label = "every past case in the prompt" + (" (measured)" if measured is not None else " (projected from this run's case sizes)")
    datasets = [{"label": label, "data": all_hist, "borderColor": "#5f6368", "backgroundColor": "#5f6368",
                 "pointRadius": 0, "borderWidth": 2, "borderDash": [] if measured is not None else [2, 3]}]
    lookup = next((metrics[a] for a in metrics if a.startswith("jev+retrieval")), None)
    if lookup and lookup.get("mean_state_bytes"):
        datasets.append({"label": "top-k lookup (k past cases)", "data": flat(lookup["mean_state_bytes"]), "borderColor": "#e37400",
                         "backgroundColor": "#e37400", "pointRadius": 0, "borderWidth": 2})
    sl = metrics.get("jev+senselab")
    senselab = round(sl["mean_state_bytes"]) if sl and sl.get("mean_state_bytes") else None
    if senselab:
        datasets.append({"label": "jev+senselab", "data": flat(senselab), "borderColor": ARM_COLORS["jev+senselab"],
                         "backgroundColor": ARM_COLORS["jev+senselab"], "pointRadius": 0, "borderWidth": 2})
    if measured is None and not senselab:
        return None
    return {"datasets": datasets, "crossing": crossing, "measured": measured, "senselab": senselab, "limit": limit}


def lifecycle(loops: dict[tuple[str, int], dict[str, Any]]) -> dict[str, Any]:
    """Fix lifecycle events by episode, per seed, for the SenseLab arm."""
    out: dict[str, Any] = {"events": [], "fixes": [], "verdicts": {}, "incidents": 0, "behaviors": 0}
    for (slug, seed), rep in loops.items():
        if slug != "senselab":
            continue
        seen: dict[str, str] = {}
        for ev in rep.get("events", []):
            kind = ev.get("kind")
            ep = ev.get("episode")
            if kind == "fix_status" and isinstance(ev.get("fix"), dict):
                f = ev["fix"]
                status = f.get("status")
                if seen.get(f["id"]) != status:
                    seen[f["id"]] = str(status)
                    out["events"].append({"seed": seed, "episode": ep, "fix": f["id"][:8], "event": status,
                                          "lever": f.get("lever"), "title": f.get("title")})
            elif kind in ("canary_start", "canary_timeout", "ship_by_hand", "test_kick"):
                out["events"].append({"seed": seed, "episode": ep, "fix": str(ev.get("fix_id"))[:8], "event": kind})
            elif kind == "canary_evaluate" and ev.get("canary_status") in ("passed", "failed") :
                out["events"].append({"seed": seed, "episode": ep, "fix": str(ev.get("fix_id"))[:8],
                                      "event": f"canary_{ev['canary_status']}"})
            elif kind == "recheck" and isinstance(ev.get("prevention"), dict) and ev["prevention"].get("status"):
                out["events"].append({"seed": seed, "episode": ep, "fix": str(ev.get("fix_id"))[:8],
                                      "event": f"prevention_{ev['prevention'].get('status')}"})
        for f in rep.get("fixes", []) or []:
            canary = f.get("canary") if isinstance(f.get("canary"), dict) else {}
            test = f.get("test") if isinstance(f.get("test"), dict) else {}
            out["fixes"].append({
                "seed": seed, "id": str(f.get("id"))[:8], "status": f.get("status"), "lever": f.get("lever"),
                "source": f.get("source"), "title": (f.get("title") or f.get("summary") or "")[:160] if isinstance(f.get("title") or f.get("summary"), str) else "",
                "test": test.get("status"), "canary": canary.get("status"),
                "canary_arms": f"{(canary.get('canary') or {}).get('graded', '-')}/{(canary.get('control') or {}).get('graded', '-')} graded, "
                               f"{(canary.get('canary') or {}).get('read', '-')} read" if canary else "",
                "shipped_at": f.get("shipped_at"), "rolled_back_at": f.get("rolled_back_at"),
            })
        vs = rep.get("verdict_summary") or {}
        out["verdicts"][str(seed)] = vs
        inc = rep.get("incidents")
        if isinstance(inc, dict):
            out["incidents"] += len(inc.get("items") or inc.get("incidents") or [])
        elif isinstance(inc, list):
            out["incidents"] += len(inc)
        beh = rep.get("behaviors")
        if isinstance(beh, dict):
            out["behaviors"] += len(beh.get("items") or beh.get("behaviors") or [])
        elif isinstance(beh, list):
            out["behaviors"] += len(beh)
    out["events"].sort(key=lambda e: (e["seed"], e["episode"] if isinstance(e["episode"], int) else 10**6))
    return out


def _fmt(v: Any, kind: str = "num") -> str:
    if v is None:
        return "–"
    if kind == "pct":
        return f"{100 * v:.0f}%"
    if kind == "usd":
        return f"${v:,.0f}"
    if kind == "int":
        return f"{v:.0f}"
    if kind == "ep":
        return f"{v:.0f}"
    return f"{v:.2f}"


def console_summary(metrics: dict[str, Any], meta: dict[str, Any], life: dict[str, Any]) -> str:
    arms = sorted(metrics, key=lambda a: ARM_ORDER.index(a) if a in ARM_ORDER else 99)
    head = f"{'arm':<26s} {'wrong':>6s} {'$ lost':>9s} {'acc':>5s} {'tacit':>6s} {'chg':>5s} {'probe':>6s} {'churn':>6s} {'rec1':>5s} {'rec2':>5s} {'conf-wrong':>10s} {'state B':>8s}"
    lines = [head, "-" * len(head)]
    for arm in arms:
        m = metrics[arm]
        lines.append(
            f"{arm:<26s} {_fmt(m['wrong'], 'int'):>6s} {_fmt(m['dollars'], 'usd'):>9s} {_fmt(m['accuracy'], 'pct'):>5s} "
            f"{_fmt(m['tacit_accuracy'], 'pct'):>6s} {_fmt(m['changed_accuracy'], 'pct'):>5s} {_fmt(m['probe_accuracy'], 'pct'):>6s} "
            f"{_fmt(m['churn_tacit'], 'pct'):>6s} {_fmt(m['recover_cycle1'], 'ep'):>5s} {_fmt(m['recover_cycle2'], 'ep'):>5s} "
            f"{_fmt(m['confidently_wrong'], 'pct'):>10s} {_fmt(m['mean_state_bytes'], 'int'):>8s}"
        )
    if life.get("fixes"):
        lines.append("")
        lines.append("repair loop fixes:")
        for f in life["fixes"]:
            lines.append(f"  s{f['seed']} {f['id']} {f['status']:<18s} lever={f['lever']} test={f['test']} canary={f['canary']} {f['canary_arms']} {f['title'][:70]}")
    return "\n".join(lines)


HTML_HEAD = """<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
body{{font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;margin:0;background:#fafafa;color:#1f1f1f}}
main{{max-width:1100px;margin:0 auto;padding:28px 20px 60px}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:19px;margin:34px 0 10px}} .sub{{color:#5f6368;margin:0 0 18px}}
table{{border-collapse:collapse;width:100%;background:#fff;font-size:14px}} th,td{{padding:7px 9px;border-bottom:1px solid #e6e6e6;text-align:right}}
th:first-child,td:first-child{{text-align:left}} th{{background:#f1f3f4;font-weight:600}}
.card{{background:#fff;border:1px solid #e6e6e6;border-radius:8px;padding:14px 16px;margin:12px 0}}
canvas{{max-height:360px}} .legend{{color:#5f6368;font-size:13px}} code{{background:#eef;padding:1px 4px;border-radius:3px}}
.ev{{font-size:13px;color:#333}} .ev b{{color:#0f9d58}}
</style></head><body><main>"""


def _js(data: Any) -> str:
    return json.dumps(data)


def sibling_history_rows(sdir: Path, scenario: str) -> list[dict[str, Any]]:
    """The ``jev+history`` rows of another directory in the same results root that ran the
    same scenario — the wall run — when this one did not run that arm."""
    for other in sorted(p for p in sdir.parent.iterdir() if p.is_dir() and p != sdir):
        meta_path = other / "meta.json"
        if not meta_path.exists():
            continue
        try:
            if json.loads(meta_path.read_text())["scenario"]["name"] != scenario:
                continue
        except (KeyError, ValueError):
            continue
        rows = [r for r in load_rows(other) if r["arm"] == "jev+history"]
        if rows:
            seed = rows[0]["seed"]
            return [r for r in rows if r["seed"] == seed]
    return []


def build_scenario(sdir: Path, *, quiet: bool = False) -> dict[str, Any] | None:
    rows = load_rows(sdir)
    meta_path = sdir / "meta.json"
    if not rows or not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    loops = load_loops(sdir)
    metrics = aggregate(rows, meta)
    cur = curves(rows)
    life = lifecycle(loops)
    resume_rows = [r for r in rows if r.get("resume")]
    resume_acc = {}
    if resume_rows:
        by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in resume_rows:
            by_arm[r["arm"]].append(r)
        resume_acc = {arm: _acc(rs) for arm, rs in by_arm.items()}
    summary = {"metrics": metrics, "lifecycle": {k: v for k, v in life.items() if k != "events"},
               "events": life["events"], "resume_accuracy": resume_acc, "meta": meta}
    (sdir / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    text = console_summary(metrics, meta, life)
    if not quiet:
        print(f"\n== {meta['scenario']['title']} ({meta.get('run_id')}) ==")
        print(text)
    history_rows = sibling_history_rows(sdir, meta["scenario"]["name"])
    (sdir / "report.html").write_text(render_scenario_html(meta, metrics, cur, life, resume_acc, rows, history_rows))
    return summary


def render_scenario_html(meta: dict[str, Any], metrics: dict[str, Any], cur: dict[str, Any],
                         life: dict[str, Any], resume_acc: dict[str, Any], rows: list[dict[str, Any]] | None = None,
                         history_rows: list[dict[str, Any]] | None = None) -> str:
    sc = meta["scenario"]
    arms = sorted(metrics, key=lambda a: ARM_ORDER.index(a) if a in ARM_ORDER else 99)
    change_at, probe_at = meta.get("change_at", -1), meta.get("probe_at", -1)
    h = [HTML_HEAD.format(title=html.escape(sc["title"]))]
    h.append(f"<h1>{html.escape(sc['title'])}</h1>")
    seeds = meta.get("seeds") or []
    seed_label = f"seed {seeds[0]}" if len(seeds) == 1 else f"seeds {seeds}"
    h.append(f"<p class='sub'>run {html.escape(str(meta.get('run_id')))} · {meta.get('episodes')} episodes · {seed_label} · "
             f"change at {change_at} · probe at {probe_at} · review noise {meta.get('noise')} · Jev {html.escape(str(meta.get('jev_model')))}</p>")
    h.append("<div class='card'><b>Tacit rule the policy misses:</b> " + html.escape(sc.get("tacit_rule", "")) +
             "<br><b>Change half-way:</b> " + html.escape(sc.get("change_rule", "")) +
             "<br><b>What outcomes have to teach:</b> " + html.escape(sc.get("expected_fix", "")) + "</div>")

    # headline table
    h.append("<h2>Whole run</h2><table><tr><th>arm</th><th>wrong decisions</th><th>dollars lost</th><th>accuracy</th>"
             "<th>tacit cases</th><th>changed cases</th><th>held-out phrasing</th><th>churn</th><th>recover (cycle 1)</th>"
             "<th>recover (cycle 2)</th><th>confidently wrong</th><th>state bytes</th><th>Jev $</th></tr>")
    for arm in arms:
        m = metrics[arm]
        rng = m.get("wrong_range")
        wrong = _fmt(m["wrong"], "int") + (f" <span class='legend'>({rng[0]:.0f}–{rng[1]:.0f})</span>" if rng and rng[0] != rng[1] else "")
        rec1 = _fmt(m["recover_cycle1"], "ep") + (f" <span class='legend'>({m['recover_cycle1_unrecovered']} never)</span>" if m.get("recover_cycle1_unrecovered") else "")
        rec2 = _fmt(m["recover_cycle2"], "ep") + (f" <span class='legend'>({m['recover_cycle2_unrecovered']} never)</span>" if m.get("recover_cycle2_unrecovered") else "")
        h.append(f"<tr><td><span style='color:{ARM_COLORS.get(arm, '#333')}'>●</span> {html.escape(arm)}</td><td>{wrong}</td>"
                 f"<td>{_fmt(m['dollars'], 'usd')}</td><td>{_fmt(m['accuracy'], 'pct')}</td><td>{_fmt(m['tacit_accuracy'], 'pct')}</td>"
                 f"<td>{_fmt(m['changed_accuracy'], 'pct')}</td><td>{_fmt(m['probe_accuracy'], 'pct')}</td><td>{_fmt(m['churn_tacit'], 'pct')}</td>"
                 f"<td>{rec1}</td><td>{rec2}</td><td>{_fmt(m['confidently_wrong'], 'pct')}</td><td>{_fmt(m['mean_state_bytes'], 'int')}</td>"
                 f"<td>${m['jev_usd']:.3f}</td></tr>")
    h.append("</table><p class='legend'>Means over seeds (ranges in parentheses). Wrong decisions and dollars are scored against the hidden truth; "
             "the arms only ever saw the review. Recover = episodes from the start of the cycle until the arm gets 4 of the next 5 such cases right. "
             "Churn = share of consecutive same-truth tacit cases on which the action flipped.</p>")

    # charts
    h.append("<h2>Learning curves</h2><div class='card'><canvas id='acc'></canvas></div>")
    h.append("<div class='card'><canvas id='usd'></canvas></div>")
    datasets_acc, datasets_usd = [], []
    for arm in arms:
        c = cur.get(arm)
        if not c:
            continue
        color = ARM_COLORS.get(arm, "#333")
        datasets_acc.append({"label": arm, "data": [{"x": e, "y": (None if v is None else round(v, 3))} for e, v in zip(c["episodes"], c["rolling_accuracy"], strict=False)],
                             "borderColor": color, "backgroundColor": color, "pointRadius": 0, "borderWidth": 2, "tension": 0.2, "spanGaps": True})
        datasets_usd.append({"label": arm, "data": [{"x": e, "y": (None if v is None else round(v, 1))} for e, v in zip(c["episodes"], c["cumulative_usd"], strict=False)],
                             "borderColor": color, "backgroundColor": color, "pointRadius": 0, "borderWidth": 2, "tension": 0.1, "spanGaps": True})
    # fix lifecycle markers on the SenseLab accuracy curve
    markers = []
    rsi = cur.get("jev+senselab")
    if rsi:
        acc_by_ep = dict(zip(rsi["episodes"], rsi["rolling_accuracy"], strict=False))
        for ev in life.get("events", []):
            ep = ev.get("episode")
            if not isinstance(ep, int) or ep < 0:
                continue
            label = ev["event"]
            if label in ("proposed", "test_passed", "test_inconclusive", "test_failed", "canary", "canary_start", "canary_passed",
                         "canary_failed", "memory_shipped", "verified", "regressed", "rolled_back", "canary_timeout"):
                markers.append({"x": ep, "y": round(acc_by_ep.get(ep) or 0, 3), "label": f"{label} ({ev['fix']}) s{ev['seed']}"})
    if markers:
        datasets_acc.append({"label": "repair loop events (RSI)", "data": markers, "type": "scatter", "pointRadius": 5,
                             "pointStyle": "rectRot", "borderColor": "#0f9d58", "backgroundColor": "#fff", "borderWidth": 2, "showLine": False})
    h.append("<script>")
    h.append(f"const ACC={_js(datasets_acc)}; const USD={_js(datasets_usd)}; const CHANGE={change_at}; const PROBE={probe_at};")
    h.append("""
const vline = {id:'vline', afterDraw(chart){const {ctx, chartArea:{top,bottom}, scales:{x}} = chart;
  for (const [v,label,color] of [[CHANGE,'change','#d93025'],[PROBE,'held-out phrasing','#8e24aa']]) { if (v<0) continue;
    const px = x.getPixelForValue(v); ctx.save(); ctx.strokeStyle=color; ctx.setLineDash([4,4]); ctx.beginPath(); ctx.moveTo(px,top); ctx.lineTo(px,bottom); ctx.stroke();
    ctx.fillStyle=color; ctx.font='12px sans-serif'; ctx.fillText(label, px+4, top+12); ctx.restore(); } } };
function mk(id, datasets, ylabel, max){ new Chart(document.getElementById(id), {type:'line', data:{datasets},
  options:{parsing:false, animation:false, responsive:true, interaction:{mode:'nearest', intersect:false},
  scales:{x:{type:'linear', title:{display:true,text:'episode'}}, y:{min:0, max:max, title:{display:true,text:ylabel}}},
  plugins:{legend:{position:'bottom'}, tooltip:{callbacks:{label:(c)=> c.raw.label ? c.raw.label : `${c.dataset.label}: ${c.parsed.y}`}}}}, plugins:[vline]}); }
mk('acc', ACC, 'rolling accuracy (window 10)', 1.0); mk('usd', USD, 'cumulative dollars lost', undefined);
</script>""")

    # memory footprint against Jev's input limit
    fp = footprint(rows or [], meta, metrics, history_rows)
    if fp:
        h.append("<h2>What fits in Jev's input</h2>")
        h.append(f"<div class='card'>Jev takes at most <b>{JEV_MAX_INPUT_TOKENS:,} input tokens</b> per call (about {JEV_MAX_STATE_CHARS:,} characters "
                 f"of plain text, fewer for field-heavy text; measured with <code>python -m core.limit</code>, the API answers "
                 f"<code>max_tokens_exceeded</code> above it). "
                 + (f"Loading every past case into the prompt grows with history: the <code>jev+history</code> arm (every past case, verbatim) "
                    f"was refused by Jev from <b>case {fp['measured']:,}</b> on, and never decided again. " if fp["measured"] is not None else
                    f"Loading every past case into the prompt grows with history: at this demo's case size it crosses the limit after about "
                    f"<b>{fp['crossing']:,} cases</b> (run <code>--arms jev,history --episodes 400</code> to watch it happen). ")
                 + "A top-k lookup stays under it by only ever showing k cases, which is also all it knows at any decision. "
                 + (f"SenseLab's memory slot is capped at {GUIDANCE_CAP_CHARS:,} characters; everything Jev received from that arm (case, policy "
                    f"and memory) averaged {fp['senselab']:,} characters per decision whatever the history behind it, because a tally like "
                    f"\"worked 9 of 20\" is the same length at 20 outcomes and at 2,000. The history itself lives outside Jev, where the "
                    f"repair loop can work on it." if fp["senselab"] else
                    "SenseLab's memory slot does not grow with history (see the main run's report for its size); the history lives outside Jev, "
                    "where the repair loop can work on it.")
                 + "</div>")
        h.append("<div class='card'><canvas id='fp'></canvas></div>")
        h.append("<script>")
        h.append(f"const FP={_js(fp['datasets'])}; const LIMIT={fp['limit']};")
        h.append("""
const hline = {id:'hline', afterDraw(chart){const {ctx, chartArea:{left,right}, scales:{y}} = chart; const py = y.getPixelForValue(LIMIT);
  if (py < chart.chartArea.top || py > chart.chartArea.bottom) return; ctx.save(); ctx.strokeStyle='#d93025'; ctx.setLineDash([6,4]); ctx.beginPath(); ctx.moveTo(left,py); ctx.lineTo(right,py); ctx.stroke();
  ctx.fillStyle='#d93025'; ctx.font='12px sans-serif'; ctx.fillText("Jev refuses the call here (32k input tokens)", left+6, py-6); ctx.restore(); } };
new Chart(document.getElementById('fp'), {type:'line', data:{datasets:FP}, options:{parsing:false, animation:false, responsive:true,
  interaction:{mode:'nearest', intersect:false}, scales:{x:{type:'linear', title:{display:true,text:'past cases'}}, y:{min:0, title:{display:true,text:'characters sent to Jev per decision'}}},
  plugins:{legend:{position:'bottom'}}}, plugins:[hline]});
</script>""")

    # lifecycle
    h.append("<h2>What the repair loop did</h2>")
    vs = life.get("verdicts") or {}
    if vs:
        graded = sum(int(v.get("graded") or 0) for v in vs.values())
        failed = sum(int(v.get("failed") or 0) for v in vs.values())
        usd = sum(float(v.get("judge_usd") or 0) for v in vs.values())
        fixes = life.get("fixes") or []
        shipped = sum(1 for f in fixes if f["status"] in ("memory_shipped", "verified"))
        tested = sum(1 for f in fixes if f.get("test") in ("passed", "failed"))
        rejected = sum(1 for f in fixes if f.get("test") == "failed")
        gate = ""
        if fixes:
            gate = (f" The repair agent drafted {len(fixes)} fix{'es' if len(fixes) != 1 else ''}; {tested} reached the replay test (Tier 1), "
                    f"which replays a fix against the runs that failed before anything is shown to live traffic. "
                    + (f"{shipped} passed and shipped. " if shipped else "")
                    + (f"{rejected} would not have fixed those runs and {'were' if rejected != 1 else 'was'} stopped there: "
                       "a draft that does not fix the failures never reaches a live decision." if rejected else ""))
        h.append(f"<div class='card'>The judge graded <b>all {graded} decisions</b> the SenseLab arm made, as they happened (${usd:.2f} of the evaluation budget). "
                 f"It flagged <b>{failed}</b> on the reviews it was given (it reads the review, not the hidden truth, so flipped reviews count), "
                 f"grouped them into <b>{life.get('behaviors') or 0} behaviours</b> and opened <b>{life.get('incidents') or 0} incident"
                 f"{'s' if (life.get('incidents') or 0) != 1 else ''}</b>.{gate}</div>")
    if life.get("fixes"):
        h.append("<table><tr><th>seed</th><th>fix</th><th>status</th><th>lever</th><th>source</th><th>Tier 1 test</th><th>canary</th><th>canary arms</th><th>what it says</th></tr>")
        for f in life["fixes"]:
            h.append(f"<tr><td>{f['seed']}</td><td><code>{f['id']}</code></td><td>{html.escape(str(f['status']))}</td><td>{html.escape(str(f['lever']))}</td>"
                     f"<td>{html.escape(str(f['source']))}</td><td>{html.escape(str(f['test']))}</td><td>{html.escape(str(f['canary']))}</td>"
                     f"<td>{html.escape(f['canary_arms'])}</td><td style='text-align:left'>{html.escape(f['title'])}</td></tr>")
        h.append("</table>")
        h.append("<div class='card ev'>" + "<br>".join(
            f"s{e['seed']} ep {e['episode']}: <b>{html.escape(str(e['event']))}</b> <code>{e['fix']}</code>" + (f": {html.escape(str(e.get('title') or ''))}" if e.get('title') else "")
            for e in life["events"]) + "</div>")
    elif not vs:
        h.append("<p class='legend'>No SenseLab arm in this run.</p>")

    if resume_acc:
        h.append("<h2>Resumed run</h2><table><tr><th>arm</th><th>accuracy on the resumed episodes</th></tr>")
        for arm, a in resume_acc.items():
            h.append(f"<tr><td>{html.escape(arm)}</td><td>{_fmt(a, 'pct')}</td></tr>")
        h.append("</table><p class='legend'>The same agent and memory, run again later: the SenseLab arms start where they left off; the others start from the written policy.</p>")

    # the policy
    h.append("<h2>The written policy every arm starts from</h2><div class='card'>" + "<br>".join(html.escape(p["text"]) for p in sc.get("policy", [])) + "</div>")
    h.append("</main></body></html>")
    return "\n".join(h)


def build_all(root: Path, *, quiet: bool = False) -> None:
    summaries: dict[str, dict[str, Any]] = {}
    for sdir in sorted(p for p in root.iterdir() if p.is_dir()):
        s = build_scenario(sdir, quiet=quiet)
        if s:
            summaries[sdir.name] = s
    if summaries:
        (root / "index.html").write_text(render_index_html(root.name, summaries))


def render_index_html(run_id: str, summaries: dict[str, dict[str, Any]]) -> str:
    h = [HTML_HEAD.format(title=f"Jev + SenseLab demos: {html.escape(run_id)}")]
    h.append(f"<h1>Jev + SenseLab demos</h1><p class='sub'>run {html.escape(run_id)}</p>")
    for name, s in summaries.items():
        meta = s["meta"]
        metrics = s["metrics"]
        arms = sorted(metrics, key=lambda a: ARM_ORDER.index(a) if a in ARM_ORDER else 99)
        h.append(f"<h2><a href='{name}/report.html'>{html.escape(meta['scenario']['title'])}</a></h2>")
        h.append(f"<p class='sub'>{html.escape(name)} · run {html.escape(str(meta.get('run_id', run_id)))} · {meta.get('episodes', '?')} episodes · "
                 f"arms: {html.escape(', '.join(arms))}</p>")
        h.append("<table><tr><th>arm</th><th>wrong decisions</th><th>dollars lost</th><th>accuracy</th><th>tacit</th><th>changed</th><th>held-out</th><th>recover 1</th><th>recover 2</th></tr>")
        for arm in arms:
            m = metrics[arm]
            h.append(f"<tr><td><span style='color:{ARM_COLORS.get(arm, '#333')}'>●</span> {html.escape(arm)}</td><td>{_fmt(m['wrong'], 'int')}</td><td>{_fmt(m['dollars'], 'usd')}</td>"
                     f"<td>{_fmt(m['accuracy'], 'pct')}</td><td>{_fmt(m['tacit_accuracy'], 'pct')}</td><td>{_fmt(m['changed_accuracy'], 'pct')}</td>"
                     f"<td>{_fmt(m['probe_accuracy'], 'pct')}</td><td>{_fmt(m['recover_cycle1'], 'ep')}</td><td>{_fmt(m['recover_cycle2'], 'ep')}</td></tr>")
        h.append("</table>")
        lc = s.get("lifecycle") or {}
        if lc.get("fixes"):
            shipped = sum(1 for f in lc["fixes"] if f["status"] in ("memory_shipped", "verified"))
            h.append(f"<p class='legend'>repair loop: {len(lc['fixes'])} fixes proposed, {shipped} shipped · incidents {lc.get('incidents')} · behaviours {lc.get('behaviors')}</p>")
    h.append("</main></body></html>")
    return "\n".join(h)


__all__ = ["aggregate", "build_all", "build_scenario", "console_summary", "curves", "lifecycle", "load_rows", "seed_metrics"]
