"""Run a demo: ``python -m core.runner --scenario lead_routing``.

Writes ``results/<run-id>/<scenario>/episodes.jsonl`` (one row per arm × seed × episode),
``loop-<arm>-s<seed>.json`` (the repair loop's record for the SenseLab arm), ``meta.json`` and
``report.html``; then ``results/<run-id>/index.html`` over every scenario in the run.
"""

from __future__ import annotations

import argparse
import json
import secrets
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import report
from . import settings as S
from .arms import DEFAULT_ARMS, LoopKnobs, RunContext, make_arm
from .eval_client import EvalClient
from .jev import Jev
from .scenario import Scenario
from .senselab import SenseLabClient

RESULTS_DIR = S.REPO_ROOT / "results"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m core.runner", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", help="lead_routing | ap_invoices | marketing | moderation")
    p.add_argument("--all", action="store_true", help="run every scenario")
    p.add_argument("--episodes", type=int, default=120)
    p.add_argument("--quick", action="store_true",
                   help="a short look: 40 episodes, drift at 20, held-out phrasing at 30 (about 8 minutes)")
    p.add_argument("--seeds", default="11", help="case-stream seed (one is enough for a demo)")
    p.add_argument("--change-at", type=int, default=60, help="episode the drift takes effect (-1: never)")
    p.add_argument("--probe-at", type=int, default=90, help="episode held-out phrasings start (-1: never)")
    p.add_argument("--noise", type=float, default=0.10, help="share of reviews flipped at random")
    p.add_argument("--arms", default=",".join(DEFAULT_ARMS),
                   help="comma-separated: jev, retrieval:meta-k8, retrieval:text-k8, history, senselab")
    p.add_argument("--agents", type=int, default=1,
                   help="handle the stream with this many instances of the agent, round-robin: lookup and history "
                        "arms keep one store per instance, the SenseLab arm is one agent with one memory and one "
                        "repair loop (default 1)")
    p.add_argument("--policy", default="auto_after_canary", choices=["auto_after_canary", "auto_after_replay", "review"])
    p.add_argument("--canary-pct", type=int, default=50)
    p.add_argument("--canary-min", type=int, default=5)
    p.add_argument("--canary-timeout", type=int, default=60, help="stop a canary after this many sessions")
    p.add_argument("--look-every", type=int, default=5)
    p.add_argument("--no-judge", action="store_true", help="SenseLab arm without the judge (debug)")
    p.add_argument("--judge-mode", choices=["continuous", "on_demand"], default="continuous",
                   help="continuous: the platform grades traces as they seal; on_demand: this process asks per trace")
    p.add_argument("--run-id", help="reuse an id (with --resume: continue on the learned memory)")
    p.add_argument("--resume", action="store_true", help="continue a finished run for --episodes more episodes")
    p.add_argument("--replay", metavar="DIR", help="rebuild the reports from an existing results directory; no keys needed")
    p.add_argument("--out", default=str(RESULTS_DIR))
    p.add_argument("--parallel", type=int, default=6, help="arm×seed jobs to run side by side")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    if args.quick:
        args.episodes, args.change_at, args.probe_at = 40, 20, 30
        args.canary_timeout = min(args.canary_timeout, 25)
    return args


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M") + "-" + secrets.token_hex(2)


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str))


def _fmt_row(row: dict[str, Any], note: dict[str, Any] | None) -> str:
    verdict = "ok  " if row["correct"] else "MISS"
    rep = "pass" if row["success"] else row["severity"]
    bits = [
        f"ep {row['episode']:03d}", f"{row['situation']:<22s}", f"truth={row['truth']:<12s}",
        f"-> {row['action']:<12s} {verdict}", f"review={rep:<16s}", f"${row['cost_usd']:>7.0f}",
    ]
    if row.get("agent") is not None:
        bits.insert(1, f"a{row['agent']}")
    if row.get("memory_strength") is not None:
        bits.append(f"mem={row.get('memory_strength')}/{row.get('memory_mode') or '-'}")
        if row.get("causal_keys"):
            bits.append("credit=" + ",".join(str(k)[:28] for k in row["causal_keys"][:2]))
        elif row.get("basis"):
            bits.append(f"basis={row['basis']}")
    if row.get("canary"):
        bits.append(f"canary={str(row['canary'])[:28]}")
    if note and note.get("fixes"):
        bits.append("fixes=" + ",".join(f"{k}:{v}" for k, v in note["fixes"].items()))
    if row.get("jev_error"):
        bits.append(f"jev_error={row['jev_error'][:40]}")
    if row.get("commit_error"):
        bits.append(f"commit_error={row['commit_error'][:40]}")
    return " ".join(bits)


def run_scenario(scenario: Scenario, args: argparse.Namespace, *, run_id: str, out_dir: Path,
                 jev: Jev, senselab: SenseLabClient | None, eval_api: EvalClient | None, arms: list[str]) -> Path:
    sdir = out_dir / scenario.name
    sdir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip()]
    meta_path = sdir / "meta.json"
    meta: dict[str, Any] = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    offset = 0
    if args.resume:
        if not meta:
            raise SystemExit(f"--resume needs an existing run at {sdir}")
        offset = int(meta.get("episodes", 0))
        change_at, probe_at = int(meta.get("change_at", args.change_at)), int(meta.get("probe_at", args.probe_at))
    else:
        change_at, probe_at = args.change_at, args.probe_at
    knobs = LoopKnobs(policy=args.policy, canary_pct=args.canary_pct, canary_min_outcomes=args.canary_min,
                      canary_timeout_sessions=args.canary_timeout, look_every=args.look_every, judge=not args.no_judge,
                      judge_mode=args.judge_mode)
    budget_before = None
    if eval_api is not None:
        try:
            budget_before = eval_api.budget()
        except Exception:  # noqa: BLE001
            budget_before = None
    if not args.resume:
        meta = {
            "run_id": run_id, "scenario": scenario.describe(), "episodes": args.episodes, "seeds": seeds,
            "change_at": change_at, "probe_at": probe_at, "noise": args.noise, "arms": arms, "agents": args.agents,
            "knobs": knobs.__dict__, "jev_model": S.JEV_MODEL, "amfs_url": S.load().amfs_url,
            "started_at": datetime.now(UTC).isoformat(), "resumes": [],
        }
    else:
        meta.setdefault("resumes", []).append({"episodes": args.episodes, "started_at": datetime.now(UTC).isoformat()})
    _write_json(meta_path, meta)

    episodes_path = sdir / "episodes.jsonl"
    total = offset + args.episodes
    write_lock = threading.Lock()
    stop = threading.Event()

    def job(seed: int, spec: str) -> None:
        """One arm on one seed, start to finish. Independent of every other job: its own
        agent id, entity path and memory, so jobs run side by side."""
        cases = scenario.cases(seed=seed, episodes=total, change_at=change_at, probe_at=probe_at)[offset:]
        ctx = RunContext(scenario=scenario, jev=jev, run_id=run_id, seed=seed, noise=args.noise,
                         senselab=senselab, eval_api=eval_api, knobs=knobs, resume=bool(args.resume))
        agents = int(meta.get("agents", args.agents) or 1) if args.resume else args.agents
        arm = make_arm(spec, ctx, agents=agents)
        head = f"[{scenario.name} s{seed} {arm.name}]"
        t0 = time.perf_counter()
        setup = arm.setup()
        if not args.quiet:
            print(f"{head} setup: {json.dumps({k: v for k, v in setup.items() if k in ('agent_id', 'workers', 'stores', 'entity_path', 'seeded', 'judge', 'settings', 'auto_propose')}, default=str)[:400]}")
        for i, case in enumerate(cases):
            if stop.is_set():
                break
            ep = offset + i
            try:
                row = arm.episode(ep, case)
            except Exception as exc:  # noqa: BLE001 - one episode's failure is a row, not the end of the run
                row = {"run_id": run_id, "scenario": scenario.name, "seed": seed, "arm": arm.name, "episode": ep,
                       **case.to_row(), "action": None, "correct": False, "success": False, "severity": "failure",
                       "cost_usd": 0.0, "noisy": False, "true_success": False, "note": "",
                       "error": f"{type(exc).__name__}: {exc}"[:300]}
            note = None
            try:
                note = arm.after_episode(ep)
            except Exception as exc:  # noqa: BLE001
                note = {"error": f"{type(exc).__name__}: {exc}"[:200]}
            if note:
                row["loop"] = note
            if args.resume:
                row["resume"] = True
            memory_text = row.pop("memory_text", None)
            with write_lock:
                if memory_text:
                    # What Jev saw in its memory slot, kept beside the rows for the curious.
                    with (sdir / f"guidance-{arm.slug}-s{seed}.jsonl").open("a") as gh:
                        gh.write(json.dumps({"episode": ep, "case_id": case.id, "situation": case.situation,
                                             "action": row.get("action"), "basis": row.get("basis"),
                                             "memory": memory_text}) + "\n")
                with episodes_path.open("a") as fh:
                    fh.write(json.dumps(row, default=str) + "\n")
            if not args.quiet:
                print(f"{head} {_fmt_row(row, note)}")
        fin = arm.finish()
        if fin:
            suffix = "-resume" if args.resume else ""
            _write_json(sdir / f"loop-{arm.slug}-s{seed}{suffix}.json", fin)
            if not args.quiet and fin.get("lifecycle") is not None:
                print(f"{head} fixes at end: {fin.get('lifecycle')}  verdicts: {fin.get('verdict_summary')}")
        print(f"{head} done in {(time.perf_counter() - t0) / 60:.1f} min")

    jobs = [(seed, spec) for seed in seeds for spec in arms]
    workers = max(1, min(args.parallel, len(jobs)))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="arm") as pool:
        futures = [pool.submit(job, seed, spec) for seed, spec in jobs]
        try:
            for fut in as_completed(futures):
                fut.result()
        except KeyboardInterrupt:
            stop.set()
            print("\ninterrupted; letting the running episodes finish")
            raise
    if eval_api is not None:
        try:
            after = eval_api.budget()
            spent = None
            if budget_before and after:
                spent = round(float(after.get("spent_usd", 0)) - float(budget_before.get("spent_usd", 0)), 4)
            meta.setdefault("platform_budget", []).append({"before": budget_before, "after": after, "spent_usd": spent})
            if spent is not None:
                print(f"[{scenario.name}] SenseLab platform spend this scenario: ${spent:.2f} "
                      f"(remaining ${float(after.get('remaining_usd', 0)):.2f} of ${float(after.get('monthly_usd_cap', 0)):.0f} monthly cap)")
        except Exception:  # noqa: BLE001
            pass
    meta["finished_at"] = datetime.now(UTC).isoformat()
    _write_json(meta_path, meta)
    report.build_scenario(sdir, quiet=args.quiet)
    return sdir


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.replay:
        root = Path(args.replay)
        if not root.exists():
            raise SystemExit(f"no such directory: {root}")
        report.build_all(root, quiet=args.quiet)
        return 0
    from demos import NAMES, load

    if not args.scenario and not args.all:
        raise SystemExit("pass --scenario <name> or --all")
    names = list(NAMES) if args.all else [args.scenario]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    settings = S.load()
    need_senselab = "senselab" in arms
    S.require(settings, jev=True, senselab=need_senselab)

    if any(a.startswith("retrieval") for a in arms):
        from baselines.retrieval import available

        if not available():
            print("note: fastembed is not installed; skipping the retrieval baseline (uv sync --extra baseline to include it)")
            arms = [a for a in arms if not a.startswith("retrieval")]

    jev = Jev(settings.typesafe_api_key or "")
    try:
        models = jev.smoke()
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Jev check failed: {exc}\nIs TYPESAFE_API_KEY valid and does the account have credit?") from exc
    print(f"Jev ok: models {models}; using {S.JEV_MODEL}")

    senselab = eval_api = None
    if need_senselab:
        senselab = SenseLabClient(settings.amfs_api_key or "", base_url=settings.amfs_url)
        try:
            senselab.memory("jev-demo-smoke").briefing(entity_path="jev-demos/smoke", compact=True)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"SenseLab check failed against {settings.amfs_url}: {exc}\nIs AMFS_API_KEY valid?") from exc
        print(f"SenseLab ok: {settings.amfs_url}")
        eval_api = EvalClient(settings.amfs_api_key or "", base_url=settings.amfs_url)
        probe = eval_api.probe()
        if not probe["ok"]:
            raise SystemExit(
                f"SenseLab repair loop not available on this key ({probe['error']}).\n"
                f"The demo needs the eval API (/api/v1/eval); check the plan on {S.DASHBOARD_URL}."
            )
        b = probe["budget"] or {}
        print(f"SenseLab repair loop ok: judges on {(probe['readiness'] or {}).get('cheap_model')}, "
              f"repair agent on {(probe['readiness'] or {}).get('strong_model')}; "
              f"platform budget ${float(b.get('spent_usd', 0)):.2f} spent of ${float(b.get('monthly_usd_cap', 0)):.0f}")

    run_id = args.run_id or new_run_id()
    out_dir = Path(args.out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"run {run_id} -> {out_dir}  arms: {', '.join(arms)}")
    try:
        for name in names:
            run_scenario(load(name), args, run_id=run_id, out_dir=out_dir, jev=jev, senselab=senselab,
                         eval_api=eval_api, arms=arms)
    except KeyboardInterrupt:
        print("\ninterrupted; building reports from what finished")
    finally:
        jev.close()
        if eval_api is not None:
            eval_api.close()
    report.build_all(out_dir, quiet=True)
    print(f"\nreport: {out_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
