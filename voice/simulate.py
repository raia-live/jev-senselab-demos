"""Calls without a microphone: the scenario's stream of callers, sent to the running server
in the shape Vapi sends it (``tool-calls`` and ``end-of-call-report`` messages).

    uv run uvicorn voice.server:app --port 8787          # in one terminal
    uv run python -m voice.simulate --calls 40           # in another

Each call registers the caller's account facts, waits a moment (the caller explaining, during
which the server opens runs for the situations the caller could be in), sends the tool call
the conversational model would send, then the end-of-call report. The server reviews each
decision the demo world's way and writes the outcome to memory, so the live agent you then
call from the browser has these calls behind it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from collections import Counter
from typing import Any

import httpx

import core.settings  # noqa: F401 - loads .env, where the webhook secret lives
from core.scenario import Case
from demos.call_handling.scenario import SCENARIO


def tool_call_message(call_id: str, case: Case) -> dict[str, Any]:
    f = case.fields
    args = {"intent": f["intent"], "summary": case.text, "amount_usd": f["amount_usd"], "sentiment": f["sentiment"]}
    return {"message": {
        "type": "tool-calls", "timestamp": int(time.time() * 1000),
        "toolCallList": [{"id": f"call_{uuid.uuid4().hex[:16]}", "type": "function",
                          "function": {"name": "decide_next_step", "arguments": args}}],
        "call": {"id": call_id, "type": "webCall"},
    }}


def end_message(call_id: str, action: str) -> dict[str, Any]:
    return {"message": {
        "type": "end-of-call-report", "endedReason": "assistant-ended-call",
        "artifact": {"transcript": f"(simulated) decision {action}"},
        "call": {"id": call_id, "type": "webCall"},
    }}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="drive the voice server with the scenario's callers")
    p.add_argument("--server", default="http://127.0.0.1:8787")
    p.add_argument("--calls", type=int, default=40)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--change-at", type=int, default=None, help="flip the demo world's change before this call (default: half way)")
    p.add_argument("--start", type=int, default=0, help="skip the first N cases of the stream (continue an earlier session)")
    p.add_argument("--wait", type=float, default=4.0, help="seconds between the call starting and the decision (prefetch window)")
    p.add_argument("--no-change", action="store_true", help="never flip the change")
    args = p.parse_args(argv)

    change_at = args.change_at if args.change_at is not None else (args.start + args.calls) // 2
    total = args.start + args.calls
    cases = SCENARIO.cases(seed=args.seed, episodes=total, change_at=(10**9 if args.no_change else change_at), probe_at=10**9)
    # The server checks Vapi's shared secret on the webhook; send the same one.
    secret = (os.environ.get("VAPI_WEBHOOK_SECRET") or "").strip()
    c = httpx.Client(base_url=args.server, timeout=90.0, headers={"x-vapi-secret": secret} if secret else {})
    try:
        cfg = c.get("/api/config").json()
    except httpx.HTTPError as exc:
        print(f"server not reachable at {args.server}: {exc}", file=sys.stderr)
        return 2
    print(f"agent {cfg['agent_id']} on {cfg['entity_path']}; {args.calls} calls from case {args.start}, change at {change_at}")
    wrong = Counter()
    by_sit: dict[str, list[bool]] = {}
    changed = bool(cfg.get("changed"))
    for case in cases[args.start:]:
        want_changed = (not args.no_change) and case.episode >= change_at
        if want_changed != changed:
            c.post("/api/world", json={"changed": want_changed})
            changed = want_changed
            print(f"--- the world changed: {'retention offer live' if changed else 'offer withdrawn'} ---")
        call_id = f"sim-{args.seed}-{case.episode}-{uuid.uuid4().hex[:6]}"
        f = case.fields
        c.post("/api/calls", json={"call_id": call_id, "caller": {
            "name": "Sim caller", "tier": f["tier"], "prior_contacts_30d": f["prior_contacts_30d"],
            "tenure_months": f["tenure_months"]}})
        time.sleep(args.wait)
        t0 = time.perf_counter()
        r = c.post("/vapi/webhook", json=tool_call_message(call_id, case)).json()
        decide_ms = (time.perf_counter() - t0) * 1000
        result = (r.get("results") or [{}])[0].get("result", "")
        action = result.split("Next step: ", 1)[-1].split(".", 1)[0] if "Next step: " in result else "?"
        c.post("/vapi/webhook", json=end_message(call_id, action))
        # The review and the seal happen off the request path; wait for the outcome to land.
        d = None
        for _ in range(60):
            state = c.get("/api/state").json()
            d = next((x for x in reversed(state["decisions"]) if x["call_id"] == call_id), None)
            if d is None or d.get("outcome"):
                break
            time.sleep(1.0)
        ok = action == case.truth
        wrong["all"] += int(not ok)
        if case.is_tacit:
            wrong["tacit"] += int(not ok)
        if case.is_changed:
            wrong["changed"] += int(not ok)
        by_sit.setdefault(case.situation, []).append(ok)
        out = (d or {}).get("outcome") or {}
        mark = "ok  " if ok else "MISS"
        pf = "pre" if (d or {}).get("memory_prefetched") else "live"
        print(f"{case.episode:3d} {mark} {case.situation:42s} {action:18s} truth {case.truth:18s} "
              f"{decide_ms:6.0f}ms {pf:4s} mem {(d or {}).get('memory_mode') or '-':14s} "
              f"{'worked' if out.get('success') else 'did not work'}: {str(out.get('note') or '')[:70]}")
    print()
    print(f"wrong: {wrong['all']} of {args.calls}; on the tacit rule {wrong['tacit']}; after the change {wrong['changed']}")
    for sit, oks in sorted(by_sit.items()):
        print(f"  {sit:42s} {sum(oks):3d} of {len(oks):3d} right   " + "".join("." if o else "x" for o in oks))
    print(json.dumps({"begin_ms_median": state.get("begin_ms_median"), "begin_count": state.get("begin_count")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
