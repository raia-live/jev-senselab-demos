"""Wrap one of your own Jev decisions with SenseLab — the whole integration in one file.

    uv run python templates/your_decision/decide.py

Edit the three marked blocks: the actions, the state you already send Jev, and how you
learn whether a decision worked. Everything else is the integration pattern the demos use:

    memory = client.memory(agent_id)              # one session per decision
    run = Run(memory); guidance = run.begin(...)   # what memory knows about cases like this
    decision = jev.decide(state + memory slot)     # Jev decides; cites the entry it relied on
    run.on_tool_result("decide", ...)              # the action, for priors
    run.learn(situation, action, worked, note)     # the lesson
    run.complete(outcome, causal_entry_keys=...)   # seal the trace; credit what drove it

Nothing here is specific to the demos' scenarios. The repair loop (judge, incident, fix,
canary, ship) runs on the SenseLab side once your agent has sealed enough traces; turn it on
from the dashboard or with `EvalClient.set_repair_settings(agent_id, auto_propose=True)`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core import settings as S  # noqa: E402
from core.arms import MEMORY_NOTE  # noqa: E402
from core.jev import Jev  # noqa: E402
from core.senselab import (  # noqa: E402
    ReviewHistory,
    SenseLabClient,
    corrections_block,
    hold_action,
    noise_guard,
    parse_record,
    situation_block,
    split_priors,
    tag_entries,
)

# ---- 1. your decision ---------------------------------------------------------------------
AGENT_ID = "my-refund-agent"                       # stable per deployment; one memory per agent
ENTITY_PATH = "myapp/refunds"                      # where the memory lives
INSTRUCTIONS = "Decide how to handle this refund request."
ACTIONS = {
    "approve": "Refund in full now.",
    "partial": "Offer a partial refund or credit.",
    "deny": "Decline the refund.",
    "escalate": "Hand to a human agent.",
}

# ---- 2. the cases: whatever you already send Jev as state --------------------------------
CASES = [
    {"id": "r-1001", "situation": "late-delivery/first-time", "request": "Order arrived 9 days late; customer asks for a refund. First order.",
     "fields": {"days_late": 9, "orders": 1, "amount": 48.0}},
    {"id": "r-1002", "situation": "damaged/repeat", "request": "Item arrived damaged, photo attached. Third claim this year.",
     "fields": {"claims_this_year": 3, "orders": 14, "amount": 120.0}},
]


# ---- 3. how you learn whether it worked -------------------------------------------------
def review(case: dict, action: str) -> tuple[bool, str, str]:
    """Return (worked, severity, note). In production this comes later — a chargeback, a
    CSAT score, a supervisor's review — and you would seal the trace when it arrives."""
    if case["id"] == "r-1002" and action == "approve":
        return False, "failure", "Chargeback filed anyway; third claim from this account was fraudulent."
    return True, "success", "Customer satisfied; no further contact."


def main() -> None:
    settings = S.load()
    S.require(settings)
    jev = Jev(settings.typesafe_api_key)
    client = SenseLabClient(settings.amfs_api_key, base_url=settings.amfs_url)
    from amfs import Run

    history = ReviewHistory()  # what the reviews said about each (situation, action); written to memory below
    for case in CASES:
        memory = client.memory(AGENT_ID)
        run = Run(memory)
        guidance = run.begin(
            case["request"], entity_path=ENTITY_PATH, model=S.JEV_MODEL, agent_version="my-refund-agent/1.0",
            runtime="my-service", candidate_actions=[f"decide:{a}" for a in ACTIONS],
            situation=case["situation"], limit=12,
        )
        # This situation's own record, read back by key: one line per action tried here, with
        # the tally and the last reviews. Retrieval ranks by similarity to the request text and
        # can rank it below rules about other situations, so it is read directly and shown first.
        prefix = ReviewHistory.prefix(case["situation"])
        own = [e for e in memory.list(ENTITY_PATH) if e.key.startswith(prefix)]
        record, record_keys = situation_block(case["situation"], own)
        # Leave the record only when nothing in it is working: hold an action that has worked
        # here and has not failed twice in a row, even when the SDK suggests exploring past it;
        # let the exploration stand when something has failed twice and nothing is working;
        # hand a record of single failures back to the written policy.
        context, priors = split_priors(guidance.text or "")
        guard = noise_guard(guidance.mode, parse_record(own))
        if guard:
            priors = hold_action(priors, guard[1])
        # The repair loop's corrections are served on a canary branch but kept out of the SDK's
        # rendered text (they are not for model training); render them first so Jev reads them.
        corrections, corrective_keys = corrections_block(guidance)
        text = "\n\n".join(part for part in (corrections, record, priors, context.strip()) if part)
        shown = corrective_keys + record_keys + [k for k in guidance.entry_keys if k not in record_keys]
        memory_text, tags = tag_entries(text, shown)
        basis = {t: f"memory entry {t}: {k.rsplit('/', 1)[-1]}" for t, k in tags.items()} or None
        if basis:
            basis["none"] = "no memory entry was decisive"

        state = {"request": case["request"], "fields": case["fields"], "memory": memory_text or "(empty)"}
        decision = jev.decide(
            state, instructions=INSTRUCTIONS + MEMORY_NOTE,
            criteria=ACTIONS, basis=basis,
        )
        memory.record_llm_call(decision.model, decision.input_tokens, decision.output_tokens, provider="typesafe",
                               latency_ms=decision.latency_ms)
        run.on_tool_result("decide", {"action": decision.action, "confidence": round(decision.confidence, 3)},
                           result=f"decision recorded: {decision.action}", action_key=f"decide:{decision.action}")

        worked, severity, note = review(case, decision.action)
        action_key = f"decide:{decision.action}"
        run.learn(case["situation"], action_key, worked, note)
        # Keep the reviews themselves: run.learn keeps one lesson per situation and the newest
        # outcome overwrites it; the record above needs the tally and the last few notes.
        memory.write(ENTITY_PATH, ReviewHistory.key(case["situation"], action_key),
                     history.add(case["situation"], action_key, worked, note), confidence=0.85, memory_type="experience")
        # Credit the entry Jev cited, and the lessons about *this situation* that claimed the
        # action works. A lesson about another situation is not charged for this outcome.
        causal = [tags[decision.basis]] if decision.basis in tags and "/seen-" not in tags[decision.basis] else []
        causal += [les["key"] for les in guidance.lessons
                   if les["situation"] == case["situation"] and les["action"] == action_key and les["worked"]]
        run.complete(severity, outcome_ref=case["id"], verified_by="review",
                     evidence={"review_note": note}, response_text=f"decision: {decision.action}. review: {note}",
                     decision_summary=f"{case['situation']}: {action_key}", causal_entry_keys=causal)
        print(f"{case['id']}: {decision.action} ({decision.confidence:.2f}) basis={decision.basis} -> {severity}: {note}")

    print(f"\nTraces for {AGENT_ID} are on your dashboard: {S.DASHBOARD_URL}")


if __name__ == "__main__":
    main()
