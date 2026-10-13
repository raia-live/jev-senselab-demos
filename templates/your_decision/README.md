# Wrap your own Jev decision

`decide.py` is the whole integration in one file, on a made-up refund decision. Copy it, and
edit the three marked blocks:

1. Your decision: the agent id, where its memory lives, and the actions Jev chooses among.
2. Your cases: whatever you already send Jev as state. Add a `situation` label from
   fields you can observe (`"late-delivery/first-time"`); lessons are keyed to it.
3. How you learn whether it worked: a chargeback, a CSAT score, a reviewer. In
   production this arrives later; seal the trace when it does.

```bash
uv run python templates/your_decision/decide.py
```

## The pattern, in order

```python
memory = client.memory(AGENT_ID)                  # one session per decision
run = Run(memory)
guidance = run.begin(request, entity_path=..., candidate_actions=[...], situation=...)

own = [e for e in memory.list(ENTITY_PATH) if e.key.startswith(ReviewHistory.prefix(situation))]
record, record_keys = situation_block(situation, own)       # this situation's tallies and last reviews
context, priors = split_priors(guidance.text)               # the SDK's entries list / its recommendation
if guard := noise_guard(guidance.mode, parse_record(own)):  # leave the record only when nothing works
    priors = hold_action(priors, guard[1])
text, tags = tag_entries(join(corrections, record, priors, context), keys)   # [M1], [M2] ... for citation

decision = jev.decide(state | {"memory": text}, criteria=ACTIONS, basis=...)   # Jev decides, cites [M#]
memory.record_llm_call(...)                                                     # cost on the trace
run.on_tool_result("decide", {...}, action_key=f"decide:{decision.action}")     # the action, for priors

run.learn(situation, action_key, worked, note)    # the lesson, when the outcome arrives
memory.write(ENTITY_PATH, ReviewHistory.key(situation, action_key), history.add(...))   # the reviews themselves
run.complete(severity, outcome_ref=..., causal_entry_keys=cited + lessons_for(situation, action))
```

Three details that matter more than they look:

- Read the recommendation against the record. SenseLab's `explore` is an exploration
  assignment for an agent retrying one task. In a stream of fresh decisions with imperfect
  reviews, `noise_guard` holds an action that has worked here and has not failed twice in a
  row, lets the exploration stand when something has failed twice and nothing is working
  (that is what a change looks like), and hands a record of single failures back to your
  written policy. One constant (`TURNED_STREAK = 2`), a few dozen lines, measured in the
  lead-routing demo.

- Credit only what drove the decision. `causal_entry_keys` names the entry Jev cited and
  the lessons that claimed the action would work. Leave it out and every entry read gets
  charged with every failure, which is how a correct rule loses confidence for sharing a
  prompt with a wrong one.
- Render the corrections. When the repair loop canaries a fix, the corrective entry is
  served on the fix's branch but kept out of the SDK's rendered text (it is not meant for
  model training). `corrections_block()` puts it back at the top of the memory slot so Jev
  reads it and can cite it.

## Turning the loop on

Once your agent has sealed traces, the repair loop runs on the SenseLab side: a judge grades
failed traces, clusters them into a behaviour, opens an incident, drafts a corrective entry,
tests it against the failed runs, canaries it against live traffic, and ships it to `main`
if the canary holds. Turn it on from the dashboard, or from code:

```python
from core.eval_client import EvalClient
api = EvalClient(settings.amfs_api_key)
api.set_repair_settings(AGENT_ID, auto_propose=True, repair_policy="auto_after_canary",
                        canary_pct=50, canary_min_outcomes=5)
```

`core/loops.py` shows the few calls a process makes to watch it (list fixes, evaluate a
canary, recheck a shipped fix); none of them are required for the loop to run.
