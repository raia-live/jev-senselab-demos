# The retrieval baseline

This is the "we could just retrieve past cases into the prompt" objection, built as well as
we could build it so the comparison is fair.

Every decision Jev makes is embedded locally (`BAAI/bge-small-en-v1.5` through
`fastembed`) together with the action taken and the review that came back. On the next
case, the `k` most similar past cases are rendered into the same `memory` slot the
SenseLab arm uses: same Jev model, same question, same state layout.

| arm | what it retrieves |
|---|---|
| `jev+retrieval:meta-k8` (default) | the 8 nearest past cases **with the same situation label**, recency-weighted |
| `jev+retrieval:text-k8` (`--arms ...,retrieval:text-k8`) | the 8 nearest past cases by text similarity alone |

The default is given the scenario's own situation label as a filter, which is more than
most deployments have; it is there so that nobody can say the baseline lost on retrieval
quality alone.

The third way of "loading past cases", every one of them verbatim, is the `jev+history`
arm in `core/arms.py`. It needs no embeddings and hits Jev's 32k-token input limit at
about case 276 of this demo (`--arms jev,history --episodes 400` shows it).

What a lookup cannot do, by construction: it has no notion of whether a past case's review
was noise or signal, no way to notice that a long-validated pattern has started failing, no
corrected rule to fall back on, no audit trail of what changed and why, and nothing that
runs when the agent is not running. The numbers in each demo's `report.html` show what that
costs.
