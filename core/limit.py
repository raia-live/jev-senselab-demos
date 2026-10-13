"""Measure how much state Jev accepts in one call.

    uv run python -m core.limit

Sends the same lead with a growing ``memory`` field until the API answers
``max_tokens_exceeded``, then bisects. Costs a few cents of Jev. The result on
``jev-1.13.0`` (October 2026) was 32,768 input tokens, about 140,000 characters of
JSON state — the number ``core.report`` draws as "Jev's input limit".
"""

from __future__ import annotations

import time

from . import settings as S
from .jev import Jev

#: Measured with this script: the largest state Jev took, in input tokens and characters.
JEV_MAX_INPUT_TOKENS = 32_768
JEV_MAX_STATE_CHARS = 140_000

CASE = {
    "lead": "We are a 250-person logistics company in Denver. Can you send pricing for 60 seats? "
            "— Dana Ortiz, VP Ops (dana@ortizfreight.com)",
    "fields": {"company_size": 250, "country": "US"},
}
CRITERIA = {"route_to_ae": "account executive", "sdr_nurture": "SDR nurture", "self_serve": "self-serve",
            "partner": "regional partner", "disqualify": "disqualify"}
LINE = ("- past case: 240-person company in Austin asked for pricing; routed to an AE; review: prospect "
        "bought online while waiting for the callback, AE cycle added a week and no value.\n")


def accepts(jev: Jev, chars: int) -> tuple[bool, int, float]:
    memory = (LINE * (chars // len(LINE) + 1))[:chars]
    d = jev.decide(dict(CASE, memory=memory), instructions="Decide how to route this inbound lead.", criteria=CRITERIA)
    return d.error is None, d.input_tokens, d.latency_ms


def main() -> None:
    settings = S.load()
    S.require(settings, senselab=False)
    jev = Jev(settings.typesafe_api_key or "")
    print(f"model {S.JEV_MODEL}")
    lo, hi, lo_tokens = 0, None, 0
    for chars in (4_000, 16_000, 64_000, 128_000, 256_000, 512_000):
        ok, toks, ms = accepts(jev, chars)
        print(f"{chars:>8,} chars -> {'ok' if ok else 'max_tokens_exceeded':<20s} input_tokens={toks:>6}  {ms:5.0f} ms")
        if ok:
            lo, lo_tokens = chars, toks
        else:
            hi = chars
            break
        time.sleep(0.3)
    if hi is None:
        print("no limit found below 512k chars")
        return
    while hi - lo > 2_000:
        mid = (lo + hi) // 2
        ok, toks, _ = accepts(jev, mid)
        if ok:
            lo = mid
            lo_tokens = toks
        else:
            hi = mid
        time.sleep(0.3)
    print(f"\nJev accepts up to ~{lo:,} chars of state (~{lo_tokens:,} input tokens); {hi:,} chars is rejected.")
    jev.close()


if __name__ == "__main__":
    main()
