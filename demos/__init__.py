"""The four demos. Each module exposes ``SCENARIO``."""

from __future__ import annotations

import importlib

from core.scenario import Scenario

NAMES = ("lead_routing", "ap_invoices", "marketing", "moderation")


def load(name: str) -> Scenario:
    if name not in NAMES:
        raise SystemExit(f"unknown scenario {name!r}; choose from {', '.join(NAMES)}")
    mod = importlib.import_module(f"demos.{name}.scenario")
    return mod.SCENARIO


__all__ = ["NAMES", "load"]
