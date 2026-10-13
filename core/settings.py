"""Environment: the two keys and the SenseLab URL, read from ``.env`` or the shell."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

DEFAULT_AMFS_URL = "https://amfs-login.sense-lab.ai"
DASHBOARD_URL = "https://amfs.sense-lab.ai"
JEV_MODEL = os.environ.get("JEV_MODEL", "jev-1.13.0")
AGENT_VERSION = "jev-senselab-demos/0.1.0"
RUNTIME = "jev-senselab-demos"


@dataclass(frozen=True)
class Settings:
    typesafe_api_key: str | None
    amfs_api_key: str | None
    amfs_url: str

    @property
    def has_jev(self) -> bool:
        return bool(self.typesafe_api_key)

    @property
    def has_senselab(self) -> bool:
        return bool(self.amfs_api_key)


def load() -> Settings:
    return Settings(
        typesafe_api_key=(os.environ.get("TYPESAFE_API_KEY") or "").strip() or None,
        amfs_api_key=(os.environ.get("AMFS_API_KEY") or "").strip() or None,
        amfs_url=(os.environ.get("AMFS_HTTP_URL") or DEFAULT_AMFS_URL).rstrip("/"),
    )


def require(settings: Settings, *, jev: bool = True, senselab: bool = True) -> None:
    missing = []
    if jev and not settings.has_jev:
        missing.append("TYPESAFE_API_KEY (Jev, https://typesafe.ai)")
    if senselab and not settings.has_senselab:
        missing.append(f"AMFS_API_KEY (SenseLab, {DASHBOARD_URL})")
    if missing:
        raise SystemExit(
            "Missing keys: " + "; ".join(missing) + "\n"
            "Copy .env.example to .env and fill them in, or export them in your shell."
        )
