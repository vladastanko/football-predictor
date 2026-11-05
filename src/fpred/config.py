"""Project paths and configuration loading."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    def ensure(self) -> "Paths":
        for p in (self.raw, self.processed, self.reports):
            p.mkdir(parents=True, exist_ok=True)
        return self


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def season_code(start_year: int) -> str:
    """2024 -> '2425' (football-data.co.uk folder naming)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    """2024 -> '2024/25'."""
    return f"{start_year}/{(start_year + 1) % 100:02d}"
