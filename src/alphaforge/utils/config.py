"""Config loading helpers."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UNIVERSE = REPO_ROOT / "config" / "universe.yaml"
DEFAULT_THRESHOLDS = REPO_ROOT / "config" / "thresholds.yaml"


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_universe(path: str | Path = DEFAULT_UNIVERSE) -> dict:
    return load_yaml(path)


def all_tickers(universe: dict) -> list[str]:
    seen: list[str] = []
    for group in universe["universe"].values():
        for t in group:
            if t not in seen:
                seen.append(t)
    return seen


def synthetic_market_cfg(thresholds_path: str | Path = DEFAULT_THRESHOLDS) -> dict:
    return load_yaml(thresholds_path).get("synthetic_market", {})
