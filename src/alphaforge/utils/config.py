"""Config loading helpers."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml

from alphaforge.state.schema import AgentConfig

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


def agent_config_from_yaml(thresholds_path: str | Path = DEFAULT_THRESHOLDS) -> AgentConfig:
    cfg = load_yaml(thresholds_path)
    s, b = cfg["statistics"], cfg["backtest"]
    u = load_yaml(DEFAULT_UNIVERSE)["data"]
    return AgentConfig(
        min_events=s["min_events"],
        bonferroni_alpha=s["bonferroni_alpha"],
        bh_fdr_q=s["bh_fdr_q"],
        bootstrap_iters=s["bootstrap_iters"],
        bootstrap_seed=s["bootstrap_seed"],
        cost_bps=b["cost_bps"],
        train_end=date.fromisoformat(b["train_end"]),
        test_start=date.fromisoformat(b["test_start"]),
        data_mode=u.get("mode", "synthetic"),
    )


def synthetic_market_cfg(thresholds_path: str | Path = DEFAULT_THRESHOLDS) -> dict:
    return load_yaml(thresholds_path).get("synthetic_market", {})
