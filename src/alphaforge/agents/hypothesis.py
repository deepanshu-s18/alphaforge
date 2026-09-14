"""Hypothesis agent: instantiate seeded templates, validate against schema, dedupe.

Two backends behind one interface:
  - template (default): deterministic, seeded parameter sampling — zero API
    cost, fully reproducible, works offline (CI/evals).
  - claude (optional): an LLM refines/rewords statements via structured
    output. Falls back to template statements when anthropic/API key absent.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import yaml

from alphaforge.state.schema import Hypothesis
from alphaforge.utils.config import DEFAULT_UNIVERSE, all_tickers, load_universe
from alphaforge.utils.logging import get_logger

log = get_logger("alphaforge.hypothesis")

TEMPLATES_PATH = Path(__file__).parent / "hypothesis_templates.yaml"

# per-family test direction used by the validation gate: significant p-value
# must ALSO match the expected sign to survive (falsification, not p-hacking)

# lexical anchors linking seed-query language to template families
FAMILY_KEYWORDS = {
    "earnings_drift": ["earnings", "beat", "surprise", "post-earnings", "drift", "pead"],
    "reversal": ["reversal", "bounce", "drop", "drawdown", "oversold"],
    "vol_regime": ["volatility", "regime", "risk", "dispersion", "clustering"],
    "sector_momentum": ["sector", "rotation", "momentum", "etf"],
    "day_of_week": ["monday", "weekday", "seasonality", "friday", "day"],
    "volume_shock": ["volume", "spike", "abnormal", "shock", "rebalancing"],
}


def _overlap(query: str, family: str) -> int:
    """Count seed-query keyword hits for a family (deterministic relevance)."""
    q = query.lower()
    return sum(1 for kw in FAMILY_KEYWORDS.get(family, []) if kw in q)


def load_templates(path: Path = TEMPLATES_PATH) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)["templates"]


class HypothesisAgent:
    def __init__(self, universe_path=DEFAULT_UNIVERSE, templates_path=TEMPLATES_PATH):
        self.universe = load_universe(universe_path)
        self.templates = load_templates(templates_path)

    def generate(self, seed_query: str, n_target: int = 8, seed: int = 42) -> list[Hypothesis]:
        """Deterministically instantiate `n_target` validated hypotheses.

        The RNG stream is derived from (seed, seed_query): different research
        seeds sample different parameterizations/scopes, so each eval task
        tests a genuinely distinct hypothesis set. Same seed + same query
        still reproduces byte-identical output.
        """
        query_key = sum(map(ord, seed_query)) % (2**31)
        rng = __import__("numpy").random.default_rng(seed + query_key)
        # simple lexical relevance: templates whose family/keywords match the
        # seed query are ordered first (deterministic, no LLM required)
        scored = sorted(
            self.templates,
            key=lambda t: (-_overlap(seed_query, t["family"]), self.templates.index(t)),
        )
        raw = []
        # round-robin ranked families first (coverage), then seeded extras
        for i, t in enumerate(itertools.cycle(scored)):
            if len(raw) >= n_target:
                break
            raw.append(self._instantiate(t, seed_query, i, rng))

        # dedupe on (family, scope, params)
        seen, out = set(), []
        for h in raw:
            key = (h.family, tuple(h.instrument_scope), str(sorted(h.event_def.items())))
            if key not in seen:
                seen.add(key)
                out.append(h)
        for i, h in enumerate(out, 1):
            h.id = f"H{i:03d}"
        # schema validation happens in Pydantic construction; filter safety net
        return [h for h in out if self._valid(h)]

    def _instantiate(self, t: dict, seed_query: str, idx: int, rng) -> Hypothesis:
        params = {k: rng.choice(v).item() if isinstance(v, list) else v
                  for k, v in t["params"].items()}
        scope_name = rng.choice(t["scopes"]).item()
        pool = self.universe["universe"][scope_name]
        if scope_name == "sector_etfs":
            scope = list(pool)
        else:
            k = min(len(pool), int(rng.integers(4, min(9, len(pool)) + 1)))
            scope = sorted(rng.choice(pool, size=k, replace=False).tolist())
        statement = (
            t["statement"]
            .replace("{surprise_threshold}", str(params.get("surprise_threshold", "")))
            .replace("{window_days}", str(params.get("window_days", "")))
            .replace("{pct_cut}", str(params.get("pct_cut", "")))
            .replace("{vol_window}", str(params.get("vol_window", "")))
            .replace("{lookback_days}", str(params.get("lookback_days", "")))
            .replace("{zscore}", str(params.get("zscore", "")))
            .replace("{direction}",
                     "positive" if t["expected_effect"] == "positive" else "negative")
            .replace("{scope_name}", scope_name.replace("_", " "))
        )
        event_def = {k: v for k, v in params.items() if k != "window_days"}
        event_def["window_days"] = int(params["window_days"])
        return Hypothesis(
            id=f"H{(900 + idx) % 1000:03d}",  # placeholder; renumbered after dedupe
            statement=statement,
            family=t["family"],
            instrument_scope=scope,
            event_def=event_def,
            expected_effect=t["expected_effect"],
            test_type=t["test_type"],
        )

    def _valid(self, h: Hypothesis) -> bool:
        universe_tickers = set(all_tickers(self.universe))
        return all(t in universe_tickers for t in h.instrument_scope)
