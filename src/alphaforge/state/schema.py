"""Shared Pydantic schemas: LangGraph state, hypotheses, tool I/O.

Every structure that flows between agents is validated here. The same
schemas are reused by the MCP servers (tool I/O) and by ForgeLM's data
pipeline (synthetic tool-call validation), so validation rules live in
exactly one place.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator

HypothesisStatus = Literal[
    "proposed", "validated", "rejected", "untestable", "backtested", "reported"
]
TestTypeEnum = Literal[
    "one_sample_ttest", "two_sample_ttest", "mannwhitney_u", "bootstrap_mean"
]

_TICKER_RE = re.compile(r"^[A-Z]{1,5}$")


class Hypothesis(BaseModel):
    """A falsifiable, testable hypothesis. Template-instantiated, LLM-refined."""

    id: str = Field(..., pattern=r"^H\d{3}$")
    statement: str = Field(..., min_length=10)
    family: Literal[
        "earnings_drift",
        "reversal",
        "vol_regime",
        "sector_momentum",
        "day_of_week",
        "volume_shock",
    ]
    instrument_scope: list[str] = Field(..., min_length=1)
    event_def: dict[str, Any]  # family-specific: thresholds, windows, direction
    expected_effect: Literal["positive", "negative"]
    test_type: TestTypeEnum
    status: HypothesisStatus = "proposed"

    @field_validator("instrument_scope")
    @classmethod
    def _tickers_valid(cls, v: list[str]) -> list[str]:
        for t in v:
            if not _TICKER_RE.match(t):
                raise ValueError(f"invalid ticker symbol: {t!r}")
        return sorted(set(v))

    @field_validator("event_def")
    @classmethod
    def _event_def_has_window(cls, v: dict[str, Any]) -> dict[str, Any]:
        if "window_days" not in v:
            raise ValueError("event_def must contain window_days")
        if int(v["window_days"]) < 1 or int(v["window_days"]) > 60:
            raise ValueError("window_days must be in [1, 60]")
        return v


class TestResult(BaseModel):
    hypothesis_id: str
    test_name: str
    statistic: float
    p_value: float = Field(..., ge=0.0, le=1.0)
    effect_size: float
    ci_low: float
    ci_high: float
    n_events: int
    bonferroni_significant: bool
    bh_fdr_significant: bool
    corrected_significant: bool


class Signal(BaseModel):
    """A validated hypothesis promoted to a tradeable event-rule."""

    signal_id: str
    hypothesis_id: str
    event_spec: dict[str, Any]
    entry_rule: str
    exit_rule: str


class BacktestResult(BaseModel):
    """Per-signal out-of-sample backtest result.

    n_train_independent / n_test_independent count non-overlapping event
    windows (events >= holding_days apart) — the honest sample size for
    overlapping event studies. Default 0 means the backtest ran but the
    count was not returned by the underlying tool (treat as unknown, not zero).
    """

    signal_id: str
    hypothesis_id: str
    train_sharpe: float
    test_sharpe: float
    test_mean_ret_bps: float
    test_max_drawdown: float
    test_win_rate: float
    n_train_events: int
    n_train_independent: int = Field(
        default=0, description="non-overlapping train events; 0 = unknown"
    )
    n_test_independent: int = Field(
        default=0, description="non-overlapping test events; 0 = unknown"
    )
    n_test_events: int
    cost_bps: float
    notes: str = ""


class AgentError(BaseModel):
    agent: str
    hypothesis_id: str | None = None
    error: str
    fallback_action: str


class AgentConfig(BaseModel):
    """Runtime knobs for one orchestrated run."""

    seed: int = 42
    max_tool_retries: int = 3
    max_hypothesis_refinements: int = 2
    min_events: int = 30
    bonferroni_alpha: float = 0.05
    bh_fdr_q: float = 0.10
    bootstrap_iters: int = 10000
    bootstrap_seed: int = 42
    cost_bps: float = 10.0
    train_end: date = date(2022, 12, 31)
    test_start: date = date(2023, 1, 1)
    data_mode: Literal["synthetic", "live"] = "synthetic"
    llm_backend: Literal["template", "claude"] = "template"
    hitl_approve: bool = True  # CI/eval runs auto-approve at the checkpoint


class ResearchState(BaseModel):
    """The single shared state flowing through the LangGraph StateGraph."""

    seed_query: str
    config: AgentConfig = AgentConfig()
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    datasets: dict[str, Any] = Field(default_factory=dict)  # key -> CachedDataset dict
    test_results: list[TestResult] = Field(default_factory=list)
    surviving_signals: list[Signal] = Field(default_factory=list)
    backtest_results: list[BacktestResult] = Field(default_factory=list)
    report_path: str | None = None
    errors: list[AgentError] = Field(default_factory=list)
    cost_usd: float = 0.0
    llm_calls: int = 0
    tool_calls: int = 0
    first_try_tool_rate: float = 0.0

    @property
    def survivors(self) -> list[TestResult]:
        return [r for r in self.test_results if r.corrected_significant]

    def to_metrics(self) -> dict[str, Any]:
        """Machine-readable run summary consumed by the eval harness."""
        return {
            "seed_query": self.seed_query,
            "n_hypotheses": len(self.hypotheses),
            "n_tested": len(self.test_results),
            "n_survivors": len(self.surviving_signals),
            "n_backtested": len(self.backtest_results),
            "n_errors": len(self.errors),
            "null_result": len(self.surviving_signals) == 0,
            "cost_usd": round(self.cost_usd, 4),
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "first_try_tool_rate": round(self.first_try_tool_rate, 4),
            "report_path": self.report_path,
        }


# ---------------------------------------------------------------------------
# Tool I/O schemas (shared with MCP servers and ForgeLM)
# ---------------------------------------------------------------------------


class OHLCVRequest(BaseModel):
    ticker: str
    start: date
    end: date
    interval: Literal["1d", "1h"] = "1d"

    @field_validator("ticker")
    @classmethod
    def _ticker(cls, v: str) -> str:
        if not _TICKER_RE.match(v):
            raise ValueError(f"invalid ticker symbol: {v!r}")
        return v.upper()

    @field_validator("end")
    @classmethod
    def _range_ok(cls, v: date, info) -> date:
        start = info.data.get("start")
        if start and v <= start:
            raise ValueError("end must be after start")
        return v


class SamplePair(BaseModel):
    sample_a: list[float] = Field(..., min_length=2)
    sample_b: list[float] = Field(..., min_length=2)
    alternative: Literal["two-sided", "less", "greater"] = "two-sided"


class BootstrapRequest(BaseModel):
    values: list[float] = Field(..., min_length=2)
    stat_fn: Literal["mean", "median"] = "mean"
    n_boot: int = Field(10000, ge=100, le=100000)
    seed: int = 42


class SignalSpec(BaseModel):
    signal_id: str = ""
    hypothesis_id: str
    family: str
    tickers: list[str]
    entry_rule: str
    exit_rule: str
    holding_days: int = Field(..., ge=1, le=60)
    event_params: dict[str, Any] = Field(default_factory=dict)


class BacktestRequest(BaseModel):
    signal: SignalSpec
    cost_bps: float = Field(10.0, ge=0.0, le=200.0)
    train_end: date
    test_start: date


def frame_to_records(df: pd.DataFrame) -> list[dict]:
    """JSON-safe records from a DataFrame (dates -> ISO strings)."""
    out = []
    for row in df.to_dict(orient="records"):
        rec = {}
        for k, v in row.items():
            if isinstance(v, (pd.Timestamp, datetime)):
                rec[k] = v.isoformat()
            elif isinstance(v, date):
                rec[k] = v.isoformat()
            elif pd.isna(v):
                rec[k] = None
            else:
                rec[k] = v
        out.append(rec)
    return out
