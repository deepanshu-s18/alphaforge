"""Backtest server: sign recovery on synthetic data with planted effects,
split integrity, cost application, and schema."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaforge.mcp_servers.backtest.server import run_event_backtest
from alphaforge.mcp_servers.market_data.server import generate_synthetic_market
from alphaforge.state.schema import BacktestRequest, SignalSpec

START, END = "2015-01-01", "2025-12-31"


@pytest.fixture(scope="module")
def synth():
    ohlcv, earnings = generate_synthetic_market(["AAPL", "MSFT", "XLE"], START, END, seed=42)
    return ohlcv, earnings


def _req(family: str, tickers: list[str], holding: int = 10, **params) -> BacktestRequest:
    return BacktestRequest(
        signal=SignalSpec(
            signal_id="S001",
            hypothesis_id="H001",
            family=family,
            tickers=tickers,
            entry_rule="close T+1",
            exit_rule=f"close T+{holding + 1}",
            holding_days=holding,
            event_params=params,
        ),
        cost_bps=10.0,
        train_end=pd.Timestamp("2022-12-31").date(),
        test_start=pd.Timestamp("2023-01-01").date(),
    )


class TestSignRecovery:
    """Synthetic market plants drift after earnings beats — backtest must find it."""

    def test_earnings_drift_positive_sharpe(self, synth):
        ohlcv, earnings = synth
        req = _req("earnings_drift", ["AAPL", "MSFT", "XLE"], surprise_threshold=2.0)
        res = run_event_backtest(req, ohlcv, earnings)
        assert res["train"]["n_events"] > 15
        assert res["test"]["n_events"] >= 5
        assert res["train"]["sharpe"] > 0
        assert res["test"]["mean_ret_bps"] > 0

    def test_reversal_bounce_positive(self, synth):
        ohlcv, _ = synth
        res = run_event_backtest(_req("reversal", ["AAPL"], decile=0.10, window_days=10),
                                 ohlcv, pd.DataFrame(columns=["ticker", "date", "surprise_pct"]))
        assert res["train"]["mean_ret_bps"] > 0

    def test_volume_shock_negative(self, synth):
        ohlcv, _ = synth
        res = run_event_backtest(_req("volume_shock", ["AAPL"], holding=5, zscore=2.0),
                                 ohlcv, pd.DataFrame(columns=["ticker", "date", "surprise_pct"]))
        assert res["train"]["mean_ret_bps"] < 0


class TestSplitIntegrity:
    def test_train_and_test_disjoint(self, synth):
        """Real disjointness check: recompute split membership from event dates."""
        ohlcv, earnings = synth
        from alphaforge.events.builders import build_events

        req = _req("day_of_week", ["AAPL"], holding=1, window_days=1)
        # scope the recomputation to the SAME tickers the backtest uses
        scoped = ohlcv[ohlcv["ticker"] == "AAPL"]
        sample = build_events("day_of_week", {"window_days": 1}, scoped, earnings)
        ev = sample.events.copy()
        ev["date"] = pd.to_datetime(ev["date"])
        # builders drop tail events whose exit bar is beyond the data edge,
        # so membership sets come from the SAME frame the backtest uses
        train = set(ev[ev["date"] <= pd.Timestamp("2022-12-31")]["date"])
        test = set(ev[ev["date"] >= pd.Timestamp("2023-01-01")]["date"])
        assert train & test == set(), "no event may be in both splits"
        assert len(train) > len(test) > 0
        res = run_event_backtest(req, ohlcv, earnings)
        assert res["train"]["n_events"] == len(train)
        assert res["test"]["n_events"] == len(test)

    def test_sharpe_annualized_by_eval_period_not_event_span(self, synth):
        """3 events in a 3-year test must annualize as 1 event/yr, not 36."""
        ohlcv, _ = synth
        from alphaforge.mcp_servers.backtest.server import _split_metrics

        # 3 events on consecutive days, 10d holding, evaluated over ~3 years
        dates = pd.to_datetime(["2023-01-05", "2023-01-06", "2023-01-09"])
        rets = np.array([0.001, 0.002, 0.0015])
        m = _split_metrics(rets, dates, cost_bps=10.0, eval_years=3.0,
                           holding_days=10)
        assert m["n_independent"] == 1  # all within one 10d window
        mean = rets.mean() - 10 / 10_000
        std = rets.std(ddof=1)
        expected = (mean / std) * np.sqrt(3 / 3.0)  # 3 events over 3y = 1/yr
        assert m["sharpe"] == pytest.approx(expected, rel=1e-9)
        # the OLD bug would compute events/yr = 3/(4days/365) ≈ 274
        assert abs(m["sharpe"]) < abs((mean / std) * np.sqrt(274.0))

    def test_independent_events_leq_raw(self, synth):
        ohlcv, _ = synth
        res = run_event_backtest(_req("reversal", ["AAPL"], holding=10, decile=0.10),
                                ohlcv, pd.DataFrame(columns=["ticker", "date", "surprise_pct"]))
        assert 0 < res["train"]["n_independent"] <= res["train"]["n_events"]

    def test_cost_reduces_returns(self, synth):
        ohlcv, earnings = synth
        free = run_event_backtest(_req("earnings_drift", ["AAPL"], surprise_threshold=2.0),
                                  ohlcv, earnings)
        # rerun with 50bps cost
        req = _req("earnings_drift", ["AAPL"], surprise_threshold=2.0)
        req.cost_bps = 50.0
        costly = run_event_backtest(req, ohlcv, earnings)
        assert costly["test"]["mean_ret_bps"] < free["test"]["mean_ret_bps"]
        assert costly["test"]["mean_ret_bps"] == pytest.approx(
            free["test"]["mean_ret_bps"] - 40.0, rel=0.01
        )


class TestMetrics:
    def test_no_events_handled(self, synth):
        ohlcv, _ = synth
        empty = pd.DataFrame(columns=["ticker", "date", "surprise_pct"])
        res = run_event_backtest(_req("earnings_drift", ["AAPL"], surprise_threshold=999.0),
                                 ohlcv, empty)
        assert "error" in res

    def test_drawdown_nonpositive_win_rate_bounded(self, synth):
        ohlcv, earnings = synth
        res = run_event_backtest(_req("earnings_drift", ["AAPL", "MSFT"], surprise_threshold=2.0),
                                 ohlcv, earnings)
        for split in ("train", "test"):
            assert res[split]["max_drawdown"] <= 0.0
            assert 0.0 <= res[split]["win_rate"] <= 1.0
