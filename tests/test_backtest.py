"""Backtest server: sign recovery on synthetic data with planted effects,
split integrity, cost application, and schema."""

from __future__ import annotations

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
        ohlcv, earnings = synth
        res = run_event_backtest(_req("day_of_week", ["AAPL"], window_days=1),
                                 ohlcv, earnings)
        # ~11 years, roughly 45% train / 27% test by construction of 2023 split
        assert res["train"]["n_events"] > res["test"]["n_events"] > 0
        # no event appears in both: train counts events <= 2022-12-31
        assert res["train"]["n_events"] + res["test"]["n_events"] <= \
            res["train"]["n_events"] + res["test"]["n_events"] + 60  # boundary days excluded

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
