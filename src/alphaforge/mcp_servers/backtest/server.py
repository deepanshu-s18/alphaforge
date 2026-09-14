"""mcp-backtest: vectorized event-study backtests with a strict train/test split.

Entry at close T+1 after the event, exit after `holding_days` sessions.
Per-event returns net of round-trip cost_bps. All metrics computed
separately on train (<= train_end) and test (>= test_start); test numbers
are the ones reported downstream.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaforge.events.builders import build_events
from alphaforge.state.schema import BacktestRequest


def _split_metrics(rets: np.ndarray, dates: pd.Series, cost_bps: float) -> dict:
    rets = np.asarray(rets, dtype=float)
    net = rets - cost_bps / 10_000.0  # round-trip cost per event
    n = len(net)
    if n < 2:
        return {"n_events": int(n), "mean_ret_bps": 0.0, "sharpe": 0.0,
                "max_drawdown": 0.0, "win_rate": 0.0}
    mean, std = net.mean(), net.std(ddof=1)
    years = max((dates.max() - dates.min()).days / 365.25, 1 / 12.0)
    events_per_year = n / years
    sharpe = (mean / std) * np.sqrt(events_per_year) if std > 0 else 0.0
    order = np.argsort(dates.to_numpy())
    eq = np.cumprod(1.0 + net[order])
    peak = np.maximum.accumulate(eq)
    max_dd = float(((eq - peak) / peak).min())
    return {
        "n_events": int(n),
        "mean_ret_bps": float(mean * 10_000),
        "sharpe": float(sharpe),
        "max_drawdown": max_dd,
        "win_rate": float((net > 0).mean()),
    }


def run_event_backtest(request: BacktestRequest, ohlcv: pd.DataFrame,
                       earnings: pd.DataFrame) -> dict:
    s = request.signal
    ohlcv = ohlcv[ohlcv["ticker"].isin(s.tickers)]
    if not earnings.empty:
        earnings = earnings[earnings["ticker"].isin(s.tickers)]
    sample = build_events(
        s.family, {**s.event_params, "window_days": s.holding_days}, ohlcv, earnings
    )
    ev = sample.events.copy()
    if ev.empty:
        return {"signal_id": s.signal_id, "hypothesis_id": s.hypothesis_id,
                "error": "no events found", "train": {}, "test": {}}
    ev["date"] = pd.to_datetime(ev["date"])

    # builders guarantee row-for-row alignment between events and treated
    treated = sample.treated
    if len(treated) != len(ev):
        raise ValueError("event/return alignment violated")
    ev["_ret"] = treated

    train = ev[ev["date"] <= pd.Timestamp(request.train_end)]
    test = ev[ev["date"] >= pd.Timestamp(request.test_start)]

    tr = _split_metrics(train["_ret"].to_numpy(), train["date"], request.cost_bps)
    te = _split_metrics(test["_ret"].to_numpy(), test["date"], request.cost_bps)
    return {
        "signal_id": s.signal_id,
        "hypothesis_id": s.hypothesis_id,
        "family": s.family,
        "train": tr,
        "test": te,
        "cost_bps": request.cost_bps,
    }


def build_server():  # pragma: no cover - requires mcp package
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("mcp-backtest")

    @mcp.tool()
    def run_backtest_tool(signal: dict, cost_bps: float, train_end: str, test_start: str) -> dict:
        """Out-of-sample event backtest for a signal spec."""
        from alphaforge.state.schema import BacktestRequest, SignalSpec

        req = BacktestRequest(
            signal=SignalSpec(**signal),
            cost_bps=cost_bps,
            train_end=pd.Timestamp(train_end).date(),
            test_start=pd.Timestamp(test_start).date(),
        )
        md = _default_market_data()
        ohlcv = md.get_ohlcv(req.signal.tickers)
        earnings = md.get_earnings_calendar(req.signal.tickers)
        return run_event_backtest(req, ohlcv, earnings)

    return mcp


def _default_market_data():  # pragma: no cover
    from alphaforge.mcp_servers.market_data.server import MarketData

    return MarketData()


if __name__ == "__main__":  # pragma: no cover
    build_server().run()
