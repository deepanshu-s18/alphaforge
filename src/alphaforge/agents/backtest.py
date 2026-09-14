"""Backtest agent: out-of-sample event backtests for surviving signals only."""

from __future__ import annotations

from alphaforge.agents.base import RunContext, ToolCallFailed, with_retries
from alphaforge.mcp_servers.backtest.server import run_event_backtest
from alphaforge.state.schema import (
    AgentError,
    BacktestRequest,
    BacktestResult,
    ResearchState,
    SignalSpec,
)
from alphaforge.utils.logging import get_logger

log = get_logger("alphaforge.backtest")


class BacktestAgent:
    def __init__(self):
        pass

    @with_retries(retries=3, base_delay=0.1)
    def _run(self, req: BacktestRequest, ohlcv, earnings):
        return run_event_backtest(req, ohlcv, earnings)

    def run(self, state: ResearchState, ctx: RunContext) -> ResearchState:
        cfg = state.config
        for sig in state.surviving_signals:
            h = next(x for x in state.hypotheses if x.id == sig.hypothesis_id)
            req = BacktestRequest(
                signal=SignalSpec(
                    signal_id=sig.signal_id,
                    hypothesis_id=sig.hypothesis_id,
                    family=h.family,
                    tickers=h.instrument_scope,
                    entry_rule=sig.entry_rule,
                    exit_rule=sig.exit_rule,
                    holding_days=int(sig.event_spec["window_days"]),
                    event_params={k: v for k, v in sig.event_spec.items()
                                  if k not in ("window_days", "family")},
                ),
                cost_bps=cfg.cost_bps,
                train_end=cfg.train_end,
                test_start=cfg.test_start,
            )
            try:
                res = self._run(req, ctx.data["ohlcv"], ctx.data.get("earnings"))
                ctx.record_tool(True)
                if "error" in res:
                    state.errors.append(AgentError(
                        agent="backtest", hypothesis_id=sig.hypothesis_id,
                        error=res["error"], fallback_action="skip_signal"
                    ))
                    continue
                te, tr = res["test"], res["train"]
                state.backtest_results.append(BacktestResult(
                    signal_id=sig.signal_id,
                    hypothesis_id=sig.hypothesis_id,
                    train_sharpe=round(tr.get("sharpe", 0.0), 3),
                    test_sharpe=round(te.get("sharpe", 0.0), 3),
                    test_mean_ret_bps=round(te.get("mean_ret_bps", 0.0), 2),
                    test_max_drawdown=round(te.get("max_drawdown", 0.0), 4),
                    test_win_rate=round(te.get("win_rate", 0.0), 4),
                    n_train_events=tr.get("n_events", 0),
                    n_test_events=te.get("n_events", 0),
                    cost_bps=cfg.cost_bps,
                ))
                h.status = "backtested"
            except ToolCallFailed as e:
                ctx.record_tool(False)
                state.errors.append(AgentError(
                    agent="backtest", hypothesis_id=sig.hypothesis_id,
                    error=str(e), fallback_action="skip_signal"
                ))
        return state
