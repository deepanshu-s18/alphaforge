"""Data agent: fetch and cache OHLCV + earnings for all hypothesis scopes."""

from __future__ import annotations

from alphaforge.agents.base import RunContext, ToolCallFailed, with_retries
from alphaforge.mcp_servers.market_data.server import MarketData
from alphaforge.state.schema import AgentError, ResearchState
from alphaforge.utils.cache import make_key
from alphaforge.utils.logging import get_logger

log = get_logger("alphaforge.data")


class DataAgent:
    def __init__(self, market_data: MarketData | None = None):
        self.md = market_data or MarketData()

    @with_retries(retries=3, base_delay=0.2)
    def _fetch(self, kind: str, tickers: list[str]):
        if kind == "ohlcv":
            return self.md.get_ohlcv(tickers)
        return self.md.get_earnings_calendar(tickers)

    def run(self, state: ResearchState, ctx: RunContext) -> ResearchState:
        needed = sorted({t for h in state.hypotheses for t in h.instrument_scope})
        state.datasets = {}
        for kind in ("ohlcv", "earnings"):
            try:
                df = self._fetch(kind, needed)
                ctx.data[kind] = df
                ctx.record_tool(True)
                state.datasets[kind] = {
                    "key": make_key(f"{kind}_run", "_".join(needed[:3]) + f"+{len(needed)}",
                                    self.md.start, self.md.end),
                    "rows": len(df),
                    "tickers": needed,
                }
            except ToolCallFailed as e:
                ctx.record_tool(False)
                log.error("data_fetch_failed", kind=kind, error=str(e))
                state.errors.append(AgentError(
                    agent="data", error=str(e), fallback_action="mark_hypotheses_untestable"
                ))
                for h in state.hypotheses:
                    h.status = "untestable"
        return state
