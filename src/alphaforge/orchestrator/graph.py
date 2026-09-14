"""LangGraph orchestrator: hypothesis → data → validation → [HITL] →
(backtest → report | null-report).

Routing:
  - 0 survivors after validation → null-report path (a FEATURE, never a crash)
  - HITL checkpoint: interrupt() when interactive; auto-approve in CI/evals
  - every node records failures into state.errors with a fallback action —
    hypotheses are never silently dropped
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from alphaforge.agents.backtest import BacktestAgent
from alphaforge.agents.base import RunContext
from alphaforge.agents.data import DataAgent
from alphaforge.agents.hypothesis import HypothesisAgent
from alphaforge.agents.report import ReportAgent
from alphaforge.agents.validation import ValidationAgent
from alphaforge.mcp_servers.market_data.server import MarketData
from alphaforge.state.schema import AgentConfig, ResearchState
from alphaforge.utils.logging import get_logger

log = get_logger("alphaforge.orchestrator")

StateDict = dict[str, Any]


class Orchestrator:
    def __init__(self, llm_backend: str | None = None, data_mode: str | None = None):
        # data_mode ("live"/"synthetic") propagates into the market-data tool
        self.hypothesis_agent = HypothesisAgent()
        self.data_agent = DataAgent(market_data=MarketData(mode=data_mode))
        self.validation_agent = ValidationAgent()
        self.backtest_agent = BacktestAgent()
        self.report_agent = ReportAgent()

    # -- node wrappers (Pydantic model <-> dict for langgraph) --------------
    def _node_hypothesis(self, state: StateDict) -> StateDict:
        rs = ResearchState(**state)
        ctx: RunContext = state["_ctx"]
        try:
            rs.hypotheses = self.hypothesis_agent.generate(
                rs.seed_query, n_target=8, seed=rs.config.seed
            )
            ctx.record_tool(True)
        except Exception as e:  # noqa: BLE001
            ctx.record_tool(False)
            rs.errors = rs.errors + [_err("hypothesis", None, str(e),
                                          "empty_hypothesis_list_fails_run")]
            rs.hypotheses = []
        return {**state, **rs.model_dump()}

    def _node_data(self, state: StateDict) -> StateDict:
        rs = ResearchState(**state)
        rs.config = _cfg_from(state)
        ctx: RunContext = state["_ctx"]
        rs = self.data_agent.run(rs, ctx)
        return {**state, **rs.model_dump()}

    def _node_validation(self, state: StateDict) -> StateDict:
        rs = ResearchState(**state)
        rs.config = _cfg_from(state)
        ctx: RunContext = state["_ctx"]
        rs = self.validation_agent.run(rs, ctx)
        return {**state, **rs.model_dump()}

    def _node_hitl(self, state: StateDict) -> StateDict:
        rs = ResearchState(**state)
        log.info(
            "hitl_checkpoint",
            survivors=len(rs.surviving_signals),
            mode="auto_approve" if rs.config.hitl_approve else "interactive",
        )
        state = {**state, "survivors_count": len(rs.surviving_signals)}
        if not rs.config.hitl_approve:
            from langgraph.types import interrupt

            decision = interrupt(
                f"{len(rs.surviving_signals)} signals passed validation. "
                f"Approve backtesting? [approve/reject]"
            )
            state["hitl_decision"] = decision
        return {**state, "hitl_decision": state.get("hitl_decision", "approve")}

    def _node_backtest(self, state: StateDict) -> StateDict:
        rs = ResearchState(**state)
        rs.config = _cfg_from(state)
        ctx: RunContext = state["_ctx"]
        if state.get("hitl_decision") == "reject":
            log.info("hitl_rejected", skipped=len(rs.surviving_signals))
            return state
        rs = self.backtest_agent.run(rs, ctx)
        return {**state, **rs.model_dump()}

    def _node_report(self, state: StateDict) -> StateDict:
        rs = ResearchState(**state)
        rs.config = _cfg_from(state)
        ctx: RunContext = state["_ctx"]
        rs = self.report_agent.run(rs, ctx, out_dir=state.get("out_dir", "reports"))
        return {**state, **rs.model_dump()}

    # -- graph ---------------------------------------------------------------
    def build_graph(self):
        g = StateGraph(StateDict)
        g.add_node("hypothesis", self._node_hypothesis)
        g.add_node("data", self._node_data)
        g.add_node("validation", self._node_validation)
        g.add_node("hitl", self._node_hitl)
        g.add_node("backtest", self._node_backtest)
        g.add_node("report", self._node_report)
        g.set_entry_point("hypothesis")
        g.add_edge("hypothesis", "data")
        g.add_edge("data", "validation")
        g.add_edge("validation", "hitl")
        g.add_conditional_edges(
            "hitl",
            lambda s: "backtest"
            if (s.get("hitl_decision") != "reject" and s.get("survivors_count", 1) != 0)
            else "report",
            {"backtest": "backtest", "report": "report"},
        )
        g.add_edge("backtest", "report")
        g.add_edge("report", END)
        return g.compile()

    def run(self, seed_query: str, config: dict | None = None,
            out_dir: str = "reports") -> ResearchState:
        cfg = AgentConfig(**(config or {}))
        # data_mode flows from the caller's AgentConfig into the market-data tool
        self.data_agent = DataAgent(market_data=MarketData(mode=cfg.data_mode))
        ctx = RunContext(config=cfg)
        graph = self.build_graph()
        final: StateDict = graph.invoke(
            {
                "seed_query": seed_query,
                "config": cfg.model_dump(mode="json"),
                "out_dir": out_dir,
                "_ctx": ctx,
            },
            config={"recursion_limit": 50},
        )
        return ResearchState(**{k: v for k, v in final.items() if not k.startswith("_")})


def _cfg_from(state: StateDict):
    from alphaforge.state.schema import AgentConfig

    return AgentConfig(**state["config"]) if isinstance(state["config"], dict) else state["config"]


def _err(agent: str, hyp_id: str | None, error: str, fallback: str):
    from alphaforge.state.schema import AgentError

    return AgentError(agent=agent, hypothesis_id=hyp_id, error=error, fallback_action=fallback)
