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
from langgraph.types import Command

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
    def __init__(self, llm_backend: str | None = None, data_mode: str | None = None,
                 llm_client_factory=None):
        # data_mode ("live"/"synthetic") propagates into the market-data tool
        self.hypothesis_agent = HypothesisAgent()
        self.data_agent = DataAgent(market_data=MarketData(mode=data_mode))
        self.validation_agent = ValidationAgent()
        self.backtest_agent = BacktestAgent()
        self.report_agent = ReportAgent()
        self._llm_client_factory = llm_client_factory or self._default_llm_client

    @staticmethod
    def _default_llm_client():
        from alphaforge.tools.llm import ClaudeClient

        return ClaudeClient()

    def _llm_client(self):
        """Raises LLMUnavailable loudly when claude is requested but not set up."""
        return self._llm_client_factory()

    # -- node wrappers (Pydantic model <-> dict for langgraph) --------------
    def _node_hypothesis(self, state: StateDict) -> StateDict:
        rs = ResearchState(**state)
        ctx: RunContext = self._active_ctx
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
        # optional LLM refinement: real API calls, real cost accounting,
        # fails loudly if the backend is requested but unavailable
        if rs.config.llm_backend == "claude" and rs.hypotheses:
            client = self._llm_client()
            rs.hypotheses = client.refine_hypotheses(rs.seed_query, rs.hypotheses)
            rs.cost_usd = round(rs.cost_usd + client.usage.cost_usd, 6)
            rs.llm_calls += client.usage.calls
            for note in client.usage.notes:
                log.warning("llm_refinement_note", note=note)
        return {**state, **rs.model_dump()}

    def _node_data(self, state: StateDict) -> StateDict:
        rs = ResearchState(**state)
        rs.config = _cfg_from(state)
        ctx: RunContext = self._active_ctx
        rs = self.data_agent.run(rs, ctx)
        return {**state, **rs.model_dump()}

    def _node_validation(self, state: StateDict) -> StateDict:
        rs = ResearchState(**state)
        rs.config = _cfg_from(state)
        ctx: RunContext = self._active_ctx
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
            # interrupts the graph (requires checkpointer, see run_interactive);
            # resumes with "approve"/"reject" via Command(resume=...)
            from langgraph.types import interrupt

            decision = interrupt(
                f"{len(rs.surviving_signals)} signals passed validation. "
                f"Approve backtesting? [approve/reject]"
            )
            state["hitl_decision"] = str(decision).lower()
        return {**state, "hitl_decision": state.get("hitl_decision", "approve")}

    def _node_backtest(self, state: StateDict) -> StateDict:
        rs = ResearchState(**state)
        rs.config = _cfg_from(state)
        ctx: RunContext = self._active_ctx
        if state.get("hitl_decision") == "reject":
            log.info("hitl_rejected", skipped=len(rs.surviving_signals))
            return state
        rs = self.backtest_agent.run(rs, ctx)
        return {**state, **rs.model_dump()}

    def _node_report(self, state: StateDict) -> StateDict:
        rs = ResearchState(**state)
        rs.config = _cfg_from(state)
        ctx: RunContext = self._active_ctx
        # reuse the same LLM client (accumulated usage) when claude backend
        if rs.config.llm_backend == "claude" and self.report_agent.llm_client is None:
            try:
                self.report_agent.llm_client = self._llm_client()
            except Exception as e:  # noqa: BLE001 - polish is optional
                log.warning("llm_polish_unavailable", error=str(e))
        before = 0.0
        if self.report_agent.llm_client is not None:
            before = self.report_agent.llm_client.usage.cost_usd
        rs = self.report_agent.run(rs, ctx, out_dir=state.get("out_dir", "reports"))
        if self.report_agent.llm_client is not None:
            delta = self.report_agent.llm_client.usage.cost_usd - before
            rs.cost_usd = round(rs.cost_usd + delta, 6)
            rs.llm_calls += self.report_agent.llm_client.usage.calls
        return {**state, **rs.model_dump()}

    # -- graph ---------------------------------------------------------------
    def build_graph(self, checkpointer=None):
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
        return g.compile(checkpointer=checkpointer)

    def run(self, seed_query: str, config: dict | None = None,
            out_dir: str = "reports") -> ResearchState:
        """Run the pipeline. Auto-approves HITL when config.hitl_approve=True.

        Interactive runs must go through run_interactive(): langgraph's
        interrupt() only pauses a graph that has a checkpointer, so silently
        calling run() with hitl_approve=False would complete WITHOUT waiting
        for a human — a silent auto-approve we refuse to do.
        """
        cfg = AgentConfig(**(config or {}))
        if not cfg.hitl_approve:
            raise ValueError(
                "hitl_approve=False requires run_interactive(); plain run() "
                "would silently auto-approve the checkpoint"
            )
        # data_mode flows from the caller's AgentConfig into the market-data tool
        self.data_agent = DataAgent(market_data=MarketData(mode=cfg.data_mode))
        self._active_ctx = RunContext(config=cfg)
        graph = self.build_graph()
        final: StateDict = graph.invoke(
            {
                "seed_query": seed_query,
                "config": cfg.model_dump(mode="json"),
                "out_dir": out_dir,
            },
            config={"recursion_limit": 50},
        )
        if "__interrupt__" in final:  # defensive: should not happen on this path
            raise RuntimeError("unexpected interrupt in auto-approve run")
        return self._finalize(final)

    def run_interactive(self, seed_query: str, config: dict | None = None,
                        out_dir: str = "reports",
                        prompt=input) -> ResearchState:
        """Run with a REAL human-in-the-loop checkpoint.

        Uses a MemorySaver checkpointer so interrupt() genuinely pauses the
        graph; `prompt` (injectable for tests) asks the human approve/reject
        and the graph resumes with their decision.
        """
        from langgraph.checkpoint.memory import MemorySaver

        cfg = AgentConfig(**{**(config or {}), "hitl_approve": False})
        self.data_agent = DataAgent(market_data=MarketData(mode=cfg.data_mode))
        self._active_ctx = RunContext(config=cfg)
        graph = self.build_graph(checkpointer=MemorySaver())
        thread = {"configurable": {"thread_id": f"hitl-{abs(hash(seed_query))}"}}
        state_in = {
            "seed_query": seed_query,
            "config": cfg.model_dump(mode="json"),
            "out_dir": out_dir,
        }
        final = graph.invoke(state_in, config={**thread, "recursion_limit": 50})
        while "__interrupt__" in final:
            answer = prompt(str(final["__interrupt__"][0].value) + " ")
            decision = "reject" if str(answer).strip().lower().startswith("r") else "approve"
            final = graph.invoke(
                Command(resume=decision),
                config={**thread, "recursion_limit": 50},
            )
        return self._finalize(final)

    def _finalize(self, final: StateDict) -> ResearchState:
        rs = ResearchState(**{k: v for k, v in final.items() if not k.startswith("_")})
        rs.tool_calls = self._active_ctx.tool_calls
        rs.first_try_tool_rate = self._active_ctx.first_try_rate
        return rs


def _cfg_from(state: StateDict):
    from alphaforge.state.schema import AgentConfig

    return AgentConfig(**state["config"]) if isinstance(state["config"], dict) else state["config"]


def _err(agent: str, hyp_id: str | None, error: str, fallback: str):
    from alphaforge.state.schema import AgentError

    return AgentError(agent=agent, hypothesis_id=hyp_id, error=error, fallback_action=fallback)
