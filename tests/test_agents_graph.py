"""State schema, agents, and graph routing tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from alphaforge.agents.base import RunContext, with_retries
from alphaforge.agents.data import DataAgent
from alphaforge.agents.hypothesis import HypothesisAgent
from alphaforge.agents.validation import ValidationAgent
from alphaforge.orchestrator.graph import Orchestrator
from alphaforge.state.schema import AgentConfig, Hypothesis, ResearchState


class TestSchemaValidation:
    def test_valid_hypothesis(self):
        h = Hypothesis(
            id="H001", statement="Earnings beats show positive drift next week.",
            family="earnings_drift", instrument_scope=["AAPL", "MSFT"],
            event_def={"window_days": 5, "surprise_threshold": 2.0},
            expected_effect="positive", test_type="one_sample_ttest",
        )
        assert h.status == "proposed"

    def test_reject_bad_ticker(self):
        with pytest.raises(ValidationError, match="invalid ticker"):
            Hypothesis(
                id="H002", statement="Something with a bad ticker symbol.",
                family="reversal", instrument_scope=["AAPL$"],
                event_def={"window_days": 5}, expected_effect="positive",
                test_type="two_sample_ttest",
            )

    def test_reject_missing_window(self):
        with pytest.raises(ValidationError, match="window_days"):
            Hypothesis(
                id="H003", statement="No window in the event definition.",
                family="reversal", instrument_scope=["AAPL"],
                event_def={"decile": 0.1}, expected_effect="positive",
                test_type="two_sample_ttest",
            )

    def test_reject_window_out_of_range(self):
        with pytest.raises(ValidationError, match="window_days"):
            Hypothesis(
                id="H004", statement="Window way out of range here.",
                family="reversal", instrument_scope=["AAPL"],
                event_def={"window_days": 200}, expected_effect="positive",
                test_type="two_sample_ttest",
            )

    def test_reject_bad_id_format(self):
        with pytest.raises(ValidationError):
            Hypothesis(
                id="HYP1", statement="Bad id format for this one.",
                family="reversal", instrument_scope=["AAPL"],
                event_def={"window_days": 5}, expected_effect="positive",
                test_type="two_sample_ttest",
            )

    def test_survivors_property(self):
        rs = ResearchState(seed_query="q")
        assert rs.survivors == []


class TestHypothesisAgent:
    def test_generates_validated_hypotheses(self):
        agent = HypothesisAgent()
        hyps = agent.generate("post-earnings drift in megacap tech", n_target=8, seed=42)
        assert 5 <= len(hyps) <= 10
        for i, h in enumerate(hyps, 1):
            assert h.id == f"H{i:03d}"
            assert h.event_def["window_days"] >= 1
            assert h.instrument_scope

    def test_deterministic_given_seed(self):
        a = HypothesisAgent().generate("sector rotation", seed=42)
        b = HypothesisAgent().generate("sector rotation", seed=42)
        assert [h.model_dump() for h in a] == [h.model_dump() for h in b]

    def test_seed_query_influences_hypotheses(self):
        """Different research seeds must test different hypothesis sets."""
        a = HypothesisAgent().generate("post-earnings drift in megacap tech", seed=42)
        b = HypothesisAgent().generate("volume spikes and next-week underperformance", seed=42)
        sig_a = {(h.family, tuple(h.instrument_scope), str(sorted(h.event_def.items())))
                 for h in a}
        sig_b = {(h.family, tuple(h.instrument_scope), str(sorted(h.event_def.items())))
                 for h in b}
        assert sig_a != sig_b

    def test_relevant_family_ranked_first(self):
        hyps = HypothesisAgent().generate("Monday seasonality in large-cap equities", seed=42)
        assert hyps[0].family == "day_of_week"

    def test_families_covered(self):
        hyps = HypothesisAgent().generate("any", n_target=8, seed=7)
        assert {h.family for h in hyps} >= {"earnings_drift", "reversal", "volume_shock"}


class TestRetryWrapper:
    def test_succeeds_after_failures(self):
        calls = {"n": 0}

        @with_retries(retries=3, base_delay=0.01)
        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ValueError("boom")
            return 42

        assert flaky() == 42
        assert calls["n"] == 3

    def test_raises_tool_call_failed_after_exhaustion(self):
        from alphaforge.agents.base import ToolCallFailed

        @with_retries(retries=2, base_delay=0.01)
        def always_fails():
            raise RuntimeError("nope")

        with pytest.raises(ToolCallFailed, match="failed after 2"):
            always_fails()


class TestDataAgentFallback:
    def test_marks_untestable_on_data_failure(self):
        class BrokenMD:
            start, end = "2015-01-01", "2025-12-31"

            def get_ohlcv(self, tickers):
                raise IOError("network down")

            def get_earnings_calendar(self, tickers):
                raise IOError("network down")

        rs = ResearchState(
            seed_query="q",
            hypotheses=[HypothesisAgent().generate("q", n_target=3, seed=1)[0]],
        )
        ctx = RunContext(config=rs.config)
        rs = DataAgent(BrokenMD()).run(rs, ctx)
        assert rs.errors and rs.errors[0].agent == "data"
        assert all(h.status == "untestable" for h in rs.hypotheses)


class TestValidationGate:
    def test_untestable_when_too_few_events(self):
        agent = ValidationAgent()
        rs = ResearchState(
            seed_query="q",
            config=AgentConfig(min_events=10_000),  # impossible gate
            hypotheses=HypothesisAgent().generate("q", n_target=3, seed=2),
        )
        ctx = RunContext(config=rs.config)
        from alphaforge.mcp_servers.market_data.server import generate_synthetic_market

        ohlcv, earnings = generate_synthetic_market(
            sorted({t for h in rs.hypotheses for t in h.instrument_scope}),
            "2015-01-01", "2025-12-31", seed=42,
        )
        ctx.data = {"ohlcv": ohlcv, "earnings": earnings}
        rs = agent.run(rs, ctx)
        assert all(h.status == "untestable" for h in rs.hypotheses)
        assert rs.test_results == []


class TestGraph:
    def test_full_run_synthetic(self, tmp_path):
        orch = Orchestrator()
        rs = orch.run(
            "post-earnings drift in megacap tech",
            config=AgentConfig(hitl_approve=True).model_dump(mode="json"),
            out_dir=str(tmp_path / "reports"),
        )
        assert len(rs.hypotheses) >= 5
        assert rs.test_results, "validation must produce test results"
        assert rs.report_path is not None
        assert (tmp_path / "reports").exists()

    def test_null_result_path_produces_report(self, tmp_path):
        """Force a null result: min_events impossible → everything untestable."""
        orch = Orchestrator()
        rs = orch.run(
            "any query",
            config=AgentConfig(min_events=10_000, hitl_approve=True).model_dump(mode="json"),
            out_dir=str(tmp_path / "reports"),
        )
        assert rs.surviving_signals == []
        assert rs.report_path is not None
        with open(rs.report_path) as f:
            text = f.read()
        assert "Null result" in text

    def test_metrics_shape(self, tmp_path):
        orch = Orchestrator()
        rs = orch.run(
            "q", config=AgentConfig().model_dump(mode="json"),
            out_dir=str(tmp_path / "reports"),
        )
        m = rs.to_metrics()
        for key in ("n_hypotheses", "n_tested", "n_survivors", "null_result",
                    "cost_usd", "llm_calls", "report_path"):
            assert key in m
        assert m["cost_usd"] == 0.0  # template mode: zero API cost
        assert m["llm_calls"] == 0


class TestHITL:
    """Regression: langgraph interrupt() without a checkpointer silently
    completes (auto-approves). Interactive runs must use run_interactive()."""

    def test_run_refuses_silent_auto_approve(self):
        from alphaforge.state.schema import AgentConfig

        with pytest.raises(ValueError, match="run_interactive"):
            Orchestrator().run(
                "q", config=AgentConfig(hitl_approve=False).model_dump(mode="json"),
                out_dir="/tmp/never",
            )

    def test_interactive_reject_skips_backtest(self, tmp_path):
        rs = Orchestrator().run_interactive(
            "post-earnings drift in megacap tech",
            out_dir=str(tmp_path / "reports"),
            prompt=lambda q: "reject",
        )
        assert rs.surviving_signals, "sanity: signals existed to approve/reject"
        assert rs.backtest_results == [], "reject must skip backtesting"
        assert rs.report_path is not None

    def test_interactive_approve_backtests(self, tmp_path):
        rs = Orchestrator().run_interactive(
            "post-earnings drift in megacap tech",
            out_dir=str(tmp_path / "reports"),
            prompt=lambda q: "approve",
        )
        assert len(rs.backtest_results) == len(rs.surviving_signals)

    def test_interactive_prompt_actually_called(self, tmp_path):
        """The checkpoint must genuinely pause and ask a human."""
        asked = []
        Orchestrator().run_interactive(
            "post-earnings drift in megacap tech",
            out_dir=str(tmp_path / "reports"),
            prompt=lambda q: asked.append(q) or "approve",
        )
        assert asked, "run_interactive completed without asking the human"
        assert "signals passed validation" in asked[0]
