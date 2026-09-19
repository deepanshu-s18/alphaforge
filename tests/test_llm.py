"""LLM backend tests — all against a mocked Anthropic client (no network/key).

Covers: refinement applied, invalid-JSON fallback, schema-violating rewording
fallback, cost accounting, unavailable-backend loud failure, prompt loading.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from alphaforge.agents.hypothesis import HypothesisAgent
from alphaforge.state.schema import AgentConfig, ResearchState
from alphaforge.tools.llm import LLMUnavailable, load_prompt


def _fake_client(payload_text: str, in_tok=1000, out_tok=200):
    """Minimal anthropic-like client returning one canned text block."""
    return SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: SimpleNamespace(
        content=[SimpleNamespace(type="text", text=payload_text)],
        usage=SimpleNamespace(input_tokens=in_tok, output_tokens=out_tok),
    )))


def _hyps():
    return HypothesisAgent().generate("post-earnings drift", n_target=3, seed=42)


class TestClaudeClient:
    def test_refine_applies_valid_statements(self):
        from alphaforge.tools.llm import ClaudeClient

        hyps = _hyps()
        payload = json.dumps({"hypotheses": [
            {"id": h.id, "statement": f"REFINED: {h.statement}"}
            for h in hyps
        ]})
        c = ClaudeClient(client=_fake_client(payload))
        out = c.refine_hypotheses("q", hyps)
        assert all(h.statement.startswith("REFINED:") for h in out)
        # cost: 1000 in * $3/M + 200 out * $15/M = $0.003 + $0.003 = $0.006
        assert c.usage.cost_usd == pytest.approx(0.006)
        assert c.usage.calls == 1

    def test_invalid_json_falls_back_to_templates(self):
        from alphaforge.tools.llm import ClaudeClient

        hyps = _hyps()
        original = [h.statement for h in hyps]
        c = ClaudeClient(client=_fake_client("not json at all"))
        out = c.refine_hypotheses("q", hyps)
        assert [h.statement for h in out] == original
        assert c.usage.rejected_outputs == 1
        assert c.usage.notes

    def test_schema_violating_rewording_kept_out(self):
        from alphaforge.tools.llm import ClaudeClient

        hyps = _hyps()
        # statement below min_length -> schema violation -> template kept
        payload = json.dumps({"hypotheses": [{"id": hyps[0].id, "statement": "x"}]})
        c = ClaudeClient(client=_fake_client(payload))
        out = c.refine_hypotheses("q", hyps)
        assert out[0].statement == hyps[0].statement  # unchanged
        assert c.usage.rejected_outputs == 1

    def test_unknown_ids_ignored(self):
        from alphaforge.tools.llm import ClaudeClient

        hyps = _hyps()
        payload = json.dumps({"hypotheses": [{"id": "H999", "statement": "junk"}]})
        c = ClaudeClient(client=_fake_client(payload))
        out = c.refine_hypotheses("q", hyps)
        assert [h.model_dump() for h in out] == [h.model_dump() for h in hyps]

    def test_codefence_json_parsed(self):
        from alphaforge.tools.llm import ClaudeClient

        hyps = _hyps()
        payload = "```json\n" + json.dumps({"hypotheses": [
            {"id": hyps[0].id, "statement": "Refined statement that is long enough."}
        ]}) + "\n```"
        c = ClaudeClient(client=_fake_client(payload))
        out = c.refine_hypotheses("q", hyps)
        assert out[0].statement == "Refined statement that is long enough."

    def test_polish_discussion_returns_text(self):
        from alphaforge.tools.llm import ClaudeClient

        c = ClaudeClient(client=_fake_client("A grounded discussion."))
        assert c.polish_discussion("ctx") == "A grounded discussion."


def _fake_gemini_client(payload_text: str, in_tok=1000, out_tok=200):
    return SimpleNamespace(
        GenerativeModel=lambda **kw: SimpleNamespace(
            generate_content=lambda user, **k: SimpleNamespace(
                text=payload_text,
                usage_metadata=SimpleNamespace(
                    prompt_token_count=in_tok,
                    candidates_token_count=out_tok,
                ),
            )
        )
    )


class TestGeminiClient:
    def test_refine_applies_valid_statements(self):
        from alphaforge.tools.llm import GeminiClient

        hyps = _hyps()
        payload = json.dumps({"hypotheses": [
            {"id": h.id, "statement": f"GEMINI: {h.statement}"}
            for h in hyps
        ]})
        c = GeminiClient(client=_fake_gemini_client(payload))
        out = c.refine_hypotheses("q", hyps)
        assert all(h.statement.startswith("GEMINI:") for h in out)
        assert c.usage.calls == 1

    def test_invalid_json_falls_back_to_templates(self):
        from alphaforge.tools.llm import GeminiClient

        hyps = _hyps()
        original = [h.statement for h in hyps]
        c = GeminiClient(client=_fake_gemini_client("not json at all"))
        out = c.refine_hypotheses("q", hyps)
        assert [h.statement for h in out] == original
        assert c.usage.rejected_outputs == 1
        assert c.usage.notes

    def test_polish_discussion_returns_text(self):
        from alphaforge.tools.llm import GeminiClient

        c = GeminiClient(client=_fake_gemini_client("A grounded Gemini discussion."))
        assert c.polish_discussion("ctx") == "A grounded Gemini discussion."


class TestUnavailableBackend:
    def test_missing_package_raises_loudly(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-dummy")
        import builtins

        real_import = builtins.__import__

        def no_anthropic(name, *a, **k):
            if name == "anthropic":
                raise ImportError("no anthropic")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", no_anthropic)
        from alphaforge.tools.llm import ClaudeClient

        with pytest.raises(LLMUnavailable, match="anthropic package"):
            ClaudeClient()

    def test_missing_key_raises_loudly(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from alphaforge.tools.llm import ClaudeClient

        with pytest.raises(LLMUnavailable, match="ANTHROPIC_API_KEY"):
            ClaudeClient()

    def test_orchestrator_claude_mode_fails_loudly_without_key(self, monkeypatch):
        """End-to-end guard: --llm claude with no key must crash with a clear
        message, never silently fall back to template mode."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        from alphaforge.orchestrator.graph import Orchestrator

        with pytest.raises(LLMUnavailable, match="ANTHROPIC_API_KEY"):
            Orchestrator().run(
                "q",
                config=AgentConfig(llm_backend="claude").model_dump(mode="json"),
                out_dir="/tmp/never",
            )

    def test_missing_gemini_key_raises_loudly(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        from alphaforge.tools.llm import GeminiClient

        with pytest.raises(LLMUnavailable, match="GEMINI_API_KEY"):
            GeminiClient()

    def test_orchestrator_gemini_mode_fails_loudly_without_key(self, monkeypatch):
        """End-to-end guard: --llm gemini with no key must crash with a clear
        message, never silently fall back to template mode."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        from alphaforge.orchestrator.graph import Orchestrator

        with pytest.raises(LLMUnavailable, match="GEMINI_API_KEY"):
            Orchestrator().run(
                "q",
                config=AgentConfig(llm_backend="gemini").model_dump(mode="json"),
                out_dir="/tmp/never",
            )


class TestPrompts:
    def test_versioned_prompts_load(self):
        for name in ("hypothesis_refine", "report_polish"):
            text = load_prompt(name)
            assert len(text) > 50, f"prompt {name} suspiciously short"

    def test_missing_prompt_raises(self):
        with pytest.raises(FileNotFoundError):
            load_prompt("does_not_exist")


class TestCostFlowsThroughState:
    def test_metrics_reflect_llm_usage(self):
        rs = ResearchState(seed_query="q", cost_usd=0.006, llm_calls=1)
        m = rs.to_metrics()
        assert m["cost_usd"] == 0.006
        assert m["llm_calls"] == 1


class TestForgeLMClient:
    def test_refine_with_fake_openai_client(self):
        from alphaforge.tools.llm import ForgeLMClient

        hyps = _hyps()
        payload = json.dumps({"hypotheses": [
            {"id": h.id, "statement": f"FORGELM: {h.statement}"}
            for h in hyps
        ]})
        mock_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=payload))],
                usage=SimpleNamespace(prompt_tokens=500, completion_tokens=150),
            )))
        )
        c = ForgeLMClient(client=mock_client)
        out = c.refine_hypotheses("q", hyps)
        assert all(h.statement.startswith("FORGELM:") for h in out)
        assert c.usage.cost_usd == 0.0  # $0 local inference cost
        assert c.usage.calls == 1

    def test_invalid_json_fallback(self):
        from alphaforge.tools.llm import ForgeLMClient

        hyps = _hyps()
        orig = [h.statement for h in hyps]
        mock_client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="invalid-json"))],
                usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20),
            )))
        )
        c = ForgeLMClient(client=mock_client)
        out = c.refine_hypotheses("q", hyps)
        assert [h.statement for h in out] == orig
        assert c.usage.rejected_outputs == 1

    def test_graph_resolves_forgelm_client(self):
        from alphaforge.orchestrator.graph import Orchestrator
        from alphaforge.tools.llm import ForgeLMClient

        client = Orchestrator._default_llm_client("forgelm")
        assert isinstance(client, ForgeLMClient)

