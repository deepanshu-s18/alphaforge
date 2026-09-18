"""Claude LLM client: structured-output hypothesis refinement + report polish.

Design rules (audit-grade honesty):
- Fails LOUDLY when the backend is requested but unavailable (missing package
  or API key) — never silently pretends to have used an LLM.
- LLM output is never trusted: refined hypotheses must pass the SAME Pydantic
  schema; anything invalid falls back to the deterministic template statement
  and is recorded.
- Cost/calls are accumulated from real token usage and flow into
  ResearchState.cost_usd / llm_calls.
- Prompts are versioned files in src/alphaforge/prompts/ (git history = audit
  trail).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alphaforge.state.schema import Hypothesis
from alphaforge.utils.logging import get_logger

log = get_logger("alphaforge.llm")

PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

# USD per million tokens (Sonnet-class default; override via constructor)


@dataclass
class LLMUsage:
    cost_usd: float = 0.0
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    rejected_outputs: int = 0
    notes: list[str] = field(default_factory=list)


class LLMUnavailable(Exception):
    """Raised when the LLM backend is requested but cannot run."""


# ---------------------------------------------------------------------------
# Pricing tables
# ---------------------------------------------------------------------------

PRICES_USD_PER_MTOK = {
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-sonnet-4-20250514": (3.0, 15.0),
    # Gemini Flash free-tier: ~$0 but we track tokens for honesty
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-2.0-flash": (0.10, 0.40),
}


def load_prompt(name: str) -> str:
    path = PROMPTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"prompt not found: {path}")
    return path.read_text()


class ClaudeClient:
    """Thin anthropic wrapper. `client` is injectable for tests."""

    def __init__(self, model: str = "claude-sonnet-4-5", client: Any | None = None):
        self.model = model
        self.usage = LLMUsage()
        self._client = client or self._make_client()

    @staticmethod
    def _make_client():
        import os

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMUnavailable(
                "llm_backend='claude' requires ANTHROPIC_API_KEY to be set"
            )
        try:
            import anthropic
        except ImportError as e:
            raise LLMUnavailable(
                "llm_backend='claude' requires the anthropic package "
                "(pip install -e '.[llm]')"
            ) from e
        return anthropic.Anthropic()

    # ------------------------------------------------------------------
    def _complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        u = getattr(resp, "usage", None)
        if u is not None:
            self.usage.calls += 1
            in_tok = int(getattr(u, "input_tokens", 0))
            out_tok = int(getattr(u, "output_tokens", 0))
            self.usage.input_tokens += in_tok
            self.usage.output_tokens += out_tok
            p_in, p_out = PRICES_USD_PER_MTOK.get(self.model, (3.0, 15.0))
            self.usage.cost_usd += in_tok / 1e6 * p_in + out_tok / 1e6 * p_out
        return "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )

    # ------------------------------------------------------------------
    def refine_hypotheses(self, seed_query: str, hypotheses: list[Hypothesis]) -> list[Hypothesis]:
        """Reword hypotheses via structured JSON; invalid outputs fall back to
        the template statement per-hypothesis (recorded, never silent)."""
        system = load_prompt("hypothesis_refine")
        user = json.dumps({
            "seed_query": seed_query,
            "hypotheses": [
                {
                    "id": h.id, "statement": h.statement, "family": h.family,
                    "instrument_scope": h.instrument_scope, "event_def": h.event_def,
                    "expected_effect": h.expected_effect,
                }
                for h in hypotheses
            ],
        })
        raw = self._complete(system, user)
        refined = self._parse_json(raw)
        if refined is None:
            self.usage.rejected_outputs += 1
            self.usage.notes.append("refinement output not valid JSON; kept templates")
            return hypotheses

        by_id = {h.id: h for h in hypotheses}
        seen: set[str] = set()
        out = list(hypotheses)
        for item in refined.get("hypotheses", []):
            hid = item.get("id")
            if hid not in by_id or hid in seen:
                continue
            new_stmt = item.get("statement", "")
            base = by_id[hid]
            try:
                candidate = base.model_copy(update={"statement": str(new_stmt)})
                Hypothesis.model_validate(candidate.model_dump())
                out[out.index(base)] = candidate
                seen.add(hid)
            except Exception:  # noqa: BLE001 - invalid rewording -> keep template
                self.usage.rejected_outputs += 1
                self.usage.notes.append(f"{hid}: invalid refinement kept template")
        return out

    def polish_discussion(self, context: str) -> str | None:
        system = load_prompt("report_polish")
        try:
            return self._complete(system, context, max_tokens=600).strip() or None
        except Exception as e:  # noqa: BLE001 - polish is optional, never fatal
            self.usage.notes.append(f"polish failed: {e}")
            return None

    @staticmethod
    def _parse_json(raw: str) -> dict | None:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


class GeminiClient:
    """Google Gemini wrapper with identical interface to ClaudeClient.

    Uses GEMINI_API_KEY (free tier at aistudio.google.com).
    Model defaults to gemini-2.0-flash — fast, free-tier eligible.
    Same honesty guarantees as ClaudeClient:
      - Fails loudly when key missing (never silently pretends)
      - Invalid JSON falls back to template statements, recorded in usage
      - Token cost tracked from real usage_metadata
    """

    def __init__(self, model: str = "gemini-2.0-flash", client: Any | None = None):
        self.model = model
        self.usage = LLMUsage()
        self._genai = client or self._make_client()

    @staticmethod
    def _make_client():
        import os
        if not os.environ.get("GEMINI_API_KEY"):
            raise LLMUnavailable(
                "llm_backend='gemini' requires GEMINI_API_KEY "
                "(free key at aistudio.google.com)"
            )
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise LLMUnavailable(
                "llm_backend='gemini' requires google-generativeai "
                "(pip install -e '.[gemini]')"
            ) from e
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        return genai

    def _complete(self, system: str, user: str, max_tokens: int = 2000) -> str:
        model = self._genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system,
        )
        resp = model.generate_content(
            user,
            generation_config={"max_output_tokens": max_tokens, "temperature": 0.0},
        )
        text = resp.text or ""
        meta = getattr(resp, "usage_metadata", None)
        if meta:
            in_tok = int(getattr(meta, "prompt_token_count", 0))
            out_tok = int(getattr(meta, "candidates_token_count", 0))
            self.usage.input_tokens += in_tok
            self.usage.output_tokens += out_tok
            p_in, p_out = PRICES_USD_PER_MTOK.get(self.model, (0.1, 0.4))
            self.usage.cost_usd += in_tok / 1e6 * p_in + out_tok / 1e6 * p_out
        self.usage.calls += 1
        return text

    def refine_hypotheses(self, seed_query: str, hypotheses: list[Hypothesis]) -> list[Hypothesis]:
        """Reword hypotheses via Gemini; invalid JSON falls back to templates."""
        system = load_prompt("hypothesis_refine")
        user = json.dumps({
            "seed_query": seed_query,
            "hypotheses": [
                {
                    "id": h.id, "statement": h.statement, "family": h.family,
                    "instrument_scope": h.instrument_scope, "event_def": h.event_def,
                    "expected_effect": h.expected_effect,
                }
                for h in hypotheses
            ],
        })
        raw = self._complete(system, user)
        refined = ClaudeClient._parse_json(raw)
        if refined is None:
            self.usage.rejected_outputs += 1
            self.usage.notes.append("gemini: output not valid JSON; kept templates")
            return hypotheses

        by_id = {h.id: h for h in hypotheses}
        seen: set[str] = set()
        out = list(hypotheses)
        for item in refined.get("hypotheses", []):
            hid = item.get("id")
            if hid not in by_id or hid in seen:
                continue
            base = by_id[hid]
            try:
                candidate = base.model_copy(update={"statement": str(item.get("statement", ""))})
                Hypothesis.model_validate(candidate.model_dump())
                out[out.index(base)] = candidate
                seen.add(hid)
            except Exception:  # noqa: BLE001
                self.usage.rejected_outputs += 1
                self.usage.notes.append(f"{hid}: invalid gemini refinement; kept template")
        return out

    def polish_discussion(self, context: str) -> str | None:
        system = load_prompt("report_polish")
        try:
            return self._complete(system, context, max_tokens=600).strip() or None
        except Exception as e:  # noqa: BLE001
            self.usage.notes.append(f"gemini polish failed: {e}")
            return None
