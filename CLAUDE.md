# CLAUDE.md — AlphaForge

Guidance for AI coding agents working in this repo. Read before making changes.

## What this repo is

A multi-agent market-research system (LangGraph) whose defining feature is
**statistical rigor inside the agent loop**: multiple-testing correction,
sign gates, out-of-sample backtests, honest null results. Every design
decision should strengthen that story, never weaken it.

## Commands

```bash
pip install -e ".[dev]"        # install (includes mcp<2 for protocol tests)
pytest -q                      # run all tests (offline, deterministic)
ruff check src tests evals     # lint (ruff==0.14.0 pinned)
python evals/harness.py        # 22-task eval suite (writes evals/results.md)
alphaforge "<seed query>"      # one full run, synthetic mode
alphaforge "<seed>" --interactive   # REAL HITL: pauses, asks approve/reject
```

- Live data needs `pip install -e ".[live]"`; Claude refinement needs
  `[llm]` + `ANTHROPIC_API_KEY`. **CI never requires either.**

## Conventions

- All randomness flows through seeded `np.random.default_rng(seed)`; never
  use unseeded randomness anywhere. 3-seed runs = 42, 123, 2024.
- Every schema lives in `src/alphaforge/state/schema.py` (Pydantic v2).
  Agents validate at boundaries; malformed data must fail loudly there.
- MCP servers are thin: tool functions are plain Python (unit-testable
  in-process); `build_server()` only wraps them for FastMCP.
- `statistics` server = scipy + numpy only. If you need a new test, add it
  there with a test against a direct scipy/statsmodels reference.
- Error handling: retries (3×, exp backoff) → fallback → mark hypothesis
  `UNTESTABLE` and record an `AgentError`. **Never silently drop a hypothesis.**
- Numbers in README/results files must come from real runs. Never type a
  metric by hand — run the harness and paste its output.
- Conventional commits (`feat:`, `fix:`, `test:`, `docs:`).

## LLM backend (claude mode)

`tools/llm.py` + versioned prompts in `prompts/*.md`. Rules that must not
regress: fails LOUDLY when `--llm claude` lacks the package/key (never
silently falls back); LLM output must pass the same Pydantic schema
(invalid rewordings fall back to templates and increment `rejected_outputs`);
token-level cost flows into `state.cost_usd`. Test all of this with the mocked
client in `tests/test_llm.py` — never with real API calls.

## MCP protocol

Servers are FastMCP (`mcp>=1,<2` — 2.x renamed it MCPServer). The protocol
tests in `tests/test_mcp_protocol.py` drive real stdio JSON-RPC sessions;
they exist so "connectable from Claude Desktop" is a tested claim, not a hope.

## Where things are

- `src/alphaforge/orchestrator/graph.py` — the LangGraph: nodes, conditional
  edges (null-result path!), HITL interrupt
- `src/alphaforge/agents/` — hypothesis (templates + optional LLM), data,
  validation (correction gates), backtest, report
- `src/alphaforge/events/builders.py` — THE shared event logic; validation and
  backtest must always use the same builders (what you validate is what you
  backtest)
- `src/alphaforge/mcp_servers/` — 4 FastMCP servers
- `evals/` — task suite, harness, results (real numbers only)

## Before you finish any change

1. `pytest -q` green, `ruff check src tests` clean
2. New statistics logic has a scipy/statsmodels reference test
3. New routing logic has a graph test (including the null path)
4. No fabricated metrics anywhere
