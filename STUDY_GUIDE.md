# AlphaForge Study Guide — internalize before the interview

You claim this project on your resume; you must be able to rebuild and defend
it from memory. Work through this out loud. Every answer should reference the
actual code (open it when checking yourself, never in the interview).

## 1. Whiteboard the architecture from memory

Draw this without looking, then diff against README:

```
seed_query → hypothesis → data → validation → [HITL] → backtest → report
                                                     ↘ null-report (0 survivors)
```

Be able to answer, cold:

- What exactly flows between agents? (Pydantic `ResearchState`; heavy DataFrames
  live in a run-scoped `RunContext` closure — NOT in the checkpointed state,
  because the langgraph checkpointer can't serialize it. That was a real bug.)
- Where is the single validation bottleneck? (`state/schema.py` — MCP tool I/O,
  hypotheses, backtest results all validate there. Why one place? ForgeLM's
  synthetic tool-call data reuses the same schemas, so training-data validation
  and runtime validation can't drift apart.)
- Name the 4 MCP servers and their tool lists. What does each depend on?
  (statistics: scipy/numpy ONLY; market-data: yfinance+parquet or deterministic
  synthetic; backtest: vectorized pandas event study; retrieval: tfidf|dense)
- Where do retries live and what happens on the 3rd failure?
  (`with_retries` decorator, exponential backoff → ToolCallFailed → agent
  records AgentError + fallback (mark UNTESTABLE / skip signal). Never silent.)

## 2. The rigor story (your differentiator — rehearse this precisely)

- Every hypothesis is tested, then Bonferroni (α=0.05, family = all tested in
  the run) AND BH-FDR (q=0.10). Survival needs corrected-significance AND the
  effect sign matching the pre-registered expectation. Why both gates?
- Null results are first-class: when 0 survive, the report explains each
  failure (sign mismatch vs p-above-gate vs untestable). Know the numbers:
  tasks produce 3–5 survivors of 8; the gates REJECT 3–5 per run by design.
- Backtests: entry close T+1 (no same-close look-ahead), strict train ≤2022 /
  test ≥2023 membership (event entry date decides the split), 10bps round-trip
  per event. Sharpe is annualized by events-per-year over the FULL evaluation
  window — know the bug story: annualizing by event-date span turned 3 events
  in 8 days into "36/yr". `n_independent` counts non-overlapping windows
  (reversal: 285 raw → ~108 independent) — overlapping events share return
  windows, so raw counts overstate sample size.
- Synthetic mode plants documented effects (PEAD +60bps/day×10d, reversal
  bounce +40bps/day×10d, volume-shock −30bps/day×5d, Monday −5bps). Tests
  assert SIGN RECOVERY of planted effects — that's how you validate a
  backtester without live data.

## 3. Questions you should be able to answer (out loud)

1. Walk me through what happens end-to-end for one seed query.
2. Why LangGraph? What would you lose with plain function calls?
   (conditional routing, checkpointed interrupts for HITL, resumability,
   state schema enforcement between agents)
3. How does your HITL checkpoint actually pause? (MemorySaver checkpointer +
   interrupt(); without a checkpointer langgraph silently completes —
   describe how you found and fixed that.)
4. Why both Bonferroni and BH-FDR? When do they disagree?
   (Bonferroni: FWER, stricter; BH: FDR, step-up. With 8 tests α=0.05:
   threshold 0.00625 vs BH's rank-dependent cutoffs.)
5. What's your false-discovery exposure even with these gates?
   (within-run protected; cross-run template-parameter specification search
   still inflates risk — the OOS split is the primary defense. SAY THIS
   unprompted.)
6. How do you know your backtester isn't lying? (planted-effect sign recovery
   tests; split-membership test recomputed independently; honest Sharpe
   annualization; cost subtractions verified exact to 40bps)
7. Why is hypothesis generation seeded templates instead of an LLM?
   (reproducible, $0, CI-safe; LLM refinement is optional and
   schema-validated; the template engine is the CONTROLLED baseline the LLM
   must beat or match)
8. What happens when the LLM returns garbage? (per-hypothesis fallback to the
   template statement, rejected_outputs counter, notes logged — never crashes,
   never silently "uses AI")
9. How would ForgeLM slot in? (it must emit hypotheses passing the SAME
   Pydantic schema + call the same statistics tools; DPO penalizes missing
   multiple-testing correction — the M3 eval metric is exactly the stats-
   correction rate. AlphaForge's schema.py is the shared dependency.)
10. What would you build next? (RL environment with a limit-order-book
    simulator to test whether agent-discovered signals survive realistic
    execution costs — the verbal flex from the spec)

## 4. Commands you must be able to run from a clean checkout

```bash
pip install -e ".[dev]" && pytest -q            # green in ~4 min
alphaforge "post-earnings drift in megacap tech"
alphaforge "volume shocks" --interactive        # HITL asks; answer approve/reject
python evals/harness.py                         # 22-task suite -> evals/results.md
docker build -t alphaforge . && docker run --rm alphaforge "any seed query"
```

Live mode (needs network only): `alphaforge "<seed>" --mode live`
LLM mode (needs key): `export ANTHROPIC_API_KEY=... && alphaforge "<seed>" --llm claude`

## 5. Known-honest limitations (offer these before they're asked)

- Synthetic results validate the pipeline, not real alpha
- Live earnings coverage is ~4y (vendor limit) → thin train samples for
  earnings-driven hypotheses
- Flat cost model; no market impact or slippage curve
- Overlapping events share windows (n_independent is reported for this)
- BH/Bonferroni protect within a run only

## 6. Rebuild drills (do each from an empty buffer)

1. Write `apply_bh_fdr` from scratch; verify against statsmodels
2. Write `_forward_returns` (entry T+1, exit T+1+window) and explain the shift
3. Write the StateGraph wiring: 6 nodes, 1 conditional edge, 1 interrupt
4. Write the Bonferroni/BH correction + sign gate block of the validation agent
5. Write the synthetic-generator planting loop (rng per ticker — explain WHY
   per-ticker streams: composition-free determinism)
