"""Validation agent: build event samples, run statistical tests, correct for
multiple testing, and emit only surviving signals.

Survival requires BOTH:
  1. corrected significance (Bonferroni alpha AND BH-FDR q)
  2. effect sign matching the hypothesis's expected direction

A system that "finds" 10/10 signals is p-hacking; this gate is designed to
kill bad ideas — null results are a feature, not a failure.
"""

from __future__ import annotations

import numpy as np

from alphaforge.agents.base import RunContext, ToolCallFailed, with_retries
from alphaforge.events.builders import build_events
from alphaforge.mcp_servers.statistics import server as st
from alphaforge.state.schema import (
    AgentError,
    Hypothesis,
    ResearchState,
    Signal,
    TestResult,
)
from alphaforge.utils.logging import get_logger

log = get_logger("alphaforge.validation")


class ValidationAgent:
    def __init__(self):
        pass

    @with_retries(retries=3, base_delay=0.1)
    def _test(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def _effect_and_ci(self, sample, baseline, h: Hypothesis, cfg) -> tuple[float, float, float]:
        """Effect size (bps) and bootstrap CI of the effect.

        Each hypothesis gets a unique but deterministic seed derived from
        cfg.bootstrap_seed + a hash of the hypothesis id, so different
        hypotheses produce different (realistic) CI widths while the overall
        run remains reproducible given the same config.
        """
        h_seed = (cfg.bootstrap_seed + sum(map(ord, h.id))) % (2**31)
        rng = np.random.default_rng(h_seed)
        if h.test_type == "one_sample_ttest":
            vals = sample
            boots = rng.choice(vals, size=(cfg.bootstrap_iters, len(vals))).mean(axis=1)
            effect = vals.mean()
        else:
            boots = (
                rng.choice(sample, size=(cfg.bootstrap_iters, len(sample))).mean(axis=1)
                - rng.choice(baseline, size=(cfg.bootstrap_iters, len(baseline))).mean(axis=1)
            )
            effect = sample.mean() - baseline.mean()
        low, high = np.percentile(boots, [2.5, 97.5])
        return float(effect * 1e4), float(low * 1e4), float(high * 1e4)

    def run(self, state: ResearchState, ctx: RunContext) -> ResearchState:
        cfg = state.config
        if ctx.data.get("ohlcv") is None:
            log.warning("validation_no_data", reason="ohlcv absent — all hypotheses untestable")
            state.errors.append(AgentError(
                agent="validation", error="ohlcv data absent from context",
                fallback_action="mark_all_untestable"
            ))
            for h in state.hypotheses:
                h.status = "untestable"
            return state

        results: list[TestResult] = []
        for h in state.hypotheses:
            try:
                sample = build_events(h.family, h.event_def, ctx.data["ohlcv"],
                                      ctx.data.get("earnings"))
                if len(sample.treated) < cfg.min_events:
                    h.status = "untestable"
                    continue
                ctx.record_tool(True)

                if h.test_type == "one_sample_ttest":
                    res = self._test(st.run_ttest_1samp, list(sample.treated), 0.0, "greater")
                elif h.test_type == "mannwhitney_u":
                    res = self._test(st.run_mannwhitney, list(sample.treated),
                                     list(sample.baseline))
                else:
                    res = self._test(st.run_ttest, list(sample.treated), list(sample.baseline))
                ctx.record_tool(True)

                effect, ci_low, ci_high = self._effect_and_ci(
                    sample.treated, sample.baseline, h, cfg
                )
                results.append(TestResult(
                    hypothesis_id=h.id, test_name=h.test_type,
                    statistic=res["stat"], p_value=res["p"],
                    effect_size=effect, ci_low=ci_low, ci_high=ci_high,
                    n_events=len(sample.treated),
                    bonferroni_significant=False, bh_fdr_significant=False,
                    corrected_significant=False,
                ))
            except (ToolCallFailed, ValueError) as e:
                ctx.record_tool(False)
                log.warning("validation_failed", hypothesis=h.id, error=str(e))
                state.errors.append(AgentError(
                    agent="validation", hypothesis_id=h.id, error=str(e),
                    fallback_action="mark_untestable"
                ))
                h.status = "untestable"

        # multiple-testing correction across ALL tested hypotheses
        pvals = [r.p_value for r in results]
        bonf = st.apply_bonferroni(pvals, cfg.bonferroni_alpha)
        bh = st.apply_bh_fdr(pvals, cfg.bh_fdr_q)
        by_id = {h.id: h for h in state.hypotheses}
        for r, b, f in zip(results, bonf, bh):
            r.bonferroni_significant = b
            r.bh_fdr_significant = f
            # spec: Bonferroni AND BH-FDR — both gates must agree.
            # Using `or` would allow either test alone to promote a signal,
            # which is exactly the p-hacking this system is designed to prevent.
            r.corrected_significant = b and f
            h = by_id[r.hypothesis_id]
            # Direct comparison avoids np.sign(0.0) == 0.0 which is neither
            # > 0 nor < 0, making an exact-zero effect silently ambiguous.
            sign_ok = (
                r.effect_size > 0
                if h.expected_effect == "positive"
                else r.effect_size < 0
            )
            if r.corrected_significant and sign_ok:
                h.status = "validated"
                state.surviving_signals.append(Signal(
                    signal_id=f"S-{r.hypothesis_id}",
                    hypothesis_id=h.id,
                    event_spec={**h.event_def, "family": h.family},
                    entry_rule="close at T+1 after event",
                    exit_rule=f"close at T+{h.event_def['window_days'] + 1}",
                ))
            else:
                h.status = "rejected"

        state.test_results = results
        return state
