"""mcp-statistics: the rigor engine. scipy + numpy only.

Exposed both as FastMCP tools (when the `mcp` package is installed) and as
plain Python functions (so tests and agents can call them in-process).
Every function is deterministic given its seed and is unit-tested against
direct scipy calls.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Core statistical functions (in-process, framework-free)
# ---------------------------------------------------------------------------


def run_ttest(sample_a: list[float], sample_b: list[float], alternative: str = "two-sided") -> dict:
    res = stats.ttest_ind(sample_a, sample_b, alternative=alternative)
    return {"stat": float(res.statistic), "p": float(res.pvalue)}


def run_ttest_1samp(values: list[float], popmean: float = 0.0,
                    alternative: str = "greater") -> dict:
    res = stats.ttest_1samp(values, popmean, alternative=alternative)
    return {"stat": float(res.statistic), "p": float(res.pvalue)}


def run_mannwhitney(sample_a: list[float], sample_b: list[float],
                    alternative: str = "two-sided") -> dict:
    res = stats.mannwhitneyu(sample_a, sample_b, alternative=alternative)
    return {"stat": float(res.statistic), "p": float(res.pvalue)}


def bootstrap_ci(
    values: list[float], stat_fn: str = "mean", n_boot: int = 10000, seed: int = 42
) -> dict:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    if stat_fn == "mean":
        boots = arr[idx].mean(axis=1)
    elif stat_fn == "median":
        boots = np.median(arr[idx], axis=1)
    else:
        raise ValueError(f"unsupported stat_fn: {stat_fn}")
    low, high = np.percentile(boots, [2.5, 97.5])
    return {"low": float(low), "high": float(high)}


def apply_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    m = len(p_values)
    return [bool(p <= alpha / m) for p in p_values]


def apply_bh_fdr(p_values: list[float], q: float = 0.10) -> list[bool]:
    """Standard Benjamini-Hochberg step-up procedure."""
    n = len(p_values)
    order = np.argsort(p_values)
    sorted_p = np.asarray(p_values)[order]
    thresholds = q * np.arange(1, n + 1) / n
    below = sorted_p <= thresholds
    if not below.any():
        return [False] * n
    k = np.max(np.nonzero(below)[0])  # largest i with p_(i) <= i*q/n; all <= k rejected
    cutoff = sorted_p[k]
    return [bool(p <= cutoff) for p in p_values]


# ---------------------------------------------------------------------------
# FastMCP server wrapper (only when the official `mcp` package is present)
# ---------------------------------------------------------------------------

TOOLS_SCHEMA = {
    "run_ttest": ["sample_a", "sample_b", "alternative"],
    "run_ttest_1samp": ["values", "popmean", "alternative"],
    "run_mannwhitney": ["sample_a", "sample_b", "alternative"],
    "bootstrap_ci": ["values", "stat_fn", "n_boot", "seed"],
    "apply_bonferroni": ["p_values", "alpha"],
    "apply_bh_fdr": ["p_values", "q"],
}


def build_server():  # pragma: no cover - requires mcp package
    from mcp.server.fastmcp import FastMCP

    mcp = FastMCP("mcp-statistics")

    mcp.tool()(run_ttest)
    mcp.tool()(run_ttest_1samp)
    mcp.tool()(run_mannwhitney)
    mcp.tool()(bootstrap_ci)
    mcp.tool()(apply_bonferroni)
    mcp.tool()(apply_bh_fdr)
    return mcp


if __name__ == "__main__":  # pragma: no cover
    build_server().run()
