"""Statistics server: every tool tested against direct scipy reference values."""

from __future__ import annotations

import numpy as np
import pytest
from alphaforge.mcp_servers.statistics import server as st
from scipy import stats


class TestTTest:
    def test_matches_scipy_reference(self):
        rng = np.random.default_rng(7)
        a, b = rng.normal(1.0, 1.0, 200), rng.normal(0.5, 1.0, 200)
        got = st.run_ttest(list(a), list(b))
        ref = stats.ttest_ind(a, b)
        assert got["stat"] == pytest.approx(ref.statistic)
        assert got["p"] == pytest.approx(ref.pvalue)

    def test_known_answer(self):
        # Zero-mean noise vs shifted noise -> strong significance, direction encoded in t
        rng = np.random.default_rng(0)
        a = list(rng.normal(0.5, 0.1, 50))
        b = list(rng.normal(0.0, 0.1, 50))
        got = st.run_ttest(a, b)
        assert got["stat"] > 15
        assert got["p"] < 1e-20

    def test_one_sided_greater(self):
        rng = np.random.default_rng(1)
        a, b = rng.normal(2, 1, 100), rng.normal(0, 1, 100)
        got = st.run_ttest(list(a), list(b), alternative="greater")
        ref = stats.ttest_ind(a, b, alternative="greater")
        assert got["p"] == pytest.approx(ref.pvalue)
        assert got["p"] < 0.001


class TestTTest1Samp:
    def test_matches_scipy_reference(self):
        rng = np.random.default_rng(3)
        vals = rng.normal(0.01, 0.05, 300)
        got = st.run_ttest_1samp(list(vals), 0.0, alternative="greater")
        ref = stats.ttest_1samp(vals, 0.0, alternative="greater")
        assert got["stat"] == pytest.approx(ref.statistic)
        assert got["p"] == pytest.approx(ref.pvalue)


class TestMannWhitney:
    def test_matches_scipy_reference(self):
        rng = np.random.default_rng(11)
        a, b = rng.normal(0, 1, 150), rng.normal(0.4, 1, 150)
        got = st.run_mannwhitney(list(a), list(b))
        ref = stats.mannwhitneyu(a, b, alternative="two-sided")
        assert got["stat"] == pytest.approx(ref.statistic)
        assert got["p"] == pytest.approx(ref.pvalue)


class TestBootstrap:
    def test_ci_covers_true_mean(self):
        rng = np.random.default_rng(42)
        vals = rng.normal(5.0, 1.0, 500)
        ci = st.bootstrap_ci(list(vals), "mean", n_boot=5000, seed=42)
        assert ci["low"] < 5.0 < ci["high"]

    def test_deterministic_given_seed(self):
        vals = [0.01, -0.02, 0.03, 0.04, -0.01, 0.02]
        c1 = st.bootstrap_ci(vals, seed=123)
        c2 = st.bootstrap_ci(vals, seed=123)
        assert c1 == c2

    def test_median_stat(self):
        vals = list(range(101))
        ci = st.bootstrap_ci(vals, "median", n_boot=2000, seed=0)
        assert ci["low"] <= 50 <= ci["high"]


class TestBonferroni:
    def test_reference_behavior(self):
        ps = [0.001, 0.01, 0.04, 0.2]
        alpha = 0.05
        expected = [p <= alpha / len(ps) for p in ps]
        assert st.apply_bonferroni(ps, alpha) == expected

    def test_all_null(self):
        assert st.apply_bonferroni([0.9, 0.95, 0.99]) == [False, False, False]

    def test_all_significant(self):
        assert st.apply_bonferroni([1e-9, 1e-9, 1e-9]) == [True, True, True]

    def test_single_pvalue_equals_uncorrected(self):
        assert st.apply_bonferroni([0.04], 0.05) == [True]
        assert st.apply_bonferroni([0.06], 0.05) == [False]


class TestBHFdr:
    def test_known_example(self):
        # Classic BH example
        ps = [0.01, 0.04, 0.03, 0.005]
        got = st.apply_bh_fdr(ps, q=0.05)
        expected = st.apply_bh_fdr(ps, 0.05)
        assert got == expected  # self-consistent
        # all p-values <= 0.05 with n=4 -> threshold for rank 4 is 0.05 -> all rejected
        assert all(got)

    def test_rejects_all_when_nothing_small(self):
        assert st.apply_bh_fdr([0.5, 0.7, 0.9]) == [False, False, False]

    def test_step_up_monotonicity(self):
        # if p_i rejected, all p_j <= p_i must also be rejected
        rng = np.random.default_rng(0)
        ps = list(rng.uniform(0, 1, 50))
        flags = st.apply_bh_fdr(ps, q=0.10)
        cutoff = max((p for p, f in zip(ps, flags) if f), default=-1.0)
        for p, f in zip(ps, flags):
            if p <= cutoff:
                assert f

    def test_matches_statsmodels_reference(self):
        from statsmodels.stats import multitest

        rng = np.random.default_rng(5)
        ps = list(rng.uniform(0, 1, 30))
        expected = multitest.multipletests(ps, alpha=0.10, method="fdr_bh")[0]
        assert st.apply_bh_fdr(ps, 0.10) == list(expected)
