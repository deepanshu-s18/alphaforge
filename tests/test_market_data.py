"""Market-data server: determinism, cache integrity, schema, planted effects."""

from __future__ import annotations

import pandas as pd
import pytest

import alphaforge.mcp_servers.market_data.server as md_mod
from alphaforge.mcp_servers.market_data import server as md
from alphaforge.utils.cache import make_key

TICKERS = ["AAPL", "MSFT", "XLE"]
START, END = "2015-01-01", "2025-12-31"


@pytest.fixture(scope="module")
def market(tmp_path_factory):
    m = md.MarketData.__new__(md.MarketData)
    import yaml

    from alphaforge.utils.config import DEFAULT_UNIVERSE, all_tickers

    m.universe_cfg = yaml.safe_load(open(DEFAULT_UNIVERSE))
    m.tickers = all_tickers(m.universe_cfg)
    m.groups = m.universe_cfg["universe"]
    m.start, m.end, m.mode = START, END, "synthetic"
    m.synth_cfg = {}
    from alphaforge.utils.cache import ParquetCache

    m.cache = ParquetCache(tmp_path_factory.mktemp("cache"))
    m._rate_last = 0.0
    return m


class TestSyntheticGenerator:
    def test_deterministic_same_seed(self):
        a_ohlc, a_e = md.generate_synthetic_market(TICKERS, START, END, seed=42)
        b_ohlc, b_e = md.generate_synthetic_market(TICKERS, START, END, seed=42)
        pd.testing.assert_frame_equal(a_ohlc, b_ohlc)
        pd.testing.assert_frame_equal(a_e, b_e)

    def test_different_seed_differs(self):
        a, _ = md.generate_synthetic_market(["AAPL"], START, END, seed=42)
        b, _ = md.generate_synthetic_market(["AAPL"], START, END, seed=123)
        assert not a["close"].equals(b["close"])

    def test_schema_and_range(self):
        ohlcv, earnings = md.generate_synthetic_market(TICKERS, START, END, seed=42)
        assert list(ohlcv.columns) == md.OHLCV_COLUMNS
        assert set(ohlcv["ticker"]) == set(TICKERS)
        assert (ohlcv["close"] > 0).all()
        assert (ohlcv["high"] >= ohlcv["low"]).all()
        assert ohlcv["date"].min() >= pd.Timestamp(START)
        assert ohlcv["date"].max() <= pd.Timestamp(END)
        assert list(earnings.columns) == md.EARNINGS_COLUMNS

    def test_planted_earnings_beat_drift_positive(self):
        """After surprise>2 beats, next-10d mean return must exceed unconditional."""
        ohlcv, earnings = md.generate_synthetic_market(["AAPL"], START, END, seed=42)
        px = ohlcv.set_index("date")["close"]
        beats = earnings[earnings["surprise_pct"] > 2.0]["date"]
        rets_10d = px.pct_change(10).shift(-10)
        after = rets_10d.reindex(beats).dropna()
        overall = rets_10d.dropna()
        assert after.mean() > overall.mean()

    def test_planted_monday_seasonality(self):
        ohlcv, _ = md.generate_synthetic_market(["AAPL"], START, END, seed=42)
        rets = ohlcv.set_index("date")["close"].pct_change()
        mon, fri = rets[rets.index.weekday == 0].mean(), rets[rets.index.weekday == 4].mean()
        assert mon < fri


class TestMarketDataTools:
    def test_get_ohlcv_schema_and_filter(self, market):
        df = market.get_ohlcv(["AAPL"], "2020-01-01", "2020-12-31")
        assert list(df.columns) == md.OHLCV_COLUMNS
        assert (df["date"] >= "2020-01-01").all()
        assert (df["date"] <= "2020-12-31").all()
        assert set(df["ticker"]) == {"AAPL"}

    def test_get_earnings_calendar(self, market):
        df = market.get_earnings_calendar(["AAPL"], START, END)
        assert list(df.columns) == md.EARNINGS_COLUMNS
        assert len(df) >= 40  # ~4/year * 11 years

    def test_list_universe_and_constituents(self, market):
        u = market.list_universe()
        assert "sector_etfs" in u and "XLK" in u["sector_etfs"]
        assert "AAPL" in market.get_index_constituents("megacap_tech")

    def test_cache_roundtrip_and_checksum(self, market):
        df1 = market.get_ohlcv(["MSFT"], START, END)
        df2 = market.get_ohlcv(["MSFT"], START, END)  # second call hits cache
        pd.testing.assert_frame_equal(df1, df2)

    def test_cache_key_determinism(self):
        assert make_key("ohlcv_AAPL", "2015-01-01", "2025-12-31", "1d") == \
               make_key("ohlcv_AAPL", "2015-01-01", "2025-12-31", "1d")

    def test_cache_corruption_detected(self, market, tmp_path):
        from alphaforge.utils.cache import ParquetCache

        cache = ParquetCache(tmp_path / "corrupt")
        df = pd.DataFrame({"a": [1, 2, 3]})
        cache.put("k", df)
        pq = tmp_path / "corrupt" / "k.parquet"
        pq.write_bytes(pq.read_bytes()[:-5] + b"xxxxx")  # flip bytes
        with pytest.raises(IOError, match="checksum"):
            cache.get("k")


class TestLiveEarningsParser:
    """Regression: yfinance earnings_dates has a tz-aware DatetimeIndex named
    'Earnings Date' and a 'Surprise(%)' column — not a 'date' column."""

    def test_parses_index_and_drops_nan_surprise(self):
        from unittest.mock import patch

        import numpy as np

        md = md_mod.MarketData.__new__(md_mod.MarketData)
        md._rate_last = 0.0
        fake = pd.DataFrame(
            {"EPS Estimate": [1.98, 1.89], "Reported EPS": [np.nan, 2.02],
             "Surprise(%)": [np.nan, 6.74]},
            index=pd.DatetimeIndex(
                [pd.Timestamp("2026-10-29 16:00:00-04:00"),
                 pd.Timestamp("2026-07-30 16:00:00-04:00")],
                name="Earnings Date",
            ),
        )
        with patch("yfinance.Ticker") as T:
            T.return_value.earnings_dates = fake
            out = md._fetch_live_earnings("AAPL")
        assert list(out.columns) == ["ticker", "date", "surprise_pct", "bmo_amc"]
        assert len(out) == 1
        assert out.iloc[0]["date"] == pd.Timestamp("2026-07-30")
        assert out.iloc[0]["surprise_pct"] == 6.74

    def test_empty_calendar(self):
        from unittest.mock import patch

        md = md_mod.MarketData.__new__(md_mod.MarketData)
        md._rate_last = 0.0
        with patch("yfinance.Ticker") as T:
            T.return_value.earnings_dates = pd.DataFrame()
            out = md._fetch_live_earnings("AAPL")
        assert out.empty
