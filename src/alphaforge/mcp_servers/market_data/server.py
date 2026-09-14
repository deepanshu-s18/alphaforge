"""mcp-market-data: OHLCV + earnings calendar over the research universe.

Two modes:
  - synthetic: deterministic seeded market with PLANTED, documented effects
    (post-earnings drift, reversal bounce, volume-shock drift, Monday
    seasonality) so statistical tests and the backtester can be verified
    against known ground truth. No API keys, no network — CI-safe.
  - live: yfinance OHLCV into a checksum-verified parquet cache, 1 req/s.

Both modes return the same schemas, so every downstream agent is
mode-agnostic. Same config => byte-identical cached data.
"""

from __future__ import annotations

import time as _time

import numpy as np
import pandas as pd

from alphaforge.state.schema import frame_to_records
from alphaforge.utils.cache import ParquetCache, make_key
from alphaforge.utils.config import (
    DEFAULT_THRESHOLDS,
    DEFAULT_UNIVERSE,
    all_tickers,
    load_universe,
    synthetic_market_cfg,
)

OHLCV_COLUMNS = ["ticker", "date", "open", "high", "low", "close", "volume"]
EARNINGS_COLUMNS = ["ticker", "date", "surprise_pct", "bmo_amc"]


# ---------------------------------------------------------------------------
# Synthetic market generator (deterministic, effects planted)
# ---------------------------------------------------------------------------


def generate_synthetic_market(
    tickers: list[str],
    start: str,
    end: str,
    seed: int = 42,
    synth_cfg: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a deterministic daily market for `tickers`.

    Planted effects (configurable via synth_cfg):
      earnings_beat_drift_bps: mean extra return per day for `holding` days after a beat
      reversal_bounce_bps:     mean bounce per day for 10 days after a <=-8% 5-day drop
      volume_shock_drift_bps:  mean drift per day for 5 days after a volume spike
      Mondays:                 -5 bps mean (seasonality)
    """
    synth_cfg = synth_cfg or {}
    drift_ann = synth_cfg.get("annual_drift", 0.05)
    vol_ann = synth_cfg.get("annual_vol", 0.25)
    beat_drift = synth_cfg.get("earnings_beat_drift_bps", 60) / 10_000
    bounce = synth_cfg.get("reversal_bounce_bps", 40) / 10_000
    shock_drift = synth_cfg.get("volume_shock_drift_bps", -30) / 10_000

    days = pd.bdate_range(start, end)
    n = len(days)

    ohlcv_frames, earnings_frames = [], []
    for t in sorted(tickers):
        # Per-ticker RNG stream: series for ticker X are identical whether X is
        # generated alone or inside any universe subset (composition-free determinism).
        rng = np.random.default_rng((seed + sum(map(ord, t))) % (2**31))
        is_monday = np.array([d.weekday() == 0 for d in days])
        mu, sigma = drift_ann / 252, vol_ann / np.sqrt(252)
        base_ret = rng.normal(mu, sigma, n)
        base_ret[is_monday] -= 5e-4  # planted Monday seasonality

        price0 = float(rng.uniform(20, 400))
        close = price0 * np.exp(np.cumsum(base_ret))
        spread = np.abs(rng.normal(0, sigma / 2, n))
        high = close * (1 + spread)
        low = close * (1 - spread)
        open_ = np.roll(close, 1) * (1 + rng.normal(0, sigma / 4, n))
        open_[0] = price0
        volume = np.round(np.exp(rng.normal(np.log(5e6), 0.35, n))).astype("int64")

        # volume spikes (~3% of days) with planted negative next-week drift
        spike = rng.random(n) < 0.03
        volume[spike] = (volume[spike] * rng.uniform(2.5, 4.0, spike.sum())).astype("int64")
        _inject_drift(base_ret, np.nonzero(spike)[0].astype(float), horizon=5, per_day=shock_drift)

        # earnings: 4 dates/year, seeded; beats plant post-earnings drift
        e_rng = np.random.default_rng((seed + sum(map(ord, t)) + 1) % (2**31))
        e_dates, e_surprise = [], []
        for year in range(days[0].year, days[-1].year + 1):
            for month in (1, 4, 7, 10):
                offset = int(e_rng.integers(15, 55))
                d = pd.Timestamp(year=year, month=month, day=1) + pd.Timedelta(days=offset)
                if days[0] <= d <= days[-1]:
                    e_dates.append(d)
                    e_surprise.append(float(e_rng.normal(0, 4.0)))
        e_idx = [days.searchsorted(d) for d in e_dates]
        for i, pos in enumerate(e_idx):
            if e_surprise[i] > 2.0 and pos + 1 < n:  # beat threshold +2%
                _inject_drift(base_ret, np.array([pos]), horizon=10, per_day=beat_drift)

        # reversal bounce: after a 5-day drop of <= -8%, bounce for 10 days
        close2 = price0 * np.exp(np.cumsum(base_ret))
        r5 = pd.Series(close2).pct_change(5).to_numpy()
        rev = np.where(np.nan_to_num(r5, nan=0.0) <= -0.08)[0]
        _inject_drift(base_ret, rev.astype(float), horizon=10, per_day=bounce)

        # NOTE: effects injected into base_ret AFTER volume/earnings but prices
        # use the final return stream below — recomputed once at the end.
        close = price0 * np.exp(np.cumsum(base_ret))
        high = close * (1 + spread)
        low = close * (1 - spread)
        open_ = np.roll(close, 1) * (1 + rng.normal(0, sigma / 4, n))
        open_[0] = price0

        ohlcv_frames.append(
            pd.DataFrame(
                {
                    "ticker": t,
                    "date": days,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )
        )
        if e_dates:
            earnings_frames.append(
                pd.DataFrame(
                    {
                        "ticker": t,
                        "date": e_dates,
                        "surprise_pct": e_surprise,
                        "bmo_amc": "amc",
                    }
                )
            )

    ohlcv = pd.concat(ohlcv_frames, ignore_index=True)
    earnings = (
        pd.concat(earnings_frames, ignore_index=True) if earnings_frames else
        pd.DataFrame(columns=EARNINGS_COLUMNS)
    )
    return ohlcv, earnings


def _inject_drift(rets: np.ndarray, event_positions: np.ndarray, horizon: int, per_day: float):
    for pos in event_positions:
        lo, hi = int(pos) + 1, min(int(pos) + 1 + horizon, len(rets))
        rets[lo:hi] += per_day


# ---------------------------------------------------------------------------
# Public data-access API (the "MCP tools")
# ---------------------------------------------------------------------------


class MarketData:
    def __init__(self, universe_path=DEFAULT_UNIVERSE, thresholds_path=DEFAULT_THRESHOLDS,
                 mode: str | None = None):
        self.universe_cfg = load_universe(universe_path)
        self.tickers = all_tickers(self.universe_cfg)
        self.groups = self.universe_cfg["universe"]
        d = self.universe_cfg["data"]
        self.start, self.end = d["start"], d["end"]
        # explicit mode overrides config file; cache key includes it so
        # synthetic and live datasets can never collide
        self.mode = mode or d.get("mode", "synthetic")
        self.synth_cfg = synthetic_market_cfg(thresholds_path)
        self.cache = ParquetCache()
        self._rate_last = 0.0

    # -- tools ------------------------------------------------------------
    def get_ohlcv(self, tickers: list[str] | None = None, start: str | None = None,
                  end: str | None = None) -> pd.DataFrame:
        tickers = tickers or self.tickers
        start = start or self.start
        end = end or self.end
        frames = [self._ohlcv_one(t, start, end) for t in tickers]
        df = pd.concat(frames, ignore_index=True)
        return df[(df["date"] >= start) & (df["date"] <= end)].reset_index(drop=True)

    def get_earnings_calendar(self, tickers: list[str] | None = None,
                              start: str | None = None, end: str | None = None) -> pd.DataFrame:
        tickers = tickers or self.tickers
        start = start or self.start
        end = end or self.end
        frames = [self._earnings_one(t) for t in tickers]
        df = pd.concat(frames, ignore_index=True)
        df = df[(df["date"] >= start) & (df["date"] <= end)]
        return df.reset_index(drop=True)

    def list_universe(self) -> dict[str, list[str]]:
        return self.groups

    def get_index_constituents(self, group: str = "megacap_tech") -> list[str]:
        return self.groups[group]

    # -- internals ----------------------------------------------------------
    def _dataset(self, kind: str, ticker: str) -> pd.DataFrame:
        # mode in the key: synthetic and live versions of the same ticker/range
        # must never collide in the cache
        key = make_key(f"{kind}_{self.mode}_{ticker}", self.start, self.end, "1d")
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        if self.mode == "live" and kind == "ohlcv":
            df = self._fetch_live_ohlcv(ticker)
        elif self.mode == "live":
            df = self._fetch_live_earnings(ticker)
        elif kind == "ohlcv":
            ohlcv, _ = generate_synthetic_market([ticker], self.start, self.end, seed=42,
                                                 synth_cfg=self.synth_cfg)
            df = ohlcv
        else:
            _, earnings = generate_synthetic_market([ticker], self.start, self.end, seed=42,
                                                    synth_cfg=self.synth_cfg)
            df = earnings
        self.cache.put(key, df)
        return df

    def _ohlcv_one(self, ticker: str, start: str, end: str) -> pd.DataFrame:
        df = self._dataset("ohlcv", ticker)
        return df[(df["date"] >= start) & (df["date"] <= end)]

    def _earnings_one(self, ticker: str) -> pd.DataFrame:
        return self._dataset("earnings", ticker)

    def _throttle(self):
        now = _time.monotonic()
        wait = 1.0 - (now - self._rate_last)
        if wait > 0:
            _time.sleep(wait)
        self._rate_last = _time.monotonic()

    def _fetch_live_ohlcv(self, ticker: str) -> pd.DataFrame:  # pragma: no cover - network
        import yfinance as yf

        self._throttle()
        raw = yf.download(ticker, start=self.start, end=self.end, interval="1d",
                          auto_adjust=False, progress=False)
        if raw is None or raw.empty:
            raise IOError(f"no OHLCV data for {ticker}")
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        cols = ["date", "open", "high", "low", "close", "volume"]
        df = raw.reset_index().rename(columns=str.lower)[cols]
        df.insert(0, "ticker", ticker)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df

    def _fetch_live_earnings(self, ticker: str) -> pd.DataFrame:  # pragma: no cover - network
        import yfinance as yf

        self._throttle()
        ed = yf.Ticker(ticker).earnings_dates
        if ed is None or ed.empty:
            return pd.DataFrame(columns=EARNINGS_COLUMNS)
        # earnings_dates: DatetimeIndex (tz-aware) named "Earnings Date",
        # columns EPS Estimate / Reported EPS / Surprise(%)
        out = pd.DataFrame({
            "ticker": ticker,
            "date": pd.to_datetime(ed.index).tz_localize(None).normalize(),
        })
        surprise = ed["Surprise(%)"] if "Surprise(%)" in ed.columns else np.nan
        out["surprise_pct"] = pd.Series(surprise).to_numpy()[: len(out)]
        out["bmo_amc"] = "unknown"
        out = out.dropna(subset=["surprise_pct"])
        return out[["ticker", "date", "surprise_pct", "bmo_amc"]]


def build_server():  # pragma: no cover - requires mcp package
    from mcp.server.fastmcp import FastMCP

    md = MarketData()
    mcp = FastMCP("mcp-market-data")

    @mcp.tool()
    def get_ohlcv(ticker: str, start: str, end: str) -> list[dict]:
        """Daily OHLCV bars for one ticker in the research universe."""
        return frame_to_records(md.get_ohlcv([ticker], start, end))

    @mcp.tool()
    def get_earnings_calendar(ticker: str, start: str, end: str) -> list[dict]:
        """Earnings event dates with surprise percent for one ticker."""
        return frame_to_records(md.get_earnings_calendar([ticker], start, end))

    @mcp.tool()
    def list_universe() -> dict:
        """Universe groups (sector -> tickers) from config."""
        return md.list_universe()

    @mcp.tool()
    def get_index_constituents(group: str) -> list[str]:
        """Tickers in a universe group, e.g. megacap_tech or sector_etfs."""
        return md.get_index_constituents(group)

    return mcp


if __name__ == "__main__":  # pragma: no cover
    build_server().run()
