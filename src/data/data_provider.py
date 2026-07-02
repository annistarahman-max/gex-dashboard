"""
Data Provider — Options Chain Fetcher
======================================
Uses yfinance to fetch live spot prices and real options chain data.
Falls back to synthetic data if yfinance fails or has no options for a ticker.
"""

from __future__ import annotations

import abc
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf


class DataProvider(abc.ABC):
    """Abstract interface every data source must satisfy."""

    @abc.abstractmethod
    def fetch_spot(self, ticker: str) -> float:
        """Return the current spot price of *ticker*."""

    @abc.abstractmethod
    def fetch_chain(self, ticker: str) -> pd.DataFrame:
        """
        Return a DataFrame with columns:
            strike           float64
            expiry           datetime64[ns]
            option_type      str   ('call' or 'put')
            open_interest    int64
            implied_volatility  float64
        """

    def available_expiries(self, ticker: str) -> list[date]:
        """Sorted list of expiry dates present in the chain."""
        chain = self.fetch_chain(ticker)
        dates = pd.to_datetime(chain["expiry"]).dt.date.unique()
        return sorted(dates)


# --------------------------------------------------------------------------- #
#  Live yfinance provider
# --------------------------------------------------------------------------- #

class YFinanceProvider(DataProvider):
    """Fetches real spot prices and options chains from Yahoo Finance."""

    def fetch_spot(self, ticker: str) -> float:
        tk = yf.Ticker(ticker.upper())
        info = tk.fast_info
        price = getattr(info, "last_price", None)
        if price is None or price == 0:
            price = getattr(info, "previous_close", None)
        if price is None or price == 0:
            hist = tk.history(period="1d")
            if not hist.empty:
                price = float(hist["Close"].iloc[-1])
        return float(price) if price else 0.0

    def fetch_chain(self, ticker: str) -> pd.DataFrame:
        tk = yf.Ticker(ticker.upper())
        expiry_strings = tk.options
        if not expiry_strings:
            return pd.DataFrame(columns=[
                "strike", "expiry", "option_type", "open_interest", "implied_volatility",
            ])

        rows: list[pd.DataFrame] = []
        for exp_str in expiry_strings:
            try:
                chain = tk.option_chain(exp_str)
            except Exception:
                continue
            expiry_ts = pd.Timestamp(exp_str)

            for opt_type, df in [("call", chain.calls), ("put", chain.puts)]:
                sub = pd.DataFrame({
                    "strike": df["strike"].astype(float),
                    "expiry": expiry_ts,
                    "option_type": opt_type,
                    "open_interest": df["openInterest"].fillna(0).astype(int),
                    "implied_volatility": df["impliedVolatility"].astype(float),
                })
                rows.append(sub)

        if not rows:
            return pd.DataFrame(columns=[
                "strike", "expiry", "option_type", "open_interest", "implied_volatility",
            ])

        result = pd.concat(rows, ignore_index=True)
        result = result[result["open_interest"] > 0]
        # Drop rows with missing/garbage IV (Yahoo's IV on illiquid strikes is
        # often NaN, ~0, or absurdly large — these poison greeks & skew charts)
        result = result[
            result["implied_volatility"].notna()
            & (result["implied_volatility"] > 0.01)
            & (result["implied_volatility"] < 3.0)
        ]
        return result


# --------------------------------------------------------------------------- #
#  Synthetic fallback (used when yfinance has no data)
# --------------------------------------------------------------------------- #

class MockDataProvider(DataProvider):
    """Generates synthetic options chain for development/fallback."""

    def __init__(self, spot: float = 100.0, seed: int = 42) -> None:
        self._spot = spot
        self._rng = np.random.default_rng(seed)

    def fetch_spot(self, ticker: str) -> float:
        return self._spot

    def fetch_chain(self, ticker: str) -> pd.DataFrame:
        spot = self._spot
        today = date.today()

        expiries = [
            today + timedelta(days=d)
            for d in [3, 7, 14, 30, 45, 60, 90, 120, 180]
        ]

        strike_step = _strike_step(spot)
        n_strikes_per_side = 20
        strikes = np.arange(
            spot - n_strikes_per_side * strike_step,
            spot + (n_strikes_per_side + 1) * strike_step,
            strike_step,
        )
        strikes = np.round(strikes / strike_step) * strike_step

        rows: list[dict] = []
        for exp in expiries:
            dte = max((exp - today).days, 1)
            for K in strikes:
                moneyness = np.log(K / spot)
                base_iv = 0.20 + 0.05 * moneyness**2 + 0.03 / np.sqrt(dte / 365.0)
                for opt_type in ("call", "put"):
                    iv = float(np.clip(
                        base_iv + self._rng.normal(0, 0.005), 0.05, 1.5
                    ))
                    oi = self._synthetic_oi(spot, K, dte, opt_type)
                    rows.append({
                        "strike": float(K),
                        "expiry": pd.Timestamp(exp),
                        "option_type": opt_type,
                        "open_interest": int(oi),
                        "implied_volatility": iv,
                    })

        return pd.DataFrame(rows)

    def _synthetic_oi(
        self, spot: float, strike: float, dte: int, opt_type: str,
    ) -> int:
        dist = abs(strike - spot) / spot
        base = 5000 * np.exp(-8 * dist**2)
        if opt_type == "put" and strike < spot:
            base *= 1.4
        if opt_type == "call" and strike > spot:
            base *= 1.2
        dte_factor = np.sqrt(max(dte, 1) / 30.0)
        oi = base * dte_factor * (1.0 + 0.3 * self._rng.standard_normal())
        return max(int(oi), 0)


def _strike_step(spot: float) -> float:
    if spot > 400:
        return 5.0
    if spot > 100:
        return 2.5
    if spot > 50:
        return 1.0
    return 0.5


def get_provider() -> DataProvider:
    """Factory — returns live yfinance provider."""
    return YFinanceProvider()
