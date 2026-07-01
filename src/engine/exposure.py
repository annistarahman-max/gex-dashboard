"""
Exposure Engine
===============
Computes dollar-denominated market-maker exposure profiles from an options chain.

Core exposures:
    GEX  — Gamma Exposure:  how the MM's delta hedge changes per $1 move in spot
    VEX  — Vanna Exposure:  how delta hedge changes per 1-pt move in implied vol
    CEX  — Charm Exposure:  how delta hedge decays over one calendar day

Sign convention (dealer model):
    Market-makers are assumed to be *short* calls and *long* puts (standard retail
    flow assumption).  Therefore call OI flips sign (+1) and put OI flips sign (−1)
    when computing the MM's net greek exposure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.engine.greeks import (
    charm,
    delta,
    gamma,
    time_to_maturity_years,
    vanna,
)


CONTRACT_MULTIPLIER: int = 100


# --------------------------------------------------------------------------- #
#  Dollar-exposure computation
# --------------------------------------------------------------------------- #

def compute_exposures(
    chain: pd.DataFrame,
    spot: float,
    r: float = 0.05,
    q: float = 0.015,
) -> pd.DataFrame:
    """
    Attach per-contract greek exposures to an options chain DataFrame.

    Expected columns in *chain*:
        strike, expiry (datetime64), option_type ('call'/'put'),
        open_interest, implied_volatility

    Returns a copy with added columns:
        T, bs_delta, bs_gamma, bs_vanna, bs_charm,
        gex, vex, cex, delta_exposure
    """
    df = chain.copy()

    now = np.datetime64("today")
    T = time_to_maturity_years(df["expiry"].values.astype("datetime64[D]"), now)
    df["T"] = T

    is_call = (df["option_type"].str.lower() == "call").values
    K = df["strike"].values.astype(np.float64)
    iv = df["implied_volatility"].values.astype(np.float64)
    oi = df["open_interest"].values.astype(np.float64)

    df["bs_delta"] = delta(spot, K, T, r, q, iv, is_call)
    df["bs_gamma"] = gamma(spot, K, T, r, q, iv)
    df["bs_vanna"] = vanna(spot, K, T, r, q, iv)
    df["bs_charm"] = charm(spot, K, T, r, q, iv, is_call)

    mm_sign = np.where(is_call, 1.0, -1.0)

    df["gex"] = oi * df["bs_gamma"] * CONTRACT_MULTIPLIER * spot**2 * mm_sign * 0.01
    df["abs_gex"] = oi * df["bs_gamma"] * CONTRACT_MULTIPLIER * spot**2 * 0.01
    df["vex"] = oi * df["bs_vanna"] * CONTRACT_MULTIPLIER * spot * mm_sign
    df["cex"] = oi * df["bs_charm"] * CONTRACT_MULTIPLIER * spot * mm_sign
    df["delta_exposure"] = oi * df["bs_delta"] * CONTRACT_MULTIPLIER * spot * mm_sign

    return df


# --------------------------------------------------------------------------- #
#  Aggregation by strike (optionally filtered to a single expiry)
# --------------------------------------------------------------------------- #

def aggregate_by_strike(
    exposures: pd.DataFrame,
    expiry_filter: str | None = None,
) -> pd.DataFrame:
    """
    Group exposures by strike and sum.  If *expiry_filter* is a date string
    (e.g. "2026-07-18"), only that expiry is included.
    """
    df = exposures
    if expiry_filter is not None:
        target = pd.Timestamp(expiry_filter)
        df = df[df["expiry"] == target]

    agg = (
        df.groupby("strike", as_index=False)
        .agg(
            gex=("gex", "sum"),
            vex=("vex", "sum"),
            cex=("cex", "sum"),
            delta_exposure=("delta_exposure", "sum"),
            call_gex=("gex", lambda s: s[df.loc[s.index, "option_type"].str.lower() == "call"].sum()),
            put_gex=("gex", lambda s: s[df.loc[s.index, "option_type"].str.lower() == "put"].sum()),
            call_abs_gex=("abs_gex", lambda s: s[df.loc[s.index, "option_type"].str.lower() == "call"].sum()),
            put_abs_gex=("abs_gex", lambda s: s[df.loc[s.index, "option_type"].str.lower() == "put"].sum()),
            call_oi=("open_interest", lambda s: s[df.loc[s.index, "option_type"].str.lower() == "call"].sum()),
            put_oi=("open_interest", lambda s: s[df.loc[s.index, "option_type"].str.lower() == "put"].sum()),
        )
        .sort_values("strike")
    )
    return agg


# --------------------------------------------------------------------------- #
#  Summary KPIs
# --------------------------------------------------------------------------- #

@dataclass
class ExposureSummary:
    net_delta: float
    net_gamma: float
    total_gex: float
    total_vex: float
    total_cex: float
    gamma_flip: float | None


def summarize(exposures: pd.DataFrame, spot: float) -> ExposureSummary:
    return ExposureSummary(
        net_delta=exposures["delta_exposure"].sum(),
        net_gamma=exposures["bs_gamma"].sum(),
        total_gex=exposures["gex"].sum(),
        total_vex=exposures["vex"].sum(),
        total_cex=exposures["cex"].sum(),
        gamma_flip=find_gamma_flip(exposures, spot),
    )


# --------------------------------------------------------------------------- #
#  Gamma Flip — the price where net GEX crosses zero
# --------------------------------------------------------------------------- #

def find_gamma_flip(
    exposures: pd.DataFrame,
    spot: float,
    price_range_pct: float = 0.20,
    n_points: int = 500,
    r: float = 0.05,
    q: float = 0.015,
) -> float | None:
    """
    Re-price net GEX across a grid of hypothetical spot prices and find
    the crossing point closest to the current spot.

    Scans ±price_range_pct around spot (default ±20%).
    Returns None if no sign change is found.
    """
    lo = spot * (1.0 - price_range_pct)
    hi = spot * (1.0 + price_range_pct)
    test_prices = np.linspace(lo, hi, n_points)

    K = exposures["strike"].values.astype(np.float64)
    T = exposures["T"].values.astype(np.float64)
    iv = exposures["implied_volatility"].values.astype(np.float64)
    oi = exposures["open_interest"].values.astype(np.float64)
    is_call = (exposures["option_type"].str.lower() == "call").values
    mm_sign = np.where(is_call, 1.0, -1.0)

    net_gex = np.empty(n_points)
    for i, s in enumerate(test_prices):
        g = gamma(s, K, T, r, q, iv)
        net_gex[i] = np.sum(oi * g * CONTRACT_MULTIPLIER * s**2 * mm_sign * 0.01)

    sign_changes = np.where(np.diff(np.sign(net_gex)))[0]
    if len(sign_changes) == 0:
        return None

    closest = sign_changes[np.argmin(np.abs(test_prices[sign_changes] - spot))]
    g0, g1 = net_gex[closest], net_gex[closest + 1]
    p0, p1 = test_prices[closest], test_prices[closest + 1]
    flip = p0 - g0 * (p1 - p0) / (g1 - g0 + 1e-15)
    return float(flip)
