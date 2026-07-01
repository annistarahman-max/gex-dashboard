"""
Black-Scholes Greeks Engine
===========================
Vectorized computation of Delta, Gamma, Vanna, and Charm for European options.

All inputs accept numpy arrays for batch computation across entire option chains.
Uses the standard Black-Scholes-Merton model under continuous dividend yield q.

Conventions:
    S  = spot price of the underlying
    K  = strike price
    T  = time to expiration in years  (must be > 0)
    r  = risk-free interest rate (annualized, continuous compounding)
    q  = continuous dividend yield (annualized)
    σ  = implied volatility (annualized)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

Float = float | NDArray[np.float64]

_TINY: float = 1e-10


def _d1(S: Float, K: Float, T: Float, r: float, q: float, sigma: Float) -> Float:
    return (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(S: Float, K: Float, T: Float, r: float, q: float, sigma: Float) -> Float:
    return _d1(S, K, T, r, q, sigma) - sigma * np.sqrt(T)


# --------------------------------------------------------------------------- #
#  Delta:  ∂V/∂S
# --------------------------------------------------------------------------- #

def delta(
    S: Float, K: Float, T: Float, r: float, q: float, sigma: Float,
    is_call: NDArray[np.bool_],
) -> Float:
    """
    Call delta = e^{-qT} N(d1)
    Put  delta = e^{-qT} [N(d1) - 1]
    """
    d1 = _d1(S, K, T, r, q, sigma)
    call_delta = np.exp(-q * T) * norm.cdf(d1)
    put_delta = np.exp(-q * T) * (norm.cdf(d1) - 1.0)
    return np.where(is_call, call_delta, put_delta)


# --------------------------------------------------------------------------- #
#  Gamma:  ∂²V/∂S²  (same for calls and puts)
# --------------------------------------------------------------------------- #

def gamma(
    S: Float, K: Float, T: Float, r: float, q: float, sigma: Float,
) -> Float:
    """
    Gamma = e^{-qT} * n(d1) / (S * σ * √T)
    where n(·) is the standard normal PDF.
    """
    d1 = _d1(S, K, T, r, q, sigma)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T) + _TINY)


# --------------------------------------------------------------------------- #
#  Vanna:  ∂²V/∂S∂σ  =  ∂Delta/∂σ  (same for calls and puts)
# --------------------------------------------------------------------------- #

def vanna(
    S: Float, K: Float, T: Float, r: float, q: float, sigma: Float,
) -> Float:
    """
    Vanna = -e^{-qT} * n(d1) * d2 / (S * σ)

    Measures how delta changes when implied volatility moves — critical for
    understanding market-maker exposure to vol-spot correlation.
    """
    d1 = _d1(S, K, T, r, q, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return -np.exp(-q * T) * norm.pdf(d1) * d2 / (S * sigma + _TINY)


# --------------------------------------------------------------------------- #
#  Charm:  ∂Delta/∂T  =  -∂²V/∂S∂t   (delta decay / delta bleed)
# --------------------------------------------------------------------------- #

def charm(
    S: Float, K: Float, T: Float, r: float, q: float, sigma: Float,
    is_call: NDArray[np.bool_],
) -> Float:
    """
    Charm (delta decay) — the rate at which delta erodes as time passes.

    Call charm = -e^{-qT} [ n(d1) * (2(r-q)T - d2 σ√T) / (2T σ√T) + q N(d1) ]
    Put  charm = -e^{-qT} [ n(d1) * (2(r-q)T - d2 σ√T) / (2T σ√T) - q N(-d1) ]

    Convention: positive charm means delta is increasing as time passes.
    """
    d1 = _d1(S, K, T, r, q, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    sqrt_T = np.sqrt(T)
    pdf_d1 = norm.pdf(d1)
    discount = np.exp(-q * T)

    common = -discount * pdf_d1 * (2.0 * (r - q) * T - d2 * sigma * sqrt_T) / (
        2.0 * T * sigma * sqrt_T + _TINY
    )

    call_charm = common - q * discount * norm.cdf(d1)
    put_charm = common + q * discount * norm.cdf(-d1)

    return np.where(is_call, call_charm, put_charm)


# --------------------------------------------------------------------------- #
#  Time-to-maturity helper
# --------------------------------------------------------------------------- #

def time_to_maturity_years(
    expiry: np.datetime64 | np.ndarray,
    now: np.datetime64 | None = None,
) -> Float:
    """
    Precise T in calendar years.  Uses actual calendar days (ACT/365).
    Floors at a small positive value to avoid division by zero on expiry day.
    """
    if now is None:
        now = np.datetime64("now")
    delta_days = (np.asarray(expiry, dtype="datetime64[D]")
                  - np.asarray(now, dtype="datetime64[D]")).astype(float)
    return np.maximum(delta_days / 365.0, _TINY)
