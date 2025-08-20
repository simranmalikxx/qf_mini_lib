# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 03:25:53 2025

@author: Simran
"""

# -*- coding: utf-8 -*-
"""
Vanilla option pricing (Analytical, Monte Carlo, Finite Difference).

Integrates with:
- market.market_data.Market
- market.utils.extract_pricing_inputs
- models.FD_solver.FiniteDifferenceEngine
- utils.enum.{PDEScheme, MonteCarloMethod}

All functions take:
    (market_data, valuation_date, expiry_date, strike, ...)

Expected by tests in tests/test_vanilla_pricing.py
"""


from typing import Tuple
import numpy as np
from scipy.stats import norm

from models.FD_solver import FiniteDifferenceEngine
from utils.enum import PDEScheme, MonteCarloMethod
from market.utils import extract_pricing_inputs


# -------------------------------
# Helpers
# -------------------------------

def _inputs(market_data, valuation_date, expiry_date, strike) -> Tuple[float, float, float, float]:
    """
    Returns (S0, r, sigma, T) via the shared extractor.
    Note: extractor returns (S0, sigma, r, T) in your latest version,
    so we re-order to (S0, r, sigma, T) for the pricing functions.
    """
    S0, sigma, r, T = extract_pricing_inputs(market_data, valuation_date, expiry_date, strike)
    return float(S0), float(r), float(sigma), float(T)


# -------------------------------
# 1) Analytical (Black–Scholes)
# -------------------------------

def price_vanilla_analytical(
    market_data,
    valuation_date,
    expiry_date,
    strike: float,
    option_type: str,
) -> float:
    """
    Black–Scholes price for European vanilla call/put under flat r and vol.
    option_type: "call" or "put"
    """
    S0, r, sigma, T = _inputs(market_data, valuation_date, expiry_date, strike)

    # Immediate payoff if expiry reached (defensive)
    if T <= 0.0:
        if option_type.lower() == "call":
            return max(S0 - strike, 0.0)
        elif option_type.lower() == "put":
            return max(strike - S0, 0.0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    if sigma <= 0.0:
        # Degenerate: forward-priced intrinsic with discounting
        fwd = S0 * np.exp(r * T)
        if option_type.lower() == "call":
            return np.exp(-r * T) * max(fwd - strike, 0.0)
        elif option_type.lower() == "put":
            return np.exp(-r * T) * max(strike - fwd, 0.0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    sqrtT = np.sqrt(T)
    d1 = (np.log(S0 / strike) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if option_type.lower() == "call":
        return S0 * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        return strike * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    else:
        raise ValueError("option_type must be 'call' or 'put'")


# -------------------------------
# 2) Monte Carlo
# -------------------------------

def price_vanilla_mc(
    market_data,
    valuation_date,
    expiry_date,
    strike: float,
    option_type: str,
    method: MonteCarloMethod,
    num_paths: int = 200_000,
) -> float:
    """
    GBM-based European vanilla pricing via Monte Carlo.
    Supports STANDARD, ANTITHETIC, CONTROL_VARIATE, QUASI_RANDOM (Sobol).
    """
    S0, r, sigma, T = _inputs(market_data, valuation_date, expiry_date, strike)

    if T <= 0.0:
        return price_vanilla_analytical(market_data, valuation_date, expiry_date, strike, option_type)

    # base normals
    rng = np.random.default_rng(1234)  # reproducible; bump/remove for stochastic CI if desired
    if method == MonteCarloMethod.QUASI_RANDOM:
        # Sobol -> inverse CDF to normals
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=1, scramble=True, seed=1234)
        u = sampler.random(num_paths).reshape(-1)
        z = norm.ppf(u)
    else:
        z = rng.standard_normal(num_paths)

    if method == MonteCarloMethod.ANTITHETIC:
        z = np.concatenate([z, -z], axis=0)

    drift = (r - 0.5 * sigma ** 2) * T
    diff = sigma * np.sqrt(T)
    ST = S0 * np.exp(drift + diff * z)

    if option_type.lower() == "call":
        payoff = np.maximum(ST - strike, 0.0)
    elif option_type.lower() == "put":
        payoff = np.maximum(strike - ST, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    if method == MonteCarloMethod.CONTROL_VARIATE:
        # Use European call as the control variate (works for both payoffs)
        control_payoff = np.maximum(ST - strike, 0.0)
        control_exact = price_vanilla_analytical(market_data, valuation_date, expiry_date, strike, "call")

        # beta = Cov(X, Y) / Var(Y)
        cov = np.cov(payoff, control_payoff, ddof=1)[0, 1]
        var_c = np.var(control_payoff, ddof=1)
        if var_c > 0:
            beta = cov / var_c
            payoff = payoff - beta * (control_payoff - control_exact)

    price = np.exp(-r * T) * np.mean(payoff)
    return float(price)


# -------------------------------
# 3) Finite Difference (PDE)
# -------------------------------

def price_vanilla_fd(
    market_data,
    valuation_date,
    expiry_date,
    strike: float,
    option_type: str,
    scheme: PDEScheme,
) -> float:
    """
    1D Black–Scholes PDE solved via FD:
      - EXPLICIT, IMPLICIT, CRANK_NICOLSON schemes via FiniteDifferenceEngine
    """
    S0, r, sigma, T = _inputs(market_data, valuation_date, expiry_date, strike)

    # Spatial grid
    Smax = max(5.0 * strike, 5.0 * S0)
    M = 400
    dS = Smax / M
    S = np.linspace(0.0, Smax, M + 1)

    # Time steps
    if scheme == PDEScheme.EXPLICIT:
        # CFL-ish stability consideration
        dt_max = dS**2 / (sigma**2 * Smax**2 + r * Smax * dS + 1e-16)
        N = max(int(T / dt_max) + 1, 1000)
    else:
        N = 400

    # Coefficients for interior points (i = 1..M-1). Pad to length M+1 for engine convenience.
    a = 0.5 * sigma**2 * S[1:-1]**2 / dS**2 - 0.5 * r * S[1:-1] / dS
    b = -sigma**2 * S[1:-1]**2 / dS**2 - r
    c = 0.5 * sigma**2 * S[1:-1]**2 / dS**2 + 0.5 * r * S[1:-1] / dS

    a = np.concatenate(([0.0], a, [0.0]))
    b = np.concatenate(([0.0], b, [0.0]))
    c = np.concatenate(([0.0], c, [0.0]))

    opt = option_type.lower()
    if opt == "call":
        payoff_fn = lambda Sarr: np.maximum(Sarr - strike, 0.0)

        def bc_fn(t, xS):
            # V(0,t)=0 ; V(Smax,t) ~ S - K e^{-r tau}
            tau = max(T - t, 0.0)
            if xS <= 0.0:
                return 0.0
            if xS >= Smax:
                return xS - strike * np.exp(-r * tau)
            return 0.0  # unused; engine uses only boundaries

    elif opt == "put":
        payoff_fn = lambda Sarr: np.maximum(strike - Sarr, 0.0)

        def bc_fn(t, xS):
            # V(0,t) ~ K e^{-r tau} ; V(Smax,t)=0
            tau = max(T - t, 0.0)
            if xS <= 0.0:
                return strike * np.exp(-r * tau)
            if xS >= Smax:
                return 0.0
            return 0.0

    else:
        raise ValueError("option_type must be 'call' or 'put'")

    engine = FiniteDifferenceEngine(Smax=Smax, M=M, N=N, method=scheme)
    S_grid, V_grid = engine.solve(a=a, b=b, c=c, T=T, payoff_fn=payoff_fn, boundary_cond_fn=bc_fn)

    # Interpolate price at S0
    return float(np.interp(S0, S_grid, V_grid))
