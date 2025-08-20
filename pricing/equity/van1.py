# -*- coding: utf-8 -*-
"""
Created on Sat Aug 16 04:03:21 2025

@author: Simran
"""

# pricing/equity/vanila.py
# -*- coding: utf-8 -*-
"""
Vanilla pricing (Analytical, Monte Carlo, Finite Difference) — curve-consistent discounting.
Uses extract_pricing_inputs for (S0, sigma, r, T) AND discount factors DF(T) from the curve.
"""


from typing import Tuple
import numpy as np
from scipy.stats import norm
from utils.error import FinError
import math

from models.FD_solver import FiniteDifferenceEngine
from utils.enum import PDEScheme, MonteCarloMethod
from market.utils import extract_pricing_inputs


def _inputs_with_df(market_data, valuation_date, expiry_date, strike):
    S0, sigma, T = extract_pricing_inputs(market_data, valuation_date, expiry_date, strike)

    # Discount factor from curve
    curve = market_data.discount_curve()
    df_T = curve.discount_factor_T(T)

    # Curve-consistent effective rate
    r_eff = -math.log(df_T) / T if T > 0 else 0.0

    return float(S0), float(sigma), float(r_eff), float(T), float(df_T), curve



# -------------------------------
# 1) Analytical (Black–Scholes) using DF
# -------------------------------

def price_vanilla_analytical(market_data, valuation_date, expiry_date, strike: float, option_type: str) -> float:
    """
    Curve-consistent Black-Scholes using forward F = S0 * (dq_T / df_T).
    Returns df_T * ( F*N(d1) - K*N(d2) ) for calls.
    """
    S0, sigma, r_eff, T, df_T, discount_curve = _inputs_with_df(market_data, valuation_date, expiry_date, strike)

    if T <= 0.0:
        return max(S0 - strike, 0.0) if option_type.lower() == "call" else max(strike - S0, 0.0)

    # Get dividend discount factor consistently from market (if available)
    dq_T = 1.0
    try:
        div_curve = market_data.dividend_curve()
        dq_T = div_curve.discount_factor_T(T)
    except Exception:
        # If no dividend curve, try market_data.dividend_yield() or fallback to 1.0
        try:
            q_flat = getattr(market_data, "dividend_yield", None)
            if q_flat is not None:
                dq_T = math.exp(-q_flat * T)
        except Exception:
            dq_T = 1.0

    if df_T <= 0:
        raise FinError("Invalid discount factor from curve")

    # Forward consistent with curves
    F = S0 * (dq_T / df_T)

    sqrtT = math.sqrt(T)
    d1 = (math.log(F / strike) + 0.5 * sigma**2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if option_type.lower() == "call":
        return float(df_T * (F * norm.cdf(d1) - strike * norm.cdf(d2)))
    else:
        return float(df_T * (strike * norm.cdf(-d2) - F * norm.cdf(-d1)))





# -------------------------------
# 2) Monte Carlo using DF
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
    """Pricing engine for European vanilla options using Monte Carlo simulation.
    
    Args:
        market_data: Market data container
        valuation_date: Pricing date
        expiry_date: Option expiry date
        strike: Option strike price
        option_type: 'call' or 'put'
        method: Monte Carlo variant (STANDARD/ANTITHETIC/CONTROL_VARIATE/QUASI_RANDOM)
        num_paths: Number of simulation paths
        
    Returns:
        Option price (float)
    """
    # Extract market parameters
    S0, sigma, r_eff, T, df_T, _ = _inputs_with_df(market_data, valuation_date, expiry_date, strike)
    
    if T <= 0.0:
        return price_vanilla_analytical(market_data, valuation_date, expiry_date, strike, option_type)

    # Path generation
    rng = np.random.default_rng(1234)
    if method == MonteCarloMethod.QUASI_RANDOM:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=1, scramble=True, seed=1234)
        z = norm.ppf(sampler.random(num_paths).reshape(-1))
    else:
        z = rng.standard_normal(num_paths)

    if method == MonteCarloMethod.ANTITHETIC:
        z = np.concatenate([z, -z])
        num_paths *= 2

    # Simulate terminal stock prices
    ST = S0 * np.exp((r_eff - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)

    # Calculate payoffs
    payoff = np.maximum(ST - strike, 0) if option_type.lower() == "call" else np.maximum(strike - ST, 0)

    # Control variate adjustment
    if method == MonteCarloMethod.CONTROL_VARIATE:
        control_payoff = np.maximum(ST - strike, 0)
        control_exact = price_vanilla_analytical(
            market_data, valuation_date, expiry_date, strike, "call") / df_T
        
        cov = np.cov(payoff, control_payoff)[0, 1]
        var_c = np.var(control_payoff)
        
        if var_c > 1e-10:
            beta = np.clip(cov / var_c, -2.0, 2.0)  # Prevent extreme values
            payoff = payoff - beta * (control_payoff - control_exact)

    # Discount and average
    return float(df_T * np.mean(payoff))

# -------------------------------
# 3) Finite Difference (PDE) using DF(τ) in boundaries
# -------------------------------
def price_vanilla_fd(
    market_data,
    valuation_date,
    expiry_date,
    strike: float,
    option_type: str,
    scheme: PDEScheme,
) -> float:
    S0, sigma, r_eff, T, df_T, curve = _inputs_with_df(market_data, valuation_date, expiry_date, strike)

    Smax = max(5.0 * strike, 5.0 * S0)
    M = 400
    dS = Smax / M
    S = np.linspace(0.0, Smax, M + 1)

    # time steps
    if scheme == PDEScheme.EXPLICIT:
        dt_max = dS**2 / (sigma**2 * Smax**2 + r_eff * Smax * dS + 1e-16)
        N = max(int(T / dt_max) + 1, 1000)
    else:
        N = 400

    # coefficients with r_eff (the PDE drift term uses the instantaneous rate)
    a = 0.5 * sigma**2 * S[1:-1]**2 / dS**2 - 0.5 * r_eff * S[1:-1] / dS
    b = -sigma**2 * S[1:-1]**2 / dS**2 - r_eff
    c = 0.5 * sigma**2 * S[1:-1]**2 / dS**2 + 0.5 * r_eff * S[1:-1] / dS

    a = np.concatenate(([0.0], a, [0.0]))
    b = np.concatenate(([0.0], b, [0.0]))
    c = np.concatenate(([0.0], c, [0.0]))

    if option_type.lower() == "call":
        payoff_fn = lambda Sarr: np.maximum(Sarr - strike, 0.0)

        def bc_fn(t, xS):
            # Use DF(τ) from curve for boundary at Smax
            tau = max(T - t, 0.0)
            df_tau = curve.discount_factor_T(tau)
            if xS <= 0.0:
                return 0.0
            if xS >= Smax:
                return xS - strike * df_tau
            return 0.0
    else:
        payoff_fn = lambda Sarr: np.maximum(strike - Sarr, 0.0)

        def bc_fn(t, xS):
            tau = max(T - t, 0.0)
            df_tau = curve.discount_factor_T(tau)
            if xS <= 0.0:
                return strike * df_tau
            if xS >= Smax:
                return 0.0
            return 0.0

    engine = FiniteDifferenceEngine(Smax=Smax, M=M, N=N, method=scheme)
    S_grid, V_grid = engine.solve(a=a, b=b, c=c, T=T, payoff_fn=payoff_fn, boundary_cond_fn=bc_fn)
    return float(np.interp(S0, S_grid, V_grid))
