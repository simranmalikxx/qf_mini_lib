# -*- coding: utf-8 -*-
"""
Asian option pricing

Implements three pricing families:
- Analytical  (methods implemented separately in AnalyticalAsianPricer)
- Monte Carlo (STANDARD, ANTITHETIC, CONTROLVARIATE, QUASI RANDOM)
- PDE         (EXPLICIT, IMPLICIT, CRANKNICOLSON)

Supports both Asian strike styles (FIXED vs FLOATING) and averaging styles (ARITHMETIC vs GEOMETRIC).
"""
from __future__ import annotations

from typing import Optional, Tuple
from types import SimpleNamespace
import math
import numpy as np
from scipy.linalg import solve_banded
from scipy.stats import qmc, norm

# Use your existing AsianOption definition (do NOT redefine it here)
from core.asian import AsianOption

# Framework imports
from utils.enum import (
    OptionType,
    ExerciseType,
    ValuationType,
    MonteCarloMethod,
    PDEScheme,
    AsianStrikeType,
    AsianAveragingType,
    AsianAnalyticalMethod,
)
from utils.math import N
from utils.helper import validate_enum
from utils.error import FinError
from utils.globals import DEFAULT_NUM_PATHS
from market.utils import extract_pricing_inputs

# -------------------------
# Small compatibility helpers (no behavior change)
# -------------------------
def _get_averaging(option: AsianOption) -> AsianAveragingType:
    # supports either .averaging or .averaging_type
    return getattr(option, "averaging", getattr(option, "averaging_type"))

def _get_n_obs(option: AsianOption) -> int:
    # supports either .n_obs or .n_avg_points
    return getattr(option, "n_obs", getattr(option, "n_avg_points"))

# =========================
# Main entry point
# =========================
class AsianPricer:
    """Unified entry point for Asian option pricing."""

    def __init__(self, market) -> None:
        self.market = market

    def price(
        self,
        option: AsianOption,
        valuation: ValuationType,
        *,
        analytical_method: Optional[AsianAnalyticalMethod] = None,
        mc_method: Optional[MonteCarloMethod] = None,
        pde_scheme: Optional[PDEScheme] = None,
        num_paths: int = DEFAULT_NUM_PATHS,
        seed: Optional[int] = None,
        time_steps_per_year: int = 252,
        fd_grid: Optional[Tuple[int, int]] = None,
    ) -> float:
        validate_enum(valuation, ValuationType)

        # Pull canonical inputs from the Market layer.
        # Expected keys: spot, discount_curve, dividend_curve (optional), vol_surface
        # For vol, we query at strike if FIXED, else at S0
        maturity = option.maturity
        strike_for_vol = option.strike if option.strike_type is AsianStrikeType.FIXED else None
        inputs = extract_pricing_inputs(self.market, self.market.as_of, maturity, strike_for_vol)
        S0 = float(inputs["spot"])
        disc_curve = inputs.get("discount_curve")
        div_curve = inputs.get("dividend_curve", None)
        vol_surface = inputs.get("vol_surface")

        if disc_curve is None or vol_surface is None:
            raise FinError("Market inputs incomplete: require discount_curve and vol_surface")

        # Risk-free rate from discount factor
        df_T = disc_curve.df(maturity)
        if df_T <= 0:
            raise FinError("Invalid discount factor from curve")
        r = -math.log(df_T) / max(maturity, 1e-14)

        # Dividend yield q from dividend curve if available
        if div_curve is not None:
            try:
                dq_T = div_curve.df(maturity)
                q = -math.log(dq_T) / max(maturity, 1e-14)
            except Exception:
                q = 0.0
        else:
            q = 0.0

        # Volatility (strike-dependent or flat)
        try:
            sigma = float(
                vol_surface.vol(
                    maturity,
                    option.strike if option.strike_type is AsianStrikeType.FIXED else S0,
                )
            )
        except Exception:
            sigma = float(vol_surface.vol(maturity))

        if sigma <= 0:
            raise FinError("Non-positive volatility from vol_surface")

        if valuation is ValuationType.ANALYTICAL:
            if analytical_method is None:
                raise FinError("analytical_method must be provided for ValuationType.ANALYTICAL")
            return AnalyticalAsianPricer.price(
                option, S0, r, q, sigma, disc_curve, analytical_method
            )

        if valuation is ValuationType.MONTE_CARLO:
            if mc_method is None:
                raise FinError("mc_method must be provided for ValuationType.MONTE_CARLO")
            return MonteCarloAsianPricer.price(
                option,
                S0,
                r,
                q,
                sigma,
                disc_curve,
                mc_method=mc_method,
                num_paths=num_paths,
                seed=seed,
                time_steps_per_year=time_steps_per_year,
            )

        if valuation is ValuationType.FINITE_DIFFERENCE:
            if pde_scheme is None:
                raise FinError("pde_scheme must be provided for ValuationType.FINITE_DIFFERENCE")
            return PDEAsianPricer.price(
                option,
                S0,
                r,
                q,
                sigma,
                disc_curve,
                pde_scheme=pde_scheme,
                grid=fd_grid,
            )

        raise FinError(f"Unsupported valuation type: {valuation}")

# =========================
# Analytical
# =========================
class AnalyticalAsianPricer:
    @staticmethod
    def price(
        option: AsianOption,
        S0: float,
        r: float,
        q: float,
        sigma: float,
        disc_curve,
        method: AsianAnalyticalMethod,
    ) -> float:
        validate_enum(method, AsianAnalyticalMethod)
        if option.exercise is not ExerciseType.EUROPEAN:
            raise FinError("Analytical Asian currently supports EUROPEAN exercise only")

        if method is AsianAnalyticalMethod.GEOMETRIC:
            return AnalyticalAsianPricer._geometric_closed_form(option, S0, r, q, sigma, disc_curve)
        elif method is AsianAnalyticalMethod.TURNBULL_WAKEMAN:
            return AnalyticalAsianPricer._turnbull_wakeman(option, S0, r, q, sigma, disc_curve)
        elif method is AsianAnalyticalMethod.CURRAN:
            return AnalyticalAsianPricer._curran(option, S0, r, q, sigma, disc_curve)
        else:
            raise FinError(f"Unknown AsianAnalyticalMethod: {method}")

    # Helpers
    @staticmethod
    def _bs_price(F: float, K: float, sigma: float, T: float, option_type: OptionType, df: float) -> float:
        """Black–Scholes style price given forward, strike, vol, maturity, and discount factor."""
        if sigma * math.sqrt(T) <= 0:
            intrinsic = max(0.0, (F - K) if option_type is OptionType.CALL else (K - F))
            return df * intrinsic

        d1_ = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
        d2_ = d1_ - sigma * math.sqrt(T)

        if option_type is OptionType.CALL:
            return df * (F * N(d1_) - K * N(d2_))
        else:
            return df * (K * N(-d2_) - F * N(-d1_))

    # Geometric closed-form (Kemna & Vorst, 1990)
    @staticmethod
    def _geometric_closed_form(option: AsianOption, S0: float, r: float, q: float, sigma: float, disc_curve) -> float:
        T = option.maturity  # this is a Date
        n = _get_n_obs(option)
        if n <= 0:
            raise FinError("n_obs / n_avg_points must be positive")
    
        # time to maturity in years
        t = disc_curve.as_of.years_between(T)
    
        # discrete sampling coefficients (equally spaced)
        a = (n + 1.0) / (2.0 * n)
        b = (n + 1.0) * (2.0 * n + 1.0) / (6.0 * n * n)
    
        # effective log-vol and drift for the geometric average
        sigma_g = sigma * math.sqrt(b)
        mu_g = (r - q - 0.5 * sigma * sigma) * a + 0.5 * sigma_g * sigma_g
    
        # discount factor
        df_T = disc_curve.discount_factor_T(t)
        if df_T <= 0:
            raise FinError("Invalid discount factor from curve")
    
        # forward (expected) level of the geometric average under the risk-neutral measure
        F_G = S0 * math.exp(mu_g * t)
    
        # FIXED strike: treat G as lognormal with vol sigma_g and forward F_G
        if option.strike_type is AsianStrikeType.FIXED:
            K = option.strike
            return AnalyticalAsianPricer._bs_price(F_G, K, sigma_g, t, option.option_type, df_T)
    
        # FLOATING strike: exchange option payoff (S_T - G)^+
        F_S = S0 * math.exp((r - q) * t)
        var_S = sigma * sigma * t
        var_G = sigma_g * sigma_g * t
        cov_SG = sigma * sigma * a * t
        var_ex = max(var_S + var_G - 2.0 * cov_SG, 1e-18)
    
        vol_ex = math.sqrt(var_ex / t)
    
        return AnalyticalAsianPricer._bs_price(F_S, F_G, vol_ex, t, option.option_type, df_T)


       # Turnbull–Wakeman (arith fixed)
    @staticmethod
    def _turnbull_wakeman(option: AsianOption, S0: float, r: float, q: float, sigma: float, disc_curve) -> float:
        if _get_averaging(option) is not AsianAveragingType.ARITHMETIC:
            raise FinError("Turnbull–Wakeman is for ARITHMETIC averaging")
        if option.strike_type is not AsianStrikeType.FIXED:
            raise FinError("Turnbull–Wakeman is for FIXED-strike Asians")

        # --- Convert dates to year fractions ---
        valuation = disc_curve.as_of
        T = valuation.years_between(option.maturity)   # maturity in years

        n = _get_n_obs(option)
        if n <= 0:
            raise FinError("n_obs / n_avg_points must be positive")

        # Equally spaced observation times (in years)
        ti = [(i * T) / n for i in range(1, n + 1)]

        # --- Mean of the arithmetic average ---
        mu = (S0 / n) * sum(math.exp((r - q) * t) for t in ti)

        # --- Variance of the arithmetic average ---
        var = 0.0
        for i in range(n):
            for j in range(n):
                tij = min(ti[i], ti[j])
                var += math.exp(2 * (r - q) * tij) * math.exp(sigma * sigma * tij)

        var = (S0 * S0 / (n * n)) * (var - math.exp(2 * (r - q) * T))

        # --- Implied lognormal volatility for forward approximation ---
        sigma_hat = math.sqrt(max(math.log(1.0 + var / (mu * mu)), 1e-18) / T)

        # --- Forward approximation ---
        F_A = mu
        df_T = disc_curve.discount_factor_T(T)

        return AnalyticalAsianPricer._bs_price(F_A, option.strike, sigma_hat, T, option.option_type, df_T)


    # Curran (arith fixed) – simple version
    @staticmethod
    def _curran(option: AsianOption, S0: float, r: float, q: float, sigma: float, disc_curve) -> float:
        if _get_averaging(option) is not AsianAveragingType.ARITHMETIC:
            raise FinError("Curran is for ARITHMETIC averaging")
        if option.strike_type is not AsianStrikeType.FIXED:
            raise FinError("Curran implemented for FIXED strike Asians")

        T = option.maturity
        n = _get_n_obs(option)
        df_T = disc_curve.df(T)

        # effective geometric approximation
        sigma_g = sigma * math.sqrt((n + 1) * (2 * n + 1) / (6 * n * n))
        mu_g = (r - q - 0.5 * sigma * sigma) * (n + 1) / (2 * n) + 0.5 * sigma_g * sigma_g

        Fg = S0 * math.exp(mu_g * T)
        d1_ = (math.log(Fg / option.strike) + 0.5 * sigma_g * sigma_g * T) / (sigma_g * math.sqrt(T))
        d2_ = d1_ - sigma_g * math.sqrt(T)

        if option.option_type is OptionType.CALL:
            return df_T * (Fg * N(d1_) - option.strike * N(d2_))
        else:
            return df_T * (option.strike * N(-d2_) - Fg * N(-d1_))

# =========================
# Monte Carlo
# =========================
class MonteCarloAsianPricer:
    @staticmethod
    def price(
        option: AsianOption,
        S0: float,
        r: float,
        q: float,
        sigma: float,
        disc_curve,
        *,
        mc_method: MonteCarloMethod,
        num_paths: int,
        seed: Optional[int],
        time_steps_per_year: int,
    ) -> float:
        validate_enum(mc_method, MonteCarloMethod)

        T = option.maturity
        n_obs = _get_n_obs(option)
        obs_times = np.linspace(T / n_obs, T, n_obs)  # observation dates (exclude 0)

        if mc_method is MonteCarloMethod.STANDARD:
            paths = _simulate_paths(S0, r, q, sigma, obs_times, num_paths, seed)
            payoffs = _asian_payoff_from_paths(paths, option)
            return float(disc_curve.df(T) * np.mean(payoffs))

        if mc_method is MonteCarloMethod.ANTITHETIC:
            paths = _simulate_paths(S0, r, q, sigma, obs_times, num_paths // 2, seed, antithetic=True)
            payoffs = _asian_payoff_from_paths(paths, option)
            return float(disc_curve.df(T) * np.mean(payoffs))

        if mc_method is MonteCarloMethod.CONTROL_VARIATE:
            # Only valid for Arithmetic Fixed Asians
            if not (_get_averaging(option) is AsianAveragingType.ARITHMETIC and option.strike_type is AsianStrikeType.FIXED):
                raise FinError("Control variate is supported only for Arithmetic Fixed Asian options")

            # simulate both arithmetic + geometric
            paths = _simulate_paths(S0, r, q, sigma, obs_times, num_paths, seed)
            A_arith, A_geom, S_T = _asian_averages_from_paths(paths)
            payoffs_arith = _asian_payoff(option, A_arith, A_geom, S_T)

            # geometric Asian analytic price (shim object; no dependency on core class ctor)
            geom_option = SimpleNamespace(
                option_type=option.option_type,
                strike_type=option.strike_type,
                averaging=AsianAveragingType.GEOMETRIC,
                strike=option.strike,
                maturity=option.maturity,
                n_obs=n_obs,
                exercise=option.exercise,
            )
            EX = AnalyticalAsianPricer._geometric_closed_form(geom_option, S0, r, q, sigma, disc_curve)
            payoffs_geom = _asian_payoff(geom_option, A_arith, A_geom, S_T)

            # control variate adjustment
            cov = np.cov(payoffs_geom, payoffs_arith, bias=True)[0, 1]
            varX = np.var(payoffs_geom)
            beta = 0.0 if varX == 0 else cov / varX
            Y_cv = payoffs_arith - beta * (payoffs_geom - EX)
            return float(disc_curve.df(T) * np.mean(Y_cv))

        if mc_method is MonteCarloMethod.QUASI_RANDOM:
            # If scipy qmc available, use Sobol; else fall back
            try:
                paths = _simulate_paths_quasi(S0, r, q, sigma, obs_times, num_paths, seed)
            except Exception:
                paths = _simulate_paths(S0, r, q, sigma, obs_times, num_paths, seed)
            payoffs = _asian_payoff_from_paths(paths, option)
            return float(disc_curve.df(T) * np.mean(payoffs))

        raise FinError(f"Unsupported MonteCarloMethod: {mc_method}")

# -------------------------
# Helpers for MC
# -------------------------
def _simulate_paths(S0, r, q, sigma, obs_times, num_paths, seed=None, antithetic=False):
    """Simulate GBM lognormal paths at given observation times."""
    if seed is not None:
        np.random.seed(seed)
    n_obs = len(obs_times)
    paths = np.zeros((num_paths * (2 if antithetic else 1), n_obs))
    for i, t in enumerate(obs_times):
        dt = t if i == 0 else obs_times[i] - obs_times[i - 1]
        drift = (r - q - 0.5 * sigma**2) * dt
        vol = sigma * np.sqrt(dt)
        Z = np.random.normal(size=num_paths)
        incr = np.exp(drift + vol * Z)
        if i == 0:
            S = S0 * incr
        else:
            S = S * incr
        paths[:num_paths, i] = S
        if antithetic:
            incr_anti = np.exp(drift - vol * Z)
            if i == 0:
                S_anti = S0 * incr_anti
            else:
                S_anti = S_anti * incr_anti
            paths[num_paths:, i] = S_anti
    return paths

def _simulate_paths_quasi(S0, r, q, sigma, obs_times, num_paths, seed=None):
    """Sobol quasi-Monte Carlo paths."""
    n_obs = len(obs_times)
    sampler = qmc.Sobol(d=n_obs, scramble=True, seed=seed)
    U = sampler.random(num_paths)
    Z = norm.ppf(U)
    paths = np.zeros((num_paths, n_obs))
    for i in range(n_obs):
        dt = obs_times[i] if i == 0 else obs_times[i] - obs_times[i - 1]
        drift = (r - q - 0.5 * sigma**2) * dt
        vol = sigma * np.sqrt(dt)
        incr = np.exp(drift + vol * Z[:, i])
        if i == 0:
            S = S0 * incr
        else:
            S = S * incr
        paths[:, i] = S
    return paths

def _asian_averages_from_paths(paths: np.ndarray):
    """Return arithmetic mean, geometric mean, and final S_T for each path."""
    A_arith = np.mean(paths, axis=1)
    A_geom = np.exp(np.mean(np.log(paths), axis=1))
    S_T = paths[:, -1]
    return A_arith, A_geom, S_T

def _asian_payoff_from_paths(paths: np.ndarray, option: AsianOption):
    """Compute payoff from simulated paths given Asian option spec."""
    A_arith, A_geom, S_T = _asian_averages_from_paths(paths)
    return _asian_payoff(option, A_arith, A_geom, S_T)

def _asian_payoff(option: AsianOption, A_arith, A_geom, S_T):
    """Compute payoff for an Asian option from averages and S_T."""
    averaging = _get_averaging(option)
    if averaging is AsianAveragingType.ARITHMETIC:
        A = A_arith
    elif averaging is AsianAveragingType.GEOMETRIC:
        A = A_geom
    else:
        raise FinError("Unsupported averaging type")

    if option.strike_type is AsianStrikeType.FIXED:
        strike = option.strike
    elif option.strike_type is AsianStrikeType.FLOATING:
        strike = A
    else:
        raise FinError("Unsupported strike type")

    if option.option_type is OptionType.CALL:
        return np.maximum(A - strike, 0.0)
    elif option.option_type is OptionType.PUT:
        return np.maximum(strike - A, 0.0)
    else:
        raise FinError("Unsupported option type")

# =========================
# PDE (Geometric, Floating-strike)
# =========================
class PDEAsianPricer:
    """
    PDE pricer for geometric-floating Asian options (W(x,τ) formulation).
    Provides three implementations:
      - _geo_floating_explicit
      - _geo_floating_implicit (backward Euler)
      - _geo_floating_crank_nicolson

    Each method returns the price (float) = S0 * W(0, τ=T).

    NOTE: This class implements GEOMETRIC averaging + FLOATING strike only.
    """

    @staticmethod
    def price(
        option: AsianOption,
        S0: float,
        r: float,
        q: float,     # q unused in W-formulation but kept for signature consistency
        sigma: float,
        disc_curve,
        *,
        pde_scheme: PDEScheme,
        grid: Optional[Tuple[int, int]] = None,
    ) -> float:
        validate_enum(pde_scheme, PDEScheme)

        # Only geometric floating supported here
        if _get_averaging(option) is not AsianAveragingType.GEOMETRIC:
            raise FinError("PDEAsianPricer supports GEOMETRIC averaging only (floating-strike).")
        if option.strike_type is not AsianStrikeType.FLOATING:
            raise FinError("PDEAsianPricer supports FLOATING strike only for the W-formulation.")

        # resolution: (M spatial, L time)
        M = int(grid[0]) if (grid and len(grid) > 0 and grid[0] is not None) else 400
        L = int(grid[1]) if (grid and len(grid) > 1 and grid[1] is not None) else 4000

        T = option.maturity
        if pde_scheme == PDEScheme.EXPLICIT:
            return PDEAsianPricer._geo_floating_explicit(S0=S0, r=r, sigma=sigma, T=T, M=M, L=L)
        elif pde_scheme == PDEScheme.IMPLICIT:
            return PDEAsianPricer._geo_floating_implicit(S0=S0, r=r, sigma=sigma, T=T, M=M, L=L)
        elif pde_scheme == PDEScheme.CRANK_NICOLSON:
            return PDEAsianPricer._geo_floating_crank_nicolson(S0=S0, r=r, sigma=sigma, T=T, M=M, L=L)
        else:
            raise FinError(f"Unsupported PDEScheme: {pde_scheme}")

    # small helper: base spatial coefficients for W PDE
    # PDE: W_tau = 0.5*sigma^2 W_xx + r W_x - r W
    @staticmethod
    def _coeff_geo_x(sigma: float, r: float, dx: float) -> tuple[float, float]:
        """Return alpha, beta used frequently in schemes (no dt factor here)."""
        alpha = 0.5 * sigma * sigma / (dx * dx)   # multiplies second difference
        beta = 0.5 * r / dx                       # multiplies first difference (with factor 2 when used)
        return alpha, beta

    # EXPLICIT scheme
    @staticmethod
    def _geo_floating_explicit(S0: float, r: float, sigma: float, T: float, *, M: int = 400, L: int = 4000, x_max: float = 3.0) -> float:
        dx = 2.0 * x_max / M
        dt = T / L
        x_grid = np.linspace(-x_max, x_max, M + 1)
        W = np.zeros((M + 1, L + 1))

        # terminal condition at tau=0
        W[:, 0] = np.maximum(1.0 - np.exp(x_grid), 0.0)

        # stability (conservative)
        stability_rhs = dx * dx / (sigma * sigma + abs(r) * dx)
        if dt > stability_rhs:
            # don't fail; warn
            print(f"Warning: explicit scheme may be unstable: dt={dt:.6e} > stable~{stability_rhs:.6e}")

        alpha, beta = PDEAsianPricer._coeff_geo_x(sigma, r, dx)

        for n in range(0, L):
            Wn = W[:, n]
            # central differences on interior nodes
            d2Wdx = (Wn[2:] - 2.0 * Wn[1:-1] + Wn[:-2])      # corresponds to /dx^2 when used with alpha
            dWdx = (Wn[2:] - Wn[:-2])                        # corresponds to /(2 dx) when used with beta

            interior_update = (alpha * d2Wdx) + (beta * 2.0 * dWdx) - (r * Wn[1:-1])
            W[1:-1, n + 1] = Wn[1:-1] + dt * interior_update

            # BCs: x->-inf => 0 ; x->+inf => discounted far-field intrinsic
            W[0, n + 1] = 0.0
            W[-1, n + 1] = math.exp(-r * (T - (n + 1) * dt)) * max(1.0 - math.exp(x_grid[-1]), 0.0)

        # interpolate at x=0
        idx = np.searchsorted(x_grid, 0.0)
        if idx == 0:
            w0 = W[0, -1]
        elif idx >= len(x_grid):
            w0 = W[-1, -1]
        else:
            theta = (0.0 - x_grid[idx - 1]) / (x_grid[idx] - x_grid[idx - 1])
            w0 = (1.0 - theta) * W[idx - 1, -1] + theta * W[idx, -1]

        return float(S0 * w0)

    # IMPLICIT (backward Euler)
    @staticmethod
    def _geo_floating_implicit(S0: float, r: float, sigma: float, T: float, *, M: int = 400, L: int = 4000, x_max: float = 3.0) -> float:
        dx = 2.0 * x_max / M
        dt = T / L
        x_grid = np.linspace(-x_max, x_max, M + 1)
        W = np.zeros((M + 1, L + 1))

        W[:, 0] = np.maximum(1.0 - np.exp(x_grid), 0.0)

        alpha, beta = PDEAsianPricer._coeff_geo_x(sigma, r, dx)

        # stencil coefficients (no dt)
        a = alpha - beta       # multiplies W_{i-1}
        b = -2.0 * alpha - r   # multiplies W_i
        c = alpha + beta       # multiplies W_{i+1}

        # Build banded matrix for (I - dt L)
        diag = 1.0 - dt * b                # = 1 + 2*alpha*dt + r*dt
        lower = -dt * a                    # = -dt*(alpha - beta)
        upper = -dt * c                    # = -dt*(alpha + beta)

        # Full banded form for M+1 nodes (we will use interior slice for solver)
        ab = np.zeros((3, M + 1))
        ab[1, :] = diag
        ab[0, 1:] = upper
        ab[2, :-1] = lower

        # enforce identity rows for boundaries
        ab[1, 0] = 1.0
        ab[0, 0] = 0.0
        ab[2, -1] = 0.0
        ab[1, -1] = 1.0

        # stepping
        for n in range(0, L):
            rhs = W[:, n].copy()
            rhs[0] = 0.0
            rhs[-1] = math.exp(-r * (T - (n + 1) * dt)) * max(1.0 - math.exp(x_grid[-1]), 0.0)

            # build interior banded system for nodes 1..M-1
            ab_in = np.zeros((3, M - 1))
            ab_in[1, :] = ab[1, 1:-1]
            if M - 1 > 1:
                ab_in[0, 1:] = ab[0, 2:-1]
                ab_in[2, :-1] = ab[2, 1:-2]

            W[1:-1, n + 1] = solve_banded((1, 1), ab_in, rhs[1:-1])

            W[0, n + 1] = rhs[0]
            W[-1, n + 1] = rhs[-1]

        # interpolate at x=0
        idx = np.searchsorted(x_grid, 0.0)
        if idx == 0:
            w0 = W[0, -1]
        elif idx >= len(x_grid):
            w0 = W[-1, -1]
        else:
            theta = (0.0 - x_grid[idx - 1]) / (x_grid[idx] - x_grid[idx - 1])
            w0 = (1.0 - theta) * W[idx - 1, -1] + theta * W[idx, -1]

        return float(S0 * w0)

    # CRANK–NICOLSON
    @staticmethod
    def _geo_floating_crank_nicolson(S0: float, r: float, sigma: float, T: float, *, M: int = 400, L: int = 4000, x_max: float = 3.0) -> float:
        dx = 2.0 * x_max / M
        dt = T / L
        x_grid = np.linspace(-x_max, x_max, M + 1)
        W = np.zeros((M + 1, L + 1))

        W[:, 0] = np.maximum(1.0 - np.exp(x_grid), 0.0)
        alpha, beta = PDEAsianPricer._coeff_geo_x(sigma, r, dx)

        # stencil no-dt coefficients
        a = alpha - beta
        b = -2.0 * alpha - r
        c = alpha + beta

        # A = I - 0.5 dt L
        diag_A = 1.0 - 0.5 * dt * b
        lower_A = -0.5 * dt * a
        upper_A = -0.5 * dt * c

        # B = I + 0.5 dt L (used on RHS)
        diag_B = 1.0 + 0.5 * dt * b
        lower_B = +0.5 * dt * a
        upper_B = +0.5 * dt * c

        # banded A (full nodes)
        ab_A = np.zeros((3, M + 1))
        ab_A[1, :] = diag_A
        ab_A[0, 1:] = upper_A
        ab_A[2, :-1] = lower_A

        # boundary identity enforcement
        ab_A[1, 0] = 1.0
        ab_A[0, 0] = 0.0
        ab_A[2, -1] = 0.0
        ab_A[1, -1] = 1.0

        # step in time
        for n in range(0, L):
            Wn = W[:, n]

            # build RHS = B * Wn (interior)
            rhs = np.empty_like(Wn)
            rhs[1:-1] = lower_B * Wn[:-2] + diag_B * Wn[1:-1] + upper_B * Wn[2:]
            rhs[0] = 0.0
            rhs[-1] = math.exp(-r * (T - (n + 1) * dt)) * max(1.0 - math.exp(x_grid[-1]), 0.0)

            # interior banded A
            ab_in = np.zeros((3, M - 1))
            ab_in[1, :] = ab_A[1, 1:-1]
            if M - 1 > 1:
                ab_in[0, 1:] = ab_A[0, 2:-1]
                ab_in[2, :-1] = ab_A[2, 1:-2]

            W[1:-1, n + 1] = solve_banded((1, 1), ab_in, rhs[1:-1])

            W[0, n + 1] = rhs[0]
            W[-1, n + 1] = rhs[-1]

        # interpolate at x=0
        idx = np.searchsorted(x_grid, 0.0)
        if idx == 0:
            w0 = W[0, -1]
        elif idx >= len(x_grid):
            w0 = W[-1, -1]
        else:
            theta = (0.0 - x_grid[idx - 1]) / (x_grid[idx] - x_grid[idx - 1])
            w0 = (1.0 - theta) * W[idx - 1, -1] + theta * W[idx, -1]

        return float(S0 * w0)
