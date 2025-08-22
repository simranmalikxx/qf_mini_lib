# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 05:27:03 2025

@author: Simran
"""

"""
Integration test:
End-to-end pricing of Asian options using analytical (geometric)
and Monte Carlo (geometric & arithmetic, fixed & floating).
Includes Monte Carlo standard error estimation outside pricer.
"""

import numpy as np
from market.market_data import Market
from market.curves.flat_discount import FlatCurve
from market.vol.flat_vol import FlatVolSurface
from utils.date import Date
from core.asian import AsianOption
from pricing.equity.asian import AnalyticalAsianPricer, MonteCarloAsianPricer
from utils.enum import (
    OptionType, ExerciseType, ValuationType,
    AsianStrikeType, AsianAveragingType,
    MonteCarloMethod
)


# ================================================================
# Market Setup
# ================================================================
def setup_market():
    S0 = 100.0
    r = 0.05
    q = 0.02
    sigma = 0.25
    valuation = Date(2025, 8, 20)
    maturity = Date(2026, 8, 20)

    disc_curve = FlatCurve(valuation, r)
    div_curve = FlatCurve(valuation, q)
    vol_surface = FlatVolSurface(sigma)

    market = Market(
        as_of=valuation,
        spot=S0,
        discount_curve=disc_curve,
        vol_surface=vol_surface,
        dividend_yield_curve=div_curve,
    )
    return S0, r, q, sigma, valuation, maturity, market


# ================================================================
# Helper for MC with error bars
# ================================================================
def mc_with_error(option, S0, r, q, sigma, disc_curve,
                  mc_method, num_paths=20_000, time_steps_per_year=12,
                  n_batches=5, base_seed=123):
    prices = []
    for i in range(n_batches):
        price = MonteCarloAsianPricer.price(
            option=option,
            S0=S0,
            r=r,
            q=q,
            sigma=sigma,
            disc_curve=disc_curve,
            mc_method=mc_method,
            num_paths=num_paths,
            time_steps_per_year=time_steps_per_year,
            seed=base_seed + i,  # different seed per batch
        )
        prices.append(price)
    mean = float(np.mean(prices))
    se = float(np.std(prices, ddof=1) / np.sqrt(n_batches))
    return mean, se


# ================================================================
# Geometric Asian Options
# ================================================================
def test_geometric_asian():
    S0, r, q, sigma, valuation, T, market = setup_market()
    n_obs = 12
    strike = 100.0

    results = {}

    # ---------- FIXED STRIKE (Analytical + MC) ----------
    geo_fixed = AsianOption(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.EUROPEAN,
        strike_type=AsianStrikeType.FIXED,
        averaging_type=AsianAveragingType.GEOMETRIC,
        strike=strike,
        maturity=T,
        n_avg_points=n_obs,
        valuation_type=ValuationType.ANALYTICAL,
        analytical_method=1,
    )

    analytical_price = AnalyticalAsianPricer._geometric_closed_form(
        geo_fixed, S0, r, q, sigma, market.discount_curve()
    )
    results["Analytical Geo Fixed"] = f"{analytical_price:.6f}"

    for mc_method in [
        MonteCarloMethod.STANDARD,
        MonteCarloMethod.ANTITHETIC,
        MonteCarloMethod.QUASI_RANDOM,
    ]:
        geo_mc = AsianOption(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike_type=AsianStrikeType.FIXED,
            averaging_type=AsianAveragingType.GEOMETRIC,
            strike=strike,
            maturity=T,
            n_avg_points=n_obs,
            valuation_type=ValuationType.MONTE_CARLO,
            mc_method=mc_method,
        )

        price, se = mc_with_error(
            option=geo_mc,
            S0=S0, r=r, q=q, sigma=sigma,
            disc_curve=market.discount_curve(),
            mc_method=mc_method,
            num_paths=20_000,
            time_steps_per_year=12,
        )
        results[f"MC Geo Fixed {mc_method.name}"] = f"{price:.6f} ± {se:.6f}"

    # ---------- FLOATING STRIKE (MC only) ----------
    for mc_method in [
        MonteCarloMethod.STANDARD,
        MonteCarloMethod.ANTITHETIC,
        MonteCarloMethod.QUASI_RANDOM,
    ]:
        geo_float = AsianOption(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike_type=AsianStrikeType.FLOATING,
            averaging_type=AsianAveragingType.GEOMETRIC,
            strike=0.0,  # not used for floating
            maturity=T,
            n_avg_points=n_obs,
            valuation_type=ValuationType.MONTE_CARLO,
            mc_method=mc_method,
        )

        price, se = mc_with_error(
            option=geo_float,
            S0=S0, r=r, q=q, sigma=sigma,
            disc_curve=market.discount_curve(),
            mc_method=mc_method,
            num_paths=20_000,
            time_steps_per_year=12,
        )
        results[f"MC Geo Float {mc_method.name}"] = f"{price:.6f} ± {se:.6f}"

    # Print results
    print("\nGeometric Asian Option Integration Test Results")
    print("=" * 70)
    print(f"{'Method':<35} | {'Price ± SE':>20}")
    print("-" * 70)
    for method, price in results.items():
        print(f"{method:<35} | {price:>20}")
    print("=" * 70)


# ================================================================
# Arithmetic Asian Options
# ================================================================
def test_arithmetic_asian():
    S0, r, q, sigma, valuation, T, market = setup_market()
    n_obs = 12
    strike = 100.0

    results = {}

    # ---------- FIXED STRIKE ----------
    for mc_method in [
        MonteCarloMethod.STANDARD,
        MonteCarloMethod.ANTITHETIC,
        MonteCarloMethod.QUASI_RANDOM,
        MonteCarloMethod.CONTROL_VARIATE,
    ]:
        arith_fixed = AsianOption(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike_type=AsianStrikeType.FIXED,
            averaging_type=AsianAveragingType.ARITHMETIC,
            strike=strike,
            maturity=T,
            n_avg_points=n_obs,
            valuation_type=ValuationType.MONTE_CARLO,
            mc_method=mc_method,
        )

        price, se = mc_with_error(
            option=arith_fixed,
            S0=S0, r=r, q=q, sigma=sigma,
            disc_curve=market.discount_curve(),
            mc_method=mc_method,
            num_paths=20_000,
            time_steps_per_year=12,
        )
        results[f"MC Arith Fixed {mc_method.name}"] = f"{price:.6f} ± {se:.6f}"

    # ---------- FLOATING STRIKE ----------
    for mc_method in [
        MonteCarloMethod.STANDARD,
        MonteCarloMethod.ANTITHETIC,
        MonteCarloMethod.QUASI_RANDOM,
    ]:
        arith_float = AsianOption(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike_type=AsianStrikeType.FLOATING,
            averaging_type=AsianAveragingType.ARITHMETIC,
            strike=0.0,
            maturity=T,
            n_avg_points=n_obs,
            valuation_type=ValuationType.MONTE_CARLO,
            mc_method=mc_method,
        )

        price, se = mc_with_error(
            option=arith_float,
            S0=S0, r=r, q=q, sigma=sigma,
            disc_curve=market.discount_curve(),
            mc_method=mc_method,
            num_paths=20_000,
            time_steps_per_year=12,
        )
        results[f"MC Arith Float {mc_method.name}"] = f"{price:.6f} ± {se:.6f}"

    # Print results
    print("\nArithmetic Asian Option Integration Test Results")
    print("=" * 70)
    print(f"{'Method':<35} | {'Price ± SE':>20}")
    print("-" * 70)
    for method, price in results.items():
        print(f"{method:<35} | {price:>20}")
    print("=" * 70)


# ================================================================
if __name__ == "__main__":
    test_geometric_asian()
    test_arithmetic_asian()
