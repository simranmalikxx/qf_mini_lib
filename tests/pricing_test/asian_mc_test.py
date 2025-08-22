# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 06:51:46 2025

@author: Simran
"""
"""
Pytest regression tests for Asian option pricing (geometric & arithmetic).
"""

import pytest
from market.market_data import Market
from market.curves.flat_discount import FlatCurve
from market.vol.flat_vol import FlatVolSurface
from utils.date import Date
from core.asian import AsianOption
from pricing.equity.asian import AnalyticalAsianPricer, MonteCarloAsianPricer
from utils.enum import (
    OptionType, ExerciseType, ValuationType,
    AsianStrikeType, AsianAveragingType,
    MonteCarloMethod, AsianAnalyticalMethod
)


# -------------------
# Fixtures
# -------------------

@pytest.fixture
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


# -------------------
# Tests
# -------------------

def test_geometric_fixed_strike_mc_vs_analytical(setup_market):
    """Geometric fixed-strike Asian: MC should converge to analytical."""
    S0, r, q, sigma, valuation, T, market = setup_market
    n_obs = 12
    strike = 100.0

    # Analytical
    geo_fixed = AsianOption(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.EUROPEAN,
        strike_type=AsianStrikeType.FIXED,
        averaging_type=AsianAveragingType.GEOMETRIC,
        strike=strike,
        maturity=T,
        n_avg_points=n_obs,
        valuation_type=ValuationType.ANALYTICAL,
        analytical_method=AsianAnalyticalMethod.GEOMETRIC
    )
    analytical = AnalyticalAsianPricer._geometric_closed_form(
        geo_fixed, S0, r, q, sigma, market.discount_curve()
    )

    # Monte Carlo (standard)
    geo_mc = AsianOption(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.EUROPEAN,
        strike_type=AsianStrikeType.FIXED,
        averaging_type=AsianAveragingType.GEOMETRIC,
        strike=strike,
        maturity=T,
        n_avg_points=n_obs,
        valuation_type=ValuationType.MONTE_CARLO,
        mc_method=MonteCarloMethod.STANDARD,
    )

    mc_price = MonteCarloAsianPricer.price(
        option=geo_mc,
        S0=S0,
        r=r,
        q=q,
        sigma=sigma,
        disc_curve=market.discount_curve(),
        mc_method=MonteCarloMethod.STANDARD,
        num_paths=50_000,
        seed=123,
        time_steps_per_year=12,
    )

    # Assert close (within 2 stdev typical Monte Carlo noise)
    assert abs(mc_price - analytical) < 0.1


def test_arithmetic_control_variate_converges(setup_market):
    """Arithmetic fixed-strike Asian with control variate should be stable."""
    S0, r, q, sigma, valuation, T, market = setup_market
    n_obs = 12
    strike = 100.0

    arith_cv = AsianOption(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.EUROPEAN,
        strike_type=AsianStrikeType.FIXED,
        averaging_type=AsianAveragingType.ARITHMETIC,
        strike=strike,
        maturity=T,
        n_avg_points=n_obs,
        valuation_type=ValuationType.MONTE_CARLO,
        mc_method=MonteCarloMethod.CONTROL_VARIATE,
    )

    price = MonteCarloAsianPricer.price(
        option=arith_cv,
        S0=S0,
        r=r,
        q=q,
        sigma=sigma,
        disc_curve=market.discount_curve(),
        mc_method=MonteCarloMethod.CONTROL_VARIATE,
        num_paths=50_000,
        seed=123,
        time_steps_per_year=12,
    )

    # The CV estimate should be close to the geometric closed form (~6.39)
    assert 6.2 < price < 6.5


def test_floating_strike_prices_positive(setup_market):
    """Floating strike Asian should produce positive, finite prices."""
    S0, r, q, sigma, valuation, T, market = setup_market
    n_obs = 12

    for avg_type in [AsianAveragingType.GEOMETRIC, AsianAveragingType.ARITHMETIC]:
        float_asian = AsianOption(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike_type=AsianStrikeType.FLOATING,
            averaging_type=avg_type,
            strike=0.0,
            maturity=T,
            n_avg_points=n_obs,
            valuation_type=ValuationType.MONTE_CARLO,
            mc_method=MonteCarloMethod.STANDARD,
        )

        price = MonteCarloAsianPricer.price(
            option=float_asian,
            S0=S0,
            r=r,
            q=q,
            sigma=sigma,
            disc_curve=market.discount_curve(),
            mc_method=MonteCarloMethod.STANDARD,
            num_paths=20_000,
            seed=123,
            time_steps_per_year=12,
        )

        assert price > 0.0
        assert price < S0  # should never exceed spot
