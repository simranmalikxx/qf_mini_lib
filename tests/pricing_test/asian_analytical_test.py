# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 09:45:55 2025

@author: Simran
"""

import math
import pytest

from market.market_data import Market
from market.curves.flat_discount import FlatCurve
from market.vol.flat_vol import FlatVolSurface
from utils.date import Date
from core.asian import AsianOption
from pricing.equity.asian import AnalyticalAsianPricer
from utils.enum import AsianAnalyticalMethod
from pricing.equity.van1 import price_vanilla_analytical
from utils.enum import OptionType, ExerciseType, ValuationType, AsianStrikeType, AsianAveragingType


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


def test_asian_analytical_methods():
    S0, r, q, sigma, valuation, T, market = setup_market()
    n_obs = 12
    strike = 100.0

    # Option definitions
    geo_fixed = AsianOption(
    option_type=OptionType.CALL,
    exercise_type=ExerciseType.EUROPEAN,
    strike_type=AsianStrikeType.FIXED,
    averaging_type=AsianAveragingType.GEOMETRIC,
    strike=strike,
    maturity=T,
    n_avg_points=n_obs,
    valuation_type=ValuationType.ANALYTICAL,
    analytical_method=AsianAnalyticalMethod.GEOMETRIC)
    
    geo_float = AsianOption(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.EUROPEAN,
        strike_type=AsianStrikeType.FLOATING,
        averaging_type=AsianAveragingType.GEOMETRIC,
        strike=strike,
        maturity=T,
        n_avg_points=n_obs,
        valuation_type=ValuationType.ANALYTICAL,
        analytical_method=AsianAnalyticalMethod.GEOMETRIC,  
    )

    arith_tw = AsianOption(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.EUROPEAN,
        strike_type=AsianStrikeType.FIXED,
        averaging_type=AsianAveragingType.ARITHMETIC,
        strike=strike,
        maturity=T,
        n_avg_points=n_obs,
        valuation_type=ValuationType.ANALYTICAL,
        analytical_method=2,  # Turnbull–Wakeman
    )

    arith_curran = AsianOption(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.EUROPEAN,
        strike_type=AsianStrikeType.FIXED,
        averaging_type=AsianAveragingType.ARITHMETIC,
        strike=strike,
        maturity=T,
        n_avg_points=n_obs,
        valuation_type=ValuationType.ANALYTICAL,
        analytical_method=3,  # Curran
    )

    # Prices
    vanilla = price_vanilla_analytical(market, valuation, T, strike, "call")
    geo_fixed_price = AnalyticalAsianPricer._geometric_closed_form(geo_fixed, S0, r, q, sigma, market.discount_curve())
    geo_float_price = AnalyticalAsianPricer._geometric_closed_form(geo_float, S0, r, q, sigma, market.discount_curve())
    tw_price = AnalyticalAsianPricer._turnbull_wakeman(arith_tw, S0, r, q, sigma, market.discount_curve())
    curran_price = AnalyticalAsianPricer._curran(arith_curran, S0, r, q, sigma, market.discount_curve())

    # Assertions (sanity hierarchy)
    assert geo_fixed_price < vanilla
    assert geo_float_price < geo_fixed_price
    assert geo_fixed_price < tw_price < vanilla
    assert geo_fixed_price <= curran_price <= vanilla
    assert math.isclose(curran_price, geo_fixed_price, rel_tol=0.05)
