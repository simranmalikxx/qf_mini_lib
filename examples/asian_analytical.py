from market.market_data import Market
from market.curves.flat_discount import FlatCurve
from market.vol.flat_vol import FlatVolSurface
from utils.date import Date 

from core.asian import AsianOption
from pricing.equity.asian import AnalyticalAsianPricer
from pricing.equity.van1 import price_vanilla_analytical
from utils.enum import OptionType, ExerciseType, ValuationType, AsianStrikeType, AsianAveragingType


def run_test_asian_all():
    # --- Market setup ---
    S0 = 100.0
    r = 0.05      # risk-free
    q = 0.02      # dividend yield
    sigma = 0.25  # volatility
    T = Date(2026, 8, 20)       # maturity
    valuation = Date(2025, 8, 20)

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

    # --- Option setup base ---
    n_obs = 12
    strike = 100.0

    option_geo_fixed = AsianOption(
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

    option_geo_float = AsianOption(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.EUROPEAN,
        strike_type=AsianStrikeType.FLOATING,
        averaging_type=AsianAveragingType.GEOMETRIC,
        strike=strike,
        maturity=T,
        n_avg_points=n_obs,
        valuation_type=ValuationType.ANALYTICAL,
        analytical_method=1,
    )

    option_arith_fixed_TW = AsianOption(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.EUROPEAN,
        strike_type=AsianStrikeType.FIXED,
        averaging_type=AsianAveragingType.ARITHMETIC,
        strike=strike,
        maturity=T,
        n_avg_points=n_obs,
        valuation_type=ValuationType.ANALYTICAL,
        analytical_method=2,  # Turnbull-Wakeman
    )

    option_arith_fixed_Curran = AsianOption(
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

    # --- Prices ---
    geo_fixed_price = AnalyticalAsianPricer._geometric_closed_form(
        option=option_geo_fixed, S0=S0, r=r, q=q, sigma=sigma, disc_curve=disc_curve
    )

    geo_float_price = AnalyticalAsianPricer._geometric_closed_form(
        option=option_geo_float, S0=S0, r=r, q=q, sigma=sigma, disc_curve=disc_curve
    )

    arith_fixed_TW_price = AnalyticalAsianPricer._turnbull_wakeman(
        option=option_arith_fixed_TW, S0=S0, r=r, q=q, sigma=sigma, disc_curve=disc_curve
    )

    arith_fixed_Curran_price = AnalyticalAsianPricer._curran(
        option=option_arith_fixed_Curran, S0=S0, r=r, q=q, sigma=sigma, disc_curve=disc_curve
    )

    vanilla_price = price_vanilla_analytical(
        market_data=market,
        valuation_date=valuation,
        expiry_date=T,
        strike=strike,
        option_type="call",
    )

    # --- Print results ---
    print("=== Asian Option Comparison (Analytical) ===")
    print(f"Spot Price      : {S0}")
    print(f"Strike (fixed)  : {strike}")
    print(f"Maturity        : {T}")
    print(f"Risk-Free Rate  : {r}")
    print(f"Dividend Yield  : {q}")
    print(f"Volatility      : {sigma}")
    print(f"Num Obs Points  : {n_obs}")
    print("--------------------------------------------")
    print(f"Vanilla Option (BS)             : {vanilla_price:.6f}")
    print(f"Geometric Asian Fixed (exact)   : {geo_fixed_price:.6f}")
    print(f"Geometric Asian Floating (exact): {geo_float_price:.6f}")
    print(f"Arithmetic Asian Fixed (T–W)    : {arith_fixed_TW_price:.6f}")
    print(f"Arithmetic Asian Fixed (Curran) : {arith_fixed_Curran_price:.6f}")
    print("--------------------------------------------")
    print("Sanity Checks:")
    print(" - Geo fixed < Vanilla (variance reduction).")
    print(" - Geo floating < Geo fixed (less optionality).")
    print(" - Arithmetic fixed (T–W, Curran) ~ between Geo fixed and Vanilla.")

if __name__ == "__main__":
    run_test_asian_all()
