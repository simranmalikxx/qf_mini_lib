# -*- coding: utf-8 -*-
"""
Asian Option Pricing Test
Tests all pricing methods and prints comparison table
"""

import math
from datetime import datetime
from utils.date import Date
from utils.enum import (
    OptionType, ExerciseType, ValuationType, 
    AsianStrikeType, AsianAveragingType,
    MonteCarloMethod, PDEScheme, AsianAnalyticalMethod
)
from core.base import BaseOption
from core.asian import AsianOption
from market.market_data import Market
from market.curves.flat_discount import FlatCurve
from market.vol.flat_vol import FlatVolSurface
from pricing.equity.asian import AsianPricer

def create_test_market():
    """Create a test market with flat rates and volatility"""
    as_of = Date(2025, 8, 20)
    spot = 100.0
    risk_free_rate = 0.05  # 5%
    dividend_yield = 0.02  # 2%
    volatility = 0.2       # 20%
    
    # Create discount curve
    discount_curve = FlatCurve(as_of, risk_free_rate)
    
    # Create dividend curve (using same structure as discount curve)
    dividend_curve = FlatCurve(as_of, dividend_yield)
    
    # Create flat volatility surface
    vol_surface = FlatVolSurface(volatility)
    
    # Create market
    market = Market(
        as_of=as_of,
        spot=spot,
        discount_curve=discount_curve,
        vol_surface=vol_surface,
        dividend_yield_curve=dividend_curve
    )
    
    return market

def create_test_options():
    """Create various Asian option configurations for testing"""
    maturity_date = Date(2026, 8, 20)  # 1 year maturity
    strike = 100.0
    n_obs = 12
    
    options = []
    
    # Fixed strike arithmetic options
    options.append({
        'name': 'Fixed-Arithmetic-Call',
        'option': AsianOption(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=strike,
            maturity=maturity_date,
            valuation_type=ValuationType.ANALYTICAL,
            strike_type=AsianStrikeType.FIXED,
            averaging_type=AsianAveragingType.ARITHMETIC,
            n_avg_points=n_obs,
            analytical_method=AsianAnalyticalMethod.TURNBULL_WAKEMAN
        )
    })
    
    options.append({
        'name': 'Fixed-Arithmetic-Put',
        'option': AsianOption(
            option_type=OptionType.PUT,
            exercise_type=ExerciseType.EUROPEAN,
            strike=strike,
            maturity=maturity_date,
            valuation_type=ValuationType.ANALYTICAL,
            strike_type=AsianStrikeType.FIXED,
            averaging_type=AsianAveragingType.ARITHMETIC,
            n_avg_points=n_obs,
            analytical_method=AsianAnalyticalMethod.TURNBULL_WAKEMAN
        )
    })
    
    # Fixed strike geometric options
    options.append({
        'name': 'Fixed-Geometric-Call',
        'option': AsianOption(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=strike,
            maturity=maturity_date,
            valuation_type=ValuationType.ANALYTICAL,
            strike_type=AsianStrikeType.FIXED,
            averaging_type=AsianAveragingType.GEOMETRIC,
            n_avg_points=n_obs,
            analytical_method=AsianAnalyticalMethod.GEOMETRIC
        )
    })
    
    options.append({
        'name': 'Fixed-Geometric-Put',
        'option': AsianOption(
            option_type=OptionType.PUT,
            exercise_type=ExerciseType.EUROPEAN,
            strike=strike,
            maturity=maturity_date,
            valuation_type=ValuationType.ANALYTICAL,
            strike_type=AsianStrikeType.FIXED,
            averaging_type=AsianAveragingType.GEOMETRIC,
            n_avg_points=n_obs,
            analytical_method=AsianAnalyticalMethod.GEOMETRIC
        )
    })
    
    # Floating strike geometric options (for PDE testing)
    # Use dummy positive strike (1.0) to satisfy BaseOption validation
    # The actual AsianPricer should ignore this for floating strike options
    options.append({
        'name': 'Floating-Geometric-Call',
        'option': AsianOption(
            option_type=OptionType.CALL,
            exercise_type=ExerciseType.EUROPEAN,
            strike=1.0,  # DUMMY POSITIVE VALUE (not used in floating strike pricing)
            maturity=maturity_date,
            valuation_type=ValuationType.FINITE_DIFFERENCE,
            strike_type=AsianStrikeType.FLOATING,
            averaging_type=AsianAveragingType.GEOMETRIC,
            n_avg_points=n_obs,
            pde_scheme=PDEScheme.CRANK_NICOLSON
        )
    })
    
    return options

def test_pricing_methods():
    """Test all pricing methods and print results"""
    print("Asian Option Pricing Test")
    print("=" * 80)
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create market and options
    market = create_test_market()
    test_options = create_test_options()
    pricer = AsianPricer(market)
    
    results = []
    
    for test_case in test_options:
        option = test_case['option']
        name = test_case['name']
        
        try:
            # Price using the specified valuation method - PASS THE VALUATION TYPE
            price = pricer.price(option, option.valuation_type)
            results.append({
                'Name': name,
                'Price': price,
                'Status': 'SUCCESS',
                'Error': None
            })
            
        except Exception as e:
            results.append({
                'Name': name,
                'Price': None,
                'Status': 'ERROR',
                'Error': str(e)
            })
    
    # Print results table
    print(f"{'Option Type':<25} {'Price':<15} {'Status':<10} {'Error':<20}")
    print("-" * 80)
    
    for result in results:
        price_str = f"{result['Price']:.4f}" if result['Price'] is not None else "N/A"
        error_str = result['Error'] or "None"
        print(f"{result['Name']:<25} {price_str:<15} {result['Status']:<10} {error_str:<20}")
    
    print()
    print("Additional Monte Carlo Tests:")
    print("-" * 40)
    
    # Test Monte Carlo methods on a fixed arithmetic call
    mc_option = AsianOption(
        option_type=OptionType.CALL,
        exercise_type=ExerciseType.EUROPEAN,
        strike=100.0,
        maturity=Date(2026, 8, 20),
        valuation_type=ValuationType.MONTE_CARLO,
        strike_type=AsianStrikeType.FIXED,
        averaging_type=AsianAveragingType.ARITHMETIC,
        n_avg_points=12,
        mc_method=MonteCarloMethod.STANDARD
    )
    
    mc_methods = [
        MonteCarloMethod.STANDARD,
        MonteCarloMethod.ANTITHETIC,
        MonteCarloMethod.CONTROL_VARIATE
    ]
    
    for method in mc_methods:
        mc_option.mc_method = method
        try:
            # PASS THE VALUATION TYPE HERE TOO
            price = pricer.price(mc_option, ValuationType.MONTE_CARLO, num_paths=10000, seed=42)
            print(f"MC {method.name:<15}: {price:.4f}")
        except Exception as e:
            print(f"MC {method.name:<15}: ERROR - {str(e)}")

if __name__ == "__main__":
    test_pricing_methods()