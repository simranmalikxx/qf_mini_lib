# tests/test_vanilla_pricing.py

import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))



from utils.enum import PDEScheme, MonteCarloMethod
from utils.date import Date
from market.curves.flat_discount import FlatCurve
from market.vol.flat_vol import FlatVolSurface
from market.market_data import Market
from pricing.equity.van1 import price_vanilla_analytical, price_vanilla_mc, price_vanilla_fd

# ------------------------------
# Market Setup
# ------------------------------
valuation_date = Date(2025, 8, 14)
expiry_date = Date(2026, 8, 14)
strike = 100.0
spot = 100.0

discount_curve = FlatCurve(Date(2025, 8, 14), 0.05)
vol_surface = FlatVolSurface(0.2)  # 20% vol

market_data = Market(
    Date(2025, 8, 14),
    spot=spot,
    discount_curve=discount_curve,
    vol_surface=vol_surface
)

# Tolerance for Monte Carlo & PDE vs Analytical
PRICE_TOL = 0.05  # 5 cents tolerance

# ------------------------------
# Analytical Pricing Unit Test
# ------------------------------
def test_analytical_prices():
    call_price = price_vanilla_analytical(market_data, valuation_date, expiry_date, strike, "call")
    put_price = price_vanilla_analytical(market_data, valuation_date, expiry_date, strike, "put")
    
    # Basic checks
    assert call_price > 0
    assert put_price > 0
    
    # Put-call parity using curve-consistent discounting
    T = valuation_date.years_between(expiry_date)
    discount_curve = market_data.discount_curve()
    df = discount_curve.discount_factor_T(T)
    discounted_strike = strike * df
    
    parity_diff = abs((call_price - put_price) - (market_data.spot() - discounted_strike))
    
    print(f"Call Price: {call_price:.6f}")
    print(f"Put Price: {put_price:.6f}")
    print(f"Spot: {market_data.spot():.6f}")
    print(f"Discounted Strike: {discounted_strike:.6f} (DF={df:.6f})")
    print(f"Parity Difference: {parity_diff:.12f}")
    
    assert parity_diff < 1e-8

# ------------------------------
# Monte Carlo Pricing Unit Test
# ------------------------------
@pytest.mark.parametrize("mc_method", list(MonteCarloMethod))
def test_monte_carlo_prices(mc_method):
    num_paths = 2**18  # Increased to 262,144 paths for better convergence
    
    # Different tolerances per method
    price_tols = {
        MonteCarloMethod.STANDARD: 0.05,
        MonteCarloMethod.ANTITHETIC: 0.03,
        MonteCarloMethod.QUASI_RANDOM: 0.02,
        MonteCarloMethod.CONTROL_VARIATE: 0.001  # Control variate should be most accurate
    }
    
    call_mc = price_vanilla_mc(market_data, valuation_date, expiry_date, strike, "call", mc_method, num_paths)
    put_mc = price_vanilla_mc(market_data, valuation_date, expiry_date, strike, "put", mc_method, num_paths)
    
    call_ana = price_vanilla_analytical(market_data, valuation_date, expiry_date, strike, "call")
    put_ana = price_vanilla_analytical(market_data, valuation_date, expiry_date, strike, "put")
    
    assert abs(call_mc - call_ana) < price_tols[mc_method]
    assert abs(put_mc - put_ana) < price_tols[mc_method]

# ------------------------------
# PDE Pricing Unit Test
# ------------------------------
@pytest.mark.parametrize("scheme", list(PDEScheme))
def test_pde_prices(scheme):
    call_fd = price_vanilla_fd(market_data, valuation_date, expiry_date, strike, "call", scheme)
    put_fd = price_vanilla_fd(market_data, valuation_date, expiry_date, strike, "put", scheme)

    call_ana = price_vanilla_analytical(market_data, valuation_date, expiry_date, strike, "call")
    put_ana = price_vanilla_analytical(market_data, valuation_date, expiry_date, strike, "put")

    # PDE prices should be reasonably close to Analytical
    assert abs(call_fd - call_ana) < PRICE_TOL
    assert abs(put_fd - put_ana) < PRICE_TOL
