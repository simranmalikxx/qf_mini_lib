# test_vanilla_integration.py

import numpy as np
from datetime import date

from market.market_data import Market
from market.curves.flat_discount import FlatCurve
from market.vol.flat_vol import FlatVolSurface
from utils.enum import PDEScheme, MonteCarloMethod
from utils.date import Date

from pricing.equity.vanila import (
    price_vanilla_fd,
    price_vanilla_analytical,
    price_vanilla_mc
)

# ------------------------------
# 1. Create dummy market objects
# ------------------------------

valuation_date = Date(2025, 8, 14)
expiry_date = Date(2026, 8, 14)
strike = 100.0
spot = 100.0

# Flat discount curve (5% annual continuously compounded)
discount_curve = FlatCurve(valuation_date, 0.05)

# Flat vol surface (20% implied vol)
vol_surface = FlatVolSurface(0.20)

# Combine into Market
market_data = Market(
    as_of=valuation_date,
    spot=spot,
    discount_curve=discount_curve,
    vol_surface=vol_surface
)

# ------------------------------
# 2. Analytical Pricing
# ------------------------------

print("--- Analytical Pricing ---")
call_ana = price_vanilla_analytical(market_data, valuation_date, expiry_date, strike, "call")
put_ana = price_vanilla_analytical(market_data, valuation_date, expiry_date, strike, "put")
print(f"Call Analytical: {call_ana:.6f}")
print(f"Put Analytical:  {put_ana:.6f}\n")

# ------------------------------
# 3. Monte Carlo Pricing
# ------------------------------

print("--- Monte Carlo Pricing ---")
for mc_method in MonteCarloMethod:
    call_mc = price_vanilla_mc(market_data, valuation_date, expiry_date, strike, "call", mc_method)
    put_mc = price_vanilla_mc(market_data, valuation_date, expiry_date, strike, "put", mc_method)
    print(f"{mc_method.name:15} -> Call: {call_mc:.6f}, Put: {put_mc:.6f}")
print()

# ------------------------------
# 4. PDE Pricing
# ------------------------------

print("--- PDE Pricing ---")
for scheme in PDEScheme:
    call_fd = price_vanilla_fd(market_data, valuation_date, expiry_date, strike, "call", scheme)
    put_fd = price_vanilla_fd(market_data, valuation_date, expiry_date, strike, "put", scheme)
    print(f"{scheme.name:15} -> Call: {call_fd:.6f}, Put: {put_fd:.6f}")
