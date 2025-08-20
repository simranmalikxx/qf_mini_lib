# -*- coding: utf-8 -*-
"""
Fixed to align with current Market, FlatCurve, and FlatVolSurface APIs.
"""

from market.market_data import Market
from utils.date import Date
from utils.error import FinError


def extract_pricing_inputs(market: Market, valuation_date: Date, maturity_date: Date, strike: float):
    if not isinstance(market, Market):
        raise FinError("market must be a Market instance")
    if not isinstance(valuation_date, Date) or not isinstance(maturity_date, Date):
        raise FinError("valuation_date and maturity_date must be utils.date.Date instances")

    S0 = market.spot()

    # Time to maturity in years (ACT/365F)
    T = valuation_date.years_between(maturity_date)
    if T <= 0:
        raise FinError("Maturity must be after valuation date")

    # Volatility (flat vol surface)
    vol_surface = market.vol_surface()
    if hasattr(vol_surface, "vol_TK"):
        sigma = vol_surface.vol_TK(T, strike)
    else:
        raise FinError("Vol surface does not implement vol_TK(T, K)")

    return S0, sigma, T
