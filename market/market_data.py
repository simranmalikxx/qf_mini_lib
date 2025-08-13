# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 14:38:26 2025

@author: Simran
"""

from utils.error import FinError
from utils.date import Date
from market.curves.base_curve import BaseCurve
from market.vol.flat_vol import FlatVolSurface
from market.curves.flat_discount import FlatCurve
from typing import Optional


class Market:
    def __init__(
        self,
        as_of: Date,
        spot: float,
        discount_curve: BaseCurve,
        vol_surface: Optional[object] = None,
        dividend_yield_curve: Optional[BaseCurve] = None
    ):
        if not isinstance(as_of, Date):
            raise FinError("as_of must be a Date instance")
        if spot <= 0.0:
            raise FinError(f"Spot must be positive. Got: {spot}")
        if not isinstance(discount_curve, BaseCurve):
            raise FinError("discount_curve must be a BaseCurve instance")

        self.as_of = as_of
        self._spot = spot
        self._discount_curve = discount_curve
        self._vol_surface = vol_surface
        self._dividend_yield_curve = dividend_yield_curve

    # ===== API =====
    def spot(self) -> float:
        return self._spot

    def discount_curve(self) -> BaseCurve:
        return self._discount_curve

    def vol_surface(self):
        if self._vol_surface is None:
            raise FinError("Volatility surface not set in Market")
        return self._vol_surface

    def dividend_yield_curve(self) -> Optional[BaseCurve]:
        return self._dividend_yield_curve

    @classmethod
    def from_manual(cls, as_of: Date, spot: float, rate: float, vol: Optional[float] = None):
        """
        Creates a Market with flat rate and optional flat vol.
        Continuous compounding for rates.
        """
        discount_curve = FlatCurve(as_of, rate)

        vol_surface = None
        if vol is not None:
            vol_surface = FlatVolSurface(vol)

        return cls(as_of, spot, discount_curve, vol_surface)
    
    def __repr__(self):
        return f"Market(as_of={self.as_of}, spot={self._spot})"