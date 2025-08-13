# -*- coding: utf-8 -*-
"""
Flat volatility surface: returns constant vol for all maturities/strikes.
"""

from utils.error import FinError
from utils.helper import to_decimal


class FlatVolSurface:
    def __init__(self, vol):
        v = to_decimal(vol)
        if v <= 0.0:
            raise FinError(f"Volatility must be positive. Got: {vol}")
        self._vol = v

    def vol_TK(self, T: float, K: float = None) -> float:
        """
        Return vol for time to maturity T (years) and optional strike K.
        """
        return self._vol

    def vol(self) -> float:
        return self._vol

    def __repr__(self):
        return f"FlatVolSurface(vol={self._vol:.4f})"
