# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 06:26:39 2025

@author: Simran
"""

import math
from market.curves.base_curve import BaseCurve
from utils.error import FinError
from utils.date import Date

class FlatCurve(BaseCurve):
    """
    A flat continuously-compounded zero-rate curve.
    """

    def __init__(self, as_of: Date, rate: float):
        super().__init__(as_of)
        if rate < 0.0:
            raise FinError("FlatCurve rate must be non-negative")
        self.rate = rate

    def discount_factor_T(self, T: float) -> float:
        """
        Continuous compounding: DF = exp(-r * T)
        """
        return math.exp(-self.rate * T)

    def __repr__(self):
        return f"FlatCurve(as_of={self.as_of}, rate={self.rate:.4f})"