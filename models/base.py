# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 04:08:35 2025

@author: Simran
"""


from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

from utils.error import FinError
from utils.date import Date


class BaseModel(ABC):
    
    def __init__(self, market):
        # avoid hard import to keep modules decoupled
        if not hasattr(market, "as_of") or not hasattr(market, "discount_curve"):
            raise FinError("market must be a Market-like object with as_of and discount_curve()")
        self.market = market


    @property
    def as_of(self) -> Date:
        return self.market.as_of

    def _time_to(self, date: Date) -> float:
        if not isinstance(date, Date):
            raise FinError("date must be a Date instance")
        return self.market.discount_curve().time_to(date)

    def forward(self, date: Date) -> float:
        """Date-based wrapper for forward_T."""
        return self.forward_T(self._time_to(date))

    def variance(self, date: Date) -> float:
        """Date-based wrapper for variance_T."""
        return self.variance_T(self._time_to(date))

    def simulate_paths(
        self,
        date: Date,
        n_paths: int,
        n_steps: int = 1,
        seed: Optional[int] = None,
        antithetic: bool = False,
        **kwargs
    ) -> np.ndarray:
        T = self._time_to(date)
        return self.simulate_paths_T(T, n_paths, n_steps, seed, antithetic, **kwargs)

    
    @abstractmethod
    def forward_T(self, T: float) -> float:
        """
        Risk-neutral forward E[S_T] for horizon T (years).
        """
        ...

    @abstractmethod
    def variance_T(self, T: float) -> float:
        """
        Model variance over [0,T]. For BS this is sigma^2 * T.
        """
        ...

    @abstractmethod
    def simulate_paths_T(
        self,
        T: float,
        n_paths: int,
        n_steps: int = 1,
        seed: Optional[int] = None,
        antithetic: bool = False,
        **kwargs
    ) -> np.ndarray:
        
        ...
        
    def calibrate(self, market_observables: dict):
        """Optional calibration hook (no-op by default)."""
        return None

    def parameters(self) -> dict:
        """Return model parameters for logging/debugging."""
        return {}

