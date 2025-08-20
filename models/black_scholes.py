# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 05:17:31 2025

@author: Simran
"""

"""
Black–Scholes (constant vol) model under risk-neutral measure.

SDE: dS_t = S_t * (r(t) - q(t)) dt + S_t * sigma dW_t

Features:
- Reads rates and dividend yields from Market via zero_rate_T().
- If market.vol_surface is present, reads vol via vol() or vol_TK() as fallback.
- simulate_paths_T supports antithetic sampling and seed control.
- Future-proof hooks (kwargs) for variance reduction flags.
"""

from typing import Optional
import numpy as np
import math

from models.base import BaseModel
from utils.error import FinError


class BlackScholesModel(BaseModel):
    def __init__(self, market, sigma: Optional[float] = None):

        super().__init__(market)

        # Resolve sigma
        if sigma is None:
            volsurf = getattr(self.market, "_vol_surface", None) or getattr(self.market, "vol_surface", None)
            if volsurf is None:
                raise FinError("sigma not provided and market.vol_surface missing")
            # Common API: prefer vol() then vol_TK
            if hasattr(volsurf, "vol"):
                sigma = volsurf.vol()
            elif hasattr(volsurf, "vol_TK"):
                sigma = volsurf.vol_TK(1.0, None)
            else:
                raise FinError("market vol surface has no 'vol' or 'vol_TK' method")

        if sigma <= 0.0:
            raise FinError("sigma must be positive")
        self.sigma = float(sigma)

    

    def _r_T(self, T: float) -> float:
        return self.market.discount_curve().zero_rate_T(T)

    def _q_T(self, T: float) -> float:
        q_curve = self.market.dividend_yield_curve()
        if q_curve is None:
            return 0.0
        return q_curve.zero_rate_T(T)


    def forward_T(self, T: float) -> float:
        if T < 0.0:
            raise FinError("T must be non-negative")
        S0 = self.market.spot()
        r = self._r_T(T)
        q = self._q_T(T)
        return S0 * math.exp((r - q) * T)

    def variance_T(self, T: float) -> float:
        if T < 0.0:
            raise FinError("T must be non-negative")
        return (self.sigma ** 2) * T


    def simulate_paths_T(
        self,
        T: float,
        n_paths: int,
        n_steps: int = 1,
        seed: Optional[int] = None,
        antithetic: bool = False,
        **kwargs
    ) -> np.ndarray:
        """
        Simulate paths with exact lognormal increments per step:

            S_{t+dt} = S_t * exp( (r - q - 0.5*sigma^2) * dt + sigma * sqrt(dt) * Z )

        """
        if T < 0.0:
            raise FinError("T must be non negative")
        if n_paths <= 0 or n_steps <= 0:
            raise FinError("n_paths and n_steps must be positive integers")

        rng = np.random.default_rng(seed)
        dt = T / n_steps if n_steps > 0 else 0.0
        S0 = self.market.spot()

        # Use horizon zero rates as simplification (if you need time-dependent r/q, sample at each grid step)
        r = self._r_T(T)
        q = self._q_T(T)
        mu = (r - q)
        sig = self.sigma

        # Pre-allocate
        S = np.empty((n_paths, n_steps + 1), dtype=float)
        S[:, 0] = S0

        # Antithetic handling
        if antithetic:
            if n_paths % 2 != 0:
                raise FinError("antithetic requires n_paths to be even")
            half = n_paths // 2
        else:
            half = n_paths

        for step in range(1, n_steps + 1):
            Z = rng.standard_normal(size=half)
            if antithetic:
                Z = np.concatenate([Z, -Z])
            increment = (mu - 0.5 * sig * sig) * dt + sig * math.sqrt(dt) * Z
            S[:, step] = S[:, step - 1] * np.exp(increment)

        return S


    def parameters(self) -> dict:
        return {"model": "BlackScholes", "sigma": self.sigma}
