# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 05:00:34 2025

@author: Simran
"""

from utils.enum import (
    AsianStrikeType,
    AsianAveragingType,
    ValuationType,
    MonteCarloMethod,
    PDEScheme,
    AsianAnalyticalMethod )
from core.base_option import BaseOption
from utils.error import FinError
from utils.helper import validate_enum


class AsianOption(BaseOption):
    def __init__(
        self,
        strike_type: AsianStrikeType,
        averaging_type: AsianAveragingType,
        analytical_method: AsianAnalyticalMethod = None,
        mc_method: MonteCarloMethod = None,
        pde_scheme: PDEScheme = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        if not validate_enum(strike_type, AsianStrikeType):
            raise FinError(f"Invalid AsianStrikeType: {strike_type}. Valid types: {list(AsianStrikeType)}")
        if not validate_enum(averaging_type, AsianAveragingType):
            raise FinError(f"Invalid AsianAveragingType: {averaging_type}. Valid types: {list(AsianAveragingType)}")
            
        self.strike_type = strike_type
        self.averaging_type = averaging_type

        if self.valuation_type == ValuationType.ANALYTICAL:
            if analytical_method is None:
                raise FinError("Analytical method required for analytical valuation")
            if not validate_enum(analytical_method, AsianAnalyticalMethod):
                raise FinError(f"Invalid AnalyticalMethod: {analytical_method}. Valid types: {list(AsianAnalyticalMethod)}")
            self.analytical_method = analytical_method
            
        elif self.valuation_type == ValuationType.MONTE_CARLO:
            if mc_method is None:
                raise FinError("Monte Carlo method required for MC valuation")
            if not validate_enum(mc_method, MonteCarloMethod):
                raise FinError(f"Invalid MonteCarloMethod: {mc_method}. Valid types: {list(MonteCarloMethod)}")
            self.mc_method = mc_method
            
        elif self.valuation_type == ValuationType.FINITE_DIFFERENCE:
            if pde_scheme is None:
                raise FinError("PDE scheme required for finite difference valuation")
            if not validate_enum(pde_scheme, PDEScheme):
                raise FinError(f"Invalid PDEScheme: {pde_scheme}. Valid types: {list(PDEScheme)}")
            self.pde_scheme = pde_scheme

    def payoff(self, spot_price: float) -> float:
        super().payoff(spot_price)  # Validates spot_price > 0
        raise NotImplementedError("Concrete AsianOption classes must implement payoff()")

    def __deepcopy__(self, memo):
        result = super().__deepcopy__(memo)
        result.strike_type = self.strike_type
        result.averaging_type = self.averaging_type
        
        if hasattr(self, 'analytical_method'):
            result.analytical_method = self.analytical_method
        if hasattr(self, 'mc_method'):
            result.mc_method = self.mc_method
        if hasattr(self, 'pde_scheme'):
            result.pde_scheme = self.pde_scheme
            
        return result

    def __str__(self):
        base_str = super().__str__()
        method_str = ""
        if hasattr(self, 'analytical_method'):
            method_str = f" | Method: {self.analytical_method.name}"
        elif hasattr(self, 'mc_method'):
            method_str = f" | Method: {self.mc_method.name}"
        elif hasattr(self, 'pde_scheme'):
            method_str = f" | Method: {self.pde_scheme.name}"
            
        return (
            f"{base_str} | StrikeType: {self.strike_type.name}, "
            f"AveragingType: {self.averaging_type.name}{method_str}"
        )