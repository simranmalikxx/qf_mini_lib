# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 05:29:30 2025

@author: Simran
"""

from utils.enum import (
    BarrierType,
    RebateType,
    ValuationType,
    BarrierAnalyticalMethod,
    MonteCarloMethod,
    PDEScheme
)
from core.base_option import BaseOption
from utils.error import FinError
from utils.helper import validate_enum


class BarrierOption(BaseOption):
    def __init__(
        self,
        barrier_type: BarrierType,
        barrier_level: float,
        rebate: float = 0.0,
        rebate_type: RebateType = RebateType.FIXED,
        analytical_method: BarrierAnalyticalMethod = None,
        mc_method: MonteCarloMethod = None,
        pde_scheme: PDEScheme = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        if not validate_enum(barrier_type, BarrierType):
            raise FinError(f"Invalid BarrierType: {barrier_type}. Valid types: {list(BarrierType)}")
        if barrier_level <= 0:
            raise FinError(f"Barrier level must be positive. Got: {barrier_level}")
        if not validate_enum(rebate_type, RebateType):
            raise FinError(f"Invalid RebateType: {rebate_type}. Valid types: {list(RebateType)}")
        if rebate < 0:
            raise FinError(f"Rebate must be non-negative. Got: {rebate}")

        self.barrier_type = barrier_type
        self.barrier_level = barrier_level
        self.rebate = rebate
        self.rebate_type = rebate_type

        if self.valuation_type == ValuationType.ANALYTICAL:
            if analytical_method is None:
                raise FinError("Analytical method required for analytical valuation")
            if not validate_enum(analytical_method, BarrierAnalyticalMethod):
                raise FinError(f"Invalid BarrierAnalyticalMethod: {analytical_method}. Valid types: {list(BarrierAnalyticalMethod)}")
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
        raise NotImplementedError("Concrete BarrierOption classes must implement payoff()")

    def __deepcopy__(self, memo):
        result = super().__deepcopy__(memo)
        result.barrier_type = self.barrier_type
        result.barrier_level = self.barrier_level
        result.rebate = self.rebate
        result.rebate_type = self.rebate_type
        
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
            f"{base_str} | BarrierType: {self.barrier_type.name}, "
            f"Level: {self.barrier_level:.2f}, Rebate: {self.rebate:.2f} "
            f"({self.rebate_type.name}){method_str}"
        )