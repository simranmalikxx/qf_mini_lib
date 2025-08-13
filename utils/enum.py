# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 11:16:23 2025

@author: Simran
"""

from enum import Enum

class OptionType(Enum):
    CALL = 1
    PUT = 2
    
class ExerciseType(Enum):
    AMERICAN = 1
    EUROPEAN = 2

class ValuationType(Enum):
    ANALYTICAL = 1
    MONTE_CARLO = 2
    FINITE_DIFFERENCE = 3
    
class MonteCarloMethod(Enum):
    STANDARD = 1
    ANTITHETIC = 2
    CONTROL_VARIATE = 3
    QUASI_RANDOM = 4

class PDEScheme(Enum):
    EXPLICIT = 1
    IMPLICIT = 2
    CRANK_NICOLSON = 3

    
# Asian Speciifc: 
class AsianStrikeType(Enum):
    FIXED = 1
    FLOATING = 2

class AsianAveragingType(Enum):
    ARITHMETIC = 1
    GEOMETRIC = 2
    
class AsianAnalyticalMethod(Enum):
    GEOMETRIC = 1
    TURNBULL_WAKEMAN = 2
    CURRAN = 3


# Barrier Specific: 
class BarrierType(Enum):
    UP_IN = 1
    UP_OUT = 2
    DOWN_IN = 3
    DOWN_OUT = 4

class BarrierAnalyticalMethod(Enum):
    REINHARDT = 1
    MERTON = 2
    BARRIERSHIFT = 3

class RebateType(Enum):
    FIXED = 1         
    PERCENTAGE = 2    