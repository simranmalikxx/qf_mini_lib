# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 11:16:23 2025

@author: Simran
"""

from enum import Enum, auto 

class OptionType(Enum):
    CALL = auto()
    PUT = auto()
    
class ExerciseType(Enum):
    AMERICAN = auto()
    EUROPEAN = auto()

class ValuationType(Enum):
    ANALYTICAL = auto()
    MONTE_CARLO = auto()
    FINITE_DIFFERENCE = auto()