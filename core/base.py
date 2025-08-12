# -*- coding: utf-8 -*-
"""
Created on Tue Aug 12 04:50:43 2025

@author: Simran
"""
from abc import ABC, abstractmethod
from copy import deepcopy
from utils.enum import OptionType, ExerciseType, ValuationType
from utils.date import Date
from utils.error import FinError
from utils.helper import validate_enum

class BaseOption(ABC):
    def __init__(
            self, 
            option_type: OptionType,
            exercise_type: ExerciseType,
            strike: float,
            maturity: Date,
            valuation_type: ValuationType):
        
        if not validate_enum(option_type, OptionType):
            raise FinError(f"Invalid Option Type: {option_type}. Valid types: {list(OptionType)}")
            
        if not validate_enum(exercise_type, ExerciseType):
            raise FinError(f"Invalid ExerciseType: {exercise_type}. Valid types: {list(ExerciseType)}")
            
        if not validate_enum(valuation_type, ValuationType):
            raise FinError(f"Invalid ValuationType: {valuation_type}. Valid types: {list(ValuationType)}")
        
        if strike <= 0:
            raise FinError(f"Strike price must be positive. Got: {strike}")

        if not isinstance(maturity, Date):
            raise FinError(f"Maturity must be a Date instance. Got: {type(maturity)}")
            
        if maturity < Date.today():
            raise FinError(f"Maturity {maturity} must be in future. Current date: {Date.today()}")

        self.option_type = option_type
        self.exercise_type = exercise_type
        self.strike = strike
        self.maturity = maturity
        self.valuation_type = valuation_type
        
    @abstractmethod
    def payoff(self, spot_price: float) -> float:
        if spot_price <= 0:
            raise FinError(f"Spot price must be positive. Got: {spot_price}")
        pass

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        result.option_type = self.option_type
        result.exercise_type = self.exercise_type
        result.strike = self.strike
        result.maturity = deepcopy(self.maturity, memo)
        result.valuation_type = self.valuation_type
        
        return result

    def __str__(self):
        return ( 
            f"{self.option_type.name} {self.exercise_type.name} Option | "
            f"Strike: {self.strike:.2f}, Maturity: {self.maturity}, "
            f"Valuation: {self.valuation_type.name}" )