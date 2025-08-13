# -*- coding: utf-8 -*-
"""
Created on Wed Aug 13 05:58:16 2025

@author: Simran
"""

from abc import ABC, abstractmethod
from utils.error import FinError
from utils.date import Date
import math 
from typing import Optional

class BaseCurve(ABC):
        
    def __init__(self, as_of: Date):
        if not isinstance(as_of, Date):
            raise FinError("as_of must be a Date instance")
        self.as_of = as_of
    
    def time_to(self, date:Date):
        if not isinstance(date, Date):
            raise FinError("date must be a Date instance")
        t = self.as_of.years_between(date)
        return max(0.0, t)
    
    @abstractmethod
    def discount_factor_T(self, T:Date):
        """
        Return DF (from as_of to as_of+T) for T in years(continuous compounding).
        """
        
        ...
        
    def discount_factor(self, date:Date):
        """Return DF(from as_of to date) using Date-based time."""
        T = self.time_to(date) #converting to years
        return self.discount_factor_T(T) #Uusing time based discounting
    
    def zero_rate_T(self, T: float) -> float:
       """
       Implied continuously-compounded zero rate from DF(T).
       By convention, r(0) = 0.
       """
       if T <= 0.0:
           return 0.0
       df = self.discount_factor_T(T)
       if df <= 0.0:
           raise FinError("Discount factor must be positive")
       return -math.log(df) / T
   
    def zero_rate(self, date: Date) -> float:
        """Date-based implied zero rate (continuous)."""
        T = self.time_to(date)
        return self.zero_rate_T(T)
    
    def __repr__(self):
        return f"{self.__class__.__name__}(as_of={self.as_of})"
    
    
    
        
    
    
        