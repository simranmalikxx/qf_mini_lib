# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 11:28:57 2025

@author: Simran
"""
from datetime import datetime, timedelta 

class Date:
    def _init_(self, year: int, month: int, day: int):
        self.dt = datetime(day, month, year)
        
    def __lt__(self, other):
        return self._dt< other._dt 
    
    def __le__(self, other):
        return self._dt <= other._dt
    
    def __eq__(self, other):
        return self._dt == other._dt

    def __gt__(self, other):
        return self._dt > other._dt

    def __ge__(self, other):
        return self._dt >= other._dt
    
    def add_days(self, n: int):
        return Date.from_datetime(self._dt + timedelta(days=n))
    
    def days_between(self, other):
        return (other._dt - self._dt).days
    
    @classmethod
    def from_datetime(cls, dt_obj):
        return cls(dt_obj.year, dt_obj.month, dt_obj.day)

    def __str__(self):
        return self._dt.strftime("%Y-%m-%d")