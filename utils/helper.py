# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 12:27:43 2025

@author: Simran
"""

from datetime import datetime

def is_percentage(value):
    
    return isinstance(value, (int, float)) and 0.0 <= value <= 100.0

def to_decimal(percentage):
    
    if not is_percentage(percentage):
        raise ValueError("Value must be between 0 and 100")
    return percentage / 100.0

def format_currency(value, decimals=2):
    
    return f"${value:.{decimals}f}"

def is_business_day(date_obj):
    
    if not isinstance(date_obj, datetime):
        raise TypeError("date_obj must be a datetime object")
    return date_obj.weekday() < 5  # Monday=0, Sunday=6

def validate_enum(value, enum_class):
    
    return value in enum_class

def parse_date(date_str, fmt="%Y-%m-%d"):
   
    try:
        return datetime.strptime(date_str, fmt)
    except ValueError:
        raise ValueError(f"Date string '{date_str}' does not match format {fmt}")
