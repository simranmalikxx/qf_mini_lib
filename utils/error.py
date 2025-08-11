# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 11:57:00 2025

@author: Simran
"""

class FinError(Exception):
    """Custom exception class for financial library errors."""
    def __init__(self, message: str):
        super().__init__(message)
