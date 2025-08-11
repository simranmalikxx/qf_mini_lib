# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 12:21:16 2025

@author: Simran
"""

# utils/math_utils.py

import math
from scipy.stats import norm
import numpy as np


#Distributions
def N(x: float) -> float:
    return norm.cdf(x)


def n(x: float) -> float:
    return norm.pdf(x)


#General 

def clamp(x: float, min_val: float, max_val: float) -> float:
    
    #Clamp value x between min_val and max_val.
    return max(min(x, max_val), min_val)


def is_close(a: float, b: float, tol: float = 1e-8) -> bool:
    
    #Check if two floats are approximately equal within a tolerance.
    return abs(a - b) < tol


def safe_log(x: float, default: float = 0.0) -> float:
    
    if x <= 0.0:
        return default
    return math.log(x)


#Black-Scholes 

def d1(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    Black-Scholes d1 term.
    """
    return (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))


def d2(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """
    Black-Scholes d2 term.
    """
    return d1(S, K, T, r, q, sigma) - sigma * math.sqrt(T)


#stats

def mean(arr: np.ndarray) -> float:
    return np.mean(arr)


def variance(arr: np.ndarray, ddof: int = 0) -> float:
    return np.var(arr, ddof=ddof)


def std_dev(arr: np.ndarray, ddof: int = 0) -> float:
    return np.std(arr, ddof=ddof)


def covar(x: np.ndarray, y: np.ndarray) -> np.ndarray:

    return np.cov(x, y)


def corr(x: np.ndarray, y: np.ndarray) -> float:
    return np.corrcoef(x, y)[0, 1]
