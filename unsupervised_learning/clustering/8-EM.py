#!/usr/bin/env python3
"""
8-EM.py
"""
import numpy as np
maximization = __import__('7-maximization').maximization
expectation = __import__('6-expectation').expectation
initialize = __import__('4-initialize').initialize


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    performs the EM algorithm for a Gaussian Mixture Model
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    n, d = X.shape
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None
    prev_l = None
    for i in range(iterations + 1):
        g, li = expectation(X, pi, m, S)
        if g is None or li is None:
            return None, None, None, None, None
        if verbose and (i % 10 == 0 or i == iterations):
            print(
                f"Log Likelihood after {i} iterations: {format_number(li)}")
        if i > 0 and prev_l is not None and abs(li - prev_l) < tol:
            if verbose:
                print(f"Log Likelihood after {i} iterations: {format_number(li)}")
            return pi, m, S, g, li
        prev_l = li
        if i < iterations:
            pi, m, S = maximization(X, g)
            if pi is None or m is None or S is None:
                return None, None, None, None, None

    return pi, m, S, g, li


def format_number(value):
    """
    Formats a float to 5 decimal places, removing trailing zeros
    """
    formatted = f"{value:.5f}"
    if '.' in formatted:
        formatted = formatted.rstrip('0').rstrip('.') 
        if formatted.endswith('0') else formatted
    return formatted
