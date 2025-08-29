#!/usr/bin/env python3
"""
Documented
"""


def summation_i_squared(n):
    """
    summation i squared
    """
    if n < 1:
        return None
    return n * (n + 1) * (2 * n + 1) // 6
