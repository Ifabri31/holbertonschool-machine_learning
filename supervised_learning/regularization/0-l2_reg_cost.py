#!/usr/bin/env python3
"""
0-l2_reg_cost.py
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    docstring
    """
    l2_cost = cost + (lambtha / (2 * m)) * sum(np.linalg.norm(
        weights['W' + str(i)], ord='fro') ** 2 for i in range(1, L + 1))
    return l2_cost
