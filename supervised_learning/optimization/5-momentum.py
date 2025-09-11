#!/usr/bin/env python3
"""
5-momentum.py
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    docstring
    """
    v = beta1 * v + (1 - beta1) * grad
    var = var - alpha * v
    return var, v
