#!/usr/bin/env python3
"""
7-RMSProp.py
"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    docstring
    """
    s = beta2 * s + (1 - beta2) * (grad ** 2)
    var = var - (alpha / (epsilon + s ** 0.5)) * grad
    return var, s
