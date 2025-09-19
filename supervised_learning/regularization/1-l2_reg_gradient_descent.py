#!/usr/bin/env python3
"""
1-l2_reg_gradient_descent.py
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    docstring
    """
    m = Y.shape[1]
    grads = {}
    dZ = cache['A' + str(L)] - Y
    for i in range(L, 0, -1):
        grads['dW' + str(i)] = (dZ @ cache['A' + str(i - 1)].T) / \
            m + (lambtha / m) * weights['W' + str(i)]
        grads['db' + str(i)] = dZ.sum(axis=1, keepdims=True) / m
        if i > 1:
            dA_prev = weights[f"W{i}"].T @ dZ
            dZ = dA_prev * (1 - cache['A' + str(i - 1)] ** 2)

    for i in range(1, L + 1):
        weights[f"W{i}"] -= alpha * grads[f"dW{i}"]
        weights[f"b{i}"] -= alpha * grads[f"db{i}"]

    return weights
