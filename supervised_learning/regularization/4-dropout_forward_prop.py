#!/usr/bin/env python3
"""
4-dropout_forward_prop.py
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    docstring
    """
    cache = {'A0': X}
    for layer in range(1, L + 1):
        A_prev = cache['A' + str(layer - 1)]
        W = weights['W' + str(layer)]
        b = weights['b' + str(layer)]
        Z = np.matmul(W, A_prev) + b
        if layer == L:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
        else:
            A = np.tanh(Z)
            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            A *= D
            A /= keep_prob
            cache['D' + str(layer)] = D.astype(np.int8)
        cache['A' + str(layer)] = A
    return cache
