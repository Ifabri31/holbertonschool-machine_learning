#!/usr/bin/env python3
"""
2-shuffle_data.py
"""
import numpy as np


def shuffle_data(X, Y):
    """
    docstring
    """
    idx = np.random.permutation(X.shape[0])
    return X[idx], Y[idx]
