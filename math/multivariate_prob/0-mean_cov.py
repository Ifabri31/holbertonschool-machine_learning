#!/usr/bin/env python3
"""
0-mean_cov
"""
import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n = X.shape[0]
    if n < 2:
        raise ValueError("X must contain multiple data points")
    mean = np.mean(X, axis=0, keepdims=True)
    deviations = X - mean
    cov = np.dot(deviations.T, deviations) / (n - 1)
    return mean, cov
