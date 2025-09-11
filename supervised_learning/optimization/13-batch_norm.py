#!/usr/bin/env python3
"""
13-batch_norm.py
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    docstring
    """
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    normalized = (Z - mean) / np.sqrt(variance + epsilon)
    return gamma * normalized + beta
