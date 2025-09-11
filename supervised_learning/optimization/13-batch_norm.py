#!/usr/bin/env python3
"""
13-batch_norm.py
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    docstring
    """
    Z_mean = Z.mean(axis=0)
    Z_var = Z.var(axis=0)
    Z_norm = (Z - Z_mean) / np,sqrt(Z_var + epsilon)
    Z_tilde = gamma * Z_norm + beta
    return Z_tilde
