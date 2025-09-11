#!/usr/bin/env python3
"""
1-normalize.py
"""
import numpy as np


def normalize(X, m, s):
    """
    Normaliza la matriz X usando la media m y desviación estándar s.
    Args:
        X (numpy.ndarray): shape (m, nx)
        m (numpy.ndarray): shape (nx,)
        s (numpy.ndarray): shape (nx,)
    Returns:
        numpy.ndarray: matriz normalizada
    """
    return (X - m) / s
