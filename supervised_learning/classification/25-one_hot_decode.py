#!/usr/bin/env python3
"""
25-one_hot_decode.py
"""
import numpy as np


def one_hot_decode(one_hot):
    """
    docstring
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None
    if not np.all((one_hot == 0) | (one_hot == 1)):
        return None
    if not np.all(one_hot.sum(axis=0) == 1):
        return None

    try:
        vector = np.argmax(one_hot, axis=0)
        return vector
    except Exception:
        return None
