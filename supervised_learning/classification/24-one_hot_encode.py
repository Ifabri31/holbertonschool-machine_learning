#!/usr/bin/env python3
"""
24-one_hot_encode.py
"""


def one_hot_encode(Y, classes):
    """
    docstring
    """
    import numpy as np
    try:
        m = Y.shape[0]
        one_hot = np.zeros((classes, m))
        one_hot[Y, np.arange(m)] = 1
        return one_hot
    except Exception:
        return None
