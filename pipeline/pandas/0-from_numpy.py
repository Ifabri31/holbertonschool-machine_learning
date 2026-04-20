#!/usr/bin/env python3
"""
0-from_numpy.py
"""
import pandas as pd


def from_numpy(array):
    """
    Create a pd.DataFrame from a np.ndarray.
    """
    df = pd.DataFrame(array)
    df.columns = [chr(i) for i in range(65, 65 + df.shape[1])]
    return df
