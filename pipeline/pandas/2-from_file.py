#!/usr/bin/env python3
"""
2-from_file.py
"""
import pandas as pd


def from_file(filename, delimiter):
    """
    Load data from a file as a pd.DataFrame.
    """

    df = pd.read_csv(filename, delimiter=delimiter)
    return df
