#!/usr/bin/env python3
"""
4-array.py
"""


def array(df):
    """
    Select the last 10 rows of the High and Close columns
    """
    data = df[['High', 'Close']].tail(10).to_numpy()
    return data
