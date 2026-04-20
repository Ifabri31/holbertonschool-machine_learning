#!/usr/bin/env python3
"""
5-slice.py
"""


def slice(df):
    """
    Extract the columns High, Low, Close, and Volume_BTC
    """
    df = df.loc[:, ["High", "Low", "Close", "Volume_(BTC)"]]
    df = df.iloc[::60]

    return df
