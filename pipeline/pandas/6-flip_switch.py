#!/usr/bin/env python3
"""
6-flip_switch.py
"""


def flip_switch(df):
    """
    Sort the data in reverse chronological order and
    transpose the sorted dataframe.
    """
    df = df.iloc[::-1]
    df = df.transpose()
    return df
