#!/usr/bin/env python3
"""
3-rename.py
"""
import pandas as pd


def rename(df):
    """
    Rename the Timestamp column to Datetime and
    convert the timestamp values to datatime values.
    """
    df = df.rename(columns={'Timestamp': 'Datetime'})
    df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')
    df = df[['Datetime', 'Close']]

    return df
