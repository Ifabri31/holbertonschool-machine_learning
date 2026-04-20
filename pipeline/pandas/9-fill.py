#!/usr/bin/env python3
"""
9-fill.py
"""


def fill(df):
    """
    Remove the Weighted_Price column, fill missing values in the Close column
    with the previous row’s value, fill missing values in the High, Low, and
    Open columns with the corresponding Close value in the same row, and set
    missing values in Volume_(BTC) and Volume_(Currency) to 0.
    """
    df = df.drop(columns=['Weighted_Price'])
    df['Close'] = df['Close'].fillna(method='ffill')
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['Open'] = df['Open'].fillna(df['Close'])
    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)
    return df
