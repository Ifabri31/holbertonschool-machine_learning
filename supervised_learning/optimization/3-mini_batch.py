#!/usr/bin/env python3
"""
3-mini_batch.py
"""
import numpy as np
shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """
    docstring
    """
    m = X.shape[0]
    mini_batches = []

    X, Y = shuffle_data(X, Y)

    full_size = m // batch_size
    for i in range(0, full_size):
        X_batch = X[i * batch_size:(i + 1) * batch_size]
        Y_batch = Y[i * batch_size:(i + 1) * batch_size]
        mini_batches.append((X_batch, Y_batch))

    if m % batch_size != 0:
        X_batch = X[full_size * batch_size:]
        Y_batch = Y[full_size * batch_size:]
        mini_batches.append((X_batch, Y_batch))

    return mini_batches
