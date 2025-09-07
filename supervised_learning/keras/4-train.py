#!/usr/bin/env python3
"""
4-train.py
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, verbose=True, shuffle=False):
    """
    docstring
    """
    network.fit(
        data, labels, batch_size=batch_size,
        epochs=epochs, verbose=verbose,
        shuffle=shuffle)
    return network.history
