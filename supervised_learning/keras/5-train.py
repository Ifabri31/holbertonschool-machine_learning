#!/usr/bin/env python3
"""
5-train.py
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
                epochs, validation_data=None, verbose=True, shuffle=False):
    """
    docstring
    """
    if validation_data is None:
        history = network.fit(data, labels,
                              batch_size=batch_size,
                              epochs=epochs, verbose=verbose,
                              shuffle=shuffle)
    else:
        history = network.fit(data, labels,
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=verbose,
                              shuffle=shuffle,
                              validation_data=validation_data)
    return history
