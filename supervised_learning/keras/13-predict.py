#!/usr/bin/env python3
"""
13-predict.py
"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    docstring
    """
    return network.predict(data, verbose=verbose)
