#!/usr/bin/env python3
"""
10-weights.py
"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """
    docstring
    """
    network.save(filename, save_format=save_format)


def load_weights(network, filename):
    """
    docstring
    """
    network.load_weights(filename)
