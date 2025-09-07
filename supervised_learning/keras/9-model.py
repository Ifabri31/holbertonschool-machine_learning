#!/usr/bin/env python3
"""
9-model.py
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    docstring
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    docstring
    """
    return K.models.load_model(filename)
