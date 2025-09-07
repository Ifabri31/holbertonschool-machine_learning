#!/usr/bin/env python3
"""
11-config.py
"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    docstring
    """
    json_config = network.to_json()
    with open(filename, 'w') as json_file:
        json_file.write(json_config)


def load_config(filename):
    """
    docstring
    """
    f = open(filename)
    model = K.models.model_from_json(f.read())
    f.close()
    return model
