#!/usr/bin/env python3
"""
2-optimize.py
"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    docstring
    """
    optimizer = K.optimizers.Adam(alpha, beta1, beta2)
    network.compile(optimizer=optimizer, loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return None
