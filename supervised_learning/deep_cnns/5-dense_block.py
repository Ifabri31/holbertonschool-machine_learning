#!/usr/bin/env python3
"""
5-dense_block.py
"""
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    docstring
    """
    concat = X
    for _ in range(layers):
        X = K.layers.BatchNormalization(axis=3)(concat)
        X = K.layers.Activation('relu')(X)
        X = K.layers.Conv2D(
            4 * growth_rate, (1, 1), padding='same',
            kernel_initializer=K.initializers.he_normal(seed=0))(X)
        X = K.layers.BatchNormalization(axis=3)(X)
        X = K.layers.Activation('relu')(X)
        X = K.layers.Conv2D(
            growth_rate, (3, 3), padding='same',
            kernel_initializer=K.initializers.he_normal(seed=0))(X)
        concat = K.layers.Concatenate(axis=3)([concat, X])

    return concat, nb_filters + layers * growth_rate
