#!/usr/bin/env python3
"""
2-identity_block.py
"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    docstring
    """
    F11, F3, F12 = filters
    initializer = K.initializers.he_normal(seed=0)
    x = K.layers.Conv2D(F11, (1, 1), padding='same',
                        kernel_initializer=initializer)(A_prev)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(F3, (3, 3), padding='same',
                        kernel_initializer=initializer)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Activation('relu')(x)
    x = K.layers.Conv2D(F12, (1, 1), padding='same',
                        kernel_initializer=initializer)(x)
    x = K.layers.BatchNormalization(axis=-1)(x)
    x = K.layers.Add()([x, A_prev])
    x = K.layers.Activation('relu')(x)
    return x
