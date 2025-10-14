#!/usr/bin/env python3
"""
6-transition_layer.py
"""
from tensorflow import keras as K


def transition_layer(X, nb_filters, compression):
    """
    docstring
    """
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    nb_filters = int(nb_filters * compression)

    X = K.layers.Conv2D(
        nb_filters, (1, 1), padding='same',
        kernel_initializer=K.initializers.he_normal(seed=0))(X)
    X = K.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))(X)

    return X, nb_filters
