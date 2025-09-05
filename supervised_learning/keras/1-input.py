#!/usr/bin/env python3
"""
1-input.py
"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    docstring
    """
    inputs = K.layers.Input(shape=(nx,))
    l2 = K.regularizers.l2(lambtha)
    output = K.layers.Dense(layers[0], activation=activations[0],
                            kernel_regularizer=l2)(inputs)

    if len(layers) > 1:
        output = K.layers.Dropout(1 - keep_prob)(output)

    for i in range(1, len(layers)):
        output = K.layers.Dense(layers[i], activation=activations[i],
                                kernel_regularizer=l2)(output)
        if i < len(layers) - 1:
            output = K.layers.Dropout(1 - keep_prob)(output)

    model = K.Model(inputs=inputs, outputs=output)
    return model
