#!/usr/bin/env python3
"""
6-dropout_create_layer.py
"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    docstring
    """
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_avg')
    dense_layer = tf.keras.layers.Dense(n, activation=activation,
                                        kernel_initializer=initializer)
    layer_output = dense_layer(prev)
    layer_output = tf.keras.layers.Dropout(
        rate=1-keep_prob)(layer_output, training=training)

    return layer_output
