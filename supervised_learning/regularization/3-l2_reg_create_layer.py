#!/usr/bin/env python3
"""
3-l2_reg_create_layer.py
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    docstring
    """
    l2_regularizer = tf.keras.regularizers.L2(lambtha)

    layer = tf.keras.layers.Dense(
        n, activation=activation,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode='fan_avg'),
        kernel_regularizer=l2_regularizer
    )
    layer_output = layer(prev)

    return layer_output
