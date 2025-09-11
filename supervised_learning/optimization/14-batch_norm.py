#!/usr/bin/env python3
"""
14-batch_norm.py
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    docstring
    """
    dense = tf.keras.layers.Dense(
        units=n,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg'),
        name="layer")(prev)
    gamma = tf.Variable(tf.constant(1.0, shape=[n]))
    beta = tf.Variable(tf.constant(0.0, shape=[n]))
    mean, variance = tf.nn.moments(dense, axes=[0])
    bn = tf.nn.batch_normalization(dense, mean=mean, variance=variance,
                                   offset=beta, scale=gamma,
                                   variance_epsilon=1e-7)

    return activation(bn)
