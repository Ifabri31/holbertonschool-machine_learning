#!/usr/bin/env python3
"""
8-RMSProp.py
"""
import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    docstring
    """
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha, rho=beta2, epsilon=epsilon)
