#!/usr/bin/env python3
"""
6-momentum.py
"""
import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    docstring
    """
    return tf.keras.optimizers.SGD(
        learning_rate=alpha, momentum=beta1)
