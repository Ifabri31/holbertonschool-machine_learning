#!/usr/bin/env python3
"""
12-learning_rate_decay.py
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_steps):
    """
    docstring
    """
    optimizer = tf.keras.optimizers.schedules.InverseTimeDecay(
        alpha, decay_steps=decay_steps,
        decay_rate=decay_rate, staircase=True)
    return optimizer
