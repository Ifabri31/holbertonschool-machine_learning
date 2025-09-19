#!/usr/bin/env python3
"""
2-l2_reg_cost.py
"""
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    docstring
    """
    l2_costs = []

    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.InputLayer):
            # Get the L2 regularization loss for this layer
            layer_l2 = tf.reduce_sum(layer.losses) + cost
            l2_costs.append(layer_l2)

    return tf.convert_to_tensor(l2_costs)
