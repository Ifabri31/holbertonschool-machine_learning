#!/usr/bin/env python3
"""
3-one_hot.py
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    docstring
    """
    if classes is None:
        classes = max(labels) + 1
    return K.utils.to_categorical(labels, num_classes=classes)
