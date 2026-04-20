#!/usr/bin/env python3
"""
3-contrast
"""
import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    randomly adjusts the contrast of an image.

    image: A 3D tf.Tensor representing the input image to adjust the contrast.
    lower: A float representing the lower bound
    of the random contrast factor range.
    upper: A float representing the upper bound
    of the random contrast factor range.
    Returns the contrast-adjusted image.
    """
    return tf.image.random_contrast(image, lower=lower, upper=upper)
