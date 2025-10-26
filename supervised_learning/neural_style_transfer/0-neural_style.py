#!/usr/bin/env python3
"""
0-neural_style.py
"""
import numpy as np
import tensorflow as tf


class NST:
    """
    class to perform neural style transfer
    """
    style_layers = [
        'block1_conv1', 'block2_conv1', 'block3_conv1',
        'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        class constructor
        """
        valid = (isinstance(style_image, np.ndarray)
                 and style_image.ndim == 3 and style_image.shape[2] == 3)
        if not valid:
            raise TypeError(
                "style_image must be a numpy.ndarray with shape (h, w, 3)")
        valid_img = (isinstance(content_image, np.ndarray)
                     and content_image.ndim == 3
                     and content_image.shape[2] == 3)
        if not valid_img:
            raise TypeError(
                "content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        scales an image such that its
        """
        valid = (isinstance(image, np.ndarray)
                 and image.ndim == 3 and image.shape[2] == 3)
        if not valid:
            raise TypeError(
                "image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        max_dim = max(h, w)
        scale = 512 / max_dim
        new_size = (int(h * scale), int(w * scale))
        # Convert to tensor and add batch dimension
        image_tensor = tf.convert_to_tensor(image)[tf.newaxis, ...]

        # Resize using bicubic interpolation (while in [0, 255] range)
        resized_image = tf.image.resize(
            image_tensor,
            new_size,
            method=tf.image.ResizeMethod.BICUBIC
        )

        scaled_image = tf.clip_by_value(resized_image / 255.0, 0.0, 1.0)

        return scaled_image
