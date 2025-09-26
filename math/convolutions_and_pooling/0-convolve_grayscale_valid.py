#!/usr/bin/env python3
"""
0-convolve_grayscale_valid.py
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    performs a valid convolution on grayscale images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    conv_h = h - kh + 1
    conv_w = w - kw + 1
    output = np.zeros((m, conv_h, conv_w))

    for i in range(conv_h):
        for j in range(conv_w):
            region = images[:, i:i+kh, j:j+kw]
            output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return output
