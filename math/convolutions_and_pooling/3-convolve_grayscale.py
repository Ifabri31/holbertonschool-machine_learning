#!/usr/bin/env python3
"""
3-convolve_grayscale.py
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    that performs a convolution on grayscale images
    """
    kh, kw = kernel.shape
    m, h, w = images.shape
    sh, sw = stride

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph = pw = 0
    else:
        ph, pw = padding

    padded_images = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw)),
        mode='constant')

    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1
    output = np.zeros((m, oh, ow))
    for i in range(oh):
        for j in range(ow):
            h_start = i * sh
            w_start = j * sw
            h_end = h_start + kh
            w_end = w_start + kw
            output[:, i, j] = np.sum(
                padded_images[:, h_start:h_end, w_start:w_end] * kernel,
                axis=(1, 2)
            )

    return output
