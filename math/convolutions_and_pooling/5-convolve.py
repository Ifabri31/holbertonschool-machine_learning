#!/usr/bin/env python3
"""
5-convolve.py
"""
import numpy as np


def convolve(images, kernel, padding='same', stride=(1, 1)):
    """
    performs a convolution on images using myltiple kernels
    """
    kh, kw, kc, nc = kernels.shape
    m, h, w, c = images.shape
    sh, sw = stride
    if kc != c:
        raise ValueError("Kernel channels must match image channels.")

    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        ph = pw = 0
    else:
        ph, pw = padding

    padded_images = np.pad(
        images,
        ((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant')

    oh = (h + 2 * ph - kh) // sh + 1
    ow = (w + 2 * pw - kw) // sw + 1
    output = np.zeros((m, oh, ow, nc))

    kernels = kernels.reshape((1, *kernels.shape))

    for i in range(oh):
        for j in range(ow):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw
            output[:, i, j, :] = np.sum(
                padded_images[:, h_start:h_end, w_start:w_end,
                              :, None] * kernels,
                axis=(1, 2, 3)
            )
    return output
