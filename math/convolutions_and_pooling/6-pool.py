#!/usr/bin/env python3
"""
6-pool.py
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    performs pooling on images
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    ph = pw = 0

    oh = (h - kh + 2 * ph) // sh + 1
    ow = (w - kw + 2 * pw) // sw + 1
    output = np.zeros((m, oh, ow, c))

    for i in range(oh):
        for j in range(ow):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw
            if mode == 'max':
                output[:, i, j, :] = np.max(
                    images[:, h_start:h_end, w_start:w_end, :],
                    axis=(1, 2)
                )
            elif mode == 'avg':
                output[:, i, j, :] = np.mean(
                    images[:, h_start:h_end, w_start:w_end, :],
                    axis=(1, 2)
                )
    return output