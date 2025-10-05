#!/usr/bin/env python3
"""
1-pool_forward.py
"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    performs forward propagation over a pooling
    layer of a neural network
    Returns: the output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    output_h = (h_prev - kh) // sh + 1
    output_w = (w_prev - kw) // sw + 1
    output = np.zeros((m, output_h, output_w, c_prev))

    for j in range(output_h):
        h_start = j * sh
        h_end = h_start + kh
        for i in range(output_w):
            w_start = i * sw
            w_end = w_start + kw
            if mode == 'max':
                output[:, j, i, :] = np.max(
                    A_prev[:, h_start:h_end, w_start:w_end, :], axis=(1, 2))
            else:
                output[:, j, i, :] = np.mean(
                    A_prev[:, h_start:h_end, w_start:w_end, :], axis=(1, 2))
    return output
