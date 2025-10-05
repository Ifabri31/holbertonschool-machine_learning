#!/usr/bin/env python3
"""
0-conv_forward.py
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    performs forward propagation over a convolutional
    layer of a neural network
    Returns: the output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    else:
        ph = (((h_prev - 1) * sh + kh - h_prev) // 2)
        pw = (((w_prev - 1) * sw + kw - w_prev) // 2)

    p_input = np.pad(A_prev,
                     ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                     mode='constant')

    oh = (h_prev + 2 * ph - kh) // sh + 1
    ow = (w_prev + 2 * pw - kw) // sw + 1
    conv_layer = np.zeros((m, oh, ow, c_new))

    W_reshaped = W[np.newaxis, ...]

    for i in range(oh):
        for j in range(ow):
            h_start = i * sh
            h_end = h_start + kh
            w_start = j * sw
            w_end = w_start + kw
            patch = p_input[:, h_start:h_end, w_start:w_end, :, np.newaxis]
            conv_layer[:, i, j, :] = activation(
                np.sum(patch * W_reshaped, axis=(1, 2, 3)) + b
            )

    return conv_layer
