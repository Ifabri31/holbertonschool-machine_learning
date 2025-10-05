#!/usr/bin/env python3
"""
2-conv_backward.py
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over
    a convolutional layer of a neural network
    Returns:
        dA_prev:
        dW:
        db: 
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == "same":
        pad_h = ((h_prev - 1) * sh + kh - h_prev) // 2 + 1
        pad_w = ((w_prev - 1) * sw + kw - w_prev) // 2 + 1
    else:
        pad_h, pad_w = 0, 0
    padded_images = np.pad(
        A_prev, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode='constant')
    dA_prev = np.zeros_like(padded_images)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for i in range(dZ.shape[1]):
        for j in range(dZ.shape[2]):
            for k in range(c_new):
                h_start = i * sh
                h_end = h_start + kh
                w_start = j * sw
                w_end = w_start + kw

                dW[:, :, :, k] += np.sum(
                    padded_images[:, h_start:h_end,
                                  w_start:w_end, :] *
                    dZ[:, i, j, k, np.newaxis, np.newaxis, np.newaxis], axis=0)
                dA_prev[:, h_start:h_end, w_start:w_end, :] += (
                    W[:, :, :, k] * dZ[:, i, j, k,
                                       np.newaxis, np.newaxis, np.newaxis])
    if padding == 'same':
        dA_prev = dA_prev[:, pad_h:-pad_h, pad_w:-pad_w, :]
    return dA_prev, dW, db
