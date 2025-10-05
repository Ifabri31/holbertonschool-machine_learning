#!/usr/bin/env python3
"""
3-pool_backward.py
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    performs back propagation over a pooling
    layer of a neural network
    """
    m, h_new, w_new, c_new = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride
    dA_prev = np.zeros_like(A_prev)
    for n in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for ch in range(c_new):
                    h_start = h * sh
                    h_end = h_start + kh
                    w_start = w * sw
                    w_end = w_start + kw

                    if mode == 'max':
                        a_prev_slice = A_prev[n,
                                              h_start:h_end, w_start:w_end, ch]
                        mask = (a_prev_slice == np.max(a_prev_slice))
                        dA_prev[n, h_start:h_end, w_start:w_end,
                                ch] += mask * dA[n, h, w, ch]
                    elif mode == 'avg':
                        average = dA[n, h, w, ch] / (kh * kw)
                        dA_prev[n, h_start:h_end, w_start:w_end,
                                ch] += np.ones((kh, kw)) * average
    return dA_prev
