#!/usr/bin/env python3
"""
0-inception_block.py
"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    docstring
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    f1 = K.layers.Conv2D(F1, kernel_size=(
        1, 1), padding='same', activation='relu')(A_prev)
    f3r = K.layers.Conv2D(F3R, kernel_size=(
        1, 1), padding='same', activation='relu')(A_prev)
    f3 = K.layers.Conv2D(F3, kernel_size=(
        3, 3), padding='same', activation='relu')(f3r)
    f5r = K.layers.Conv2D(F5R, kernel_size=(
        1, 1), padding='same', activation='relu')(A_prev)
    f5 = K.layers.Conv2D(F5, kernel_size=(
        5, 5), padding='same', activation='relu')(f5r)
    fpp = K.layers.MaxPooling2D(pool_size=(
        3, 3), strides=(1, 1), padding='same')(A_prev)
    fpp = K.layers.Conv2D(FPP, kernel_size=(
        1, 1), padding='same', activation='relu')(fpp)
    output = K.layers.Concatenate(axis=-1)([f1, f3, f5, fpp])
    return output
