#!/usr/bin/env python3
"""
Module
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    Doc
    """
    return np.concatenate((mat1, mat2), axis=axis)
