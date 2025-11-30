#!/usr/bin/env python3
"""0-pca"""
import numpy as np


def pca(X, var=0.95):
    """
    performs PCA on a dataset
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
           n: number of data points
           d: number of dimensions in each data point
        var: float, the fraction of variance that the PCA transformation
             should maintain
    Returns:
        W: numpy.ndarray of shape (d, k) containing the PCA transformation
           matrix, where k is the number of dimensions required to
           maintain the specified variance var
    """
    SVD = np.linalg.svd(X, full_matrices=False)
    U = SVD[0]
    S = SVD[1]
    Vt = SVD[2]
    total_variance = np.sum(S)
    for i in range(len(S)):
        variance = np.sum(S[:i + 1])
        if variance / total_variance >= var:
            return Vt[:i + 1].T
    return Vt.T
