#!/usr/bin/env python3
"""
3-specificity.py
"""
import numpy as np


def specificity(confusion):
    """
    calculates the specificity for each class in a confusion matrix
    """
    classes = confusion.shape[0]
    specificity = np.zeros(classes)
    for i in range(classes):
        tp = confusion[i, i]
        fp = np.sum(confusion[:, i]) - tp
        fn = np.sum(confusion[i, :]) - tp
        tn = np.sum(confusion) - tp - fp - fn
        total_negatives = tn + fp

        specificity[i] = tn / total_negatives
    return specificity
