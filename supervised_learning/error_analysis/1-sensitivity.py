#!/usr/bin/env python3
"""
1-sensitivity.py
"""
import numpy as np


def sensitivity(confusion):
    """
    calculates the sensitivity for each class in a confusion matrix
    """
    classes = confusion.shape[0]
    sensitivity = np.zeros(classes)
    for i in range(classes):
        true_positives = confusion[i, i]
        total = np.sum(confusion[i, :])
        sensitivity[i] = true_positives / total
    return sensitivity
