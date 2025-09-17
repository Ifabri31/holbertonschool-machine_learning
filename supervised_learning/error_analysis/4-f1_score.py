#!/usr/bin/env python3
"""
4-f1_score.py
"""


def f1_score(confusion):
    """
    calculates the F1 score of a confusion matrix
    """
    sensitivity = __import__('1-sensitivity').sensitivity
    precision = __import__('2-precision').precision
    precision = precision(confusion)
    sensitivity = sensitivity(confusion)
    f1 = (2 * (precision * sensitivity)) / (precision + sensitivity)
    return f1
