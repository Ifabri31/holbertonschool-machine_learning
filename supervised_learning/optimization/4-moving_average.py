#!/usr/bin/env python3
"""
4-moving_average.py
"""


def moving_average(data, beta):
    """
    docstring
    """
    average = 0
    moving_averages = []
    for i, data in enumerate(data):
        average = beta * average + (1 - beta) * data
        average_corrected = average / (1 - beta ** (i + 1))
        moving_averages.append(average_corrected)
    return moving_averages
