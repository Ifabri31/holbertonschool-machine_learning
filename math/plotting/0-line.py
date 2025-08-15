#!/usr/bin/env python3
"""
0-line.py
"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Documented
    """

    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.xlim(0, 10)
    plt.plot(range(0, 11), y, color='red', linestyle='-')
    plt.show()
