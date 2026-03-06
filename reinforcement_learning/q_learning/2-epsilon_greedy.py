#!/usr/bin/env python3
"""
2-epsilon_greedy.py
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    uses epsilon-greedy to determine the next action
    """
    p = np.random.uniform()
    if p < epsilon:
        return np.random.randint(Q.shape[1])
    return np.argmax(Q[state, :])
