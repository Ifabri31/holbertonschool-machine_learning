#!/usr/bin/env python3
"""
11-learning_rate_decay.py
"""
import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    docstring
    """
    return alpha / (1 + decay_rate * (global_step // decay_step))
