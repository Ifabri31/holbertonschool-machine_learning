#!/usr/bin/env python3
"""
1-q_init.py
"""
import numpy as np


def q_init(env):
    """
    Initialize the Q-learning environment.
    """
    states = env.observation_space.n
    actions = env.action_space.n
    return np.zeros((states, actions))
