#!/usr/bin/env python3
"""
0-load_env.py
"""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Load the FrozenLake environment with specified parameters."""
    env = gym.make(
        "FrozenLake-v1", desc=desc, map_name=map_name, is_slippery=is_slippery)
    return env
