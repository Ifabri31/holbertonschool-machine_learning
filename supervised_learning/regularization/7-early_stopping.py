#!/usr/bin/env python3
"""
7-early_stopping.py
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    docstring
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    if count >= patience:
        return True, count

    return False, count
