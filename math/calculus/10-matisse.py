#!/usr/bin/env python3
"""
10-matisse.py
"""


def poly_derivative(poly):
    """
    poly derivation
    """
    if poly == [0]:
        return [0]
    if poly == []:
        return None
    if type(poly) is not list:
        return None
    if len(poly) == 1:
        return [0]
    if len(poly) == 2:
        return [poly[0]]
    return [i * coeff for i, coeff in enumerate(poly)][1:]
