#!/usr/bin/env python3
"""
documented
"""


def poly_integral(poly, C=0):
    """
    poly integral
    """
    if not isinstance(poly, list):
        return None
    if len(poly) == 0:
        return None
    if not isinstance(C, (int, float)):
        return None
    res = [int(C) if float(C).is_integer() else C] + [
        int(r) if r.is_integer() else r
        for r in (poly[i] / (i + 1) for i in range(len(poly)))
    ]
    if res[-1] == 0:
        res.pop()
    return res
