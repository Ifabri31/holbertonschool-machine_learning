#!/usr/bin/env python3
"""
Module documented.
"""


def matrix_transpose(matrix):
    """
    Documented
    """
    transpose = []

    for i in range(len(matrix[0])):
        new_row = []
        for j in range(len(matrix)):
            new_row.append(matrix[j][i])
        transpose.append(new_row)
    return transpose
