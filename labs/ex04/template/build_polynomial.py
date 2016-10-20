# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function:
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    x = np.array(x)  # to make it safe.
    _x = np.ones((x.shape[0], degree + 1))
    for i in range(degree):
        _x[:, i + 1:degree + 1] *= x[:, np.newaxis]
    return _x