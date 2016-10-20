# -*- coding: utf-8 -*-
"""Exercise 3.

Ridge Regression
"""

import numpy as np
from .costs import compute_mse

def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
    # Hes = tx.T * tx + 2*N*lambda * I_m
    G = np.eye(tx.shape[1])
    G[0, 0] = 0
    hes = np.dot(tx.T, tx) + lamb * G
    weight = np.linalg.solve(hes, np.dot(tx.T, y))
    mse = compute_mse(y, tx, weight)
    return mse, weight