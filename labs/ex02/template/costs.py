# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np

"""calculate the cost.
you can calculate the cost by mse or mae.
"""
def compute_cost_MSE(y, tx, w):
    y = np.array(y)
    return np.linalg.norm(y - np.dot(tx, w)) ** 2 / (2 * y.shape[0])

def compute_cost_MAE(y, tx, w):
    y = np.array(y)
    return np.sum(abs(y - np.dot(tx, w))) / y.shape[0]

def compute_cost_MSE_for_Ridge(y, tx, w, lamb):
    return compute_cost_MSE(y,tx,w) + lamb * np.linalg.norm(w) / 2 / len(y)