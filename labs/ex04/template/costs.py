# -*- coding: utf-8 -*-
"""A function to compute the cost."""


import numpy as np

def compute_mse(y, tx, beta):
    """compute the loss by mse."""
    e = y - tx.dot(beta)
    mse = e.dot(e) / (2 * len(e))
    return mse

def compute_mae(y, tx, w):
    y = np.array(y)
    return np.sum(abs(y - np.dot(tx, w))) / y.shape[0]

def compute_mse_for_ridge(y, tx, w, lamb):
    return compute_mse(y,tx,w) + lamb * np.sum(w**2) /(2 * len(y))

def compute_rmse(mse):
    return np.sqrt(mse * 2)