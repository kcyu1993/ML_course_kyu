# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def calculate_mse(y, tx, w):
    """Calculate the mse for vector e."""
    e = y - tx.dot(w)
    return 1/2*np.mean(e**2)


def calculate_mae(y, tx, w):
    """Calculate the mae for vector e."""
    e = y - tx.dot(w)
    return np.mean(np.abs(e))


def compute_loss(y, tx, w, error='mse'):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    e = y - tx.dot(w)
    if error is 'mse':
        return calculate_mse(e)
    elif error is 'mae':
        return calculate_mae(e)
    else:
        raise NotImplementedError


def calculate_rmse(y, tx, w):
    return np.sqrt(calculate_mse(y, tx, w) * 2)


def loss_logistic(y, tx, w):
    """compute the cost by negative log likelihood."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    #
    # ***************************************************
    loss = 0
    N = len(y)
    for index in range(len(tx)):
        e = np.dot(np.transpose(tx[index, :]), w)
        loss += np.log(1 + np.exp(e)) - y[index] * e
    return loss / N

def compute_misclass_rate(y, tx, model):
    pred_y = model(tx)

def get_loss_function(error='mse'):
    if error is 'mse':
        return calculate_mse
    elif error is 'mae':
        return calculate_mae
    elif error is 'rmse':
        return calculate_rmse
    elif error is 'logistic':
        return loss_logistic
