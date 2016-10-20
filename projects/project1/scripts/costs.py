# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)


def calculate_mae(e):
    """Calculate the mae for vector e."""
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

def compute_misclass_rate(y, tx, model):
    pred_y = model(tx)

def get_loss_function(error='mse'):
    if error is 'mse':
        return compute_loss

