'''
Python files contains the learning models.
The whole idea is to have learning model separated from optimizer
Would be called by LinearRegression model.

Available model listed:
    least_squares
    sigmoid
    least_squares_GD
'''
from __future__ import absolute_import

from gradient import *
from costs import *
import numpy as np

error = 'mse'

def least_squares(y, tx):
    return np.linalg.solve(np.dot(tx.T,tx), np.dot(tx.T,y))


def sigmoid(t):
    """apply sigmoid function on t."""
    return 1 / (1 + np.exp(-t))


def least_squares_GD(y, tx, gamma, max_iters):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [np.random.randn(tx.shape[1])]
    losses = []
    w = ws[0]

    for n_iter in range(max_iters):
        grad = gradient_least_square(y, tx, w, error)
        loss = compute_loss(y, tx, w, error)
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
    return losses, ws


