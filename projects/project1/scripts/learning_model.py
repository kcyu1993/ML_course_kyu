'''
Python files contains the learning models.
The whole idea is to have learning model separated from optimizer

How to do this?

Available model listed:
    least_squares
    ridge_regression
    logistic_regression
    reg_logistic_regression

Available optimizer:
    GD: gradient descent
    SGD: stochastic gradient descent
        Supported more features like, momentum,
'''
from __future__ import absolute_import
from abc import ABCMeta, abstractmethod
import numpy as np
from .gradient import *
from .costs import *
from data_utils import batch_iter

error = 'mse'

def least_squares(y, tx):
    return np.linalg.solve(np.dot(tx.T,tx), np.dot(tx.T,y))


def sigmoid(t):
    """apply sigmoid function on t."""
    return 1 / (1 + np.exp(-t))


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression:
    # ***************************************************
    # Hes = tx.T * tx + 2*N*lambda * I_m
    G = np.eye(tx.shape[1])
    G[0, 0] = 0
    hes = np.dot(tx.T, tx) + lambda_ * G
    return np.linalg.solve(hes, np.dot(tx.T, y))


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


def least_squares_SGD(y,tx,gamma,max_iters):
    return least_squares_SGD(y, tx, gamma, max_iters, 16)


def least_squares_SGD(y, tx, gamma, max_iters, batch_size=16):
    print('Least square training with SGD with batch size {b}, learning rate {g}'.
          format(b=batch_size, g=gamma))
    print('Data size: {size}'.format(size=len(y)))
    ws = [np.random.randn(tx.shape[1])]
    losses = []
    w = ws[0]
    for epoch in range(max_iters):
        # Generate the patch
        print('Epoch {e} in {m}'.format(e=epoch, m=max_iters))
        for batch_y, batch_x in batch_iter(y, tx, batch_size):
            grad = stoch_gradient_least_square(batch_y, batch_x, w, error)
            w = w - gamma * grad
        loss = compute_loss(y, tx, w, error)
        print('Loss = {l}'.format(l=loss))
        ws.append(w)
        losses.append(loss)
    return losses, ws
