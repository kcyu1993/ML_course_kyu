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

import datetime
import numpy as np
from model import *
from gradient import *
from costs import *
from data_utils import compose_complex_features_further

error = 'mse'


def least_squares(y, tx):
    """calculate the least squares solution."""
    A = np.dot(tx.T, tx)
    b = np.dot(tx.T, y)
    # Compute solution
    w_opt = np.linalg.solve(A, b)
    # Compute loss
    e = y - (tx).dot(w_opt)
    N = len(e)
    MSE_opt = 1 / (2 * N) * np.dot(e.T, e)
    return w_opt, MSE_opt


def least_squares_GD(y, tx, gamma, max_iters):
    """Gradient descent algorithm."""
    # Initialisation
    D = tx.shape[1]  # number of features
    initial_w = np.zeros([D, 1])

    # Start gradient descent.
    start_time = datetime.datetime.now()
    gradient_losses, ws = gradient_descent(y, tx, initial_w, gamma, max_iters)
    end_time = datetime.datetime.now()

    # Print result
    exection_time = (end_time - start_time).total_seconds()
    print("Gradient Descent: execution time={t:.3f} seconds".format(t=exection_time))

    # The last element is the optimum one
    return ws[-1]


def least_squares_SGD(y, tx, gamma, max_iters):
    """Gradient descent algorithm."""
    # Initialisation
    D = tx.shape[1]  # number of features
    initial_w = np.zeros([D, 1])
    batch_size = 1
    # Start SGD.
    start_time = datetime.datetime.now()
    gradient_losses, ws = stochastic_gradient_descent(
        y, tx, initial_w, batch_size, gamma, max_iters)
    end_time = datetime.datetime.now()
    # Print result
    exection_time = (end_time - start_time).total_seconds()
    print("SGD: execution time={t:.3f} seconds".format(t=exection_time))
    # The last element is the optimum one
    return ws[-1]


def ridge_regression(y, tx, lamb):
    """implement ridge regression."""

    D = tx.shape[1]  # number of features
    N = len(y)  # Number of measurements

    A = np.dot(tx.T, tx) + 2 * N * lamb * np.eye(D)
    b = np.dot(tx.T, y)

    # Obtain optimal solution
    w_opt = np.linalg.solve(A, b)

    # Compute the loss
    e = y - np.dot(tx, w_opt)
    MSE = 1 / (2 * N) * np.dot(e.T, e)
    RMSE = np.sqrt(2 * MSE)
    return w_opt, RMSE


def logistic_regression(y, tx, gamma, max_iters):
    """ Logistic regression basic version """
    model = LogisticRegression((tx, y))
    return model.train(lr=gamma, decay=1, max_iters=max_iters)


def reg_logistic_regression(y, tx, lambda_, gamma, max_iters):
    """ L2 reg logistic basic version """
    model = LogisticRegression((tx, y), regularizer="Ridge", regularizer_p=lambda_)
    return model.train(lr=gamma, decay=1, max_iters=max_iters)


def lasso_logistic_regression(y, tx, lambda_, gamma, max_iters):
    """
    L1 reg logistic logistic basic version
    :param y:           given y
    :param tx:          data matrix
    :param lambda_:     given parameter for lasso logistic regression
    :param gamma:       initial learning rate
    :param max_iters:   maximum iterations
    :return:
    """
    model = LogisticRegression((tx, y), regularizer="Lasso", regularizer_p=lambda_)
    return model.train(lr=gamma, decay=1, max_iters=max_iters)


def logistic_regression_best(y, tx, lambda_, gamma, max_iters):
    """
    Implement the best logistic regression.
    :param y:
    :param tx:
    :param lambda_:
    :param gamma:
    :param max_iters:
    :return:
    """
    model = LogisticRegression((tx, y), regularizer="Lasso", regularizer_p=lambda_)
    return model.train(lr=gamma, decay=1, max_iters=max_iters)
