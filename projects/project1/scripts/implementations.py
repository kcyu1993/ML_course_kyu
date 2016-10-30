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
from model import *
from gradient import *
from costs import *
from data_utils import compose_complex_features_further

error = 'mse'


def least_squares(y, tx):
    return np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))


def least_squares_GD(y, tx, gamma, max_iters):
    """Gradient descent algorithm."""


def least_squares_SGD(y, tx, gamma, max_iters):
    """Gradient descent algorithm."""


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression:
    # ***************************************************
    model = LinearRegression((tx, y), regularizer='Ridge', regularizer_p=lambda_)
    return model.train()


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
