'''
Python file required.
This file consists of all six method required by the project sheet
In addition, it contains one method that would generate the best result.
Note that, you should give RAW data matrix, i.e. in shape of (nb_sample, 30) as
input to all the methods here.
All the methods except the logistic_regression_best is aiming to produce the baseline
statistics. Thus, standardization is the only data manipulation.
For more information on tests, please refer to test.py, we compose a complex experiment
phases there.
'''
from __future__ import absolute_import

import datetime
import numpy as np
from model import *
from gradient import *
from costs import *
from data_utils import compose_interactions_for_transforms, standardize

error = 'mse'


def least_squares(y, tx):
    """calculate the least squares solution."""
    tx, _, _ = standardize(tx)
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
    tx, _, _ = standardize(tx)
    D = tx.shape[1]  # number of features
    initial_w = np.zeros((D,))
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
    tx, _, _ = standardize(tx)
    D = tx.shape[1]  # number of features
    initial_w = np.zeros((D,))
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
    tx, _, _ = standardize(tx)

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
    tx, _, _ = standardize(tx)
    model = LogisticRegression((tx, y))
    return model.train(lr=gamma, decay=1, max_iters=max_iters)


def reg_logistic_regression(y, tx, lambda_, gamma, max_iters):
    """ L2 reg logistic basic version """
    tx, _, _ = standardize(tx)
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
    tx, _, _ = standardize(tx)
    model = LogisticRegression((tx, y), regularizer="Lasso", regularizer_p=lambda_)
    return model.train(lr=gamma, decay=1, max_iters=max_iters)


def logistic_regression_best(y, tx, lambda_=0.1, gamma=0.01, max_iters=2000):
    """
    Implement the best logistic regression.
    This function would process the data according to our best result.
    Hyper-parameters:
        decay=0.5, decay interval = 100, early_stop = 1000, batch_size=128,
    :param y:       raw data prediction [250000,]
    :param tx:      raw data matrix [250000, 30]
    :param lambda_: lambda = 0.1
    :param gamma:   gamma = 0.01 for step size
    :param max_iters: 2000
    :return:
    """

    tx = compose_interactions_for_transforms(tx)
    model = LogisticRegression((tx, y), regularizer="Ridge", regularizer_p=lambda_)
    return model.train(lr=gamma, decay=0.5, max_iters=max_iters, early_stop=1000, decay_intval=100)


def compose_best_submission_feature(tx):
    """
    This is to be called if you want to repreduce the feature of our best submission
    :param tx: raw data matrix
    :return:
    """
    data, _, _ = compose_interactions_for_transforms(tx)
    return data
