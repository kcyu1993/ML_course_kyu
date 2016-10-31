"""
This file consists of operation related to gradient descent generally,
would be referenced by model.py and implementations.py
"""
import numpy as np
from data_utils import batch_iter
from costs import compute_loss



def compute_gradient(y, tx, w):
    """Compute the gradient."""

    e = y - (tx).dot(w)
    N = len(e)
    gradient = -1 / N * (tx.T).dot(e)

    return gradient


def gradient_least_square(y, tx, w, cost='mse'):
    """Compute the gradient."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute gradient and loss
    # ***************************************************
    if cost is 'mse':
        N = y.size
        e = y - np.dot(tx, w)
        return -1 / N * np.dot(tx.T, e)
    elif cost is 'mae':
        e = y - np.dot(tx, w)
        return np.dot(tx.T, (-1) * np.sign(e)) / y.size
    else:
        raise Exception

def gradient_descent(y, tx, initial_w, gamma, max_iters):
    """Gradient descent algorithm."""
    threshold = 1e-3  # determines convergence. To be tuned

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    method = 'mse'
    for n_iter in range(max_iters):
        current_grad = gradient_least_square(y, tx, w)
        current_loss = compute_loss(y, tx, w, method)
        # Moving in the direction of negative gradient
        w = w - gamma * current_grad
        # Store w and loss
        ws.append(w)
        losses.append(current_loss)
        # Convergence criteria
        if len(losses) > 1 and np.abs(current_loss - losses[-1]) < threshold:
            break
        print("Gradient Descent({bi}): loss={l}".format(
            bi=n_iter, l=current_loss))
    return losses, ws


def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, gamma, max_iters):
    """Stochastic gradient descent algorithm."""
    threshold = 1e-3  # determines convergence. To be tuned

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size, 1, True):
            current_grad = compute_gradient(minibatch_y, minibatch_tx, w)
            current_loss = compute_loss(y, tx, w)
            # Moving in the direction of negative gradient
            w = w - gamma * current_grad
            # store w and loss
            ws.append(np.copy(w))
            losses.append(current_loss)
            # Convergence criteria
            if len(losses) > 1 and np.abs(current_loss - losses[-1]) < threshold:
                break
        print("Gradient Descent({bi}): loss={l}".format(
            bi=n_iter, l=current_loss))
    return losses, ws
