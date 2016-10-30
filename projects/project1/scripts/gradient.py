import numpy as np


def gradient_least_square(y, tx, w, cost='mse'):
    """Compute the gradient."""
    if cost is 'mse':
        N = y.size
        e = y - np.dot(tx,w)
        return -1/N * np.dot(tx.T,e)
    elif cost is 'mae':
        e = y - np.dot(tx, w)
        return np.dot(tx.T, (-1) * np.sign(e)) / y.size
    else:
        raise Exception


def stoch_gradient_least_square(batch_y, batch_x, w, cost='mse'):
    N = batch_y.shape[0]
    grad = np.empty(len(w))
    for _y, _x in zip(batch_y, batch_x):
        grad = grad + gradient_least_square(_y, _x, w, cost)
    return grad / N

