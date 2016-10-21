# -*- coding: utf-8 -*-
"""some helper functions."""
import numpy as np
import os


def standardize(x, mean_x=None, std_x=None, intercept=True):
    """Standardize the original data set."""
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x>0] = x[:, std_x>0] / std_x[std_x>0]
    if intercept is True:
        tx = np.hstack((np.ones((x.shape[0], 1)), x))
    else:
        tx = x
    return tx, mean_x, std_x


def batch_iter(y, tx, batch_size, num_batches=None, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)
    num_batches_max = int(np.ceil(data_size/batch_size))
    if num_batches is None:
        num_batches = num_batches_max
    else:
        num_batches = min(num_batches, num_batches_max)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # polynomial basis function:
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    x = np.array(x)  # to make it safe.
    _x = np.ones((x.shape[0], degree + 1))
    for i in range(degree):
        _x[:, i + 1:degree + 1] *= x[:, np.newaxis]
    return _x

def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    # ***************************************************
    # INSERT YOUR CODE HERE
    # split the data based on the given ratio:
    # ***************************************************
    # Random shuffle the index by enumerate.
    pair = np.c_[x,y]
    np.random.shuffle(pair)
    index = np.round(x.size * ratio, 0).astype('int16')
    p1, p2 = np.split(pair, [index])
    x1, y1 = zip(*p1)
    x2, y2 = zip(*p2)
    return x1, y1, x2, y2


def split_data_general(*args, ratio=[0.5], seed=1):
    np.random.seed(seed=seed)
    split_pos = [int(r * len(args[0])) for r in ratio]
    index = np.random.permutation(range(len(args[0])))
    split_indices = np.split(index, split_pos)
    split_result = []
    for split_index in split_indices:
        group = []
        for arg in args:
            arg = np.array(arg)
            if len(arg.shape) > 1:
                group.append(arg[split_index, :])
            else:
                group.append(arg[split_index,])
        split_result.append(group)
    return split_result

def get_dataset_dir():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    current_dir, _ = os.path.split(current_dir)
    # data_dir = os.path.join('..', current_dir)
    return os.path.join(current_dir, 'dataset')

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)