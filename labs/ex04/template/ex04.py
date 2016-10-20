# Useful starting lines
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
# %load_ext autoreload
# %autoreload 2
from sklearn import linear_model

# from __future__ import absolute_import
from labs.ex03.template import helpers

from labs.ex04.template.costs import compute_rmse, compute_mse
from labs.ex04.template.costs import compute_mse_for_ridge
from labs.ex04.template.ridge_regression import ridge_regression
from labs.ex04.template.build_polynomial import build_poly
from labs.ex04.template.plots import cross_validation_visualization
from labs.ex04.template.plots import cross_validation_visualization_for_degree

from labs.ex04.template.least_squares import least_squares
from labs.ex04.template.split_data import split_data
from labs.ex04.template.plots import bias_variance_decomposition_visualization


# load dataset
def data_load():
    ''' Return x, y '''
    return helpers.load_data()

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, k_indices, k, lamb, degree, rmse=False):
    """return the loss of ridge regression."""
    # ***************************************************
    # Split data into K groups according to indices
    # get k'th subgroup in test, others in train:
    # ***************************************************
    x = np.array(x)
    y = np.array(y)

    train_ind = np.concatenate((k_indices[:k], k_indices[k+1:]), axis=0)
    train_ind = np.reshape(train_ind, (train_ind.size,))

    test_ind = k_indices[k]
    # Note: different from np.ndarray, tuple is name[index,]
    # ndarray is name[index,:]
    train_x = x[train_ind,]
    train_y = y[train_ind,]
    test_x = x[test_ind,]
    test_y = y[test_ind,]

    # ***************************************************
    # INSERT YOUR CODE HERE
    # form data with polynomial degree:
    # ***************************************************
    train_x = build_poly(train_x, degree)
    test_x = build_poly(test_x, degree)

    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression:
    # ***************************************************
    loss_tr, weight = ridge_regression(train_y, train_x, lamb)

    # Test with sklearn ridge solve.
    clf = linear_model.ridge_regression(train_x, train_y, alpha=lamb)
    # weight = clf

    # ***************************************************
    # INSERT YOUR CODE HERE
    # calculate the loss for train and test data: TODO
    # ***************************************************
    ''' Compute MSE by ridge weights '''
    loss_tr = compute_mse_for_ridge(train_y, train_x, weight,lamb)
    loss_te = compute_mse_for_ridge(test_y, test_x, weight, lamb)
    # loss_tr = compute_mse(train_y, train_x, weight)
    # loss_te = compute_mse(test_y, test_x, weight)

    if rmse is True:
        loss_tr = compute_rmse(loss_tr)
        loss_te = compute_rmse(loss_te)
    return loss_tr, loss_te

def cross_validation_demo():
    seed = 1
    degree = 7
    k_fold = 4
    lambdas = np.logspace(-4, 2, 30)
    y,x = data_load()
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    mse_tr = []
    mse_te = []
    # ***************************************************
    # INSERT YOUR CODE HERE
    # cross validation:
    # ***************************************************
    for lamb in lambdas:
        _mse_tr = []
        _mse_te = []
        for k in range(k_fold):
            loss_tr, loss_te = cross_validation(y,x,k_indices,k,lamb,degree, rmse=True)
            _mse_tr += [loss_tr]
            _mse_te += [loss_te]
        avg_tr = np.average(_mse_tr)
        avg_te = np.average(_mse_te)
        mse_tr += [avg_tr]
        mse_te += [avg_te]

    cross_validation_visualization(lambdas, mse_tr, mse_te)
    print(mse_tr, mse_te)

def cross_validation_demo_degree():
    seed = 1
    degrees = range(2,11)
    k_fold = 4
    lamb = 0.5
    y,x = data_load()
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    mse_tr = []
    mse_te = []
    # ***************************************************
    # INSERT YOUR CODE HERE
    # cross validation:
    # ***************************************************
    for degree in degrees:
        _mse_tr = []
        _mse_te = []
        for k in range(k_fold):
            loss_tr, loss_te = cross_validation(y,x,k_indices,k,lamb, degree, rmse=True)
            _mse_tr += [loss_tr]
            _mse_te += [loss_te]
        avg_tr = np.average(_mse_tr)
        avg_te = np.average(_mse_te)
        mse_tr += [avg_tr]
        mse_te += [avg_te]

    cross_validation_visualization_for_degree(degrees, mse_tr, mse_te)
    print(mse_tr, mse_te)

def bias_variance2(y, x, weight, variance_e):
    '''
    For linear model bias-variance calculation. The dimension is len(weight)
    :param y:
    :param x:
    :param weight: beta of linear model
    :param function:
    :param variance_e:
    :return:
    '''
    # N = len(x)
    # res = np.dot(x, weight)
    # error = variance_e * (len(weight) / N) + np.sum( (y - np.dot(x, weight)) **2 )/ N
    # return compute_rmse(error)
    return compute_rmse(compute_mse(y,x,weight) + 1 + len(weight)/ len(x))


def bias_variance(function, x, weight, variance_e):
    '''
    For linear model bias-variance calculation. The dimension is len(weight)
    :param y:
    :param x:
    :param weight: beta of linear model
    :param function:
    :param variance_e:
    :return:
    '''
    y = function(x[:,1])
    # N = len(x)
    # res = np.dot(x, weight)
    # error = variance_e * (len(weight) / N) + np.sum( (y - np.dot(x, weight)) **2 )/ N
    # return compute_rmse(error)
    return compute_rmse(compute_mse(y,x,weight))

def bias_variance_demo():
    """The entry."""
    # define parameters
    seeds = range(100)
    num_data = 10000
    ratio_train = 0.005
    degrees = range(1, 10)

    # define list to store the variable
    rmse_tr = np.empty((len(seeds), len(degrees)))
    rmse_te = np.empty((len(seeds), len(degrees)))

    for index_seed, seed in enumerate(seeds):
        np.random.seed(seed)
        x = np.linspace(0.1, 2 * np.pi, num_data)
        y = np.sin(x) + 0.3 * np.random.randn(num_data).T
        # ***************************************************
        # INSERT YOUR CODE HERE
        # split data with a specific seed: TODO
        # ***************************************************
        train_x, train_y, test_x, test_y = split_data(x,y,ratio_train,seed)
        # ***************************************************
        # INSERT YOUR CODE HERE
        # bias_variance_decomposition: TODO
        # ***************************************************
        for ind_degree, degree in enumerate(degrees):
            # Use least square
            x_tr = build_poly(train_x, degree)
            x_te = build_poly(test_x, degree)
            mse, weight = least_squares(train_y, x_tr)
            rmse_tr[index_seed][ind_degree] = bias_variance(np.sin, x_tr, weight, 1)
            rmse_te[index_seed][ind_degree] = bias_variance(np.sin, x_te, weight, 1)
            # rmse_tr[index_seed][ind_degree] = bias_variance2(train_y, x_tr, weight, 1)
            # rmse_te[index_seed][ind_degree] = bias_variance2(test_y, x_te, weight, 1)

    bias_variance_decomposition_visualization(degrees, rmse_tr, rmse_te)



# cross_validation_demo()
# degree = 5.
# cross_validation_demo_degree()


bias_variance_demo()
print()