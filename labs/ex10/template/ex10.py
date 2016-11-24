# Useful starting lines
# %matplotlib inline

import numpy as np
import scipy
import scipy.io
import scipy.sparse as sp
import matplotlib.pyplot as plt
# %load_ext autoreload
# %autoreload 2

from helpers import load_data, preprocess_data
from plots import plot_raw_data
from plots import plot_train_test_data
from helpers import calculate_mse
from helpers import build_index_groups


def split_data(ratings, num_items_per_user, num_users_per_item,
               min_num_ratings, p_test=0.1):
        """split the ratings to training data and test data.
        Args:
            min_num_ratings:
                all users and items we keep must have at least min_num_ratings per user and per item.
        """
        # set seed
        np.random.seed(988)

        # select user and item based on the condition.
        valid_users = np.where(num_items_per_user >= min_num_ratings)[0]
        valid_items = np.where(num_users_per_item >= min_num_ratings)[0]
        valid_ratings = ratings[valid_items, :][:, valid_users]

        # ***************************************************
        # INSERT YOUR CODE HERE
        # split the data and return train and test data.
        # NOTE: we only consider users and movies that have more
        # than 10 ratings
        # ***************************************************
        train = sp.lil_matrix(valid_ratings.shape)
        test = sp.lil_matrix(valid_ratings.shape)
        rows, cols = valid_ratings.nonzero()
        for i, j in zip(rows, cols):
            if np.random.random() > p_test:
                train[i,j] = valid_ratings[i,j]
            else:
                test[i,j] = valid_ratings[i,j]

        print("Total number of nonzero elements in origial data:{v}".format(v=ratings.nnz))
        print("Total number of nonzero elements in train data:{v}".format(v=train.nnz))
        print("Total number of nonzero elements in test data:{v}".format(v=test.nnz))
        return valid_ratings, train, test


def compute_error_mse(prediction, original):
    rows, cols = original.nonzero()
    ind = zip(rows, cols)
    mse = 0
    for x,y in ind:
        mse += (prediction[x,y] - original[x,y]) ** 2
    return mse


def baseline_item_mean(train, test):
    """baseline method: use item means as the prediction."""
    mse = 0
    num_items, num_users = train.shape
    mean = train.mean(1)
    prediction = sp.lil_matrix((num_items, num_users))
    for i in range(num_users):
        prediction[:, i] = mean
    # Use test to calcualte RMSE
    mse = compute_error_mse(prediction, test)
    print('item mean mse ' + str(mse))
    return mse



def baseline_user_mean(train, test):
    """baseline method: use the user means as the prediction."""

    num_items, num_users = train.shape
    mean = train.mean(0)
    prediction = sp.lil_matrix((num_items, num_users))
    for i in range(num_items):
        prediction[i, :] = mean
    # Use test to calcualte RMSE
    mse = compute_error_mse(prediction, test)
    print('user mean mse ' + str(mse))
    return mse
    # print('baseline_item_mean: ' + str(mse))


def baseline_global_mean(train, test):
    """baseline method: use the global mean."""
    num_items, num_users = train.shape
    mean = train.mean(0)
    mean = np.mean(mean)
    prediction = sp.lil_matrix((num_items, num_users))
    for i in range(num_items):
        for j in range(num_users):
            prediction[i, j] = mean
    # Use test to calcualte RMSE
    mse = compute_error_mse(prediction, test)
    print('global mean mse ' + str(mse))
    return mse


def init_MF(train, num_features):
    """init the parameter for matrix factorization."""

    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # you should return:
    #     user_features: shape = num_features, num_user
    #     item_features: shape = num_features, num_item
    # ***************************************************
    num_item, num_user = train.shape
    user_features = sp.lil_matrix(np.random.random((num_features, num_user)))
    item_features = sp.lil_matrix(np.random.random((num_features, num_item)))
    return user_features, item_features


def matrix_factorization_SGD(train, test):
    """matrix factorization by SGD."""
    # define parameters
    gamma = 0.01
    num_features = 20  # K in the lecture notes
    lambda_user = 0.1
    lambda_item = 0.7
    num_epochs = 20  # number of full passes through the train set
    errors = [0]

    # set seed
    np.random.seed(988)

    # init matrix in shape (K, N), (K, D)
    user_features, item_features = init_MF(train, num_features)

    # find the non-zero ratings indices
    nz_row, nz_col = train.nonzero()
    nz_train = list(zip(nz_row, nz_col))
    nz_row, nz_col = test.nonzero()
    nz_test = list(zip(nz_row, nz_col))

    print("learn the matrix factorization using SGD...")
    for it in range(num_epochs):
        # shuffle the training rating indices
        np.random.shuffle(nz_train)

        # decrease step size
        gamma /= 1.2
        # item - d - W || user - n - Z
        for d, n in nz_train:
            # pred = item_features.T.dot(user_features)
            pred = item_features[:, d].T.dot(user_features[:, n])
            p = -(train[d,n] - pred[0,0])
            grad_w = p * user_features[:,n]
            grad_z = p * item_features[:,d]
            user_features[:,n] -= gamma * grad_z
            item_features[:,n] -= gamma * grad_w
            #
            # for k in range(num_features):
            #     # Update each w(d,k) z(n,k)
            #     # for w -> item_feature
            #     grad_w = -(train[d,n] - pred[d,n])*user_features[k,n]
            #     # for z -> user_feature
            #     grad_z = -(train[d,n] - pred[d,n])*item_features[k,d]
            #     user_features[k,n] -= gamma * grad_z
            #     item_features[k,d] -= gamma * grad_w
            # user_features -= lambda_user * user_features
            # item_features -= lambda_item * item_features

        rmse = compute_error(train, user_features, item_features, nz_train)
        print("iter: {}, RMSE on training set: {}.".format(it, rmse))

        errors.append(rmse)
    rmse = compute_error(test, user_features, item_features, nz_test)
    print("RMSE on test data: {}.".format(rmse))


def compute_error(data, user_features, item_features, nz):
    """
    compute the loss (MSE) of the prediction of nonzero elements.
    :param data: X
    :param user_features: W
    :param item_features: Z
    :param nz:  non-zero indices
    :return:
    """
    pred = item_features.T.dot(user_features)
    assert data.shape == pred.shape
    mse = 0
    for i,j in nz:
        mse += (data[i,j] - pred[i,j])**2
    mse /= 2 * len(nz)
    return np.sqrt(mse)*2


def update_user_feature(
        train, item_features, lambda_user,
        nnz_items_per_user, nz_user_itemindices):
    """update user feature matrix."""


def update_item_feature(
        train, user_features, lambda_item,
        nnz_users_per_item, nz_item_userindices):
    """update item feature matrix."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # update and return item feature.
    # ***************************************************
    raise NotImplementedError


def ALS(train, test):
    """Alternating Least Squares (ALS) algorithm."""
    # define parameters
    num_features = 20  # K in the lecture notes
    lambda_user = 0.1
    lambda_item = 0.7
    stop_criterion = 1e-4
    change = 1
    error_list = [0, 0]

    # set seed
    np.random.seed(988)

    # init ALS
    user_features, item_features = init_MF(train, num_features)

    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO
    # start you ALS-WR algorithm.
    # ***************************************************
    raise NotImplementedError


if __name__ == '__main__':
    path_dataset = "movielens100k.csv"
    ratings = load_data(path_dataset)

    num_items_per_user, num_users_per_item = plot_raw_data(ratings)
    print("min # of items per user = {}, min # of users per item = {}.".format(
            min(num_items_per_user), min(num_users_per_item)))
    # Ex 1
    valid_ratings, train, test = split_data(
        ratings, num_items_per_user, num_users_per_item, min_num_ratings=10, p_test=0.1)
    # plot_train_test_data(train, test)

    # Ex2
    # baseline_global_mean(train, test)
    # Ex3
    # baseline_user_mean(train, test)
    # baseline_item_mean(train, test)
    # Ex 4
    matrix_factorization_SGD(train, test)
    # ALS(train, test)
