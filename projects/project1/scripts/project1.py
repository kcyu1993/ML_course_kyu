from __future__ import absolute_import

# Useful starting lines
# %matplotlib inline
# import numpy as np
# import matplotlib.pyplot as plt
# %load_ext autoreload
# %autoreload 2

from projects.project1.scripts.proj1_helpers import *
from projects.project1.scripts.helpers import *
from projects.project1.scripts.learning_model import *
from projects.project1.scripts.data_clean import *
from projects.project1.scripts.model import *
from projects.project1.scripts.network import Network
import os, datetime


def test_Network():
    training_data, valid_data = load_train_data_neural(validation=True, validation_ratio=0.1)
    network = Network([30, 100, 20, 2])
    network.SGD(training_data=training_data, epochs=100, lr=0.05, mini_batch_size=32,
                evaluation_data=valid_data,
                halve_learning_rate=10, momentum=0.8,
                monitor_evaluation_accuracy=True, monitor_evaluation_cost=True,
                monitor_training_accuracy=True, monitor_training_cost=True)


# test_Network()


def test_split():
    x = np.array(range(0, 9)).reshape((3, 3))
    y = np.array(range(0, 3))
    result = split_data_general(x, y)
    print(result)
    [tr_x, tr_y], [te_x, te_y] = result
    print(tr_x, tr_y, te_x, te_y)


# test_split()

def test_Model():
    y, x, _ = load_train_data()
    # tr_x, tr_y, te_x, te_y = split_data(y, x, ratio=0.7, seed=6)
    # lin_model = LinearRegression([tr_y, tr_x], validation=[te_y, te_x])
    x = standardize(x)
    lin_model = LinearRegression([y, x[0]])
    # lin_model = LogisticRegressionSK([y, x[0]])
    lin_model.train()


# test_Model()

def test1():
    data_dir = get_dataset_dir()
    train_path = os.path.join(data_dir, 'train.csv')
    print(data_dir)

    DATA_TRAIN_PATH = train_path #download train data and supply path here
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

    print(len(y))
    # data clean
    tx_clean = remove_outlier(tX)


    ''' Machien learning stuff here '''

    '''
    Potential ways to improve the results?
        Apply momentum and decays to SGD?
        Adaptive learning rate when the error plateaus
    '''
    losses, weights = least_squares_SGD(y[tx_clean,], tX[tx_clean,:], 0.1, 10)


    test_path = os.path.join(data_dir, 'test.csv')
    DATA_TEST_PATH = test_path # TODO: download train data and supply path here
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)


    OUTPUT_PATH = os.path.join(data_dir, 'output.csv') # TODO: fill in desired name of output file for submission
    y_pred = predict_labels(weights[-1], tX_test)
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

    # create_csv_submission(ids_test, y_pred, 'output1')

# test1()


# test_lsq_sgd_with_cleand_data()




