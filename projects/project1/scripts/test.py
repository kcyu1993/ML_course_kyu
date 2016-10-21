from __future__ import absolute_import
from projects.project1.scripts.proj1_helpers import *
# from projects.project1.scripts.helpers import *
# from projects.project1.scripts.learning_model import *
# from projects.project1.scripts.data_clean import *
from projects.project1.scripts.model import LogisticRegression
from projects.project1.scripts.network import Network
from projects.project1.scripts.test_logistic import test
import os, datetime


def test_Network():
    training_data, valid_data = load_train_data_neural(validation=True, validation_ratio=0.1, clean=False)
    network = Network([30, 100, 20, 2])
    result = network.SGD(training_data=training_data, epochs=50, lr=0.1, mini_batch_size=32,
                         momentum=0.8, regular_p=0.1, no_improve=10, halve_learning_rate=16,
                         evaluation_data=valid_data,
                         monitor_evaluation_accuracy=True, monitor_evaluation_cost=True,
                         monitor_training_accuracy=True, monitor_training_cost=True)
    _, test_data, ids = load_test_data(clean=False)

    print("predicting the test data")
    test_predict = network.predict(test_data.T)
    print("finish {} prediction and saving results".format(len(test_predict)))
    create_csv_submission(ids, test_predict, 'neural-mean_fill_output.csv')


def test_lsq_sgd_with_cleand_data():
    b_time = datetime.datetime.now()
    print('Begining reading data')
    data_dir = get_dataset_dir()
    train_path = os.path.join(data_dir, 'cleaned_train.csv')
    DATA_TRAIN_PATH = train_path  # download train data and supply path here
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    print("Finish loading in {s} seconds".
          format(s=(datetime.datetime.now() - b_time).total_seconds()))
    tX = standardize(tX)
    # Begin the least square sgd
    e_time = datetime.datetime.now()
    print("Finish data reading in {s} seconds".
          format(s=(e_time - b_time).total_seconds()))
    # losses, weights = least_squares_SGD(y, tX[0],
    #                                     gamma=0.1, max_iters=10, batch_size=16)
    # print(losses)

def test_logistic():
    b_time = datetime.datetime.now()
    print('Begining reading data')
    DATA_TRAIN_PATH = get_filepath('train')
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    print("Finish loading in {s} seconds".
          format(s=(datetime.datetime.now() - b_time).total_seconds()))
    tX = standardize(tX)
    # Begin the least square sgd
    e_time = datetime.datetime.now()
    print("Finish data reading in {s} seconds".
          format(s=(e_time - b_time).total_seconds()))
    logistic = LogisticRegression((y, tX[0]), regularizer="Ridge", regularizer_p=0.1)
    result = logistic.train(lr=0.1, batch_size=32, max_iters=1000)
    print(result)
    # print(losses)


test_logistic()

# truncate_csv(1000)
# clean_save_data_with_filling(train_filename)
# clean_save_data_with_filling(test_filename)

# test_Network()
# test()
