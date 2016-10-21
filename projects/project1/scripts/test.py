from __future__ import absolute_import
from projects.project1.scripts.proj1_helpers import *
# from projects.project1.scripts.helpers import *
# from projects.project1.scripts.learning_model import *
# from projects.project1.scripts.data_clean import *
# from projects.project1.scripts.model import *
from projects.project1.scripts.network import Network
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
    test_predict = network.predict(test_data)
    print("finish {} prediction and saving results".format(len(test_predict)))
    create_csv_submission(ids, test_predict, 'neural-mean_fill_output.csv')


def test_logistic():

# test_Network()

# truncate_csv(1000)
# clean_save_data_with_filling(train_filename)
# clean_save_data_with_filling(test_filename)

# test_Network()
