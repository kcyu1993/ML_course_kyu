# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
from projects.project1.scripts.data_clean import remove_outlier
import os

def get_dataset_dir():
    """
    Utility function: get the absolute dataset directory based on project dir.
    :return:
    """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    current_dir, _ = os.path.split(current_dir)
    # data_dir = os.path.join('..', current_dir)
    data_dir = os.path.join(current_dir, 'dataset')
    return data_dir

def load_train_data(sub_sample=False, clean=True, validation=False):
    filename = 'train.csv'
    if clean is True:
        filename = 'cleaned_' + filename
    path = os.path.join(get_dataset_dir(), filename)
    return load_csv_data(path, sub_sample=sub_sample, original_y=False)



def load_csv_data(data_path, sub_sample=False, original_y=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    if original_y is False:
        # convert class labels from strings to binary (-1,1)
        yb = np.ones(len(y))
        yb[np.where(y=='b')] = -1
    
    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]
    if original_y is False:
        return yb, input_data, ids
    else:
        return y, input_data, ids


def predict_labels(weights, data):
    """Generates class predictions given weights, and a test data matrix"""
    y_pred = np.dot(data, weights)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    
    return y_pred


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in csv format for submission to kaggle
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id':int(r1),'Prediction':int(r2)})


def get_csv_header(data_path):
    """
    Get header line of a given csv file, for data manipulation purpose
    :param data_path: given data path, should be absolute path to make sure.
    :return: header as a tuple
    """
    f = open(data_path, 'rb')
    reader = csv.reader(f)
    return reader.next()

def save_data_as_original_format(y, ids, x, headers, data_path):
    """
    Save the data after modification to a csv file
    :param y:       y as predictions as array
    :param ids:     id numbers as array
    :param x:       data as np.ndarray
    :param headers: headers of data objects
    :param data_path: output data path to be saved
    :return: None
    """
    with open(data_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=headers)
        writer.writeheader()
        _x = x.tolist()
        for row in zip(ids, y, _x):
            _row = row[:2] + tuple(row[2])
            writer.writerow({header: r for header, r in zip(headers,_row)})

def clean_save_data_without_invalid(filename):
    """
    Clean the data with removing all the invalid samples.
    :param filename:
    :return:
    """
    data_dir = get_dataset_dir()
    train_path = os.path.join(data_dir, filename)
    print(data_dir)

    DATA_TRAIN_PATH = train_path  # download train data and supply path here
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH, original_y=True)

    print(len(y))
    # data clean

    tx_clean = remove_outlier(tX)
    header = get_csv_header(train_path)
    cleaned_filename = 'cleaned_' + filename
    cleaned_path = os.path.join(data_dir, cleaned_filename)
    save_data_as_original_format(y[tx_clean,], ids[tx_clean,], tX[tx_clean, :], header, cleaned_path)

def clean_save_data_with_filling(filename, method='mean'):
    """
    Filling missing value with mean, median, mode, etc.
    :param filename:
    :return:
    """

