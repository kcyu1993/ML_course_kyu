# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
from projects.project1.scripts.data_clean import remove_outlier, fill_missing
import os, sys
from .helpers import standardize, split_data_general

# train_filename = 'reduced_train.csv'
# test_filename = 'reduced_test.csv'
# train_filename = 'train.csv'
# test_filename = 'train.csv'
train_filename = 'mean_fill_train.csv'
test_filename = 'mean_fill_test.csv'


# train_filename = 'reduced_mean_fill_train.csv'
# test_filename = 'reduced_mean_fill_test.csv'


# train_filename = 'cleaned_train.csv'
# test_filename = 'cleaned_test.csv'


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


def get_plot_path(filename=''):
    """
        Utility function: get the absolute dataset directory based on project dir.
        :return:
        """
    current_dir = os.path.dirname(os.path.realpath(__file__))
    current_dir, _ = os.path.split(current_dir)
    # data_dir = os.path.join('..', current_dir)
    plot_dir = os.path.join(current_dir, 'plots')
    plot_path = os.path.join(plot_dir, filename)
    return plot_path


def get_filepath(file='train'):
    '''
    Get the file path,
    :param file:        {train, test}
    :return:            return absolute path to file
    '''
    if file is 'train':
        return os.path.join(get_dataset_dir(), train_filename)
    elif file is 'test':
        return os.path.join(get_dataset_dir(), test_filename)
    else:
        raise NotImplementedError


def load_train_data(sub_sample=False, clean=True, original_y=False, validation=False):
    """
    wrapper for loading trainign data sample
    :param sub_sample:      subsample flag
    :param clean:           clean with mean-filled
    :param original_y:      return original y label
    :param validation:      validation flag. not implemented
    :return:    [tuple:]    y, input_data, ids
    """
    filename = train_filename
    if clean is True:
        filename = 'cleaned_' + filename
    path = os.path.join(get_dataset_dir(), filename)
    print('loading training data from {}'.format(path))
    return load_csv_data(path, sub_sample=sub_sample, original_y=original_y)


def load_test_data(sub_sample=False, clean=True, original_y=False, validation=False):
    """
    wrapper for loading test data sample
    :param sub_sample:      subsample flag
    :param clean:           clean with mean-filled
    :param original_y:      return original y label
    :param validation:      validation flag. not implemented
    :return:    [tuple:]    y, input_data, ids
    """
    filename = test_filename
    if clean is True:
        filename = 'cleaned_' + filename
    path = os.path.join(get_dataset_dir(), filename)
    print("loading test data from {}".format(path))
    return load_csv_data(path, sub_sample=sub_sample, original_y=original_y)


def load_train_data_neural(sub_sample=False, clean=True, validation=False, validation_ratio=0.3):
    filename = train_filename
    if clean is True:
        filename = 'cleaned_' + filename
    path = os.path.join(get_dataset_dir(), filename)
    tr_y, tr_data, ids = load_csv_data(path, sub_sample=sub_sample, original_y=False)
    tr_data, tr_mean, tr_std = standardize(tr_data, intercept=False)
    dim = tr_data.shape[1]
    training_inputs = [np.reshape(x, (dim, 1)) for x in tr_data]
    training_results = [vectorize_result(y) for y in tr_y]
    if validation is False:
        return zip(training_inputs, training_results)
    [tr_x, tr_y], [te_x, te_y] = split_data_general(training_inputs, training_results, ratio=[1 - validation_ratio])

    print("Spliting data into training and validation with size {tr}, {te}".format(tr=len(tr_x), te=len(te_x)))
    tr_data = zip(tr_x, tr_y)
    te_data = zip(te_x, te_y)
    return tr_data, te_data


def vectorize_result(y):
    """
    From [-1, 1] into binary codes.
    :param y: either -1 or 1
    :return: [1, 0] for -1, [0, 1] for 1
    """
    res = np.zeros((2, 1))
    if y == -1:
        res[0] = 1.0
    elif y == 1:
        res[1] = 1.0
    return res


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
    f = open(data_path)
    reader = csv.reader(f)
    header = next(reader)
    print(header)
    return header


def truncate_csv(line=10000):
    y, x, ids = load_train_data(clean=False, original_y=True)
    file_path = os.path.join(get_dataset_dir(), train_filename)
    header = get_csv_header(file_path)
    truncate_filename = 'reduced_' + train_filename
    truncate_path = os.path.join(get_dataset_dir(), truncate_filename)
    save_data_as_original_format(y[:line, ], ids[:line, ], x[:line, :], header, truncate_path)
    y, x, ids = load_test_data(clean=False, original_y=True)
    file_path = os.path.join(get_dataset_dir(), test_filename)
    header = get_csv_header(file_path)
    truncate_filename = 'reduced_' + test_filename
    truncate_path = os.path.join(get_dataset_dir(), truncate_filename)
    save_data_as_original_format(y[:line, ], ids[:line, ], x[:line, :], header, truncate_path)


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
    print('Writing data as original format to {}'.format(data_path))
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
    print("clean and save {} to {}".format(filename, train_path))

    DATA_TRAIN_PATH = train_path  # download train data and supply path here
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH, original_y=True)
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

    train_path = os.path.join(get_dataset_dir(), filename)
    y, tX, ids = load_csv_data(train_path, original_y=True)

    # fill the data
    clean_tx = fill_missing(tX, missing=-999.0, method=method)
    header = get_csv_header(train_path)
    cleaned_name = 'mean_fill_' + filename
    cleaned_path = os.path.join(get_dataset_dir(), cleaned_name)
    save_data_as_original_format(y, ids, clean_tx, header, cleaned_path)


class Logger(object):
    def __init__(self, filename=(get_plot_path("log/Default.log"))):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
