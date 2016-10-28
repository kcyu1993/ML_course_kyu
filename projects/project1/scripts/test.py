from __future__ import absolute_import
from projects.project1.scripts.proj1_helpers import *
# from projects.project1.scripts.helpers import *
# from projects.project1.scripts.learning_model import *
from projects.project1.scripts.data_clean import *
from projects.project1.scripts.model import LogisticRegression
from projects.project1.scripts.network import Network

import os, datetime, sys


def test_Network():
    training_data, valid_data = load_train_data_neural(validation=True, validation_ratio=0.1, clean=False)
    network = Network([30, 20, 2])
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
    logistic = LogisticRegression((y, tX[0]), regularizer="Lasso", regularizer_p=0.1)
    result = logistic.train(lr=0.05, batch_size=128, max_iters=1000)
    print(result)
    # print(losses)


def test_k_fold_logistic():
    np.set_printoptions(precision=4)
    b_time = datetime.datetime.now()
    print('Begining reading data')
    DATA_TRAIN_PATH = get_filepath('train')
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    print("Finish loading in {s} seconds".
          format(s=(datetime.datetime.now() - b_time).total_seconds()))
    tX = remove_dimensions(tX)
    tX = standardize(tX)

    e_time = datetime.datetime.now()
    print("Finish data reading in {s} seconds".
          format(s=(e_time - b_time).total_seconds()))

    # Lambda space
    lambdas = np.logspace(-3, 1, 10)
    logistic = LogisticRegression((y, tX[0]), regularizer='Lasso', regularizer_p=0.1)
    best_lambda, (tr_err, te_err) = logistic.cross_validation(5, lambdas, lambda_name='regularizer_p', max_iters=6000)
    print('best lambda {}'.format(best_lambda))
    save_path = get_plot_path(test_k_fold_logistic.__name__)
    tr_err = np.array(tr_err)
    te_err = np.array(te_err)
    np.save(save_path + "tr_err", tr_err)
    np.save(save_path + "te_err", te_err)


def test_pca_logistic():
    """
    According to the PCA first 3 component test, the selected index:
        3,8,5,9,7,10,2,1,6,0,4
        0-10
    :return:
    """


    b_time = datetime.datetime.now()
    print('Begining reading data')
    DATA_TRAIN_PATH = get_filepath('train')
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

    print("Finish loading in {s} seconds".
          format(s=(datetime.datetime.now() - b_time).total_seconds()))
    data, x_mean, x_std = standardize(tX)
    print("test bias")
    test_bias(y)
    nb_pc = 5
    print("test the PCA with {} elements".format(nb_pc))
    pcs, pc_data = pca(data, nb_pc, concatenate=False)

    print("get interactions")
    interaction = interactions(data, range(0, 10))
    interaction, _, _ = standardize(interaction)
    print("select first 10 data entry with pc data")
    data = np.c_[data[:, 0:10], pc_data]

    data = np.c_[data, interaction]
    # Begin the least square sgd
    e_time = datetime.datetime.now()
    print("Finish data reading in {s} seconds".
          format(s=(e_time - b_time).total_seconds()))
    # logistic = LogisticRegression((y, tX))
    logistic = LogisticRegression((y, data), regularizer="Lasso", regularizer_p=0.)
    # result = logistic.train(lr=0.1, batch_size=32, max_iters=6000)
    result = logistic.cross_validation(4, [0.01, 0.5, 0.1], 'regularizer_p',
                                       lr=0.1, batch_size=32, max_iters=6000, early_stop=1000)
    print(result)

    # print(losses)


def test_bias(inpt):
    target = inpt
    target[np.where(target < 0.5)] = 0
    target[np.where(target >= 0.5)] = 1
    count = np.sum(target)
    print("positive {} total {} ratio {}".format(count, len(inpt), float(count) / len(inpt)))
    return count



def test_normal():
    b_time = datetime.datetime.now()
    print('Begining reading data')
    DATA_TRAIN_PATH = get_filepath('train')
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    print("Finish loading in {s} seconds".
          format(s=(datetime.datetime.now() - b_time).total_seconds()))
    tX = standardize(tX)
    result = normaltest(tX[0])
    print(result)


def test_box_cos():
    b_time = datetime.datetime.now()
    print('Begining reading data')
    DATA_TRAIN_PATH = get_filepath('train')
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    print("Finish loading in {s} seconds".
          format(s=(datetime.datetime.now() - b_time).total_seconds()))
    tX = fill_missing(tX)
    tX = standardize(tX, intercept=False)
    header = get_csv_header(DATA_TRAIN_PATH)
    data = tX[0]
    (N, degree) = data.shape
    result = []
    best = []
    func_space = [lambda x: x, lambda x: x ** 2, lambda x: 1 / x,
                  lambda x: np.exp(x), lambda x: np.sqrt(x), lambda x: np.log(x)]
    for index in range(degree):
        # For each variable, test the
        group = []
        print("Box-cox analysis for variable {} ".format(header[index + 2]))
        _data = np.reshape(data[:, index], (N,))
        for i, func in enumerate(func_space):
            _data_trans = func(_data)
            # _result =
            group.append(normaltest(_data_trans).__getattribute__('pvalue'))
        print("p-values {}".format(group))
        print("Best transform {} with p-value {}".format(np.argmax(group), np.max(group)))
        result.append(group)
        _best = np.argmax(group)
        best.append(func_space[_best](_data))

    best = np.array(best)
    best_data = np.reshape(best, (N, degree))
    save_data_as_original_format(y, ids, best_data, header,
                                 os.path.join(get_dataset_dir(), 'trans_' + train_filename))
    print(result)


def box_cos(x, degree):
    if degree == 0:
        if (np.min(x) < 0):
            x += np.min(x) + 0.1
        return np.log(x)
    else:
        if abs(degree) < 1.0:
            if (np.min(x) < 0):
                x += np.min(x) + 0.1
        return (np.power(x, degree) - 1) / degree


if __name__ == '__main__':
    # test_box_cos()
    # test_normal()
    # sys.stdout = Logger(get_plot_path("log/test_k_fold.log"))
    test_pca_logistic()
    # truncate_csv(10000)
    # test_logistic()

    # test_k_fold_logistic()
    # clean_save_data_with_filling(train_filename)
    # clean_save_data_with_filling(test_filename)

    # test_Network()
