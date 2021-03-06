"""
This python file contains every test case based on our logistic
regression model.
The best selected model is put inside the implementations.py
You could skip this class, if you want to run one of the method,
change the last __main__ method.
We provide you a sample run, you could directly run this in your 
terminal.

python \path\to\test.py 

"""

from __future__ import absolute_import
from data_utils import *
from helpers import *
from implementations import *
from model import LogisticRegression
import os, datetime



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
    pcs, pc_data = pca_transform(data, nb_pc, concatenate=False)

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
    result = logistic.cross_validation(4, [0.5], 'regularizer_p',
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


def test_box_cos():
    """
    Potentially testing box_cox transformation of data entry.
    Discarded.
    :return:
    """
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
            # group.append(normaltest(_data_trans).__getattribute__('pvalue'))
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


def test_complex():
    b_time = datetime.datetime.now()
    print('Begining reading data')
    DATA_TRAIN_PATH = get_filepath('train')
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    print("Finish loading in {s} seconds".
          format(s=(datetime.datetime.now() - b_time).total_seconds()))
    tX, _, _ = standardize(tX, intercept=False)
    complex_tx, _, _ = compose_complex_features(tX, intercept=True,
                                                interaction=True,
                                                log=True,
                                                sqrt=False,
                                                pca=True)
    test_bias(y)
    logistic = LogisticRegression((y, complex_tx), regularizer="Lasso", regularizer_p=0.5)
    # result = logistic.train(lr=0.1, batch_size=32, max_iters=6000)
    result = logistic.cross_validation(4, [0.5], 'regularizer_p',
                                       lr=0.1, batch_size=32, max_iters=6000, early_stop=1000)


def test_draw():
    """
    Draw balanced sample, but result worse result.
    """
    b_time = datetime.datetime.now()
    print('Begining reading data')
    DATA_TRAIN_PATH = get_filepath('train')
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    print("Finish loading in {s} seconds".
          format(s=(datetime.datetime.now() - b_time).total_seconds()))

    # data, _, _ = standardize(tX)

    # test_bias(y)
    # nb_pc = 5
    # print("test the PCA with {} elements".format(nb_pc))
    # pcs, pc_data = pca_transform(data, nb_pc, concatenate=False)
    #
    # print("get interactions")
    # interaction = interactions(data, range(0, 10))
    # interaction, _, _ = standardize(interaction)
    # print("select first 10 data entry with pc data")
    # data = np.c_[data[:, 0:10], pc_data]
    # data = np.c_[data, interaction]
    # # Begin the least square sgd
    # e_time = datetime.datetime.now()
    # print("Finish data reading in {s} seconds".
    #       format(s=(e_time - b_time).total_seconds()))
    data, _, _ = compose_complex_features_further(tX, intercept=True,
                                                  interaction=True,
                                                  log=True,
                                                  sqrt=True,
                                                  power=True,
                                                  pca=True)
    train, valid = draw_balanced_subsample(y, data, trainsize=6000)
    # t_data, _, _ = compose_complex_features_further(test_x, intercept=True,
    #                                                 interaction=True,
    #                                                 log=True,
    #                                                 sqrt=True,
    #                                                 power=True,
    #                                                 pca=True)


    logistic = LogisticRegression(train=train, validation=valid, regularizer='Lasso', regularizer_p=0.5)
    result = logistic.train(lr=0.01, decay=0.5, early_stop=400, max_iters=2000)
    print(result)


def test_final():
    b_time = datetime.datetime.now()
    print('Begining reading data')
    DATA_TRAIN_PATH = get_filepath('train')
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    print("Finish loading in {s} seconds".
          format(s=(datetime.datetime.now() - b_time).total_seconds()))

    data, _, _ = standardize(tX)

    nb_pc = 5
    print("test the PCA with {} elements".format(nb_pc))
    pcs, pc_data = pca_transform(data, nb_pc, concatenate=False)

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
    # train, valid = split_train_valid(0.8, data, labels=y)
    logistic = LogisticRegression((y, data), regularizer='Lasso', regularizer_p=0.)
    result = logistic.cross_validation(4, [0.], 'regularizer_p',
                                       lr=0.1, batch_size=32, max_iters=1200, early_stop=400)
    weight = result[0]

    print("loading the test set")
    _, test_data, test_ids = load_test_data(clean=False)
    # Feature transform
    data, _, _ = standardize(test_data)
    nb_pc = 5
    print("test the PCA with {} elements".format(nb_pc))
    pcs, pc_data = pca_transform(data, nb_pc, concatenate=False)

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
    y_pred = []
    for w in weight:
        _y_pred = logistic.__call__(data, w)
        y_pred += _y_pred
    y_pred = np.average(y_pred)
    y_pred[np.where(y_pred <= 0.5)] = -1
    y_pred[np.where(y_pred > 0.5)] = 1
    output_path = get_dataset_dir() + '/second_submission.csv'
    create_csv_submission(test_ids, y_pred, output_path)


def test_pca_logistic2():
    """
    According to the PCA first 3 component test, the selected index:
        3,8,5,9,7,10,2,1,6,0,4
        0-10
    :return:
    """
    print('Submission added test full')

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
    pcs, pc_data = pca_transform(data, nb_pc, concatenate=False)

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
    result = logistic.cross_validation(4, [0.5], 'regularizer_p',
                                       lr=0.1, batch_size=32, max_iters=1000,
                                       early_stop=1000, skip=True)

    weight = result[0]
    _, test_x, test_ids = load_test_data(clean=False)
    test_data, x_mean, x_std = standardize(test_x)
    pcs, pc_data = pca_transform(test_data, nb_pc, concatenate=False)

    print("get interactions")
    interaction = interactions(test_data, range(0, 10))
    interaction, _, _ = standardize(interaction)
    print("select first 10 data entry with pc data")
    test_data = np.c_[test_data[:, 0:10], pc_data]
    test_data = np.c_[test_data, interaction]

    y_pred = []
    for w in weight:
        _y_pred = logistic.__call__(test_data, w)
        y_pred.append(_y_pred)
    y_pred = np.average(y_pred, axis=0)
    y_pred[np.where(y_pred <= 0.5)] = -1
    y_pred[np.where(y_pred > 0.5)] = 1
    output_path = get_dataset_dir() + \
                  '/submission/pca_test{}.csv'.format(
                      datetime.datetime.now().__str__())
    create_csv_submission(test_ids, y_pred, output_path)
    # print(losses)


def test_baseline():
    print("base line testing")
    b_time = datetime.datetime.now()
    print('Begining reading data')
    DATA_TRAIN_PATH = get_filepath('train')
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

    print("Finish loading in {s} seconds".
          format(s=(datetime.datetime.now() - b_time).total_seconds()))
    data = baseline_logistic(tX)

    logistic = LogisticRegression((y, data))
    weight = logistic.train(lr=0.01, decay=1)

    plot = True
    if plot:
        from plots import cross_validation_visualization

    _, test_x, test_ids = load_test_data(clean=False)
    t_data = baseline_logistic(test_x)
    pred_label = predict_labels(weight, t_data)
    create_csv_submission(test_ids, pred_label, get_dataset_dir() + '/submission/logistic_baseline.csv')


def test_data_model():
    title = 'complex_full_before_ddl_interactions_full'
    print("Base line testing for model " + title)
    b_time = datetime.datetime.now()
    print('Beginning reading data')
    DATA_TRAIN_PATH = get_filepath('train')
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    print("Finish loading in {s} seconds".
          format(s=(datetime.datetime.now() - b_time).total_seconds()))
    _, test_x, test_ids = load_test_data(clean=False)


    data = compose_interactions_for_transforms(tX)
    t_data = compose_interactions_for_transforms(test_x)

    # Test 1 Ridge 0.1
    logistic = LogisticRegression((y, data), regularizer='Ridge', regularizer_p=0.1)
    weight = logistic.train(lr=0.01, decay=0.5, max_iters=2000, early_stop=1000, decay_intval=100)
    # weight, _, _ = logistic.cross_validation(4, [0.1, 0.5, 0.05], 'regularizer_p', max_iters=2000)
    pred_label = predict_labels(weight, t_data)
    create_csv_submission(test_ids, pred_label, get_dataset_dir() +
                          '/submission/removed_outlier_{}.csv'.format(title + 'Ridge01'))

    # Test 2 Lasso 0.1
    logistic = LogisticRegression((y, data), regularizer='Lasso', regularizer_p=0.1)
    weight = logistic.train(lr=0.01, decay=0.5, max_iters=2000, early_stop=1000, decay_intval=100)
    # weight, _, _ = logistic.cross_validation(4, [0.1, 0.5, 0.05], 'regularizer_p', max_iters=2000)
    pred_label = predict_labels(weight, t_data)
    create_csv_submission(test_ids, pred_label, get_dataset_dir() +
                          '/submission/removed_outlier_{}.csv'.format(title + '-Lasso0.1'))

    # Test 3 No penalized
    logistic = LogisticRegression((y, data))
    weight = logistic.train(lr=0.01, decay=0.5, max_iters=2000, early_stop=1000)
    # weight, _, _ = logistic.cross_validation(4, [0.1, 0.5, 0.05], 'regularizer_p', max_iters=2000)
    pred_label = predict_labels(weight, t_data)
    create_csv_submission(test_ids, pred_label, get_dataset_dir() +
                          '/submission/removed_outlier_{}.csv'.format(title))


def test_implementations():
    title = 'final baseline for '
    print("Base line testing for model " + title)
    b_time = datetime.datetime.now()
    print('Beginning reading data')
    DATA_TRAIN_PATH = get_filepath('train')
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    print("Finish loading in {s} seconds".
          format(s=(datetime.datetime.now() - b_time).total_seconds()))
    _, test_x, test_ids = load_test_data(clean=False)

    data = standardize(tX)[0]
    t_data = standardize(test_x)[0]

    # Test 1 Linear least squares
    weight, _ = least_squares(y, tX)
    pred_label = predict_labels(weight, t_data)
    create_csv_submission(test_ids, pred_label, get_dataset_dir() +
                          '/submission/{}.csv'.format(title + 'least_squares'))

    # Test 2 Least_squares_GD
    weight = least_squares_GD(y, tX, gamma=0.01, max_iters=1000)
    pred_label = predict_labels(weight, t_data)
    create_csv_submission(test_ids, pred_label, get_dataset_dir() +
                          '/submission/{}.csv'.format(title + 'least_squaresGD'))

    # Test 3 Least_squares_SGD
    weight = least_squares_SGD(y, tX, gamma=0.01, max_iters=1000)
    pred_label = predict_labels(weight, t_data)
    create_csv_submission(test_ids, pred_label, get_dataset_dir() +
                          '/submission/{}.csv'.format(title + 'least_squaresSGD'))

    # Test 4 Ridge_regression
    weight, _ = ridge_regression(y, tX, lamb=0.00001)
    pred_label = predict_labels(weight, t_data)
    create_csv_submission(test_ids, pred_label, get_dataset_dir() +
                          '/submission/{}.csv'.format(title + 'Ridge_regression'))


if __name__ == '__main__':
    # test_box_cos()
    # test_normal()
    # sys.stdout = Logger(get_plot_path("log/test_k_fold.log"))
    # test_pca_logistic2()
    # truncate_csv(10000)
    test_complex()
    # test_draw()
    # test_final()
    # test_logistic()
    # test_baseline()
    # test_data_model()
    # test_cross_valid()
    # test_implementations()
    # test_k_fold_logistic()
    # clean_save_data_with_filling(train_filename)
    # clean_save_data_with_filling(test_filename)