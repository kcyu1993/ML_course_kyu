from __future__ import absolute_import
from projects.project1.scripts.helpers import *
# from projects.project1.scripts.helpers import *
# from projects.project1.scripts.learning_model import *
from projects.project1.scripts.data_utils import *
from projects.project1.scripts.model import LogisticRegression
from projects.project1.scripts.network import Network
from projects.project1.scripts.plots import *
import os, datetime, sys


def test_pca_plot():
    title = "pca test"
    headers = get_csv_header(get_filepath('train'))
    _, x, _ = load_train_data(clean=False)
    pc, transfx = pca_transform(x, 2)
    # pca_plot(headers, pc[0], pc[1], title)
    pca_plot_general(headers, pc, (0, 1), title='train PC1 vs PC2')
    pca_plot_general(headers, pc, (1, 2), title='train PC1 vs PC3')
    pca_plot_general(headers, pc, (0, 2), title='train PC2 vs PC3')

    headers = get_csv_header(get_filepath('test'))
    _, x, _ = load_test_data(clean=False)
    pc, transfx = pca_transform(x, 2)
    # pca_plot(headers, pc[0], pc[1], title)
    pca_plot_general(headers, pc, (0, 1), title='test PC1 vs PC2')
    pca_plot_general(headers, pc, (1, 2), title='test PC1 vs PC3')
    pca_plot_general(headers, pc, (0, 2), title='test PC2 vs PC3')


def pca_combined():
    title = "pca test"
    headers = get_csv_header(get_filepath('train'))
    _, tr_x, _ = load_train_data(clean=False)
    tr_pc, transfx = pca_transform(tr_x, 3)

    headers = get_csv_header(get_filepath('test'))
    _, te_x, _ = load_test_data(clean=False)
    te_pc, transfx = pca_transform(te_x, 3)

    pca_plot_general(headers, tr_pc, te_pc, index=(0, 1), title='train-test PC1 vs PC2', print_name=True)
    pca_plot_general(headers, tr_pc, te_pc, index=(1, 2), title='train-test PC1 vs PC3', print_name=True)
    pca_plot_general(headers, tr_pc, te_pc, index=(0, 2), title='train-test PC2 vs PC3', print_name=True)

    pca_plot_general(headers, tr_pc, te_pc, index=(0, 1), title='train-test PC1 vs PC2 clean', print_name=False)
    pca_plot_general(headers, tr_pc, te_pc, index=(1, 2), title='train-test PC1 vs PC3 clean', print_name=False)
    pca_plot_general(headers, tr_pc, te_pc, index=(0, 2), title='train-test PC2 vs PC3 clean', print_name=False)


def test_histogram():
    title = "histotest"
    headers = get_csv_header(get_filepath('train'))
    tr_y, tr_x, _ = load_train_data(clean=False)
    te_y, te_x, _ = load_test_data(clean=False)
    # tr_x, _, _ = standardize(tr_x, intercept=False)
    # te_x, _, _ = standardize(te_x, intercept=False)
    histogram(tr_y, tr_x, headers, filename='train')
    # histogram(te_y, te_x, headers, filename='test', outlier=True)


if __name__ == '__main__':
    # test_pca_plot()
    # pca_combined()
    test_histogram()
