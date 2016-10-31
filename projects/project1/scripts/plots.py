# -*- coding: utf-8 -*-
"""a function of ploting figures."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from helpers import get_plot_path

""" Lab 3 """


def plot_fitted_curve(y, x, weights, ax):
    """plot the fitted curve. x, weights should align dimension """
    ax.scatter(x, y, color='b', s=12, facecolors='none', edgecolors='r')
    xvals = np.arange(min(x) - 0.1, max(x) + 0.1, 0.1)
    f = x.dot(weights)
    ax.plot(xvals, f)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Fitted curve for x y")


def plot_train_test(train_errors, test_errors, names=['', ''], xlabel='', ylabel='',
                    lambdas=None, filename=''):
    """
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a ridge regression on the train set
    * test_errors[0] = RMSE of the parameter found by ridge regression applied on the test set

    degree is just used for the title of the plot.
    """
    plt.semilogx(lambdas, train_errors, color='b', marker='*', label=names[0])
    plt.semilogx(lambdas, test_errors, color='r', marker='*', label=names[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(filename)
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    plt.show()
    plt.savefig(get_plot_path("train_test " + filename))


""" Lab 4 """


def cross_validation_visualization(params, mse_tr, mse_te, params_name='', title='', error_name=''):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(params, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(params, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("Parameters: " + params_name)
    plt.ylabel("Error: " + error_name)
    plt.title("cross validation" + title)
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig(get_plot_path("cross_validation_" + title))
    plt.show()


def bias_variance_decomposition_visualization(models, rmse_tr, rmse_te, model_names=[]):
    """visualize the bias variance decomposition."""
    rmse_tr_mean = np.expand_dims(np.mean(rmse_tr, axis=0), axis=0)
    rmse_te_mean = np.expand_dims(np.mean(rmse_te, axis=0), axis=0)
    degrees = np.array(range(len(models)))
    plt.plot(
        degrees,
        rmse_tr.T,
        'b',
        linestyle="-",
        color=([0.7, 0.7, 1]),
        label='train',
        linewidth=0.3)
    plt.plot(
        degrees,
        rmse_te.T,
        'r',
        linestyle="-",
        color=[1, 0.7, 0.7],
        label='test',
        linewidth=0.3)
    plt.plot(
        degrees,
        rmse_tr_mean.T,
        'b',
        linestyle="-",
        label='train',
        linewidth=3)
    plt.plot(
        degrees,
        rmse_te_mean.T,
        'r',
        linestyle="-",
        label='test',
        linewidth=3)
    # plt.ylim(0.2, 0.7)
    plt.xlabel("degree")
    plt.ylabel("error")
    plt.title("Bias-Variance Decomposition")
    plt.savefig(get_plot_path("bias_variance"))
    plt.show()


def pca_plot(headers, pc1, pc2, title=''):
    """ Plot pca accordingly """
    fig, ax = plt.subplots()
    ax.scatter(pc1, pc2)
    for i, name in enumerate(headers[2:]):
        ax.annotate(name, (pc1[i], pc2[i]))

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("PCA plot for data")
    fig.show()
    fig.savefig(get_plot_path("pca_plot " + title))


def pca_plot_general(headers, pcs, pcs2=None, index=(0, 1), title='', color=['b', 'r'], print_name=False):
    pc1 = pcs[index[0]]
    pc2 = pcs[index[1]]
    fig, ax = plt.subplots()
    sct = ax.scatter(pc1, pc2, color=color[0])
    for i, name in enumerate(headers[2:]):
        if not print_name:
            ax.annotate('{}'.format(i - 2), (pc1[i], pc2[i]))
        else:
            ax.annotate('{}-'.format(i - 2) + name, (pc1[i], pc2[i]))

    if pcs2 is not None:
        fig.hold()
        pc1 = pcs2[index[0]]
        pc2 = pcs2[index[1]]
        sct2 = ax.scatter(pcs2[index[0]], pcs2[index[1]], color=color[1])
        for i, name in enumerate(headers[2:]):
            if not print_name:
                ax.annotate('{}'.format(i - 2), (pc1[i], pc2[i]))
            else:
                ax.annotate('{}-'.format(i - 2) + name, (pc1[i], pc2[i]))
        fig.legend((sct, sct2), ('train', 'test'))

    ax.set_xlabel("PC {}".format(index[0]))
    ax.set_ylabel("PC {}".format(index[1]))
    ax.set_title("PCA plot for data")
    fig.show()
    fig.savefig(get_plot_path("pca_plot " + title))


def histogram(label, data, headers=None, colors=['b', 'r'], print_name=True, transform=None, filename='Default.plt',
              outlier=False):
    """
    Build up histogram regarding to labels, via each dimensions.
    Stored in the path: plots/histogram
    :param ids:         index
    :param label:       y
    :param data:        data matrix
    :param headers:     headers accordingly
    :param print_name:  print name on the histogram
    :return:
    """
    hist_path = get_plot_path() + '/histogram/'
    # Generate positive and negative index
    negative_index = np.where(label < 0)[0]
    positive_index = np.where(label > 0)[0]
    nega_data = data[negative_index, :]
    posi_data = data[positive_index, :]
    if transform is None:
        transform = [lambda x: x, lambda x: np.log(x + 0.01 - np.min(x)), lambda x: np.sqrt(np.abs(x)),
                     lambda x: np.power(x, 2)]
    trans_labels = ['linear', 'log', 'sqrt|abs|', 'power']
    # Plot according to each dimensions
    headers = headers[2:]
    # Hard coded
    gs = gridspec.GridSpec(2, 2)
    assert len(headers) == len(data[0])
    for index, header in enumerate(headers):
        # fig, axs = plt.subplots(1, len(transform))
        for f_ind, f_trans in enumerate(transform):
            ax = plt.subplot(gs[int(f_ind / 2), f_ind % 2])
            ax.set_aspect('auto')
            if outlier:
                ax.hist(f_trans(nega_data[:, index]).T, bins=100, color=colors[0], alpha=0.5)
            else:
                # nega_ind = np.where(nega_data[:, index] != -999.0)
                ax.hist(f_trans(nega_data[np.where(nega_data[:, index] != -999.0)[0], index]).T,
                        bins=100, color=colors[0], alpha=0.5)
            ax.hold(True)
            if outlier:
                ax.hist(f_trans(posi_data[:, index]).T, bins=100, color=colors[1], alpha=0.8)
            else:
                ax.hist(f_trans(posi_data[np.where(posi_data[:, index] != -999.0)[0], index]).T,
                        bins=100, color=colors[1], alpha=0.8)
            if print_name:
                ax.set_xlabel("{}({})".format(trans_labels[f_ind], header))
            else:
                ax.set_xlabel(trans_labels[f_ind])
            ax.hold(False)
        plt.savefig(hist_path + "{}-{}_{}.png".format(filename, index, header))
        plt.close()
