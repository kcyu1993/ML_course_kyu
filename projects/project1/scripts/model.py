from __future__ import absolute_import
from abc import ABCMeta, abstractmethod
import numpy as np
from .gradient import *
from .costs import *
from .helpers import batch_iter, build_k_indices
from .learning_model import ridge_regression


# Test
from sklearn import svm
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.feature_selection import RFECV


class Model(object):
    def __init__(self, train_data, validation=None):
        """
        Initializer of all learning models.
        :param train_data: training data.
        :param validation_data:
        """
        self.train_x = train_data[1]
        self.train_y = train_data[0]

        # Set validation here.
        self.validation = False
        if validation is not None:
            self.valid_x = validation[1]
            self.valid_y = validation[0]
            self.validation = True
            self.valid_losses = []
            self.valid_misclass_rate = []

        ''' Define the progress of history here '''
        self.losses = []
        self.iterations = 0
        self.weights = []
        self.misclass_rate = []
        self.loss_function = None
        self.loss_function_name = None

    @abstractmethod
    def __call__(self, **kwargs):
        """Define the fit function and get prediction"""
        raise NotImplementedError

    @abstractmethod
    def get_gradient(self, y, x, weight):
        raise NotImplementedError

    @abstractmethod
    def predict(self, x, weight):
        raise NotImplementedError

    @abstractmethod
    def compute_weight(self, y, tx, lamb):
        raise NotImplementedError

    def get_history(self):
        """
        Get the training history of current model
        :return: list as [iterations, [losses], [weights], [mis_class]]
        """
        return self.iterations, self.losses, self.weights, self.misclass_rate

    def train(self, optimizer='sgd', cross_valid=0, loss_function='mse', **kwargs):
        self.loss_function = get_loss_function(loss_function)
        self.loss_function_name = loss_function
        if cross_valid == 0:
            if optimizer is 'sgd':
                self.sgd()
            elif optimizer is 'gd':
                self.gd()
        elif cross_valid > 0:
            self.cross_validadtion(cross_valid)


    """ Begining of the optimize Routines """

    def sgd(self, lr=0.001, momentum=0.9, decay=0.2, max_iters=100, batch_size=10):
        '''Define the SGD algorithm here'''
        w = self.weights[0]
        for epoch in range(max_iters):
            print('Epoch {e} in {m}'.format(e=epoch+1, m=max_iters), end="\t")
            for batch_y, batch_x in batch_iter(self.train_y, self.train_x, batch_size):
                grad = self.get_gradient(batch_y, batch_x, w)
                w = w - lr * grad
            loss = self.compute_loss(self.train_y, self.train_x, w)
            mis_class = self.compute_metrics(self.train_y, self.train_x, w)

            self.weights.append(w)
            self.losses.append(loss)
            self.misclass_rate.append(mis_class)
            if self.validation is True:
                valid_loss = self.compute_loss(self.valid_y, self.valid_x, w)
                valid_mis_class = self.compute_metrics(self.valid_y, self.valid_x, w)
                self.valid_losses.append(valid_loss)
                self.valid_misclass_rate.append(valid_mis_class)
                print('Train Loss {t_l}, Train mis-class {t_m}, valid loss {v_l}, valid mis-class {v_m}'.
                      format(t_l=loss, t_m=mis_class, v_l=valid_loss, v_m=valid_mis_class))
            else:
                print('Train Loss {t_l}, Train mis-class {t_m}'.
                      format(t_l=loss, t_m=mis_class))

    def normalequ(self):
        '''Define Gradient descent here'''
        raise NotImplementedError

    def cross_validation(self, cv, lambdas, *args):
        """
        Cross validation method to acquire the best prediction parameters.
        It will use the train_x y as data and do K-fold cross validation.
        :param cv:
        :param lambdas:
        :param args:
        :return:
        """
        k_indices = build_k_indices(y, cv)
        # define lists to store the loss of training data and test data
        mse_tr = []
        mse_te = []
        self.lambdas = lambdas

        for lamb in self.lambdas:
            _mse_tr = []
            _mse_te = []
            for k in range(cv):
                loss_tr, loss_te = self._loop_cross_validation(self.train_y, self.train_x, k_indices, k, lamb)
                _mse_tr += [loss_tr]
                _mse_te += [loss_te]
            avg_tr = np.average(_mse_tr)
            avg_te = np.average(_mse_te)
            mse_tr += [avg_tr]
            mse_te += [avg_te]

        # Select the best parameter during the cross validations.
        print('K-fold cross validation result: ', mse_tr, mse_te)

    def _loop_cross_validation(self,y, x, k_indices, k, lamb):
        train_ind = np.concatenate((k_indices[:k], k_indices[k + 1:]), axis=0)
        train_ind = np.reshape(train_ind, (train_ind.size,))

        test_ind = k_indices[k]
        # Note: different from np.ndarray, tuple is name[index,]
        # ndarray is name[index,:]
        train_x = x[train_ind,]
        train_y = y[train_ind,]
        test_x = x[test_ind,]
        test_y = y[test_ind,]
        weight = self.compute_weight(train_y, train_x, lamb)

        # Compute the metrics and return
        loss_tr = self.compute_loss(train_y, train_x, weight)
        loss_te = self.compute_loss(test_y, test_x, weight)

        return weight, loss_tr, loss_te


    def compute_metrics(self, target, data, weight):
        """
        Compute the following metrics
                Misclassification rate
        """
        pred = self.predict(data, weight)
        assert len(pred) == len(target)
        # Calculate the mis-classification rate:
        N = len(pred)
        nb_misclass = np.count_nonzero(target - pred)
        return nb_misclass / N

    def compute_loss(self, y, x, weight):
        return self.loss_function(y, x, weight)



class LinearRegression(Model):


    def __init__(self, train, validation=None, initial_weight=None, regularizer=None, regularizer_p=None):
        # Initialize the super class with given data.
        super(LinearRegression, self).__init__(train, validation)
        degree = self.train_x.shape[1]

        # Initialize the weight for linear model.
        if initial_weight is not None:
            self.weights.append(initial_weight)
        else:
            self.weights.append(np.random.rand(degree))

        self.regularizer = regularizer
        self.regularizer_p = regularizer_p
        if self.regularizer is not None:
            if self.regularizer_p is None:
                self.regularizer_p = 0.01



    def __call__(self, x):
        return np.dot(x, self.weights[-1])

    def get_gradient(self, batch_y, batch_x, weight):
        N = batch_y.shape[0]
        grad = np.empty(len(weight))
        for index in range(N):
            _y = batch_y[index]
            _x = batch_x[index]
            grad = grad + gradient_least_square(_y, _x, weight, self.loss_function_name)
        grad /= N
        return grad / N

    def predict(self, x, weight):
        """Prediction function"""
        pred = np.dot(x, weight)
        pred[np.where(pred <= 0)] = -1
        pred[np.where(pred > 0)] = 1
        return pred

class RidgeRegression(Model):

    def __init__(self, train, lambdas=np.logspace(-4,2,30), validation=None):
        # Initialize the super class with given data.
        super(RidgeRegression, self).__init__(train, validation)
        degree = self.train_x.shape[1]
        # no need to init the weights. Instead, choose the lambda.
        self.lambdas = lambdas


    def __call__(self, x):
        return np.dot(x, self.weights[-1])

    def get_gradient(self, batch_y, batch_x, weight):
        raise NotImplementedError

    def predict(self, x, weight):
        """Prediction function"""
        pred = np.dot(x, weight)
        pred[np.where(pred <= 0)] = -1
        pred[np.where(pred > 0)] = 1
        return pred

    def train(self, optimizer=None, cross_valid=4, loss_function='mse', **kwargs):
        """
        Override the train function, to a simple direct solution with selection based on
        misclassification rate, with default cross validation
        :param optimizer:
        :param cross_valid:
        :param loss_function:
        :param kwargs:
        :return:
        """



    def compute_weight(self, y, x, lamb):
        _, weight = ridge_regression(y, x, lamb)
        return weight




class SupportVectorMachineSK(Model):
    def __init__(self, train, validation=None):
        super(SupportVectorMachineSK, self).__init__(train, validation)
        self.clf = svm.SVC()

    def get_gradient(self, y, x, weight):
        raise NotImplementedError

    def predict(self, x, weight):
        pred = self.clf.predict(x)
        pred[np.where(pred <= 0)] = -1
        pred[np.where(pred > 0)] = 1
        return pred

    def train(self, optimizer='sgd', cross_valid=0, loss_function='mse', **kwargs):
        self.clf.fit(self.train_x, self.train_y)
        mis_class = self.compute_metrics(self.train_y, self.train_x, None)
        self.misclass_rate.append(mis_class)
        print('Misclassification rate for SVM {m}'.format(m=mis_class))

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

class LogisticRegressionSK(Model):
    def __init__(self, train, validation=None):
        super(LogisticRegressionSK, self).__init__(train, validation)
        self.clf = SGDClassifier(loss='log')

    def get_gradient(self, y, x, weight):
        raise NotImplementedError

    def predict(self, x, weight):
        pred = self.clf.predict(x)
        # pred[np.where(pred <= 0)] = -1
        # pred[np.where(pred > 0)] = 1
        return pred

    def train(self, optimizer='sgd', cross_valid=0, loss_function='mse', **kwargs):

        estimator = svm.SVR(kernel='linear')
        selector = RFECV(estimator=estimator, step=1, cv=5)
        selector = selector.fit(self.train_x, self.train_y)
        print(selector.support_)
        selected = []
        for index, select in enumerate(selector.support_):
            if select[index] is True:
                selected.append(index)
        print(selected)

        self.clf.fit(self.train_x[:,selected], self.train_y)
        mis_class = self.compute_metrics(self.train_y, self.train_x, None)
        self.misclass_rate.append(mis_class)
        print('Misclassification rate for Logistic regression {m}'.format(m=mis_class))

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
