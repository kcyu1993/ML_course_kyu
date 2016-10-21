
import numpy as np

class Regularizer(object):

    def get_parameter(self):
        return self.p

    def set_parameter(self, p):
        self.p = p

    def get_loss(self, loss):
        return loss

    def get_gradient(self, weight):
        return 0

    @staticmethod
    def get_regularizer(regularizer=None, parameter=None):
        if regularizer is None:
            return Regularizer()
        elif regularizer is 'Ridge':
            return Ridge(parameter=parameter)
        elif regularizer is 'Lasso':
            return Lasso(parameter=parameter)
        else:
            raise NotImplementedError

class Ridge(Regularizer):

    def __init__(self, parameter):
        if parameter is None:
            self.p = 0.01
        self.p = parameter

    def get_gradient(self, weight):
        """
        This is for calculate the gradient of l2 normalization, for
        gradient descents.
        :param weight:
        :return:
        """
        return 2 * self.p * weight

    def get_linear_regression_exact(self, y, tx):
        """
        This is to get the exact solution for simple linear regression,
        for normal equation method.
        :param y:
        :param tx:
        :return:
        """
        G = np.eye(tx.shape[1])
        G[0, 0] = 0
        hes = np.dot(tx.T, tx) + self.p * G
        return np.linalg.solve(hes, np.dot(tx.T, y))

class Lasso(Regularizer):

    def __init__(self, parameter):
        if parameter is None:
            self.p = 0.01
        self.p = parameter

    def get_gradient(self, weight):
        return self.p * np.sign(weight)

