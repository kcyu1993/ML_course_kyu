import numpy as np

class Regularizer(object):
    """
    Implement L1 L2 regularizer, define loss, gradient for each regularizer
    Implement static method get_regularizer to create a regularizer and return
    """
    def get_parameter(self):
        return self.p

    def set_parameter(self, p):
        self.p = p

    def get_loss(self, loss):
        return loss

    def get_gradient(self, weight):
        """ Return 0 for no regularize effect """
        return 0

    @staticmethod
    def get_regularizer(regularizer=None, parameter=None):
        """
        Get specific regularizer, if no regularizer specified,
        it would return a Regularizer object which would always
        have 0 gradient during training, i.e. no penalized.

        :param regularizer: string, 'Ridge', 'Lasso' or None
        :param parameter:   given parameter of initial setup
        :return:
        """
        if regularizer is None:
            return Regularizer()
        elif regularizer is 'Ridge':
            return Ridge(parameter=parameter)
        elif regularizer is 'Lasso':
            return Lasso(parameter=parameter)
        else:
            raise NotImplementedError


class Ridge(Regularizer):
    """
    L2, Ridge regularizer.
    """
    def __init__(self, parameter):
        self.p = parameter
        self.name = 'Ridge'
        if parameter is None:
            self.p = 0.01

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
        For numerical stability, the first entry of Eye matrix is set to 0
        :param y:
        :param tx:
        :return:
        """
        G = np.eye(tx.shape[1])
        G[0, 0] = 0
        hes = np.dot(tx.T, tx) + self.p * G
        return np.linalg.solve(hes, np.dot(tx.T, y))


class Lasso(Regularizer):
    """ Lasso regularizer """
    def __init__(self, parameter):
        self.name = Lasso
        if parameter is None:
            self.p = 0.01
        self.p = parameter

    def get_gradient(self, weight):
        return self.p * np.sign(weight)

