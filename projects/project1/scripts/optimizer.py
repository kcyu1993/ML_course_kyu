import numpy as np

class Optimizer(object):
    '''
    Class for optimizer, including SGD, GD for the time being
    Data structures:
        weights, updates shall be in form of np.ndarray.
    '''
    def __init__(self):
        ''' initialize function '''
        self.updates = []
        self.weights = []

    def get_state(self):
        '''
        Get the state of current training
        :return: updates as np.ndarray
        '''
        return self.updates

    def set_state(self, values):
        # assert len(self.updates) == len(values)
        assert self.updates.shape == values.shape
        self.updates = values

    def set_weights(self, weights):
        assert self.weights.shape == weights.shape
        self.weights = weights

    def get_updates(self):
        return self.updates

    def set_gradient_function(self, grad_func):
        self.gradient_function = grad_func

    def get_gradients(self, loss, weights):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, lr=0.1, momentum=0.0, decay=0.0):
        super(SGD, self).__init__()
        self.iterations = 0
        self.lr = lr
        self.momentum = momentum
        self.decay = decay

    def get_updates(self, cost, ):
        '''
        Calculate the corresponding change in weight vector.
        :param params:
        :param constraints:
        :param loss:
        :return:
        '''
        # grads =
        self.updates = []

# class GD(Optimizer):
