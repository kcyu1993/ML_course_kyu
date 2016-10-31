from __future__ import absolute_import

import copy
from data_utils import build_k_indices
from learning_model import *
from regularizer import *
from helpers import save_numpy_array

class Model(object):
    """
    Machine learning model engine
    Implement the optimizers
        sgd
        normal equations
        cross-validation of given parameters
    Support:
        L1, L2 normalization
    """
    def __init__(self, train_data, validation=None, initial_weight=None,
                 loss_function_name='mse', cal_weight='gradient',
                 regularizer=None, regularizer_p=None):
        """
        Initializer of all learning models.
        :param train_data: training data.
        :param validation_data:
        """
        self.train_x = train_data[1]
        self.train_y = train_data[0]

        self.set_valid(validation)

        ''' Define the progress of history here '''
        self.losses = []
        self.iterations = 0
        self.weights = []
        self.misclass_rate = []

        ''' Define loss, weight calculation, regularizer '''
        self.loss_function = get_loss_function(loss_function_name)
        self.loss_function_name = loss_function_name
        self.calculate_weight = cal_weight
        self.regularizer = Regularizer.get_regularizer(regularizer, regularizer_p)
        self.regularizer_p = regularizer_p

        # Asserting degree
        if len(self.train_x.shape) > 1:
            degree = self.train_x.shape[1]
        else:
            degree = 1

        # Initialize the weight for linear model.
        if initial_weight is not None:
            self.weights.append(initial_weight)
        else:
            self.weights.append(np.random.rand(degree))

    def set_valid(self, validation):
        # Set validation here.
        self.validation = False
        self.valid_x = None
        self.valid_y = None
        self.valid_losses = None
        self.valid_misclass_rate = None
        if validation is not None:
            (valid_y, valid_x) = validation
            self.valid_x = valid_x
            self.valid_y = valid_y
            self.validation = True
            self.valid_losses = []
            self.valid_misclass_rate = []

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
    def normalequ(self, **kwargs):
        ''' define normal equation method to calculate optimal weights'''
        raise NotImplementedError

    def compute_weight(self, y, x, test_x=None, test_y=None, **kwargs):
        """ Return weight under given parameter """
        model = copy.copy(self)
        model.__setattr__('train_y', y)
        model.__setattr__('train_x', x)
        if test_x is not None and test_y is not None:
            model.set_valid((test_y, test_x))
        _kwargs = []
        for name, value in kwargs.items():
            # Recognize parameter "
            if name is "regularizer_p":
                model.__setattr__(name, value)
                model.regularizer.set_parameter(value)
            else:
                _kwargs.append((name, value))
        _kwargs = dict(_kwargs)
        if model.calculate_weight is 'gradient':
            return model.sgd(**_kwargs)
        # elif model.calculate_weight is 'newton':
        #     return model.newton(**_kwargs)
        elif model.calculate_weight is 'normalequ':
            return model.normalequ(**_kwargs)

    def get_history(self):
        """
        Get the training history of current model
        :return: list as [iterations, [losses], [weights], [mis_class]]
        """
        if self.validation:
            return self.iterations, (self.losses, self.valid_losses), \
                   (self.weights), (self.misclass_rate, self.valid_misclass_rate)
        return self.iterations, self.losses, self.weights, self.misclass_rate

    def train(self, optimizer='sgd', loss_function='mse', **kwargs):
        """
        Train function to perform one time training
        Will based optimizer to select.
            TODO: Would add 'newton' in the future
        This
        :param optimizer: only support 'sgd'
        :param loss_function: loss_function name {mse, mae, logistic}
        :param kwargs: passed into sgd
        :return: best weight
        """
        self.loss_function = get_loss_function(loss_function)
        self.loss_function_name = loss_function

        if optimizer is 'sgd':
            self.sgd(**kwargs)

        return self.weights[-1]

    """===================================="""
    """ Beginning of the optimize Routines """
    """===================================="""
    def sgd(self, lr=0.01, momentum=0.9, decay=0.5, max_iters=1000,
            batch_size=128, early_stop=150, decay_intval=50, decay_lim=9):
        """
        Define the SGD algorithm here

        :param lr:          learning rate
        :param momentum:    momentum TODO
        :param decay:       weight decay after fix iterations
        :param max_iters:   maximum iterations
        :param batch_size:  batch_size
        :param early_stop:  early_stop after no improvement
        :return: final weight vector
        """
        np.set_printoptions(precision=4)
        w = self.weights[0]
        loss = self.compute_loss(self.train_y, self.train_x, w)
        best_loss = loss
        best_counter = 0
        decay_counter = 0
        # print("initial loss is {} ".format(loss))
        for epoch in range(max_iters):

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
            # Display every 25 epoch
            if (epoch + 1) % 25 == 0:
                print('Epoch {e} in {m}'.format(e=epoch + 1, m=max_iters), end="\t")
                if self.validation is True:
                    # print('\tTrain Loss {0:0.4f}, \tTrain mis-class {0:0.4f}, '
                    #       '\tvalid loss {0:0.4f}, \tvalid mis-class {0:0.4f}'.
                    #       format(loss, mis_class, valid_loss, valid_mis_class))
                    print('\tTrain Loss {}, \tTrain mis-class {}, '
                          '\tvalid loss {}, \tvalid mis-class {}'.
                          format(loss, mis_class, valid_loss, valid_mis_class))
                else:
                    print('\tTrain Loss {}, \tTrain mis-class {}'.
                          format(loss, mis_class))
            # judge the performance
            if best_loss - loss > 0.000001:
                best_loss = loss
                best_counter = 0
            else:
                best_counter += 1
                if best_counter > early_stop:
                    print("Learning early stop since loss not improving for {} epoch.".format(best_counter))
                    break
                if best_counter % decay_intval == 0:
                    print("weight decay by {}".format(decay))
                    lr *= decay
                    decay_counter += 1
                    if decay_counter > decay_lim:
                        print("decay {} times, stop".format(decay_lim))
                        break
        return self.weights[-1]

    def newton(self, lr=0.01, max_iters=100):
        # TODO: implement newton method later
        raise NotImplementedError

    def cross_validation(self, cv, lambdas, lambda_name, seed=1, skip=False, plot=False, **kwargs):
        """
        Cross validation method to acquire the best prediction parameters.
        It will use the train_x y as data and do K-fold cross validation.
        :param cv:              cross validation times
        :param lambdas:         array of lambdas to be validated
        :param lambda_name:     the lambda name tag
        :param seed:            random seed
        :param skip:            skip the cross validation, only valid 1 time
        :param plot             plot cross-validation plot, if machine not support
                                matplotlib.pyplot, set to false.
        :param kwargs:          other parameters could pass into compute_weight
        :return: best weights, best_lambda, (training error, valid error)
        """
        np.set_printoptions(precision=4)
        k_indices = build_k_indices(self.train_y, cv, seed)
        # define lists to store the loss of training data and test data
        err_tr = []
        err_te = []
        weights = []
        print("K-fold ({}) cross validation to examine [{}]".
              format(cv, lambdas))
        for lamb in lambdas:
            print("For lambda: {}".format(lamb))
            _mse_tr = []
            _mse_te = []
            _weight = []
            for k in range(cv):
                print('Cross valid iteration {}'.format(k))
                weight, loss_tr, loss_te = self._loop_cross_validation(self.train_y, self.train_x,
                                                                       k_indices, k,
                                                                       lamb, lambda_name, **kwargs)
                _mse_tr += [loss_tr]
                _mse_te += [loss_te]
                _weight.append(weight)
                if skip:
                    break
            avg_tr = np.average(_mse_tr)
            avg_te = np.average(_mse_te)
            err_tr += [avg_tr]
            err_te += [avg_te]
            weights.append(_weight)
            print("\t train error {}, \t valid error {}".
                  format(avg_tr, avg_te))
        # Select the best parameter during the cross validations.
        print('K-fold cross validation result: \n {} \n {}'.
              format(err_tr, err_te))
        # Select the best based on least err_te
        min_err_te = np.argmin(err_te)
        print('Best err_te result {}, lambda {}'.
              format(err_te[min_err_te], lambdas[min_err_te]))
        if plot:
            from plots import cross_validation_visualization
            cross_validation_visualization(lambdas, err_tr, err_te, title=lambda_name,
                                           error_name=self.loss_function_name)
        else:
            save_numpy_array(lambdas, err_tr, err_te, names=['lambda', 'err_tr', 'err_te'], title=self.regularizer.name)

        return weights[min_err_te], lambdas[min_err_te], (err_tr, err_te)

    def _loop_cross_validation(self, y, x, k_indices, k, lamb, lambda_name, **kwargs):
        """
        Single loop of cross validation
        :param y:           train labels
        :param x:           train data
        :param k_indices:   indices array
        :param k:           number of cross validations
        :param lamb:        lambda to use
        :param lambda_name: lambda_name to pass into compute weight
        :return:            weight, mis_tr, mis_te
        """
        train_ind = np.concatenate((k_indices[:k], k_indices[k + 1:]), axis=0)
        train_ind = np.reshape(train_ind, (train_ind.size,))

        test_ind = k_indices[k]
        # Note: different from np.ndarray, tuple is name[index,]
        # ndarray is name[index,:]
        train_x = x[train_ind,]
        train_y = y[train_ind,]
        test_x = x[test_ind,]
        test_y = y[test_ind,]
        # Insert one more kwargs item
        kwargs[lambda_name] = lamb
        # print("_loop_cv kwargs: ", end="")
        # print(kwargs)

        weight = self.compute_weight(train_y, train_x, test_x, test_y, **kwargs)

        # Compute the metrics and return
        loss_tr = self.compute_metrics(train_y, train_x, weight)
        loss_te = self.compute_metrics(test_y, test_x, weight)

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
        pred = np.reshape(pred, (N,))
        target = np.reshape(target, (N,))
        nb_misclass = np.count_nonzero(target - pred)
        return nb_misclass / N

    def compute_loss(self, y, x, weight):
        return self.loss_function(y, x, weight)


class LinearRegression(Model):
    """ Linear regression model """

    def __init__(self, train, validation=None, initial_weight=None,
                 regularizer=None, regularizer_p=None,
                 loss_function_name='mse', calculate_weight='normalequ'):
        # Initialize the super class with given data.
        super(LinearRegression, self).__init__(train, validation,
                                               initial_weight=initial_weight,
                                               loss_function_name=loss_function_name,
                                               cal_weight=calculate_weight,
                                               regularizer=regularizer,
                                               regularizer_p=regularizer_p)

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
        grad += self.regularizer.get_gradient(weight)
        return grad

    def predict(self, x, weight):
        """Prediction function"""
        pred = np.dot(x, weight)
        pred[np.where(pred <= 0)] = -1
        pred[np.where(pred > 0)] = 1
        return pred

    def normalequ(self):
        """ Normal equation to get parameters """
        tx = self.train_x
        y = self.train_y
        if self.regularizer is None:
            return np.linalg.solve(np.dot(tx.T, tx), np.dot(tx.T, y))
        elif self.regularizer.name is 'Ridge':
            G = np.eye(tx.shape[1])
            G[0, 0] = 0
            hes = np.dot(tx.T, tx) + self.regularizer_p * G
            return np.linalg.solve(hes, np.dot(tx.T, y))
        else:
            raise NotImplementedError


class LogisticRegression(Model):
    """ Logistic regression """

    def __init__(self, train, validation=None, initial_weight=None,
                 loss_function_name='logistic',
                 calculate_weight='gradient',
                 regularizer=None, regularizer_p=None):
        """
        Constructor of Logistic Regression model
        :param train:           tuple (y, x)
        :param validation:      tuple (y, x)
        :param initial_weight:  weight vector, dim align x
        :param loss_function:   f(x, y, weight)
        :param regularizer:     "Ridge" || "Lasso"
        :param regularizer_p:   parameter
        """
        # Initialize the super class with given data.
        # Transform the y into {0,1}
        y, tx = train
        y[np.where(y < 0)] = 0
        train = (y, tx)
        if validation:
            val_y, val_tx = validation
            val_y[np.where(val_y < 0)] = 0
            validation = (val_y, val_tx)
        super(LogisticRegression, self).__init__(train, validation,
                                                 initial_weight=initial_weight,
                                                 loss_function_name=loss_function_name,
                                                 cal_weight=calculate_weight,
                                                 regularizer=regularizer,
                                                 regularizer_p=regularizer_p)
        # Set predicted label
        self.pred_label = [-1, 1]

    def __call__(self, x, weight=None):
        """Define the fit function and get prediction"""
        if weight is None:
            weight = self.weights[-1]
        return sigmoid(np.dot(x, weight))

    def get_gradient(self, y, x, weight):
        """ calculate gradient given data and weight """
        y = np.reshape(y, (len(y),))
        return np.dot(x.T, sigmoid(np.dot(x, weight)) - y) \
               + self.regularizer.get_gradient(weight)

    def get_hessian(self, y, x, weight):
        # TODO: implement hessian for newton method
        raise NotImplementedError

    def predict(self, x, weight=None, cutting=0.5):
        """ Prediction of event {0,1} """
        if weight is None: weight = self.weights[-1]
        pred = sigmoid(np.dot(x, weight))
        pred[np.where(pred <= cutting)] = 0
        pred[np.where(pred > cutting)] = 1
        return pred

    def predict_label(self, x, weight=None, cutting=0.5, predict_label=None):
        """ Prediction result with labels """
        if predict_label is None:
            predict_label = self.pred_label
        if weight is None: weight = self.weights[-1]
        pred = self.predict(x, weight, cutting)
        pred[np.where(pred == 0)] = predict_label[0]
        pred[np.where(pred == 1)] = predict_label[1]
        return pred

    def train(self, loss_function='logistic',
              lr=0.1, momentum=0.9, decay=0.5, max_iters=3000, batch_size=128, **kwargs):
        """ Make the default loss logistic """
        return super(LogisticRegression, self).train('sgd', loss_function,
                                                     lr=lr, momentum=momentum,
                                                     decay=decay, max_iters=max_iters,
                                                     batch_size=batch_size, **kwargs)

    def normalequ(self, **kwargs):
        """ Should never call """
        raise NotImplementedError

