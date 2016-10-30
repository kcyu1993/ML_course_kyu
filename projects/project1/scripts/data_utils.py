import numpy as np
from helpers import standardize

""" Data cleaning class """


def calculate_valid_mean(x, outlier):
    raise NotImplementedError


def draw_balanced_subsample(labels, data, trainsize=100000, seed=1):
    """
    Draw subsample.
    :param labels:
    :return: ind_tr, ind_valid
    """

    positive = np.where(labels > 0)[0]
    negative = np.where(labels <= 0)[0]
    # Random subsamples of trainsize/2
    np.random.seed(seed=seed)
    split_pos = [trainsize / 2]
    # Generate randomized index for positive and negative array
    index_posi = np.random.permutation(range(len(positive)))
    index_nega = np.random.permutation(range(len(negative)))

    posi_tr, posi_te = np.split(index_posi, split_pos)
    nega_tr, nega_te = np.split(index_nega, split_pos)
    # ind_tr = np.concatenate(positive[posi_tr], negative[nega_tr])
    # ind_te = np.concatenate(positive[posi_te], negative[nega_te])
    ind_tr = np.concatenate((positive[posi_tr], negative[nega_tr]))
    ind_te = np.concatenate((positive[posi_te], negative[nega_te]))
    x_tr = data[ind_tr, :]
    x_te = data[ind_te, :]
    y_tr = labels[ind_tr]
    y_te = labels[ind_te]
    return (y_tr, x_tr), (y_te, x_te)


def split_train_valid(ratio, data, labels, seed=1):
    """ Split train validation set with given ratio """
    if ratio > 1: ratio = 1
    if ratio < 0: ratio = 0
    np.random.seed(1)
    split_pos = int(ratio * len(labels))
    index = np.random.permutation(range(len(labels)))
    ind_tr, ind_te = np.split(index, [split_pos])
    x_tr = data[ind_tr, :]
    x_te = data[ind_te, :]
    y_tr = labels[ind_tr]
    y_te = labels[ind_te]
    return (y_tr, x_tr), (y_te, x_te)


def fill_missing(data, missing=-999.0, method='mean'):
    '''
    Fill the missing data
    :param data: np.ndarray with data vector.
    :param missing: pre-defined missing value
    :return: np.ndarray same structure with all missing value filled
            by mean of its column.
    '''
    # fill the missing data -999 with the mean.
    mask_x = np.ma.masked_values(data, missing)
    for i in range(mask_x.shape[1]):
        if method is 'mean':
            mask_x[:, i] = mask_x[:, i].filled(mask_x[:, i].mean())
    return mask_x._get_data()


def remove_outlier(data, missing=-999.0):
    '''
    Return the tx with all valid values, no invalid values.
    :param data: data matrix
    :return:
    '''
    valid = []
    mask_x = np.ma.masked_values(data, missing)
    for index in range(data.shape[0]):
        if np.sum(mask_x[index, :]._get_mask()) == 0:
            valid.append(index)

    print(len(valid), ' valid training samples)')
    return valid


def remove_dimensions(data, missing=-999.0):
    """
    Remove the dimension with invalid samples.
    :param data:
    :param missing:
    :return:
    """
    valid = []
    mask_x = np.ma.masked_values(data, missing)
    for index in range(data.shape[1]):
        if np.sum(mask_x[:, index]._get_mask()) == 0:
            valid.append(index)
    print(len(valid), 'valid training column, with index \n \t', end='')
    print(valid)
    tx = data[:, valid]
    return tx


def pca_transform(data, nbNewColumn=5, concatenate=True, **kwargs):
    """
    Author: Sina, Kaicheng
    Modified for return type
    :param data:        data input, with dimension [nb_sample, features]
    :param nbNewColumn: number of PC output
    :param concatenate:
    :return:   PC decompositions, the transformed PC features
    """
    S = np.cov(data.T)  # Coompute the covariance matrix
    eig_val, eig_vec = np.linalg.eig(S)  # Get the eigenvalues and eigenvectors
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    idx = np.argsort(eig_val)[::-1]
    eig_val = eig_val[idx]
    # sort eigenvectors according to same index
    eig_vec = eig_vec[:, idx]
    matrix_w = eig_vec[:, :nbNewColumn]
    transformedX = np.dot(data, matrix_w)
    if concatenate:
        transformedX = np.c_[data, transformedX]
    return eig_vec, transformedX


def interactions(data, indices=None):
    """
    Generate interactions, within given columns.
    :param data: data marix
    :param indices: if None, generate interactions across all data matrix
    :return: result interactions
    """
    if len(data.shape) > 1:
        degree = data.shape[1]
    else:
        degree = 1
        print("Cannot generate interactions for single column")
        return None
    if indices is None:
        indices = np.array(range(degree))
    inter_degree = len(indices)
    interaction = np.ones((data.shape[0], inter_degree * (inter_degree - 1) / 2), dtype=np.float32)
    # Generate interactions according to the columns
    counter = 0
    for index, column in enumerate(indices):
        for i in range(index + 1, len(indices)):
            interaction[:, counter] *= data[:, column] * data[:, indices[i]]
            counter += 1
    return interaction


def sqrt_transform(data, indices=[3, 2, 0], method='abs', **kwargs):
    """
    Indices:
        24, 27, 12, 6,5,4,3,2,0
    :param data:        data matrix [nb_sample, nb_features]
    :param indices:     indices to be log-transformed
    :param method:      shiftmin, abs
    :return:
    """
    if method is 'shiftmin':
        sqrt_transform = lambda x: np.sqrt(x + 0.01 - np.min(x))
    elif method is 'abs':
        sqrt_transform = lambda x: np.sqrt(np.abs(x))
    else:
        sqrt_transform = lambda x: np.sqrt(x)

    sqrt_data = data[:, indices]
    for i in range(len(indices)):
        sqrt_data[i] = sqrt_transform(sqrt_data[i])
    return sqrt_data


def log_transform(data, indices=[1, 3, 8, 9, 10, 13], method='shiftmin', **kwargs):
    """
    log transform the data according to its indices
    asserting by the plot.
    default indices is:
        0,1,2,3,4,5,8,9,10,13
    :param data:        data matrix [nb_sample, nb_features]
    :param indices:     indices to be log-transformed
    :param method:      shiftmin, abs
    :return:
    """
    if method is 'shiftmin':
        log_transform = lambda x: np.log(x + 0.01 - np.min(x))
    elif method is 'abs':
        log_transform = lambda x: np.log(np.abs(x))
    else:
        log_transform = lambda x: np.log(x)

    log_data = data[:, indices]
    for i in range(len(indices)):
        log_data[i] = log_transform(log_data[i])

    return log_data


def polynomial_tranform(data, degrees=[2], indices=None):
    """
    Polynomial tranform, of given data based on indices.
    :param data:
    :param degrees:
    :param indices:
    :return:
    """
    if indices is None:
        power_data = data
    else:
        power_data = data[:, indices]
    result = power_data
    for degree in degrees:
        result = np.c_[result, np.power(power_data, degree)]
    return result[:, len(power_data[0]):]


def compose_complex_features_further(data, interaction=False, log=False,
                                     sqrt=False, intercept=True, pca=False,
                                     standardize_first=True, power=False,
                                     **kwargs):
    """
    Feature engineering:
        further expansion of data.
    Read the input of original data matrix, with all the in-valid
        Discard them to produce the first factor

    Use mean_fill for the data then compose interactions of valid data
    Use log transform for default indices
    Use sqrt transform for default indices

    :param data: given data matrix
    :param flags
    :return:
        standardized version with intercept
        [valid_data, interactions[0:10],
        log[default], sqrt[default],
        polynomial[default], pca[default=5]
    """
    _valid_data = remove_dimensions(data)
    # valid_data = _valid_data #copy

    mean_fill = fill_missing(data)
    # Delete the complete useless part.
    _mean_fill = np.delete(mean_fill, [4, 5, 6, 12, 26, 27, 28], axis=1)
    valid_data = _mean_fill
    if standardize_first:
        mean_fill, _, _ = standardize(mean_fill, intercept=False)
    # valid_data = mean_fill[:,0:10]

    if power:
        data_power = polynomial_tranform(_valid_data, degrees=[2, 3, 4, 5, 6, 7, 8, 9],
                                         **kwargs)
        valid_data = np.c_[valid_data, data_power]

    if interaction:
        data_interact = interactions(mean_fill)
        valid_data = np.c_[valid_data, data_interact]

    if log:
        data_log = log_transform(mean_fill, **kwargs)
        valid_data = np.c_[valid_data, data_log]

    if sqrt:
        data_sqrt = sqrt_transform(mean_fill, **kwargs)
        valid_data = np.c_[valid_data, data_sqrt]

    if pca:
        _, data_pca = pca_transform(data, nbNewColumn=10, concatenate=False, **kwargs)
        valid_data = np.c_[valid_data, data_pca]

    valid_data, _, _ = standardize(valid_data)
    # result = np.c_[mean_fill[:, (0,2,4,5,6,7)], valid_data]
    # result, result_mean, result_std = standardize(valid_data, intercept=intercept)
    return valid_data, None, None


def compose_complex_features(data, interaction=False, log=False,
                             sqrt=False, intercept=True, pca=False,
                             standardize_first=True, power=False,
                             **kwargs):
    """
    Feature engineering:
        further improvement of the data varieties.
    Read the input of original data matrix, with all the in-valid
        Discard them to produce the first factor

    Use mean_fill for the data then compose interactions [0:10]
    Use log transform for default indices
    Use sqrt transform for default indices

    :param data:
    :return:
        standardized version with intercept
        [valid_data, interactions[0:10],
        log[default], sqrt[default], pca[default=5]
    """
    valid_data = remove_dimensions(data)
    _valid_data = valid_data  # copy
    mean_fill = fill_missing(data)
    if standardize_first:
        mean_fill, _, _ = standardize(mean_fill, intercept=False)
    # valid_data = mean_fill[:,0:10]

    if power:
        data_power = polynomial_tranform(_valid_data, degrees=[2, 3, 4, 5, 6, 7, 8, 9],
                                         **kwargs)
        valid_data = np.c_[valid_data, data_power]

    if interaction:
        data_interact = interactions(_valid_data)
        valid_data = np.c_[valid_data, data_interact]

    if log:
        data_log = log_transform(mean_fill, **kwargs)
        valid_data = np.c_[valid_data, data_log]

    if sqrt:
        data_sqrt = sqrt_transform(mean_fill, **kwargs)
        valid_data = np.c_[valid_data, data_sqrt]

    if pca:
        _, data_pca = pca_transform(data, concatenate=False, **kwargs)
        valid_data = np.c_[valid_data, data_pca]

    valid_data, _, _ = standardize(valid_data)
    # result = np.c_[mean_fill[:, (0,2,4,5,6,7)], valid_data]
    # result, result_mean, result_std = standardize(valid_data, intercept=intercept)
    return valid_data, None, None


def baseline_logistic(data):
    """
    Data generation method:
        Base-line logistic regression data manipulation.
        Only standardization is involved.
    :param data:
    :return:
    """
    tx = remove_dimensions(data)
    tx = standardize(tx)
    return tx[0]


def interactions_logistic(data):
    """
    Data generation method: logistic with interactions only
    :param data:
    :return:
    """
    tx = remove_dimensions(data)
    tx, _, _ = standardize(tx, intercept=False)

    tx_inter = interactions(tx)
    tx_inter = standardize(tx_inter, intercept=False)

    return np.c_[tx, tx_inter[0]]


def pca_interactions_logistics(data):
    """

    :param data:
    :return:
    """
    data, x_mean, x_std = standardize(data, intercept=False)
    nb_pc = 5
    print("test the PCA with {} elements".format(nb_pc))
    pcs, pc_data = pca_transform(data, nb_pc, concatenate=False)

    print("get interactions")
    interaction = interactions(data, range(0, 10))
    interaction, _, _ = standardize(interaction)
    print("select first 10 data entry with pc data")
    data = np.c_[data, pc_data]
    data = np.c_[data, interaction]
    return data


def box_cos(x, degree):
    """
    Box-cox transformation
    :param x:
    :param degree:
    :return:
    """
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
    """ Testing purpose """
    a = np.array(range(0, 9))
    a = np.reshape(a, (3, 3))
    # np.set_printoptions(4, suppress=True)
    # result, _, _ = compose_complex_features(a, intercept=True, interaction=True, log=True, sqrt=True, pca=True,
    #                                   nbNewColumns=2)
    # result = polynomial_tranform(a, degrees=[2,4], indices=[0,2])
    # print(result)
#     print(interactions(a))
#     print(interactions(a, [0,2]))
