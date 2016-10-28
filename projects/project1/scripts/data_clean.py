import numpy as np
from scipy.stats.mstats import normaltest


def calculate_valid_mean(x, outlier):
    raise NotImplementedError


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
        if np.sum(mask_x[index,:]._get_mask()) == 0:
            valid.append(index)

    print(len(valid),' valid training samples')
    return valid


def remove_dimensions(data, missing=-999.0):
    valid = []
    mask_x = np.ma.masked_values(data, missing)
    for index in range(data.shape[1]):
        if np.sum(mask_x[:, index]._get_mask()) == 0:
            valid.append(index)
    print(len(valid), 'valid training column')
    tx = data[:, valid]
    return tx


def normal(data, axis=0):
    return normaltest(data, axis)


def pca(data, nbNewColumn, concatenate=True):
    """
    Author: Sina
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


def interactions(data, columns=None):
    if len(data.shape) > 1:
        degree = data.shape[1]
    else:
        degree = 1
        print("Cannot generate interactions for single column")
        return None
    if columns is None:
        columns = np.array(range(degree))
    inter_degree = len(columns)
    interaction = np.ones((data.shape[0], inter_degree * (inter_degree - 1) / 2), dtype=np.float32)
    # Generate interactions according to the columns
    counter = 0
    for index, column in enumerate(columns):
        for i in range(index + 1, len(columns)):
            interaction[:, counter] *= data[:, column] * data[:, columns[i]]
            counter += 1
    return interaction

# if __name__ == '__main__':
#     a = np.array(range(0,9))
#     a = np.reshape(a, (3,3))
#     np.set_printoptions(4, suppress=True)
#     print(interactions(a))
#     print(interactions(a, [0,2]))
