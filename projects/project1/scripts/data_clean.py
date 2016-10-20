import numpy as np

def calculate_valid_mean(x, outlier):
    raise NotImplementedError

def fill_missing(data, missing=-999.0):
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
        mask_x[:, i] = mask_x[:, i].filled(mask_x[:, i].mean())
    return mask_x._get_data

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

def perform_PCA(y, tx):
    raise NotImplementedError