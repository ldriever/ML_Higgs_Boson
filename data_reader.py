'''

Contains useful functions for reading different files used in this project

'''

import numpy as np

def load_csv_data(data_path, sub_sample=False):
    """
    Loads data and returns y (class labels), tX (features) and ids (event ids)
    Function provided by the TAs
    """
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(y.shape[0])
    yb[np.where(y=='b')] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def load_clean_csv(data_path, adjust_range=True):
    """
    Loads data and converts the y values to the range [0, 1]

    INPUTS
     - data_path                string specifying where the file can be found
     - adjust_range = True      if True this means that the values are moved from the range [-1,1] to [0,1]

    OUTPUTS

    """
    data = np.genfromtxt(data_path, delimiter=",")

    ids = data[:, 0].astype(np.int)
    y = data[:, 1].astype(np.int)
    tX = data[:, 2:-1]

    # if desired shift scope from [-1, 1] to [0, 1]
    if adjust_range:
        y[np.where(y == -1)] = 0

    return y, tX, ids


def load_weights(data_path):
    '''
    Function that reads and returns the weights stored in a csv file

    INPUTS
     - data_path        file wherein the weights are stored

    OUTPUTS
     - weights          array of weights read from the file, reshaped as column vector
    '''
    weights = np.genfromtxt(data_path, delimiter=",")
    return np.reshape(weights, (len(weights), 1))