# -*- coding: utf-8 -*-
'''
Functions supporting implementations.py
'''
import numpy as np

def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
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

# ************************************************************************************************
# Helper functions for implementation.py *********************************************************
# ************************************************************************************************

def sigmoid(t):
    """Sigmoid function

    INPUT VARIABLES:
    - t:        Given variable
    OUTPUT VARIABLES:
    - sigmoid:  The value of sigmoid function given variable t
    """

    sigmoid = 1/(1+np.exp(-t))
    return sigmoid


def compute_mean_square_error(y, tx, w):
    """Calculate the mean square error.

    INPUT VARIABLES:
    - y:     Observed data (Vector: Nx1)
    - tx:    Input data (Matrix: NxD)
    - w:     Weigths (Vector: Dx1)

    LOCAL VARIABLES:
    - N:     Number of datapoints
    - e:     Error (Vector: Nx1)
    OUPUT VARIABLES:
    - mse:   Mean square error (Scalar)
    """
    y = np.reshape(y,(len(y),1))
    N = len(y)
    # Loss by MSE (Mean Square Error)
    try:
        e = y - tx@w
    except ValueError:
        print(np.shape(tx), np.shape(w))
    mse = (1/(2*N))*e.T@e
    return mse


def compute_least_squares_gradient(y, tx, w):
    """Compute the gradient.

    INPUT VARIABLES:
    - y:     Observed data (Vector: Nx1)
    - tx:    Input data (Matrix: NxD)
    - w:     Weigths (Vector: Dx1)

    LOCAL VARIABLES:
    - N:     Number of datapoints
    - e:     Error (Vector: Nx1)
    OUPUT VARIABLES:
    - gradient:    Gradient (Vector: Dx1)
    """
    y = np.reshape(y, (len(y), 1))
    N = len(y)
    e = y-tx@w
    gradient = -(1/N)*tx.T@e
    return gradient


def compute_negative_log_likelihood_loss(y, tx, w):
    """Compute a stochastic gradient

    INPUT VARIABLES:
    - y:           Observed data (Vector: Nx1)
    - tx:          Input data (Matrix: NxD)
    - w:           Weights (Vector: Dx1)

    OUTPUT VARIABLES:
    - loss:        Loss for given w

    """
    loss = np.sum(np.log(np.exp(tx @ w) + 1) - y * (tx @ w))
    return loss
