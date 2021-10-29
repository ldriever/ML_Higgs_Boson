import numpy as np
from proj1_helpers import compute_mean_square_error

#--------------- Writing Weights -----------------#
def weight_writer(w, DATA_SAVE_PATH):
    w = np.reshape(w, (1, len(w)))[0]
    with open(DATA_SAVE_PATH, 'w') as f:
        f.write(str(w[0]))
        for weight in w[1:]:
            f.write("," + str(weight))
    print("file_written")


#--------------- Building the k indices for cross validation -----------------#
def build_k_indices(y, k_fold, seed = 1):
    """ Function that builds the k-indices for k-fold cross validation

    Credit for writing this function goes to the TAs of the 2021-2022 ML course at EPFL

    INPUT VARIABLES:
    - y                     1D array of data points. Used to get the right dimensions for the indices array
    - k_fold                Number of groups into which the data is to be split for ridge regression (integer)
    - seed                  Integer or float to initialize the random number generator

    OUTPUT VARIABLES:
    - k_indices             Array containing shuffled indices. Has dimension k x int(len(y)/k)

    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

#--------------- Split data for cross validation -----------------#
def split_data(x, y, k_indices, k):
    """
    Splits data into k subsets for cross validation
    """

    # Find number of columns of x
    #     -> necessary to account for special case of x being 1D
    if len(np.shape(x)) == 1:
        cols = 1
    else:
        cols = len(x[0])

    # Extract test set for y and x
    #      -> Reshaping is necessary in case that we are working with 1D arrays (i.e. for y and for some cases of x)
    y_test = np.reshape(y[k_indices[k]], (len(k_indices[0]), 1))
    x_test = np.reshape(x[k_indices[k]], (len(k_indices[0]), cols))


    # Extract remaining data
    #      -> flattening and reshaping is necessary to ensure the correct output shape
    if k == len(k_indices)-1:
        y_train = np.reshape(y[k_indices[:k]].flatten(), ((len(k_indices) - 1) * len(k_indices[0]), 1))
        x_train = np.reshape(x[k_indices[:k]].flatten(), ((len(k_indices) - 1) * len(k_indices[0]), cols))
    else:
        y_train = np.reshape(np.vstack((y[k_indices[:k]], y[k_indices[k+1:]])).flatten(), ((len(k_indices) - 1) * len(k_indices[0]), 1))
        x_train = np.reshape(np.vstack((x[k_indices[:k]], x[k_indices[k+1:]])).flatten(), ((len(k_indices) - 1) * len(k_indices[0]), cols))

    return x_train, y_train, x_test, y_test

#--------------- Cross Validation -----------------#
def cross_validation(y, x, k_fold, method, *args):
    """

    Performs k-fold cross validation for the specified method (can pass different ML models to this function)

    --> returns the loss of ridge regression.

    """

    # Create shuffled indices for splitting the data into k groups
    k_indices = build_k_indices(y, k_fold)

    # Set up lists for training and test losses to be appended to for each iteration of the cross validation
    tr_mse_lst = []
    te_mse_lst = []

    # Loop over all different data splits
    for k in range(k_fold):

        # Split data into training and test sets
        x_train, y_train, x_test, y_test = split_data(x, y, k_indices, k)

        # Compute the weights using the specified method and any related args
        w, _ = method(y_train, x_train, *args)

        # Compute the errors and append them to the respective lists
        if sum(abs(w)) == 0:
            te_mse_lst.append(np.array([[np.inf]]))
        else:
            te_mse_lst.append(compute_mean_square_error(y_test, x_test, w))

    # Finally take the averages over the collected lists

    tr_mse = sum(tr_mse_lst) / k_fold
    te_mse = sum(te_mse_lst) / k_fold

    return tr_mse, te_mse


#--------------- Polynomial Feature Expansion -----------------#
def build_poly(x, degree):
    """Creates polynomial basis functions for input data x, for j=0 up to j=degree.

    INPUT VARIABLES:
    - x                     Data array of size n x m where m can be equal to or greater than 1
    - verbose               Polynomial degree up to which the features shall be extended

    OUTPUT VARIABLES:
    - x_return              Array with the same number of rows as A but with d*m + 1 columns
    """
    if degree == 0:
        return x
    # Reshape x in case that it is 1-dimensional
    if np.shape(x)[1] == 1:
        x = x.reshape(len(x), 1)

    degree = int(degree)
    # Add one column of 1s to x. Only one column is needed (not one per original feature)
    x_return = np.hstack((np.ones((len(x),1)), x))

    # Extend x by extending it by powers of the original features
    for i in range(2, degree + 1):
        x_return = np.hstack((x_return, x ** i))

    return x_return
