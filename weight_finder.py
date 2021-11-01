'''
The functions in this file are used to optimize hyperparameters using cross validation

The functions further train the models and write the found weights to files
'''

import numpy as np

from implementations import least_squares, least_squares_GD, least_squares_SGD, ridge_regression, logistic_regression_SGD, reguralized_logistic_regression
from weight_helpers import *
from data_reader import load_clean_csv

#--------------- Function Implementations -----------------#
def train_least_squares(DATA_SAVE_PATH, TRAIN_DATA_PATH = "data/clean_train_data.csv"):
    """ Training the least squares

    INPUT VARIABLES:
    - DATA_SAVE_PATH:           The path to where the data will be saved
    - TRAIN_DATA_PATH:          The path to where we will collect our data from
    
    """
    y, tX, ids = load_clean_csv(TRAIN_DATA_PATH)

    # Actual training of the model to find the weights
    w, _ = least_squares(y, tX)

    # Writing the weights to the DATA_SAVE_PATH file, in csv format
    weight_writer(w, DATA_SAVE_PATH)


def train_least_squares_GD(DATA_SAVE_PATH, gamma_array, TRAIN_DATA_PATH = "data/clean_train_data.csv"):
    """ Training the least squares gradient descent

    INPUT VARIABLES:
    - DATA_SAVE_PATH:           The path to where the data will be saved
    - gamma_array:              The array containing the different gamma hyperparameters
    - TRAIN_DATA_PATH:          The path to where we will collect our data from
    
    """
    print("now training least squares GD")

    y, tX, ids = load_clean_csv(TRAIN_DATA_PATH)

    test_loss = []
    # Lopping over the gamma values to find the most optimal value for gamma
    for gamma in gamma_array:
        # Will run fold cross validation on the unqiue combination of gamma and lambda value
        # This will then output the loss and add it to the test_loss list
        _, test_ls = cross_validation(y, tX, 3, least_squares_GD, gamma)
        if np.isnan(test_ls): test_ls = np.array([[100]])
        if np.isinf(test_ls): test_ls = np.array([[100]])
        if test_ls > 101: test_ls = np.array([[100]])
        test_loss.append(test_ls[0,0])
        print("finished round ", np.where(gamma_array==gamma))

    # The most optimal gamma value given the lowest test loss
    g_star = gamma_array[np.argmin(test_loss)]

    # Actual training of the model to find the weights
    w, _ = least_squares_GD(y, tX, g_star)

    # Writing the weights to the DATA_SAVE_PATH file, in csv format
    weight_writer(w, DATA_SAVE_PATH)


def train_least_squares_SGD(DATA_SAVE_PATH, gamma_array, TRAIN_DATA_PATH = "data/clean_train_data.csv"):
    """ Training the least squares stochastic gradient descent

    INPUT VARIABLES:
    - DATA_SAVE_PATH:           The path to where the data will be saved
    - gamma_array:              The array containing the different gamma hyperparameters
    - TRAIN_DATA_PATH:          The path to where we will collect our data from
    
    """
    print("now training least squares SGD")
    y, tX, ids = load_clean_csv(TRAIN_DATA_PATH)

    test_loss = []
    # Lopping over the gamma values to find the most optimal value for gamma
    for gamma in gamma_array:
        # Will run fold cross validation on the unqiue combination of gamma and lambda value
        # This will then output the loss and add it to the test_loss list
        _, test_ls = cross_validation(y, tX, 3, least_squares_SGD, gamma)
        if np.isnan(test_ls): test_ls = np.array([[10000]])
        if np.isinf(test_ls): test_ls = np.array([[10000]])
        test_loss.append(test_ls[0,0])
        print("finished round ", np.where(gamma_array==gamma)[0])

    # The most optimal gamma value, given lowest loss from cross validation
    g_star = gamma_array[np.argmin(test_loss)]

    # Actual training of the model to find the weights
    w, _ = least_squares_SGD(y, tX, g_star)

    # Writing the weights to the DATA_SAVE_PATH file, in csv format
    weight_writer(w, DATA_SAVE_PATH)

def train_ridge_regression(DATA_SAVE_PATH, lambda_array, TRAIN_DATA_PATH = "data/clean_train_data.csv"):
    """ Training the ridge regression

    INPUT VARIABLES:
    - DATA_SAVE_PATH:           The path to where the data will be saved
    - lambda_array:             The array containing the different lambda hyperparameters
    - TRAIN_DATA_PATH:          The path to where we will collect our data from
    
    """

    print("now training ridge regression")
    y, tX, ids = load_clean_csv(TRAIN_DATA_PATH)

    test_loss = []
    # Lopping over the lambda values to find the most optimal value for lambda
    for lambda_ in lambda_array:
        # Will run fold cross validation on the unqiue combination of gamma and lambda value
        # This will then output the loss and add it to the test_loss list
        _, test_ls = cross_validation(y, tX, 3, ridge_regression, lambda_)
        if np.isnan(test_ls): test_ls = np.array([[10000]])
        if np.isinf(test_ls): test_ls = np.array([[10000]])
        test_loss.append(test_ls[0,0])
        print("finished round ", np.where(lambda_array==lambda_)[0])

    # The most optimal lambda value, given loss from cross validation
    l_star = lambda_array[np.argmin(test_loss)]
    print("hi", test_loss)

    # Actual training of the model to find the weights
    w, _ = ridge_regression(y, tX, l_star)

    # Writing the weights to the DATA_SAVE_PATH file, in csv format
    weight_writer(w, DATA_SAVE_PATH)


def train_logistic_SGD(DATA_SAVE_PATH, gamma_array, TRAIN_DATA_PATH = "data/clean_train_data.csv"):
    """ Training the logistic stochastic gradient descent

    INPUT VARIABLES:
    - DATA_SAVE_PATH:           The path to where the data will be saved
    - gamma_array:              The array containing the different gamma hyperparameters
    - TRAIN_DATA_PATH:          The path to where we will collect our data from
    
    """

    print("now training logistic SGD")
    y, tX, ids = load_clean_csv(TRAIN_DATA_PATH)

    test_loss = []
    
    # Lopping over the gamma values to find the most optimal value for gamma
    for gamma in gamma_array:
        # Will run fold cross validation on the unqiue combination of gamma and lambda value
        # This will then output the loss and add it to the test_loss list
        _, test_ls = cross_validation(y, tX, 3, logistic_regression_SGD, gamma)
        if np.isnan(test_ls): test_ls = np.inf
        test_loss.append(test_ls[0,0])
        print("finished round ", np.where(gamma_array==gamma)[0])

    # The most optimal gamma value
    g_star = gamma_array[np.argmin(test_loss)]

    # Actual training of the model to find the weights, using 500 000 iterations.
    w, _ = logistic_regression_SGD(y, tX, g_star, max_iters=500000)

    # Writing the weights to the DATA_SAVE_PATH file, in csv format
    weight_writer(w, DATA_SAVE_PATH)


def train_regularized_logistic_SGD(DATA_SAVE_PATH, lambda_array, gamma_array, TRAIN_DATA_PATH = "data/clean_train_data.csv"):
    """ Training the reguralized logistic stochastic gradient descent

    INPUT VARIABLES:
    - DATA_SAVE_PATH:           The path to where the data will be saved
    - lambda_array:             The array containing the different lambda hyperparameters
    - gamma_array:              The array containing the different gamma hyperparameters
    - TRAIN_DATA_PATH:          The path to where we will collect our data from

    """

    print("now training regularized logistic SGD")
    y, tX, ids = load_clean_csv(TRAIN_DATA_PATH)

    test_loss = []

    # Looping over the lambda_ values and the gamma values in a double for loop
    for lambda_ in lambda_array:
        loss_saver = []
        for gamma in gamma_array:

            # Will run fold cross validation on the unqiue combination of gamma and lambda value
            # This will then output the loss and add it to the test_loss list
            _, test_ls = cross_validation(y, tX, 3, reguralized_logistic_regression, lambda_, gamma)
            if np.isnan(test_ls): test_ls = np.array([[np.inf]])
            loss_saver.append(test_ls[0,0])
            print("finished sub_round ", np.where(gamma_array==gamma)[0], "-", np.where(lambda_array==lambda_)[0])

        test_loss.append(loss_saver)
        print("finished round ", np.where(lambda_array==lambda_)[0])

    # Collecting the location of the lowest loss
    location = np.argwhere(test_loss == min(np.array(test_loss).flatten()))[0]

    # The most optimal lambda value and gamma value
    l_star = lambda_array[location[0]]
    g_star = gamma_array[location[1]]

    # Actual training of the model to find the weights, using 500 000 iterations.
    w, _ = reguralized_logistic_regression(y, tX, l_star, g_star, max_iters=500000)

    # Writing the weights to the DATA_SAVE_PATH file, in csv format
    weight_writer(w, DATA_SAVE_PATH)
