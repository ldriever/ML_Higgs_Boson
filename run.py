'''

Running this file will do all the steps necessary in order to reproduce the best-scoring predictions

In this case the best scoring predictions correspond to ridge regression with lambda = 4.64 x10^-5
    -> (note that the gamma value is found in a different step and taken as input here)

'''

from data_cleaner import data_maker
from weight_finder import *
from prediction_writer import prediction_writer
from further_optimization import double_optimize_ridge_regression, ridge_regression_jet_optimization


## ------------------------------------------------------------------------------------##
# Creating the cleaned data files that have all -999 entries removed - these files are needed for the final submission
#           ---         ---         ---         ---         ---         ---         ---#

data_maker("data/train.csv", "data/clean_train_data_jets.csv", jets=True)
data_maker("data/test.csv", "data/clean_test_data_jets.csv", jets=True)

## ------------------------------------------------------------------------------------##
# Creating the cleaned data files that have all -999 entries removed - these files are needed for all other predictions
# -> uncomment these lines when trying to apply any other models in any other way than using ridge_regression_jet_optimization
#           ---         ---         ---         ---         ---         ---         ---#

# data_maker("data/train.csv", "data/clean_train_data.csv")
# data_maker("data/test.csv", "data/clean_test_data.csv")


## ------------------------------------------------------------------------------------##
# Uncomment the lines below to train the different models and write their weights
#           ---         ---         ---         ---         ---         ---         ---#

# train_least_squares("data/weights/least_squares_weights.csv")
# train_least_squares_GD("data/weights/least_squares_GD_weights.csv", np.logspace(-5.1,-4.7,5))
# train_least_squares_SGD("data/weights/least_squares_SGD_weights.csv", np.logspace(-10.2,-9,5))
# train_ridge_regression("data/weights/ridge_regression_weights.csv", np.logspace(-9,-3,10))
# train_logistic_SGD("data/weights/logistic_SGD_weights.csv", np.logspace(-6, -4.5, 5))
# train_regularized_logistic_SGD("data/weights/regularized_logistic_SGD_weights.csv", [30, 35, 50, 60], np.logspace(-15,-1,30))


## ------------------------------------------------------------------------------------##
# Uncomment the lines below to write the predictions for the different models
# -> requires that the weights have been written
#           ---         ---         ---         ---         ---         ---         ---#

# prediction_writer("data/clean_test_data.csv", "data/weights/least_squares_weights.csv", "data/predictions/least_squares_predictions.csv")
# prediction_writer("data/clean_test_data.csv", "data/weights/ridge_regression_weights.csv", "data/predictions/ridge_regression.csv")
# prediction_writer("data/clean_test_data.csv", "data/weights/least_squares_GD_weights.csv", "data/predictions/least_squares_GD.csv")
# prediction_writer("data/clean_test_data.csv", "data/weights/least_squares_SGD_weights.csv", "data/predictions/least_squares_SGD.csv")
# prediction_writer("data/clean_test_data.csv", "data/weights/logistic_SGD_weights.csv", "data/predictions/logistic_SGD.csv")
# prediction_writer("data/clean_test_data.csv", "data/weights/regularized_logistic_SGD_weights.csv", "data/predictions/regularized_logistic_SGD.csv")


## ------------------------------------------------------------------------------------##
# Uncomment the following lines to compute the results for ridge regression with polynomial feature expansion
#           ---         ---         ---         ---         ---         ---         ---#

# double_optimize_ridge_regression("data/weights/optimized_ridge_weights", np.linspace(1, 3, 3), np.logspace(-5,-4,6))
# prediction_writer("data/clean_test_data.csv", "data/weights/optimized_ridge_weights", "data/predictions/optimized_ridge_regression.csv", degree=1)

## ------------------------------------------------------------------------------------##
# The lines below write the final submission file with the best score
#           ---         ---         ---         ---         ---         ---         ---#

ridge_regression_jet_optimization(4.64e-5, "data/predictions/RR_jet_split.csv")
