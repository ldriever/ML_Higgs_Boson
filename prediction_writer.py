import numpy as np
from data_reader import load_weights, load_clean_csv
from weight_helpers import build_poly

# Reading the weights from file
def prediction_writer(TEST_DATA_PATH, WEIGHTS_PATH, DATA_SAVE_PATH, degree=None):

    # Loading the weights
    w = load_weights(WEIGHTS_PATH)

    # Loading the test data
    _, tX, ids = load_clean_csv(TEST_DATA_PATH, False)

    # Building the polynomial features if applicable
    if degree:
        tX = build_poly(tX, degree)

    # generating predictions

    y_good_guess = tX @ w

    # discretizing into -1 and 1
    y_good_guess[np.where(y_good_guess < 0.5)] = -1
    y_good_guess[np.where(y_good_guess >= 0.5)] = 1

    # Write the prediction file in the desired format
    count = 0
    with open(DATA_SAVE_PATH, 'w') as f:
        f.write("Id,Prediction\n")
        for i in range(len(y_good_guess)):
            f.write(str(ids[i]) + "," + str(y_good_guess[i,0]) + "\n")
            if y_good_guess[i,0] == 1: count += 1
            

