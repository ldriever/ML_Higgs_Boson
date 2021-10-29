import numpy as np

from implementations import ridge_regression
from data_reader import load_clean_csv
from weight_helpers import weight_writer, cross_validation, build_poly


def double_optimize_ridge_regression(DATA_SAVE_PATH, d_array, lambda_array, TRAIN_DATA_PATH = "data/clean_train_data.csv"):
    '''
    This function performs a two-parameter optimization for ridge regression
    -> the method uses cross-validation
    -> the optimized parameters are the polynomial degree of feature expansion and the hyperparameter lambda
    '''

    print("now optimizing ridge regression")

    # Load the data
    y, tX, ids = load_clean_csv(TRAIN_DATA_PATH)

    # Do the two-fold optimization
    test_loss = []
    for d in d_array:
        loss_saver = []

        tX_local = build_poly(tX, d)

        for l in lambda_array:
            _, test_ls = cross_validation(y, tX_local, 3, ridge_regression, l, True)
            if np.isnan(test_ls): test_ls = np.array([[np.inf]])
            loss_saver.append(test_ls[0,0])
            print("finished sub_round ", np.where(lambda_array==l)[0][0] + 1, "-", d)

        test_loss.append(loss_saver)
        print("finished round ", d)

    location = np.argwhere(test_loss == min(np.array(test_loss).flatten()))[0]
    print(test_loss)
    print(location)
    d_star = d_array[location[0]]
    l_star = lambda_array[location[1]]

    tX = build_poly(tX, d_star)


    w, _ = ridge_regression(y, tX, l_star, poly=True)

    weight_writer(w, DATA_SAVE_PATH)


def ridge_regression_jet_optimization(lambda_, PREDICTION_SAVE_PATH, TEST_DATA_PATH="data/clean_test_data_jets.csv", TRAIN_DATA_PATH="data/clean_train_data_jets.csv"):
    '''

    The function splits the train set by jet number, trains four models, applies these models to the split...
        ...test data, and writes the joint prediction file

    '''

    y_tr, tX_tr, _ = load_clean_csv(TRAIN_DATA_PATH)
    _, tX_te, ids_te = load_clean_csv(TEST_DATA_PATH)

    w_0, _ = ridge_regression(y_tr[np.where(tX_tr[:, -2] == 0)], tX_tr[np.where(tX_tr[:, -2] == 0)], lambda_)
    w_1, _ = ridge_regression(y_tr[np.where(tX_tr[:, -2] == 1)], tX_tr[np.where(tX_tr[:, -2] == 1)], lambda_)
    w_2, _ = ridge_regression(y_tr[np.where(tX_tr[:, -2] == 2)], tX_tr[np.where(tX_tr[:, -2] == 2)], lambda_)
    w_3, _ = ridge_regression(y_tr[np.where(tX_tr[:, -2] == 3)], tX_tr[np.where(tX_tr[:, -2] == 3)], lambda_)

    pred_0 = tX_te[np.where(tX_te[:, -2] == 0)] @ w_0
    pred_0[np.where(pred_0 < 0.5)] = -1
    pred_0[np.where(pred_0 >= 0.5)] = 1
    dat_0 = np.hstack((np.reshape(ids_te[np.where(tX_te[:, -2] == 0)], (len(np.where(tX_te[:, -2] == 0)[0]), 1)), pred_0))

    pred_1 = tX_te[np.where(tX_te[:, -2] == 1)] @ w_1
    pred_1[np.where(pred_1 < 0.5)] = -1
    pred_1[np.where(pred_1 >= 0.5)] = 1
    dat_1 = np.hstack((np.reshape(ids_te[np.where(tX_te[:, -2] == 1)], (len(np.where(tX_te[:, -2] == 1)[0]), 1)), pred_1))

    pred_2 = tX_te[np.where(tX_te[:, -2] == 2)] @ w_2
    pred_2[np.where(pred_2 < 0.5)] = -1
    pred_2[np.where(pred_2 >= 0.5)] = 1
    dat_2 = np.hstack((np.reshape(ids_te[np.where(tX_te[:, -2] == 2)], (len(np.where(tX_te[:, -2] == 2)[0]), 1)), pred_2))

    pred_3 = tX_te[np.where(tX_te[:, -2] == 3)] @ w_3
    pred_3[np.where(pred_3 < 0.5)] = -1
    pred_3[np.where(pred_3 >= 0.5)] = 1
    dat_3 = np.hstack((np.reshape(ids_te[np.where(tX_te[:, -2] == 3)], (len(np.where(tX_te[:, -2] == 3)[0]), 1)), pred_3))

    output = np.vstack((dat_0, dat_1, dat_2, dat_3))
    output = output[output[:,0].argsort()]

    count = 0
    with open(PREDICTION_SAVE_PATH, 'w') as f:
        f.write("Id,Prediction\n")
        for i in range(len(output)):
            f.write(str(output[i,0]) + "," + str(output[i,1]) + "\n")
            if output[i,1] == 1: count += 1
        print("file written")
