## **Machine Learning Project 1: Higgs Boson Detection**

### **Group members: H. Soland, F. Flaate, L.X. Driever**

_**Note:** In order to run different parts of the code, please look no further than the wonderful file **run.py**. By default this will produce the final submission file that lead to the best predictions, but by uncmmenting parts of the file it is possible to also run the other steps of this project_

The aim of this project is to use different machine learning methods in order to predict whether or not a Higgs boson signal was present in data available from the CERN ATLAS experiment. This project is part of the EPFL machine learning course challenge 2021. For more info see https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/

The investigated machine learning methods are:
-  least squares (solved with normal equations)
-  least squares (solved with gradient descent)
-  least squares (solved with stochastic gradient descent)
-  ridge regression (solved with normal equations)
-  logistic reggression (solved with stochastic gradient descent)
-  regularized logistic reggression (solved with stochastic gradient descent)

Overall the project consisted of the following steps:
1) Data processing
2) Hyperparameter optimization and model training
3) Writing of predictions for all models
4) Testing of predictions
5) Further improvement of best model
6) Assessment of final performance

In step (1) all columns with overly many missing values were removed. Missing entries for the parameter DER_mass_MMC where predicted using linear regression with the parameter DER_mass_vis as the two of a strong linear correlation greater than 0.9. The code for this step is mostly found in data_cleaner.py.

In step (2) cross validation was used to optimize the hyperparameters for the different ML models. Using the optimum hyperparameter values, the models were then trained and the weights stored.

In step (3) the found weights are used to generate predictions for the test data. The code for this step is mostly found in prediction_writer.py

Step (4) was conducted using the assessment system of the competition, as can be found on https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/

Step (5) looked at two ways to further optimize the best model, ridge regression. The first way is polynomial feature expansion, the second way is to split the data into subsets according to the jet number. The code for this step can be found in further_optimization.py

Step (6) was again conducted using the external platform https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/
