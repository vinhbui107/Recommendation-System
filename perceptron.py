import numpy as np
import pandas as pd
from collaborative_filtering import CF
from demographic_filtering import DF
from get_data import (
    get_users_data,
    get_ratings_data,
    get_rating_base_data,
    get_rating_test_data,
)
from sklearn.metrics.pairwise import cosine_similarity


class Perceptron:
    def __init__(self, learning_rate, n_iters):
        self.learning_rate = learning_rate
        self.n_iters = n_iters

    def fit(self):
        pass

    def predict(self, cf, df):
        pass


#######################################################################################
USERS_DATA = get_users_data()
RATINGS_DATA = get_ratings_data()
RATE_TRAIN = get_rating_base_data().values
RATE_TEST = get_rating_test_data().values

RATE_TRAIN[:, :2] -= 1  # start from 0
RATE_TEST[:, :2] -= 1

n_trains = RATE_TRAIN.shape[0]
learning_rate = 1 / n_trains

PLA = Perceptron(learning_rate, n_trains)
CF = CF(RATE_TRAIN, 25)
DF = DF(USERS_DATA, RATE_TRAIN, 25, cosine_similarity)

CF.fit()
DF.fit()
PLA.fit()

n_tests = RATE_TEST.shape[0]
SE = 0
for n in range(n_tests):
    cf_pred = CF.pred(RATE_TEST[n, 0], RATE_TEST[n, 1])
    df_pred = DF.pred(RATE_TEST[n, 0], RATE_TEST[n, 1])
    pred = PLA.pred(df_pred, cf_pred)
    SE += (pred - RATE_TEST[n, 2]) ** 2

RMSE = np.sqrt(SE / n_tests)
print("Perceptron Learning Algorithm, RMSE: ", RMSE)
