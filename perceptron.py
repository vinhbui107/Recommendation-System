import numpy as np
from collaborative_filtering import CF
from demographic_filtering import DF
from get_data import (
    get_users_data,
    get_ratings_data,
    get_rating_base_data,
    get_rating_test_data,
)


class Perceptron:
    def __init__(self, learning_rate, n_iters):
        self.learning_rate = learning_rate
        self.n_iters = n_iters

    def fit(self, X, y):
        """Calculate"""

    def net_input(self, X):
        """Calculate net input"""

    def activation(self, X):
        """Compute linear activation"""

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(X) >= 0.0, 1, -1)


#######################################################################################
USERS_DATA = get_users_data()
RATINGS_DATA = get_ratings_data().values
RATE_TRAIN = get_rating_base_data().values
RATE_TEST = get_rating_test_data().values

print(USERS_DATA)
print(RATINGS_DATA)

# DF.fit()
# PLA.fit()

# n_tests = RATE_TEST.shape[0]
# SE = 0
# for n in range(n_tests):
#     cf_pred = CF.pred(RATE_TEST[n, 0], RATE_TEST[n, 1])
#     df_pred = DF.pred(RATE_TEST[n, 0], RATE_TEST[n, 1])
#     real_rate = RATE_TEST[n, 2]
#     pred = PLA.learning(df_pred, cf_pred, real_rate)
#     SE += (pred - RATE_TEST[n, 2]) ** 2

# RMSE = np.sqrt(SE / n_tests)
# print("Perceptron Learning Algorithm, RMSE: ", RMSE)
