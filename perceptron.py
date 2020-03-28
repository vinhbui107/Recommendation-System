import numpy as np
import random
from collaborative_filtering import CF
from demographic_filtering import DF
from get_data import (
    get_users_data,
    get_rating_base_data,
    get_rating_test_data,
)


class Perceptron:
    def __init__(self, dataset, learning_rate, n_iters):
        self.dataset = dataset
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.w1 = random.uniform(0, 1)
        self.w2 = random.uniform(0, 1)

    def fit(self):
        """Calculate"""
        adj = self.dataset[0, 0] * self.w1 + self.dataset[0, 1] * self.w2
        for i in range(1, self.n_iters):
            # update weight
            w1_stamp = self.w1 + self.learning_rate * self.dataset[i - 1, 0] * (dataset[i - 1, 2] - adj)
            w2_stamp = self.w2 + self.learning_rate * self.dataset[i - 1, 1] * (dataset[i - 1, 2] - adj)
            if np.abs(self.w1 - w1_stamp) <= 0.0001 and np.abs(self.w2 - w2_stamp) <= 0.0001:
                break
            else:
                self.w1 = w1_stamp
                self.w2 = w2_stamp
            adj = self.dataset[i, 0] * self.w1 + self.dataset[i, 1] * self.w2

    def predict(self):
        """Predict rating based on w"""
        new_predicted_ratings = []
        for row in self.dataset:
            new_predicted_rating = row[0] * self.w1 + row[1] * self.w2
            new_predicted_ratings.append(new_predicted_rating)
        return new_predicted_ratings


#######################################################################################
# RATE_TRAIN = get_rating_base_data().values
# RATE_TEST = get_rating_test_data().values

# RATE_TRAIN[:, :2] -= 1
# RATE_TEST[:, :2] -= 1

# ids = np.where(RATE_TEST[:, 0] == 0)[0].astype("int32")
# scores = RATE_TEST[ids, 2]
# learning_rate = 0.001
# n_iters = len(ids)

# dataset = []
# for row in RATE_TEST[ids, :]:
#     predicted_rating_cf = DF.pred(0, row[1])
#     predicted_rating_df = CF.pred(0, row[1])
#     true_rating = row[2]
#     dataset.append([predicted_rating_cf, predicted_rating_df, true_rating])
# dataset = np.asarray(dataset)

# PLA = Perceptron(dataset, learning_rate, n_iters)
# PLA.fit()
# predicted_ratings_pla = PLA.predict()

# print("Rated movie ids : ", ids)
# print("True Ratings    : ", scores)
# print("Predicted Rating: ", np.round(predicted_ratings_pla, 2))
