import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import correlation, cdist
from sklearn.metrics.pairwise import cosine_similarity

# from sklearn.metrics import
from scipy.stats import pearsonr
from get_data import (
    get_ratings_data,
    get_users_data,
    get_rating_base_data,
    get_rating_test_data,
)

np.seterr(divide="ignore", invalid="ignore")


class CF(object):
    """ Docstring for DF """

    def __init__(self, Y_data, k, dist_func=correlation):
        self.Y_data = Y_data
        self.k = k
        self.dist_func = dist_func

        # number of users and items. Remember to add 1 since id starts from 0
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1

        self.Ybar_data = self.Y_data.copy()  # normalized

    def _normalize_Y(self):
        """
        Normalize data rating of users
        """
        users = self.Y_data[:, 0]  # all users - first col of the Y_data
        self.Ybar_data = self.Y_data.copy()
        self.mu = np.zeros((self.n_users,))

        for n in range(self.n_users):
            # row indices of rating done by user n
            # since indices need to be integers, we need to convert
            ids = np.where(users == n)[0].astype(np.int32)  # row = [0,...,n]

            # and the corresponding ratings
            ratings = self.Y_data[ids, 2]  # ratings [n,...n], n: [0, 5]

            # take mean
            m = np.mean(ratings)
            if np.isnan(m):
                m = 0  # to avoid empty array and nan value
            self.mu[n] = m

            # normalize
            self.Ybar_data[ids, 2] = ratings - self.mu[n]

        ###############################################################################
        # i create a matrix user and item, it's diff with Ybar in demographic_filtering
        # we need it because it will help us alot when we calc pearsonr correlation.
        # correlation(u, v)
        self.Ybar = sparse.coo_matrix(
            (self.Ybar_data[:, 2], (self.Ybar_data[:, 1], self.Ybar_data[:, 0])),
            (self.n_items, self.n_users),
        )

        self.Ybar = self.Ybar.tocsr()

    def _calc_similarity(self):
        """
        Calculate sim values of user with all users
        """
        Ybar_copy = self.Ybar.copy().toarray()
        self.S = np.corrcoef(Ybar_copy.T)

    def add(self, new_data):
        """
        Update Y_data matrix when new ratings come.
        For simplicity, suppose that there is no new user or item.
        """

    def fit(self):
        """
        Normalize data and calculate similarity matrix again (after
        some few ratings added)
        """
        self._normalize_Y()
        self._calc_similarity()

    def pred(self, u, i):
        """
        Predict the rating of user u for item i
        """
        # find users rated i
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)

        # find similarity btw current user and others
        # who rated i
        sim = self.S[u, users_rated_i]

        # find the k most similarity users
        a = np.argsort(sim)[-self.k :]

        nearest_s = sim[a]

        # ratings of nearest users rated item i
        r = self.Ybar[i, users_rated_i[a]]

        return (r * nearest_s)[0] / (np.abs(nearest_s).sum() + 1e8) + self.mu[u]

    def recommend(self, u):
        """
        The decision is made based on all i such that:
        self.pred(u, i) > 0. Suppose we are considering items which
        have not been rated by u yet.
        """
        ids = np.where(self.Y_data[:, 0] == u)[0]
        items_rated_by_u = self.Y_data[ids, 1].tolist()
        recommended_items = []
        for i in range(self.n_items):
            if i not in items_rated_by_u:
                rating = self.pred(u, i)
                if rating > 0:
                    recommended_items.append(i)
        return recommended_items

    def display(self):
        """
        Display all items which should be recommend for each user
        """
        print("Recommendation: ")
        for u in range(self.n_users):
            recommended_items = self.recommend(u)
            print("Recommend item(s): {0} to user {1}".format(recommended_items, u))


#######################################################################################
# i call function from another get_data module so please check it.

# RATINGS = get_ratings_data().values  # convert from dataframe to matrix
# RATINGS[:, :2] -= 1  # start from 0
# RS = CF(RATINGS, 50)
# RS.fit()
# RS.display()
#######################################################################################

RATE_TRAIN = get_rating_base_data().values  # convert to matrix
RATE_TEST = get_rating_test_data().values  # convert to matrix


RATE_TRAIN[:, :2] -= 1  # start from 0
RATE_TEST[:, :2] -= 1  # start from 0

RS = CF(RATE_TRAIN, k=25)
RS.fit()

n_tests = RATE_TEST.shape[0]
SE = 0
for n in range(n_tests):
    pred = RS.pred(RATE_TEST[n, 0], RATE_TEST[n, 1])
    SE += (pred - RATE_TEST[n, 2]) ** 2

RMSE = np.sqrt(SE / n_tests)
print("Collaborative Filtering, RMSE: ", RMSE)
