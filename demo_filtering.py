import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from get_data import (
    get_ratings_data,
    get_users_data,
    get_rating_base_data,
    get_rating_test_data,
)


class DemographicFiltering(object):
    """ Docstring for DF """

    def __init__(self, users, ratings, k):
        self.users = users
        self.ratings = ratings
        self.k = k

        self.n_users = self.users.user_id.count()

        self.ratings_normalized = None
        self.users_features = None
        self.similarities = None
        self.mu = None

    def _get_users_features(self):
        """
        convert demographic data of user to binary
        """
        self.users_features = self.users.copy()
        # First i convert age follow:
        # 1:  "Under 18"
        # 18:  "18-24"
        # 25:  "25-34"
        # 35:  "35-44"
        # 45:  "45-49"
        # 50:  "50-55"
        # 56:  "56+"
        self.users_features["age"] = self.users_features.age.map(
            lambda x: 1
            if int(x) >= 1 and int(x) < 18
            else (
                18
                if int(x) >= 18 and int(x) < 25
                else (
                    25
                    if int(x) >= 25 and int(x) < 35
                    else (
                        35
                        if int(x) >= 35 and int(x) < 45
                        else (
                            45
                            if int(x) >= 45 and int(x) < 50
                            else (50 if int(x) >= 50 and int(x) < 56 else 56)
                        )
                    )
                )
            )
        )
        # convert sex: if M == 1 else F == 0
        self.users_features["sex"] = self.users_features.sex.map(
            lambda x: 1.0 * (x == "M")
        )
        self.users_features.drop(["zip_code",], axis=1, inplace=True)  # we dont need it

        # The get_dummies() function is used to convert categorical variable
        # into dummy/indicator variables.
        self.users_features = pd.get_dummies(
            self.users_features, columns=["age", "occupation"]
        )
        # i set index of users_features dataframe is user_id
        self.users_features.set_index("user_id", inplace=True)

    def _calc_similarity(self):
        """
        calculate sim values of user with all users
        """
        # now i convert from dataframe to array for calculate cosine
        self.users_features = self.users_features.to_numpy()
        # calculate similarity
        self.similarities = cosine_similarity(self.users_features, self.users_features)

    def _normalize_ratings(self):
        """
        normalize data rating of users
        """
        users = self.ratings[:, 0]
        self.ratings_normalized = self.ratings.copy()
        self.mu = np.zeros((self.n_users,))

        for n in range(self.n_users):  # user + 1 because user_id from 1
            # row indices of rating done by user n
            # since indices need to be integers, we need to convert
            ids = np.where(users == n)[0].astype(np.int32)  # row = [0,...,n]

            # items were rated by u
            item_ids = self.ratings[ids, 1]  # items = [1,..,n]

            # and the corresponding ratings
            ratings = self.ratings[ids, 2]  # ratings [n,...n], n: [0, 5]

            # take mean
            m = np.mean(ratings)
            if np.isnan(m):
                m = 0  # to avoid empty array and nan value
            self.mu[n] = m

            # normalize
            self.ratings_normalized[ids, 2] = ratings - self.mu[n]

    def fit(self):
        self._get_users_features()
        self._calc_similarity()
        self._normalize_ratings()

    def pred(self, u, i):
        """
        predict the rating tof user u for item i
        """
        # find users rated i
        ids = np.where(self.ratings[:, 1] == i)[0].astype(np.int32)  # row = [0,...,n]
        users_rated_i = (self.ratings[ids, 0]).astype(np.int32)  # users_id = [0,..,n]

        # find similarity btw current user and others
        # who rated i
        sim = self.similarities[u, users_rated_i]  # sims = [f,...,f] f: [0, 1]

        # find the k most similarity users
        a = np.argsort(sim)[-self.k :]

        nearest_s = sim[a]

        # ratings of nearest users rated item i
        r = self.ratings_normalized[,]

        return (r * nearest_s)[0] / (nearest_s.sum() + 1e-8) + self.mu[u]

    def recommend(self, u):
        pass

    def display(self):
        pass


#######################################################################################
# users, items start from 1 -> n.
#

# i call function from another get_data module so please check it.
RATINGS = get_ratings_data().values  # convert from dataframe to matrix
RATINGS[:, :2] -= 1  # start from 0 will easy calculate

# not convert to matrix because
# dataframe will easy to change values and get users features
USERS = get_users_data()  # dataframe


RS = DemographicFiltering(USERS, RATINGS, 3)
RS.fit()
print(RS.ratings)
print(RS.ratings_normalized)
RS.display()
RS.pred(0, 3)

#######################################################################################
RATINGS_BASE = get_rating_base_data().values
RATINGS_TEST = get_rating_test_data().values
