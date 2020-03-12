from get_data import *
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity


class DemographicFiltering:
    """ Docstring for DF """

    def __init__(self, users, ratings, k):
        self.users = users
        self.ratings = ratings
        self.k = k  # number neighbors

        self.n_users = self.users.user_id.count()
        self.users_features = None
        self.similarities = None  # this is a matrix similarities of users

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

    def _similarity(self):
        """
        calculate sim values of user with all users
        """
        # call it because we need data for calculate similarity
        self._get_users_features()
        # now i convert to array for calculate cosine
        self.users_features = self.users_features.to_numpy()
        # calculate similarity
        self.similarities = cosine_similarity(self.users_features, self.users_features)
        # now we have matrix similarity

    def _pred(self):
        """
        Predict
        """


# i call function from another get_data module so please check it.
# and it return dataframe
RATINGS = get_ratings_data()
USERS = get_users_data()


DF = DemographicFiltering(USERS, RATINGS, 2)
DF._similarity()
