from recommendation import Recommendation


class CollaborativeFiltering:
    def __init__(self):
        pass

    def normalize_data(self):
        """
        Normalization data of users
        """
        self.ratings_data_normalized = self.ratings_data.copy()

        for user_id in range(1, self.n_users + 1):
            all_ratings_of_user = []
            for i in range(len(self.ratings_data)):
                if self.ratings_data[i, 0] == user_id:
                    all_ratings_of_user.append(self.ratings_data[i, 2])

            m = np.mean(all_ratings_of_user)
            if np.isnan(m):
                m = 0

            for i in range(len(self.ratings_data_normalized)):
                if (
                    self.ratings_data_normalized[i, 0] == user_id
                    and self.ratings_data_normalized[i, 2] != 0
                ):
                    self.ratings_data_normalized[i, 2] = (
                        self.ratings_data_normalized[i, 2] - m
                    )
        return self.ratings_data_normalized
