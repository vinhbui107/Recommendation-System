import numpy as np
import pandas as pd
from scipy import sparse
from get_data import (
    get_ratings_data,
    get_users_data,
    get_items_date,
    get_rating_base_data,
    get_rating_test_data
)

# load data and convert it to matrix
RATING = get_ratings_data()
RATING = np.asmatrix(RATING)

USERS = get_users_data()
USERS = np.asmatrix(USERS)

ITEMS = get_items_data()
ITEMS = np.asmatrix(ITEMS)

BASE = get

class Recommendation(object):
    """
    Docstring for Recommendation
    """

    def __init__(self, ratings_data, users_data, items_data, k=None):
        """
        Docstring for __init__
        """
        self.ratings_data = ratings_data
        self.k = k  # number of neighbor points

        self.n_users = int(np.max(self.ratings_data[:, 0]))
        self.n_items = int(np.max(self.ratings_data[:, 1]))

        self.ratings_data_normalized = None  # use it for normalize function


    def predict(self, users, item):
        """
        Predict items for user
        """

    def add(self, new_data):
        """
        Add new user into system
        """

    def update(self):
        """
        Calculate again when create a new user
        """


