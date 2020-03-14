"""
we have this file because these datas can use for some file in RS.
"""

import pandas as pd
import numpy as np


def get_users_data():
    """
    Get demographic data of users
    Output: matrix users
    """
    _user_cols = ["user_id", "age", "sex", "occupation", "zip_code"]
    # users = pd.read_csv("./data_small/user.csv", sep="|", names=_user_cols)
    users = pd.read_csv("./ml-100k/u.user", sep="|", names=_user_cols)

    return users


def get_ratings_data():
    """
    Get the ratings data
    Output: dataframe ratings of all user
    """
    _data_cols = ["user_id", "item_id", "rating", "timestamp"]
    # data = pd.read_csv("./data_small/data.csv", sep="\t", names=_data_cols)
    data = pd.read_csv("./ml-100k/u.data", sep="\t", names=_data_cols)
    return data


def get_items_data():
    """
    Get items data
    Output: dataframe items
    """
    _item_cols = [
        "movie_id",
        "movie_title",
        "release_date",
        "video_release_date",
        "imdb_url",
        "unknown",
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "filmNoir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "SciFi",
        "Thriller",
        "War",
        "Western",
    ]
    # items = pd.read_csv('./ml-100k/u.item',
    # sep='|',
    # names=_item_cols,
    # encoding='latin-1')
    #
    items = pd.read_csv(
        "./data_small/item.csv", sep="|", names=_item_cols, encoding="latin-1"
    )
    return items


def get_rating_test_data():
    """
    Get rating_test data
    Output: dataframe rating_test
    """
    _rating_cols = ["user_id", "item_id", "rating", "timestamp"]
    rating_test = pd.read_csv(
        "./ml-100k/ua.test", sep="\t", names=_rating_cols, encoding="latin-1"
    )
    return rating_test


def get_rating_base_data():
    """
    Get rating_base data
    Output: dataframe rating_base
    """
    _rating_cols = ["user_id", "item_id", "rating", "timestamp"]
    rating_base = pd.read_csv(
        "./ml-100k/ua.base", sep="\t", names=_rating_cols, encoding="latin-1"
    )
    return rating_base
