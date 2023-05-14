import os
import sqlite3

import pandas as pd


TEST_DATABASE = "test.db"


def movie_dataset():
    return pd.DataFrame(
        columns=["title", "year", "score"],
        data=[["Mulan", 2021, 9], ["Lion King", 1990, 10]],
    )


def setup():
    connection = sqlite3.connect(TEST_DATABASE)
    try:
        movie_dataset().to_sql("movie", connection, index=False)
    except ValueError:
        pass
    return connection


def teardown(connection):
    connection.close()
    os.remove(TEST_DATABASE)
