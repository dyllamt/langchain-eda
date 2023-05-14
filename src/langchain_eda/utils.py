import os
import sqlite3

import pandas as pd

TEST_DATABASE = "test.db"


def movie_dataset() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["title", "year", "score"],
        data=[["Mulan", 2021, 9], ["Lion King", 1990, 10]],
    )


def setup() -> sqlite3.Connection:
    connection = sqlite3.connect(TEST_DATABASE)
    try:
        movie_dataset().to_sql("movie", connection, index=False)
    except ValueError:
        pass
    return connection


def teardown(connection: sqlite3.Connection):
    connection.close()
    os.remove(TEST_DATABASE)
