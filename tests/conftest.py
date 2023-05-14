import os
import sqlite3

import pandas as pd
import pytest

TEST_DATABASE = "test.db"


@pytest.fixture(scope="module")
def movie_dataset():
    return pd.DataFrame(
        columns=["title", "year", "score"],
        data=[["Mulan", 2021, 9], ["Lion King", 1990, 10]],
    )


@pytest.fixture(scope="module")
def sqlite_uri(movie_dataset):
    # setup
    connection = sqlite3.connect(TEST_DATABASE)
    movie_dataset.to_sql("movie", connection, index=False)

    # fixture
    yield f"sqlite:///{TEST_DATABASE}"

    # teardown
    connection.close()
    os.remove(TEST_DATABASE)
