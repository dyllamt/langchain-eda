import pytest

import langchain_eda.utils as utils


@pytest.fixture(scope="module")
def sqlite_uri():
    # setup
    connection = utils.setup()

    # fixture
    yield f"sqlite:///{utils.TEST_DATABASE}"

    # teardown
    utils.teardown(connection)


@pytest.fixture(scope="module")
def dataframe():
    yield utils.movie_dataset()
