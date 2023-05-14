import streamlit as st

import langchain_eda.app.components as components
import langchain_eda.utils as utils

TEST_DATABASE = "test.db"


def main():
    # sets page layout
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Langchain EDA",
    )

    # setup test database
    connection = utils.setup()

    # single column for data exploration
    components.data_explorer(connection=connection)

    # teardown test database
    utils.teardown(connection)


if __name__ == "__main__":
    main()
