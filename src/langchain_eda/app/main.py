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

    # application contents
    c1, c2 = st.columns(2)
    with c1:  # sql generation and query execution
        st.header("Data Explorer")
        user_query, sql_query = components.generate_sql_query()
        components.run_sql_query(connection, sql_query)

    # teardown test database
    utils.teardown(connection)


if __name__ == "__main__":
    main()
