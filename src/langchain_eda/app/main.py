import asyncio

import streamlit as st

TEST_DATABASE = "test.db"


def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()


def main():
    # solves runtime error w/ marvin dependency
    loop = get_or_create_eventloop()
    asyncio.set_event_loop(loop)
    import langchain_eda.app.components as components  # need to import after setting event loop
    import langchain_eda.utils as utils

    # sets page layout
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Langchain EDA",
    )

    # setup test database
    connection = utils.app_setup()

    # displays application components
    col1, col2 = st.columns(2)
    with col1:
        df = components.data_explorer(connection=connection)
    with col2:
        components.data_visualization(df=df)

    # teardown test database
    utils.teardown(connection)


if __name__ == "__main__":
    main()
