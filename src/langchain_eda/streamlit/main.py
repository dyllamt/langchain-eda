import streamlit as st

import langchain_eda.streamlit.components as components


def main():
    # sets page layout
    st.set_page_config(
        layout="wide",
        initial_sidebar_state="auto",
        page_title="Langchain EDA",
    )

    # displays application components
    components.item()


if __name__ == "__main__":
    main()
