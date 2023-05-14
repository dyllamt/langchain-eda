import pandas as pd
import streamlit as st

from langchain_eda.chains.sql_query import sql_query_chain


def generate_sql_query():
    user_query = st.text_area(label="What would you like to know?", value="Give me all the info on movies")
    sql_query = st.text_area(label="Generated SQL query:", value=sql_query_chain("sqlite:///test.db").run(user_query))
    return user_query, sql_query


def run_sql_query(connection, sql_query):
    result = pd.read_sql(sql_query, connection)
    st.dataframe(result, use_container_width=True)
