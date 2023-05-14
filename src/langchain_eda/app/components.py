from typing import Any

import pandas as pd
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler

from langchain_eda.chains.sql_query import sql_query_chain


class TextLog(BaseCallbackHandler):
    text: str = ""

    def on_text(self, text: str, **kwargs: Any) -> Any:
        self.text += "> Prompt:\n"
        self.text += (
            text.replace("[0m", "")
            .replace("[32;1m", "")
            .replace("[1;3m", "")
            .replace("", "")
            .replace("Prompt after formatting:", "")
        )
        self.text += "\n"

    def on_chain_end(self, outputs, *, run_id, parent_run_id=None, **kwargs) -> Any:
        self.text += "\n> Response:\n\n"
        for k in outputs.keys():
            self.text += outputs[k] + "\n\n"
        self.text += "-----\n"


def generate_sql_query():
    handler = TextLog()
    user_query = st.text_area(label="What would you like to know?", value="Give me all the info on movies")
    sql_query = st.text_area(
        label="Generated SQL query:", value=sql_query_chain("sqlite:///test.db").run(user_query, callbacks=[handler])
    )
    return user_query, sql_query, handler


def run_sql_query(connection, sql_query):
    result = pd.read_sql(sql_query, connection)
    st.dataframe(result, use_container_width=True)
