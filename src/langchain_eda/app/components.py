import json
import sqlite3
from typing import Any, Callable, Tuple

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler

from langchain_eda.chains.sql_query import sql_query_chain
from langchain_eda.chains.plotly_express import plotly_express_chain


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


def _lang_log(handler: TextLog):
    with st.expander(label="*Lang log*"):
        st.text(handler.text)


def _generate_sql_query() -> Tuple[str, str, TextLog]:
    handler = TextLog()
    user_query = st.text_area(label="What would you like to know?", value="Give me all the info on movies.")
    sql_query = st.text_area(
        label="Generated SQL query:", value=sql_query_chain("sqlite:///test.db").run(user_query, callbacks=[handler])
    )
    return user_query, sql_query, handler


def _run_sql_query(connection: sqlite3.Connection, sql_query: str) -> pd.DataFrame:
    result = pd.read_sql(sql_query, connection)
    st.dataframe(result, use_container_width=True)
    return result


def data_explorer(connection: sqlite3.Connection) -> pd.DataFrame:
    """Element that translates a natural language prompt into a SQL query and executes it against a database.

    Additionally, the element will transcribe a log of the natural language communication.

    Parameters
    ----------
    connection
        Sqlite connection to a database.

    Returns
    -------
    The results of the SQL query.
    """
    st.header("Data Explorer")
    user_query, sql_query, handler = _generate_sql_query()
    result = _run_sql_query(connection, sql_query)
    st.markdown("---")
    with st.expander(label="*Lang log*"):
        st.text(handler.text)
    return result


def _generate_plotly_spec(df: pd.DataFrame) -> Tuple[str, Callable, str, TextLog]:
    handler = TextLog()
    user_query = st.text_area(label="What would you like to visualize?", value="Scatter plot of score over time.")
    results = plotly_express_chain(df)(user_query, callbacks=[handler])
    return user_query, results["plot"], results["kwargs"], handler


def _run_plotly_spec(df: pd.DataFrame, func: Callable, kwargs: str) -> go.Figure:
    fig = func(df, **json.loads(kwargs))
    st.plotly_chart(fig)
    return fig


def data_visualization(df: pd.DataFrame) -> go.Figure:
    """Element that translates a natural language prompt into a set of kwargs to pass to a plotly express call.

    Additionally, the element will transcribe a log of the natural language communication.

    Parameters
    ----------
    df
        Data that you are visualizing.

    Returns
    -------
    Does not return
    """
    st.header("Visualization")
    user_query, plot_func, kwargs, handler = _generate_plotly_spec(df)
    fig = _run_plotly_spec(df, plot_func, kwargs)
    st.markdown("---")
    with st.expander(label="*Lang log*"):
        st.text(handler.text)
    return fig
