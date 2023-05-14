from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers.list import CommaSeparatedListOutputParser
from langchain.prompts.prompt import PromptTemplate
from langchain.sql_database import SQLDatabase
from pydantic import Field

_selector_prompt = """Given the below input question and list of potential tables, output a comma separated list of the
table names that may be necessary to answer this question.

Question: {input}

Table Names: {table_names}

Relevant Table Names:"""
selector_prompt = PromptTemplate(
    input_variables=["input", "table_names"],
    template=_selector_prompt,
    output_parser=CommaSeparatedListOutputParser(),
)


_sqlite_prompt = """You are a SQLite expert. Given an input question, create a syntactically correct SQLite query to
run. Never query for all columns from a table. You must query only the columns that are needed to answer the question.
Wrap each column name in double quotes (") to denote them as delimited identifiers.Pay attention to use only the column
names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to
which column is in which table. Pay attention to use date('now') function to get the current date, if the question
involves "today".

Use the following format:

Question: Question here
SQLQuery: SQL Query to run

Only use the following tables:
{table_info}

Question: {input}
SQLQuery:"""
sqlite_prompt = PromptTemplate(
    input_variables=["input", "table_info"],
    template=_sqlite_prompt,
)


class SQLQueryChain(Chain):
    """Inspects the tables in a SQL database to generate an appropriate query for a user's question."""

    database: SQLDatabase
    llm: BaseLanguageModel = Field(
        default_factory=lambda: ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)  # type: ignore
    )

    @property
    def input_keys(self) -> List[str]:
        return ["prompt"]

    @property
    def output_keys(self) -> List[str]:
        return ["result"]

    @property
    def selected_tables_chain(self):
        return LLMChain(llm=self.llm, prompt=selector_prompt)

    @property
    def sql_query_chain(self):
        return LLMChain(llm=self.llm, prompt=sqlite_prompt)

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        # configures run manager
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()

        # determines which tables to use
        table_names = self.database.get_usable_table_names()
        _lowercase_table_names = [n.lower() for n in table_names]
        selected_tables_inputs = {
            "input": inputs[self.input_keys[0]],
            "table_names": ", ".join(table_names),
        }
        selected_tables = self.selected_tables_chain.predict_and_parse(**selected_tables_inputs)
        table_names_to_use = [name for name in selected_tables if name.lower() in _lowercase_table_names]

        # generates sql query
        sql_query_inputs = {
            "input": inputs[self.input_keys[0]],
            "table_info": self.database.get_table_info(table_names=table_names_to_use),
            "stop": ["\nQuestion:"],
        }
        sql_query = self.sql_query_chain.predict(callbacks=_run_manager.get_child(), **sql_query_inputs).strip()

        # returns sql query
        return dict(result=sql_query)


def sql_query_chain(uri: str) -> SQLQueryChain:
    return SQLQueryChain(database=SQLDatabase.from_uri(database_uri=uri))
