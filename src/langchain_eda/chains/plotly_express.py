from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import pydantic
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from marvin import ai_model
from pydantic import Field

# defines figures supported for plotting
supported_figures = {
    "scatter": px.scatter,
    # "line": px.line,
    "bar": px.bar,
    "histogram": px.histogram,
    "violin": px.violin,
}


@ai_model(llm_model_name="gpt-3.5-turbo", llm_model_temperature=0.0)
class PlottingVariables(pydantic.BaseModel):
    """Variable names to plot in a figure."""

    x: Optional[str] = None
    y: Optional[str] = None
    color: Optional[str] = None


# defines prompts and chain
_figure_selection_prompt = """You are a data scientist that is skilled in programming and data visualization. Given an
input question, determine an appropriate plotting type from the comma separated list below:

{plot_choices}

You must choose an exact string match from one of the options above. Remember, you MUST match one of the visualization
choices EXACTLY in your response. Do not include any additional information in your response, except for one of the
visualization choices.

# question
{query}
"""
figure_selection_prompt = PromptTemplate(
    template=_figure_selection_prompt,
    input_variables=["query"],
    partial_variables={"plot_choices": ", ".join(supported_figures.keys())},
)


_column_replacement_prompt = """You are an expert in manipulating json data structures. Your job is to replace leaf
values in json data structures. You must only replace leaf values in the data structure and you must replace all of
them. The choices for replacement are as follows:

{allowed_columns}

You must strictly replace the leaf values with values from the allowed list above.

Use the following rules for replacement:
1. if a value in the json is conceptually similar or related to one of the allowed replacements, replace it with the
allowed replacement. Be lenient with synonyms. For example, month is similar to time, year, minute, etc.
2. if no value in the allowed replacements is similar to the json value, replace it with null

Remember that you must replace ALL of the leaf values and ONLY the leaf values. Remember that you must ONLY return the
replaced json structure with no additional explanation. Remember that ONLY the allowed list of columns can be used. You
must not leave any values un-replaced.

# json structure
{json_structure}
"""
column_replacement_prompt = PromptTemplate(
    template=_column_replacement_prompt,
    input_variables=["allowed_columns", "json_structure"],
)


class PlotlyExpressChain(Chain):
    dataframe: pd.DataFrame
    llm: BaseLanguageModel = Field(
        default_factory=lambda: ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0)  # type: ignore
    )

    @property
    def input_keys(self) -> List[str]:
        return ["prompt"]

    @property
    def output_keys(self) -> List[str]:
        return ["plot", "kwargs"]

    @property
    def figure_selection_chain(self):
        return LLMChain(llm=self.llm, prompt=figure_selection_prompt)

    @property
    def plot_kwargs_chain(self):
        return PlottingVariables

    @property
    def replaced_kwargs_chain(self):
        return LLMChain(llm=self.llm, prompt=column_replacement_prompt)

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        # configures run manager
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()

        # determines type of plot
        figure_selection_inputs = {
            "query": inputs[self.input_keys[0]],
            "stop": ["\nQuestion:"],
        }
        figure_selection = (
            self.figure_selection_chain.predict(callbacks=_run_manager, **figure_selection_inputs).strip().lower()
        )
        print(figure_selection)

        # determines plotly kwargs
        initial_kwargs = self.plot_kwargs_chain(inputs[self.input_keys[0]]).json()

        # string replaces with columns from dataframe
        replaced_kwargs_inputs = {
            "allowed_columns": ", ".join(self.dataframe.columns),
            "json_structure": initial_kwargs,
        }
        replaced_kwargs = self.replaced_kwargs_chain.predict(callbacks=_run_manager, **replaced_kwargs_inputs).strip()

        # returns kwargs for the selected plotting function
        return dict(plot=supported_figures[figure_selection], kwargs=replaced_kwargs)


def plotly_express_chain(dataframe: pd.DataFrame) -> PlotlyExpressChain:
    return PlotlyExpressChain(dataframe=dataframe)
