from typing import Any, Callable, Dict, Type, Dict, Optional, List
import inspect
import io

import pandas as pd
import plotly.express as px
import pydantic
from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import Field


# dynamically creates up-to-date schemas for prompt formatting
supported_figures = {
    "scatter": px.scatter,
    "line": px.line,
    "bar": px.bar,
    "histogram": px.histogram,
    "violin": px.violin,
}


def create_model_from_signature(
    func: Callable, model_name: str, base_model: Type[pydantic.BaseModel] = pydantic.BaseModel
):
    args, _, varkw, defaults, kwonlyargs, kwonlydefaults, annotations = inspect.getfullargspec(func)
    defaults = defaults or []
    args = args or []

    non_default_args = len(args) - len(defaults)
    defaults = (...,) * non_default_args + defaults

    keyword_only_params = {param: kwonlydefaults.get(param, Any) for param in kwonlyargs}
    params = {param: (annotations.get(param, Any), default) for param, default in zip(args, defaults)}

    class Config:
        extra = "allow"

    # Allow extra params if there is a **kwargs parameter in the function signature
    config = Config if varkw else None

    return pydantic.create_model(
        model_name,
        **params,
        **keyword_only_params,
        __base__=base_model,
        __config__=config,
    )


supported_schemas = {i: create_model_from_signature(j, j.__name__) for i, j in supported_figures.items()}


# defines prompts and chain
_figure_selection_prompt = """You are a data scientist that is skilled in programming and data visualization. Given an
input question, determine an appropriate plotting visualization from the comma separated list below.

Visualization choices: {plot_choices}

Use the following format:

Question: Question here
Visualization: One of the visualization choices

You must match one of the visualizations exactly.

Question: {query}
Visualization: 
"""
figure_selection_prompt = PromptTemplate(
    template=_figure_selection_prompt,
    input_variables=["query"],
    partial_variables={"plot_choices": ", ".join(supported_schemas.keys())},
)


_plotly_kwargs_prompt = """You are a data scientist that is skilled in python and data visualization. You need to
create a visualization to answer a user's question based on the pandas DataFrame shown below. The visualization should
be a {plot_type} chart.

{format_instructions}

Use the following format:

Question: Question here
Dataframe info: Dataset info here
Output schema: Output schema here

Question: {query}
Dataframe info: {dataframe_info}
Output schema: 
"""
plotly_kwargs_prompt = PromptTemplate(
    template=_plotly_kwargs_prompt,
    input_variables=["query", "plot_type", "dataframe_info", "format_instructions"],
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
        return ["result"]

    @property
    def figure_selection_chain(self):
        return LLMChain(llm=self.llm, prompt=figure_selection_prompt)

    @property
    def plotly_kwargs_chain(self):
        return LLMChain(llm=self.llm, prompt=plotly_kwargs_prompt)

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        # configures run manager
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()

        # determines type of plot
        figure_selection_inputs = {
            "query": inputs[self.input_keys[0]],
            "stop": ["\nQuestion:"],
        }
        figure_selection = self.figure_selection_chain.predict(
            callbacks=_run_manager, **figure_selection_inputs
        ).strip().lower()
        print(figure_selection)

        # determines relevant plotly kwargs
        dataframe_info = io.StringIO()
        self.dataframe.info(buf=dataframe_info)
        dataframe_info = dataframe_info.getvalue()  # get string value
        print(dataframe_info)

        parser = PydanticOutputParser(pydantic_object=supported_schemas[figure_selection])
        plotly_kwargs_inputs = {
            "query": inputs[self.input_keys[0]],
            "plot_type": figure_selection,
            "dataframe_info": dataframe_info,
            "format_instructions": parser.get_format_instructions(),
        }
        plotly_kwargs = self.plotly_kwargs_chain.predict(callbacks=_run_manager, **plotly_kwargs_inputs).strip()

        # returns kwargs for plotly plot
        return dict(result=plotly_kwargs)
        # return dict(result=parser.parse(plotly_kwargs))


def plotly_express_chain(dataframe: pd.DataFrame) -> PlotlyExpressChain:
    return PlotlyExpressChain(dataframe=dataframe)


if __name__ == "__main__":
    import langchain_eda.utils as utils
    dataframe = utils.movie_dataset()
    result = plotly_express_chain(dataframe).run("Scatter plot of movie score vs time")
    print(result)
