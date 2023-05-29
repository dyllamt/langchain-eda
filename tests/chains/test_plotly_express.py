import plotly.express as px

from langchain_eda.chains.plotly_express import plotly_express_chain


def test_plotly_express(dataframe):
    result = plotly_express_chain(dataframe)("How does movie score vary over time?")
    assert result["plot"].__name__ == px.line.__name__
    assert result["kwargs"] == '{"x": "year", "y": "score", "color": null}'
