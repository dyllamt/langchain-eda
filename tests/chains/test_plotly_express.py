from langchain_eda.chains.plotly_express import plotly_express_chain


def test_plotly_express(dataframe):
    result = plotly_express_chain(dataframe).run("Scatter plot of movie score vs time")
    print(result)
