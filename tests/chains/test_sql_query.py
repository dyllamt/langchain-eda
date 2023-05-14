from langchain_eda.chains.sql_query import sql_query_chain


def test_sql_query(sqlite_uri):
    result = sql_query_chain(sqlite_uri).run("How many videos were made after 1992?")
    assert result == "SELECT COUNT(*) FROM movie WHERE year > 1992;"
