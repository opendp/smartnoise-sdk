import pytest

@pytest.mark.parametrize("project", [{"params": {"dataset_name": "reddit", "budget": .2,
                                      "query": "SELECT ngram, COUNT(*) as n FROM reddit.reddit GROUP BY ngram"},
                                      "uri": "modules/sql-module"}])
def test_execute_run(execution_client, project):
    execution_client.submit(params=project["params"],
                            uri=project["uri"])
