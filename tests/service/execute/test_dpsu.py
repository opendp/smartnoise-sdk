import pytest

@pytest.mark.parametrize("project", [{"params": {"dataset_name": "PUMS", "budget": .2,
                                      "query": "SELECT married, educ FROM PUMS.PUMS GROUP BY married, educ"},
                                      "uri": "modules/sql-module"}])
def test_execute_run(execution_client, project):
    execution_client.submit(params=project["params"],
                            uri=project["uri"])
