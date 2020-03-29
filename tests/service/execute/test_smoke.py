import pytest

@pytest.mark.parametrize("project", [{"params": {"text": "this text"}, "uri": "modules/module-example"},
                                     {"params": {"dataset_name": "example", "budget": .2,
                                      "query": "SELECT COUNT(*) FROM example.example"},
                                      "uri": "modules/sql-module"}])
def test_execute_run(execution_client, project):
    execution_client.submit(params=project["params"],
                            uri=project["uri"])

@pytest.mark.dataverse_token
@pytest.mark.parametrize("project", [{"params": {"dataset_name": "demo_dataverse", "budget": .2,
                                      "query": "SELECT COUNT(*) FROM demo_dataverse.demo_dataverse"},
                                      "uri": "modules/sql-module"}])
def test_execute_w_dataverse(execution_client, project):
    execution_client.submit(params=project["params"],
                            uri=project["uri"])

