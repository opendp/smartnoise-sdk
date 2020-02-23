import pytest

@pytest.mark.parametrize("project", [{"params": {"dataset_name": "iris", "budget": 2.5,
                                         "x_features": '["sepal length (cm)"]', "y_targets":'["petal width (cm)"]' },
                                      "uri": "modules/lin-reg-module"}])

def test_execute_run(execution_client, project):
    execution_client.submit(params=project["params"],
                            uri=project["uri"])
