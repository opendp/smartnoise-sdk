import json

import pytest

@pytest.mark.parametrize("project", [{"params": {"alpha": .4}, "uri": "https://github.com/mlflow/mlflow-example"},
                                     {"params": {"text": "this text"}, "uri": "modules/module-example"},
                                     {"params": {"dataset_name": "example", "column_name": "a", "budget": .2},
                                      "uri": "modules/psi-count-module"}])
def test_execute_run(execution_client, project):
    execution_client.submit(params=project["params"],
                            uri=project["uri"])
