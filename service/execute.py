import json
import os

import mlflow
from mlflow.tracking.client import MlflowClient


def run(details):
    params = json.loads(details["params"])
    params["server_path"] = os.getcwd()
    project_uri = details["project_uri"]
    if not(project_uri.startswith("http://") or project_uri.startswith("https://")):
        assert project_uri.replace(".", "").replace("/", "").startswith("modules"), "Only support modules dir"

        project_uri = os.path.join(os.path.dirname(__file__), project_uri)

    submitted_run = mlflow.projects.run(project_uri,
                                        parameters=params,
                                        use_conda=False)
    out = MlflowClient().get_run(submitted_run.run_id).to_dictionary()
    return {"result": json.dumps(out)}
