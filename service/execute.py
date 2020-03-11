import json
import os

import mlflow
from mlflow.tracking.client import MlflowClient


def run(details):
    """Execute the specified module with the given parameters.

    :param details: A dictionary of additional configuration parameters.
    :type details: dict {"project_uri": str, "params": dict{str: str}
    :return: Container string for the run files.
    :rtype: str
    """

    params = json.loads(details["params"])
    project_uri = details["project_uri"]  # TODO only support whitenoise modules
    if not(project_uri.startswith("http://") or project_uri.startswith("https://")):
        assert project_uri.replace(".", "").replace("/", "").startswith("modules"), "Only support modules dir"

        project_uri = os.path.join(os.path.dirname(__file__), project_uri)

    submitted_run = mlflow.projects.run(project_uri,
                                        parameters=params,
                                        use_conda=False)
    path = MlflowClient().download_artifacts(submitted_run.run_id, "result.json")  # TODO move to report.json
    with open(path, "r") as stream:
        json_str = stream.read()
    return {"result": json.dumps(json.loads(json_str))}
