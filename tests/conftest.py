import os
import subprocess
import sys
import time
import sklearn.datasets
import pandas as pd


from opendp.whitenoise.metadata import CollectionMetadata
from opendp.whitenoise.metadata.collection import Table, Float

from subprocess import Popen, PIPE
from threading import Thread

import pytest

from requests import Session

from opendp.whitenoise.client import _get_client
from opendp.whitenoise.client.restclient.rest_client import RestClient
from opendp.whitenoise.client.restclient.models.secret import Secret
DATAVERSE_TOKEN_ENV_VAR = "WHITENOISE_DATAVERSE_TEST_TOKEN"

# Add the utils directory to the path
root_url = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()
sys.path.append(os.path.join(root_url, "utils"))

from service_utils import run_app  # NOQA

iris_dataset_path = os.path.join(root_url, "service", "datasets", "iris.csv")
if not os.path.exists(iris_dataset_path):
    sklearn_dataset = sklearn.datasets.load_iris()
    sklearn_df = pd.DataFrame(data=sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    sklearn_df.to_csv(iris_dataset_path)


iris_schema_path = os.path.join(root_url, "service", "datasets", "iris.yaml")
if not os.path.exists(iris_schema_path):
    iris = Table("iris", "iris", 150, [
                Float("sepal length (cm)", 4, 8),
                Float("sepal width (cm)", 2, 5),
                Float("petal length (cm)", 1, 7),
                Float("petal width (cm)", 0, 3)
    ])
    schema = CollectionMetadata([iris], "csv")
    schema.to_file(iris_schema_path, "iris")


@pytest.fixture(scope="session")
def client():
    url = os.environ.get("WHITENOISE_SERVICE_URL", "localhost")
    port = int(os.environ.get("WHITENOISE_SERVICE_PORT", 5001))

    client = _get_client()
    if DATAVERSE_TOKEN_ENV_VAR in os.environ:
        import pdb; pdb.set_trace()
        client.secretsput(Secret(name="dataverse:{}".format("demo_dataverse"),
                                 value=os.environ[DATAVERSE_TOKEN_ENV_VAR]))
    return client

def test_client(client):
    pass
