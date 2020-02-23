import os
import json
from requests import Session

from .restclient.models.project_run_details import ProjectRunDetails
from .restclient.rest_client import RestClient
from .restclient.models.dataset_read_request import DatasetReadRequest


class _MockCredentials(object):
    def signed_session(self, session=None):
        return session if session is not None else Session()


def _get_client():
    url = os.environ.get("BURDOCK_SERVICE_URL", "localhost")
    port = int(os.environ.get("BURDOCK_SERVICE_PORT", 5000))

    base_url = "http://{}:{}/api/".format(url, port)
    client = RestClient(_MockCredentials(), base_url)
    return client

class ExecutionClient(object):
    def submit(self, params, uri):
        client = _get_client()
        details = ProjectRunDetails(params=json.dumps(params),
                                    project_uri=uri)
        return client.executerun(details)

class DatasetClient(object):
    def read(self, dataset_name, budget):
        client = _get_client()
        read_request = DatasetReadRequest(dataset_name=dataset_name)
        return client.datasetread(read_request)


def get_dataset_client():
    return DatasetClient()

def get_execution_client():
    return ExecutionClient()

__all__ = ["get_dataset_client", "get_execution_client"]
