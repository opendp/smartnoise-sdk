import os
import json
import logging
import pkg_resources
from requests import Session

from .restclient.models.project_run_details import ProjectRunDetails
from .restclient.rest_client import RestClient
from .restclient.models.dataset_read_request import DatasetReadRequest
from .restclient.models.dataset_put_document import DatasetPutDocument

module_logger = logging.getLogger(__name__)

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
    def register(self, dataset):
        client = _get_client()
        register_request = DatasetPutDocument(dataset_name=dataset['dataset_name'], dataset_type=dataset['dataset_type'], host=dataset['host'], schema=dataset['schema'], budget=dataset['budget'], key=dataset['key'], token=dataset['token'])
        return client.datasetregister(register_request)

    def read(self, dataset_name, budget):
        client = _get_client()
        read_request = DatasetReadRequest(dataset_name=dataset_name, budget=budget)
        return client.datasetread(read_request)

def get_dataset_client():
    client_overrides = [entrypoint for entrypoint in pkg_resources.iter_entry_points("opendp_whitenoise_dataset_client")]
    if len(client_overrides) == 1:

        try:
            entrypoint = client_overrides[0]
            extension_class = entrypoint.load()
            return extension_class()
        except Exception as e:  # pragma: no cover
                msg = "Failure while loading {} with exception {}.".format(
                    entrypoint, e)
                module_logger.warning(msg)
    else:
                module_logger.warning("Multiple client overrides found {}".format(client_overrides))
    return DatasetClient()

def get_execution_client():
    return ExecutionClient()


__all__ = ["get_dataset_client", "get_execution_client"]
