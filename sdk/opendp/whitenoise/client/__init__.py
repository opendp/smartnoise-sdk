import os
import json
import logging
import pkg_resources
from requests import Session

from .restclient.models.project_run_details import ProjectRunDetails
from .restclient.rest_client import RestClient
from .restclient.models.dataset_read_request import DatasetReadRequest
from .restclient.models.dataset_document import DatasetDocument

module_logger = logging.getLogger(__name__)

KNOWN_DATASET_TYPE_KEYS = ["csv_details", "dataverse_details"]

class _MockCredentials(object):
    def signed_session(self, session=None):
        return session if session is not None else Session()


def _get_client():
    port = os.environ.get("WHITENOISE_SERVICE_PORT", "5000")
    url = os.environ.get("WHITENOISE_SERVICE_URL", "localhost:{}".format(port))

    base_url = "{}/api/".format(url)
    base_url = base_url if base_url.startswith("http") else "http://" + base_url
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

        for key in KNOWN_DATASET_TYPE_KEYS:
            if not key in dataset:
                dataset[key]=None

        register_request = DatasetDocument(dataset_name=dataset['dataset_name'], \
            dataset_type=dataset['dataset_type'], \
            budget=dataset['budget'], \
            csv_details=dataset['csv_details'], \
            dataverse_details=dataset['dataverse_details'])
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
        if len(client_overrides) > 1:
                module_logger.warning("Multiple client overrides found {}".format(client_overrides))
    return DatasetClient()

def get_execution_client():
    return ExecutionClient()


__all__ = ["get_dataset_client", "get_execution_client"]
