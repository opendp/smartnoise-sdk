import os
import json
import uuid
import logging
import pkg_resources
from requests import Session

from .restclient.models.project_run_details import ProjectRunDetails
from .restclient.rest_client import RestClient
from .restclient.models.dataset_read_request import DatasetReadRequest
from .restclient.models.dataset_document import DatasetDocument
from .restclient.models.release_dataset_document import ReleaseDatasetDocument
from .restclient.models.dataset_read_release_request import DatasetReadReleaseRequest

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

def _guid_header():
    """
    Generates a guid header for client, to be tracked by client/server.

    :return: A UUID
    :rtype: str
    """
    return str(uuid.uuid4())

class ExecutionClient(object):
    def submit(self, params, uri):
        client = _get_client()
        details = ProjectRunDetails(params=json.dumps(params),
                                    project_uri=uri)
        return client.executerun(details)

class DatasetClient(object):
    """
    A client for registering, reading and releasing differentially private
    datasets using the opendp-whitenoise service
    """
    def __init__(self):
        # Tag requests with this custom header for now
        self._guid = _guid_header()
        self.custom_headers = {'client_guid': self._guid}

    def _register_release_request_helper(self, dataset):
        """
        Helper for register/release, 
        both of which use DatasetDocuments as request formats
        """
        for key in KNOWN_DATASET_TYPE_KEYS:
            if not key in dataset:
                dataset[key]=None

        request = DatasetDocument(dataset_name=dataset['dataset_name'], \
            dataset_type=dataset['dataset_type'], \
            budget=dataset['budget'], \
            release_cost=dataset['release_cost'], \
            csv_details=dataset['csv_details'], \
            dataverse_details=dataset['dataverse_details'], \
            authorized_users=dataset['authorized_users'])
        
        return request

    def release(self, dataset):
        """
        Generates a DatasetDocument and sends it to the service.
        Requests the release of a Differentially Private DatasetDocument, with budget
        (to authorized users)
        Tags the request with Client guid.
        """
        client = _get_client()
        release_request = self._register_release_request_helper(dataset)
        return client.datasetrelease(release_request, custom_headers=self.custom_headers)

    def register(self, dataset):
        """
        Generates a DatasetDocument and sends it to the service.
        Requests the registration of this private DatasetDocument
        Tags the request with Client guid.
        """
        client = _get_client()
        register_request = self._register_release_request_helper(dataset)
        return client.datasetregister(register_request, custom_headers=self.custom_headers)

    def read(self, dataset_name, budget):
        """
        Generates a DatasetReadRequest and sends it to the service.
        Reads from a private DatasetDocument
        Tags the request with Client guid.
        """
        client = _get_client()
        read_request = DatasetReadRequest(dataset_name=dataset_name, budget=budget)
        return client.datasetread(read_request, custom_headers=self.custom_headers)
    
    def read_released(self, dataset_name):
        """
        Generates a DatasetReadReleaseRequest and sends it to the service.
        Tags the request with Client guid.
        """
        client = _get_client()
        read_released_request = DatasetReadReleaseRequest(dataset_name=dataset_name)
        return client.datasetreadreleased(read_released_request, custom_headers=self.custom_headers)

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
