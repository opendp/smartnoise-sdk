import os
from requests import Session

from burdock.restclient.rest_client import RestClient
from burdock.restclient.models.dataset_read_request import DatasetReadRequest


class _MockCredentials(object):
    def signed_session(self, session=None):
        return session if session is not None else Session()


def _get_client():
    port = int(os.environ.get("BURDOCK_SERVICE_PORT", 5000))

    base_url = "http://localhost:{}/api/".format(port)
    client = RestClient(_MockCredentials(), base_url)
    return client


class DatasetClient(object):
    def read(self, dataset_name, budget):
        client = _get_client()
        read_request = DatasetReadRequest(dataset_name=dataset_name)
        return client.datasetread(read_request)


def get_dataset_client():
    return DatasetClient()


__all__ = "get_dataset_client"
