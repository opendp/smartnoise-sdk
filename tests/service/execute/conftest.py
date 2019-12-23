import pytest

from burdock.restclient.models.project_run_details import ProjectRunDetails
from burdock.client import get_execution_client

@pytest.fixture(scope="session")
def execution_client(client):
    return get_execution_client()
