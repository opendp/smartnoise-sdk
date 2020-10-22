import pytest

from opendp.smartnoise.client import get_execution_client

@pytest.fixture(scope="session")
def execution_client(client):
    return get_execution_client()
