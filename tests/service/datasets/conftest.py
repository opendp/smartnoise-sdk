import pytest

from opendp.whitenoise.client import get_dataset_client


@pytest.fixture(scope="session")
def dataset_client(client):
    return get_dataset_client()
