import pytest

from burdock.client import get_dataset_client

@pytest.fixture(scope="session")
def dataset_client():
    return get_dataset_client()
    
