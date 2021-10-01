import pytest
import os
from opendp.smartnoise.client import _get_client
from opendp.smartnoise.client.restclient.models.secret import Secret

DATAVERSE_TOKEN_ENV_VAR = "SMARTNOISE_DATAVERSE_TEST_TOKEN"

@pytest.fixture(scope="session")
def client():
    client = _get_client()
    if DATAVERSE_TOKEN_ENV_VAR in os.environ:
        import pdb; pdb.set_trace()
        client.secretsput(Secret(name="dataverse:{}".format("demo_dataverse"),
                                 value=os.environ[DATAVERSE_TOKEN_ENV_VAR]))
    return client

from .setup.dataloader import TestDbCollection, download_data_files

download_data_files()

dbcol = TestDbCollection()
print(dbcol)

@pytest.fixture(scope="module")
def test_databases():
    return dbcol

def test_client(client):
    pass
