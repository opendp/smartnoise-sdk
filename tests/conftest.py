import pytest
import pandas as pd

from .setup.dataloader import TestDbCollection, _get_client

@pytest.fixture(scope="session")
def client():
    client = _get_client()
    if DATAVERSE_TOKEN_ENV_VAR in os.environ:
        import pdb; pdb.set_trace()
        client.secretsput(Secret(name="dataverse:{}".format("demo_dataverse"),
                                 value=os.environ[DATAVERSE_TOKEN_ENV_VAR]))
    return client

dbcol = TestDbCollection()
print(dbcol)

@pytest.fixture(scope="module")
def test_databases():
    return dbcol


def test_client(client):
    pass
