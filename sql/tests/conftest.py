import pytest

from .setup.dataloader import TestDbCollection, download_data_files

download_data_files()

dbcol = TestDbCollection()
print(dbcol)

@pytest.fixture(scope="module")
def test_databases():
    return dbcol

def test_client(client):
    pass
