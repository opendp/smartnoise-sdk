import pytest

from .setup.dataloader import DbCollection, download_data_files

download_data_files()

dbcol = DbCollection()
print(dbcol)

@pytest.fixture(scope="module")
def test_databases():
    return dbcol

def test_client(client):
    pass
