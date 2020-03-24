import os
import json
import pandas as pd

import pytest

from opendp.whitenoise.client import get_dataset_client
from opendp.whitenoise.data.adapters import load_dataset

@pytest.mark.parametrize("dataset_name", ["example_csv"])
def test_read_local_csv(dataset_client, dataset_name):
    """
    READ TEST
    Checks that a service/service owner owner can read existing dataset.
    """
    dataset_document = dataset_client.read(dataset_name)
    local_csv_details = dataset_document.csv_details

    with open(local_csv_details.local_path, "r") as stream:
        text = stream.read()
        assert len(text) > 0

@pytest.mark.parametrize("dataset_name", ["example_csv"])
def test_load_local_csv(dataset_client, dataset_name):
    """
    READ TEST
    Checks that a service/service owner can load the dataset into a dataframe
    """
    dataset_document = dataset_client.read(dataset_name)
    df = load_dataset(dataset_document)
    assert isinstance(df, pd.pandas.core.frame.DataFrame)

@pytest.mark.parametrize("dataset_name", ["example_csv"])
@pytest.mark.parametrize("budget", [10.0])
def test_release_local_csv(dataset_client, dataset_name, budget):
    """
    RELEASE TEST
    Checks that the service can release a dataset
    """
    response = dataset_client.release(dataset_name, budget)
    assert response.budget == budget

@pytest.mark.parametrize("dataset_name", ["example_csv"])
@pytest.mark.parametrize("budget", [1.0])
def test_readrelease_budget(dataset_client, dataset_name, budget):
    """
    READ (RELEASE) TEST
    Checks that a new user can load a released dataset, incurring budget
    """
    dataset_client._guid = 'mock_creds'
    dataset_document = dataset_client.readreleased(dataset_name, budget)
    df = load_dataset(dataset_document)
    assert isinstance(df, pd.pandas.core.frame.DataFrame)

@pytest.mark.parametrize("dataset_name", ["demo_dataverse"])
@pytest.mark.parametrize("budget", [100.0])
def test_readrelease_budget_exceeded(dataset_client, dataset_name, budget):
    """
    READ (RELEASE) TEST
    Checks that a new user who attempts to use too much budget raises exception
    """
    dataset_client._guid = 'mock_creds'
    with pytest.raises(Exception) as error:
        dataset_client.readreleased(dataset_name, budget)
    
    assert error.typename == "HttpOperationError"

@pytest.mark.parametrize("dataset_name", ["example_csv"])
@pytest.mark.parametrize("budget", [100.0])
def test_readrelease_no_penalty(dataset_client, dataset_name, budget):
    """
    READ (RELEASE) TEST
    Further readrelease calls do not incure budget
    """
    dataset_client._guid = 'mock_creds'
    dataset_document = dataset_client.readreleased(dataset_name, budget)
    df = load_dataset(dataset_document)
    assert isinstance(df, pd.pandas.core.frame.DataFrame)

@pytest.mark.parametrize("dataset", [{
                        "dataset_name": "another_csv",
                        "dataset_type": "csv_details",
                        "csv_details": {
                            "local_path": os.path.join(os.path.dirname(__file__), "test_datasets", "example.csv")
                        },
                        "budget":3.0,
                        "authorized_users":['mock_creds']}])
def test_register_csv(dataset_client, dataset):
    """
    REGISTER TEST
    Checks that a service owner can register a new dataset (local csv)
    """
    response = get_dataset_client().register(dataset)
    assert response.result == dataset['dataset_name']

@pytest.mark.parametrize("dataset_name", ["another_csv"])
@pytest.mark.parametrize("budget", [100.0])
def test_release_registered_csv(dataset_client, dataset_name, budget):
    """
    RELEASE TEST
    Checks that the service can release a dataset (newly registered csv)
    """
    dataset_client._guid = 'mock_creds'
    response = dataset_client.release(dataset_name, budget)
    assert response.dataset_name == dataset_name

@pytest.mark.parametrize("dataset_name", ["another_csv"])
def test_release_exception_rerelease(dataset_client, dataset_name):
    """
    RELEASE TEST
    Checks an exception when trying to re release a released dataset
    """
    dataset_client._guid = 'mock_creds'
    with pytest.raises(Exception) as error:
        response = dataset_client.release(dataset_name, 10.0)
    
    assert error.typename == "HttpOperationError"

@pytest.mark.parametrize("dataset", [{
                        "dataset_name": "another_dataverse",
                        "dataset_type": "dataverse_details",
                        "dataverse_details": {
                            "local_metadata_path": os.path.join(os.path.dirname(__file__),
                                                                "datasets",
                                                                "dataverse",
                                                                "demo_dataverse.yml"),
                            "host": "https://demo.dataverse.org/api/access/datafile/395811",
                            "schema": '{"fake":"schema"}',
                            "token": {'name':'another_dataverse', 'value': 42}
                        },
                        "budget":3.0,
                        "authorized_users":['mock_creds']}])
def test_register_dataverse(dataset_client, dataset):
    """
    REGISTER TEST
    Checks that a service owner can register a new dataset (dataverse remote)
    """
    response = get_dataset_client().register(dataset)
    assert response.result == dataset['dataset_name']

@pytest.mark.parametrize("dataset_name", ["another_dataverse"])
def test_release_dataverse(dataset_client, dataset_name):
    """
    RELEASE TEST
    Checks that the service can release a dataset (newly registered dataverse)
    """
    dataset_client._guid = 'mock_creds'
    response = dataset_client.release(dataset_name, 10.0)
    assert response.dataset_name == dataset_name

@pytest.mark.parametrize("dataset", [{
                        "dataset_name": "authorized_csv",
                        "dataset_type": "csv_details",
                        "csv_details": {
                            "local_path": os.path.join(os.path.dirname(__file__), "test_datasets", "example.csv")
                        },
                        "budget":10.0,
                        "authorized_users":[]}])
def test_auth_register_release_csv(dataset):
    """
    Tests registering with authorized guids, check successful reads for them
    + unsuccessful reads otherwise

    Check that subsequent reads don't incur budget
    """
    # Generate some valid/invalid clients
    external_clients = [get_dataset_client() for _ in range(0,9)]
    valid_clients = external_clients[5:9]
    invalid_clients = external_clients[0:5]

    valid_guids = [c._guid for c in valid_clients]

    service_client = get_dataset_client()
    dataset["authorized_users"] = valid_guids
    print(dataset)

    # Register dataset takes an optional list of valid ids 5-9
    response = service_client.register(dataset)
    assert response.result == dataset['dataset_name']

    # Release dataset
    response = service_client.release(dataset['dataset_name'], 100.0)
    assert response.dataset_name == dataset['dataset_name']

    # Attempt to read from released dataset with valid clients
    for c in valid_clients:
        dataset_document = c.readreleased(dataset['dataset_name'], 1.0)
        df = load_dataset(dataset_document)
        assert isinstance(df, pd.pandas.core.frame.DataFrame)

    # Attempt to read from released dataset with invalid clients
    for c in invalid_clients:
        with pytest.raises(Exception) as error:
            service_client.readreleased(dataset['dataset_name'], 1.0)
        
        assert error.typename == "HttpOperationError"
    
    prev_budget = service_client.read(dataset['dataset_name']).budget

    # Execute a second read with the same clients - should not incur budget
    for c in valid_clients:
        dataset_document = c.readreleased(dataset['dataset_name'], 1.0)
        df = load_dataset(dataset_document)
        assert isinstance(df, pd.pandas.core.frame.DataFrame)

    assert prev_budget == service_client.read(dataset['dataset_name']).budget

# TODO: Add register exception tests
# TODO: Add release exception tests