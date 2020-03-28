import os
import json
import pandas as pd

import pytest

from opendp.whitenoise.client import get_dataset_client
from opendp.whitenoise.data.adapters import load_dataset

## READ TESTS ##
@pytest.mark.parametrize("dataset_name", ["example"])
@pytest.mark.parametrize("budget", [1.0])
def test_read_local_csv(dataset_client, dataset_name, budget):
    dataset_document = dataset_client.read(dataset_name, budget)
    local_csv_details = dataset_document.csv_details

    with open(local_csv_details.local_path, "r") as stream:
        text = stream.read()
        assert len(text) > 0

@pytest.mark.parametrize("dataset_name", ["example"])
@pytest.mark.parametrize("budget", [1.0])
def test_load_local_csv(dataset_client, dataset_name, budget):
    dataset_document = dataset_client.read(dataset_name, budget)
    df = load_dataset(dataset_document)
    assert isinstance(df, pd.pandas.core.frame.DataFrame)

@pytest.mark.parametrize("dataset_name", ["example"])
@pytest.mark.parametrize("budget", [100.0])
def test_budget_exceeded(dataset_client, dataset_name, budget):
    with pytest.raises(Exception) as error:
        dataset_client.read(dataset_name, budget)

    assert error.typename == "HttpOperationError"

## READ RELEASED TESTS ##

@pytest.mark.parametrize("dataset_name", ["example_released_csv"])
def test_read_release_no_penalty(dataset_client, dataset_name):
    """
    READ (RELEASE) TEST
    Further readrelease calls do not incur budget
    """
    dataset_client.custom_headers = {'client_guid': 'mock_user_guid'} 
    dataset_document = dataset_client.read_released(dataset_name)
    df = load_dataset(dataset_document)
    assert isinstance(df, pd.pandas.core.frame.DataFrame)

## RELEASE TESTS ##
@pytest.mark.parametrize("release_request", [{
                        "dataset_name": "local_csv_again",
                        "dataset_type": "csv_details",
                        "csv_details": {
                            "local_path": os.path.join(os.path.dirname(__file__), "datasets", "example.csv")
                        },
                        "release_cost":1.0,
                        "budget":3.0,
                        "authorized_users":['mock_user_guid']}])
def test_release_csv(dataset_client, release_request):
    """
    RELEASE TEST
    Proper usage
    """
    dataset_client.custom_headers = {'client_guid': 'mock_user_guid'} 
    budget_check = release_request["budget"] - release_request["release_cost"]
    dataset = dataset_client.release(release_request)
    assert dataset.dataset_name == "local_csv_again"
    assert dataset.budget == budget_check

@pytest.mark.parametrize("release_request", [{
                        "dataset_name": "demo_released_dataverse",
                        "dataset_type": "dataverse_details",
                        "dataverse_details": {
                            "local_metadata_path": os.path.join(os.path.dirname(__file__),
                                                                "datasets",
                                                                "dataverse",
                                                                "demo_dataverse.yml"),
                            "host": "https://demo.dataverse.org/api/access/datafile/395811",
                            "schema": '{"fake":"schema"}'
                        },
                        "release_cost":10.0,
                        "budget":100.0,
                        "authorized_users":['mock_user_guid']}])
def test_release_exception_rerelease(dataset_client, release_request):
    """
    RELEASE TEST
    Checks an exception when trying to release a dataset with the same name
    """
    dataset_client.custom_headers = {'client_guid': 'mock_user_guid'} 
    with pytest.raises(Exception) as error:
        response = dataset_client.release(release_request)
    
    assert error.typename == "HttpOperationError"

## REGISTER TESTS ##

@pytest.mark.parametrize("dataset", [{
                        "dataset_name": "another_csv",
                        "dataset_type": "csv_details",
                        "csv_details": {
                            "local_path": os.path.join(os.path.dirname(__file__), "test_datasets", "example.csv")
                        },
                        "release_cost":3.0,
                        "budget":3.0,
                        "authorized_users":['mock_user_guid']}])
def test_register_csv(dataset_client, dataset):
    """
    REGISTER TEST
    Checks that a service owner can register a new dataset (local csv)
    """
    response = get_dataset_client().register(dataset)
    assert response.result == dataset['dataset_name']

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
                        "release_cost":3.0,
                        "budget":10.0,
                        "authorized_users":['mock_user_guid']}])
def test_register_dataverse(dataset_client, dataset):
    """
    REGISTER TEST
    Checks that a service owner can register a new dataset (dataverse remote)
    """
    response = get_dataset_client().register(dataset)
    assert response.result == dataset['dataset_name']

## SYSTEM TESTS ##

@pytest.mark.parametrize("dataset", [{
                        "dataset_name": "authorized_csv",
                        "dataset_type": "csv_details",
                        "csv_details": {
                            "local_path": os.path.join(os.path.dirname(__file__), "test_datasets", "example.csv")
                        },
                        "release_cost":3.0,
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

    # Register dataset takes an optional list of valid ids 5-9
    response = service_client.register(dataset)
    assert response.result == dataset['dataset_name']

    # Fake DP Perturb, similar release will be done in module
    retrieved_dataset = service_client.read(dataset["dataset_name"], 1.0)
    retrieved_dataset.dataset_name = "release_authorize_csv"
    release_doc = {
        "dataset_name": "release_authorize_csv",
        "dataset_type": retrieved_dataset.dataset_type,
        retrieved_dataset.dataset_type: retrieved_dataset.csv_details,
        "release_cost":retrieved_dataset.release_cost,
        "budget":retrieved_dataset.budget,
        "authorized_users":retrieved_dataset.authorized_users}
    release_dataset = service_client.release(release_doc)
    assert release_dataset.dataset_name == "release_authorize_csv"

    # Should have same authorized users
    assert release_dataset.authorized_users == valid_guids

    # Attempt to read from released dataset with valid clients
    for c in valid_clients:
        dataset_document = c.read_released(release_dataset.dataset_name)
        df = load_dataset(dataset_document)
        assert isinstance(df, pd.pandas.core.frame.DataFrame)

    # Attempt to read from released dataset with invalid clients
    for c in invalid_clients:
        with pytest.raises(Exception) as error:
            c.read_released(release_dataset.dataset_name)
        
        assert error.typename == "HttpOperationError"