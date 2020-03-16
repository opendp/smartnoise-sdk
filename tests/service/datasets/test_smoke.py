import os
import json
import pandas as pd

import pytest

from opendp.whitenoise.client import get_dataset_client
from opendp.whitenoise.data.adapters import load_dataset

@pytest.mark.parametrize("dataset_name", ["example_csv"])
@pytest.mark.parametrize("budget", [1.0])
def test_read_local_csv(dataset_client, dataset_name, budget):
    dataset_document = dataset_client.read(dataset_name, budget)
    local_csv_details = dataset_document.csv_details

    with open(local_csv_details.local_path, "r") as stream:
        text = stream.read()
        assert len(text) > 0

@pytest.mark.parametrize("dataset_name", ["example_csv"])
@pytest.mark.parametrize("budget", [1.0])
def test_load_local_csv(dataset_client, dataset_name, budget):
    dataset_document = dataset_client.read(dataset_name, budget)
    df = load_dataset(dataset_document)
    assert isinstance(df, pd.pandas.core.frame.DataFrame)

@pytest.mark.parametrize("dataset_name", ["example_csv"])
@pytest.mark.parametrize("budget", [100.0])
def test_budget_exceeded(dataset_client, dataset_name, budget):
    with pytest.raises(Exception) as error:
        dataset_client.read(dataset_name, budget)
    
    assert error.typename == "HttpOperationError"

@pytest.mark.parametrize("dataset", [{
                        "dataset_name": "another_csv",
                        "dataset_type": "csv_details",
                        "csv_details": {
                            "local_path": os.path.join(os.path.dirname(__file__), "test_data", "example.csv")
                        },
                        "budget":3.0}])
def test_register_csv(dataset_client, dataset):
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
                        "budget":3.0}])
def test_register_dataverse(dataset_client, dataset):
    response = get_dataset_client().register(dataset)
    assert response.result == dataset['dataset_name']

# TODO: Add register exception tests