import json
import pandas as pd

import pytest

from opendp.smartnoise.client import get_dataset_client
from opendp.smartnoise.data.adapters import load_dataset

@pytest.mark.dataverse_token
@pytest.mark.parametrize("dataset_name", ["demo_dataverse"])
@pytest.mark.parametrize("budget", [0.1])
def test_read_dataverse(dataset_client, dataset_name, budget):
    dataset_document = dataset_client.read(dataset_name, budget)
    details = dataset_document.dataverse_details

    with open(details.local_metadata_path, "r") as stream:
        text = stream.read()
        assert len(text) > 0


@pytest.mark.dataverse_token
@pytest.mark.parametrize("dataset_name", ["demo_dataverse"])
@pytest.mark.parametrize("budget", [0.1])
def test_load_dataverse_dataset_file(dataset_client, dataset_name, budget):
    dataset_document = dataset_client.read(dataset_name, budget)
    df = load_dataset(dataset_document)
    assert isinstance(df, pd.pandas.core.frame.DataFrame)
