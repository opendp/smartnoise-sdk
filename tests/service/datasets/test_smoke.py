import json
import pandas as pd

import pytest

from burdock.client import get_dataset_client
from burdock.data.adapters import load_dataset

@pytest.mark.parametrize("dataset_name", ["example"])
def test_read_local_csv(dataset_client, dataset_name):
    dataset_document = dataset_client.read(dataset_name, None)
    local_csv_details = dataset_document.csv_details

    with open(local_csv_details.local_path, "r") as stream:
        text = stream.read()
        assert len(text) > 0


@pytest.mark.parametrize("dataset_name", ["example"])
def test_load_local_csv(dataset_client, dataset_name):
    dataset_document = dataset_client.read(dataset_name, None)
    df = load_dataset(dataset_document)
    isinstance(df, pd.pandas.core.frame.DataFrame)

