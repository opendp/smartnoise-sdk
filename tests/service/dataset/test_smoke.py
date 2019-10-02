import json

import pytest

from burdock.restclient.models.dataset_read_request import DatasetReadRequest

@pytest.mark.parametrize("dataset_id", ["0"])
def test_read_local_csv(client, dataset_id):
    read_request = DatasetReadRequest(dataset_id=dataset_id)
    dataset_document = client.datasetread(read_request)
    local_csv_details = dataset_document.csv_details

    with open(local_csv_details.local_path, "r") as stream:
        text = stream.read()
        assert len(text) > 0

