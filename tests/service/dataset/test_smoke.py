import json

import pytest

from burdock.restclient.models.dataset_read_request import DatasetReadRequest

@pytest.mark.parametrize("dataset_id", ["0"])
def test_execute_run(client, dataset_id):
    read_request = DatasetReadRequest(dataset_id=dataset_id)
    client.datasetread(read_request)

