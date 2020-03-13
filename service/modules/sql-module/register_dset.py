import mlflow
import json
import sys

import pandas as pd

from burdock.client import get_dataset_client
from pandasql import sqldf

if __name__ == "__main__":
    new_dataset = {
        "dataset_name": "new",
        "dataset_type": "dataverse",
        "host": "https://me.com",
        "schema": '{"fake_schema": "schema"}',
        "budget": 3.0,
        "key": "dataverse_details",
        "token": '{"name":"new", "value":0}'
    }
    response = get_dataset_client().register(new_dataset)
    print(response)