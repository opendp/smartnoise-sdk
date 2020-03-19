import mlflow
import json
import sys
import os

import pandas as pd

from opendp.whitenoise.client import get_dataset_client
from opendp.whitenoise.data.adapters import load_reader, load_metadata, load_dataset
from opendp.whitenoise.sql.private_reader import PrivateReader
from pandasql import sqldf


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    dataset_type = sys.argv[2]
    budget = sys.argv[3]
    json_flag = sys.argv[4]

    details = {}

    if json_flag:
        # Dump the json
        pass
    else:
        for arg in sys.argv[5:]:
            details[arg.split("=")[0]] = arg.split("=")[1]

    with mlflow.start_run():
        # Optionally load the schema from a json? 
        new_dataset = {
            "dataset_name": dataset_name,
            "dataset_type": dataset_type,
            dataset_type: details,
            "budget": budget
        }

        dataset_document = get_dataset_client().register(new_dataset)

        # with open("result.json", "w") as stream:
        #     json.dump(df.to_dict(), stream)
        # mlflow.log_artifact("result.json")
