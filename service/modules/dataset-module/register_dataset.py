import mlflow
import json
import sys
import os

import pandas as pd

from opendp.whitenoise.client import get_dataset_client
from opendp.whitenoise.data.adapters import load_reader, load_metadata, load_dataset
from opendp.whitenoise.sql.private_reader import PrivateReader
from pandasql import sqldf

"""
Sample Command line:

python register_dataset.py private_csv csv_details 10.0 False local_path=serverside/path/to/example.csv
"""

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    dataset_type = sys.argv[2]
    budget = sys.argv[3]
    json_flag = sys.argv[4]

    details = {}

    # If json flag, assume 5th argument is valid json
    # Else, load each specified detail
    if json_flag == "True":
        with open(sys.argv[5], "r") as f:
            details = json.load(f)
    else:
        # Args split by spaces, then key=value
        for arg in sys.argv[5:]:
            details[arg.split("=")[0]] = arg.split("=")[1]
    
    with mlflow.start_run():
        new_dataset = {
            "dataset_name": dataset_name,
            "dataset_type": dataset_type,
            dataset_type: details,
            "budget": budget,
            "authorized_users": []
        }

        dataset_document = get_dataset_client().register(new_dataset)

        # TODO: Perform basic dataset_document validation