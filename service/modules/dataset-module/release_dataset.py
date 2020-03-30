import mlflow
import json
import sys
import os
import yaml

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
    private_dataset_name = sys.argv[1]
    release_dataset_name = sys.argv[2]
    budget = sys.argv[3]
    
    with mlflow.start_run():
        service_client = get_dataset_client()
        dataset_document = service_client.read(private_dataset_name, budget)

        input_dataframe = load_dataset(dataset_document)
        new_dataframe = pd.DataFrame(input_dataframe) # DP stuff here? "make_new_df"?

        prev_path = dataset_document['csv_details']['path']
        new_path = os.path.join(os.path.dirname(prev_path), release_dataset_name + '.csv')
        
        prev_schema = dataset_document['csv_details']['schema']
        new_schema = os.path.join(os.path.dirname(prev_schema), release_dataset_name + '.yaml')

        new_dataframe.to_csv(new_path, index=False)
        with open(prev_schema, 'w') as yaml_path:
            yaml.dump(prev_schema, yaml_path, default_flow_style=False)

        # Create the new details
        # TODO: Add type inference
        new_dataset = {
            "dataset_name": release_dataset_name,
            "dataset_type": "local_csv",
            "local_csv": {},
            "budget": budget,
            "authorized_users": []
        }

        with open("result.json", "w") as stream:
            json.dump({"released_dataset_name": release_dataset_name}, stream)
        mlflow.log_artifact("result.json")
        
        # TODO: Perform basic dataset_document validation
        service_client.release(new_dataset)