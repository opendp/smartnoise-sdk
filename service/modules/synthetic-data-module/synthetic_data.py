#%%
import mlflow
import json
import sys
import os
import yaml
import subprocess

import numpy as np
import pandas as pd
from opendp.whitenoise.client import get_dataset_client
from opendp.whitenoise.data.adapters import load_reader, load_metadata, load_dataset
from opendp.whitenoise.sql.private_reader import PrivateReader
from pandasql import sqldf

from sdgym.constants import CONTINUOUS
from sdgym.synthesizers.utils import Transformer

from opendp.whitenoise.synthesizers.mwem import MWEMSynthesizer

# List of supported DP synthesizers
SUPPORTED_SYNTHESIZERS = {'MWEMSynthesizer': MWEMSynthesizer}

# Maintain a dataset client
DATASET_CLIENT = get_dataset_client()

def load_data(dataset_name, budget):
    """
    Only works with categorical/ordinal columns as of now
    
    SQL scenario...?
    """
    # Load dataset from service (dataset is pd.DataFrame)
    dataset_document = DATASET_CLIENT.read(dataset_name, budget)
    dataset = load_dataset(dataset_document)
    schema = load_metadata(dataset_document)

    # NOTE: As of right now, any data clipping per schema is not
    # required for the supported synthetic data scenarios

    # TODO: Support categorical, ordinal and continuous specification through schema 
    categorical_columns = []
    ordinal_columns = [] # range(0,len(data))

    # TODO: Temporary support for dropping id column
    if 'pid' in dataset.columns:
        dataset.drop('pid', axis=1, inplace=True)
    if 'income' in dataset.columns:
        dataset.drop('income', axis=1, inplace=True)

    return dataset, dataset_document, {'categorical_columns': categorical_columns, 'ordinal_columns': ordinal_columns}, schema

def release_data(release_dataset_name, dataset_type, details, budget, release_cost, auth_users):
    # Create the new details
    # TODO: Add type inference
    dataset_to_release = {
        "dataset_name": release_dataset_name,
        "dataset_type": dataset_type,
        dataset_type: details,
        "budget": budget,
        "release_cost": release_cost,
        "authorized_users": auth_users
    }

    response = DATASET_CLIENT.release(dataset_to_release)
    return response.dataset_name == release_dataset_name

if __name__ == "__main__":
    # Example run args: "PUMS" "MWEMSynthesizer" 1000 3.0 1.0 pums_released
    # NOTE: MWEMSynthesizer actually ignores sample_size, 
    # returns synthetic_samples_size == real_dataset_size
    dataset_name = sys.argv[1]
    synthesizer_name = sys.argv[2]
    sample_size = sys.argv[3]
    budget = sys.argv[4]
    release_cost = sys.argv[5]
    release_dataset_name = sys.argv[6]

    with mlflow.start_run():
        dataset, dataset_document, synth_schema, prev_schema = load_data(dataset_name, budget)
            
        # Try to get an instance of synthesizer
        try:
            synthesizer = SUPPORTED_SYNTHESIZERS[synthesizer_name]()
        except:
            raise ValueError('Specified synthesizer is not supported.')
        
        # TODO: Add check to validate dataset.to_numpy
        synthesizer.fit(dataset.to_numpy(), synth_schema['categorical_columns'], synth_schema['ordinal_columns'])
        synthetic_data = synthesizer.sample(int(sample_size))
        

        # Create new synthetic dataframe
        df = pd.DataFrame(synthetic_data, 
            index=dataset.index,
            columns=dataset.columns)
        
        # Retrieve dataset details
        details = getattr(dataset_document, dataset_document.dataset_type)

        # Release dataset first, if successful, add the dataframe csv to the path, as well as schema 
        if release_data(release_dataset_name, "csv_details", {'local_path': details.local_path}, budget, release_cost, []):
            # TODO: Only supports csv scenario as of now
            df.to_csv(os.path.join(os.path.dirname(details.local_path), release_dataset_name + '.csv'), index=False)
            with open(os.path.join(os.path.dirname(details.local_path), release_dataset_name + "_schema.yaml"), 'w') as yaml_path:
                yaml.dump(prev_schema, yaml_path, default_flow_style=False)

        with open("result.json", "w") as stream:
            json.dump({"released_dataset_name": release_dataset_name}, stream)
        mlflow.log_artifact("result.json")
        
