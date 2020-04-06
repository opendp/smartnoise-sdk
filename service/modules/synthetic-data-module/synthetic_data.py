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

from privbn import PrivBNSynthesizer
from mwem import MWEMSynthesizer

# List of supported DP synthesizers
SUPPORTED_SYNTHESIZERS = {'PrivBNSynthesizer': PrivBNSynthesizer, 'MWEMSynthesizer': MWEMSynthesizer}

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

    # NOTE: As of right now, any data clipping per schema is not
    # required for the supported synthetic data scenarios

    # TODO: Support categorical, ordinal and continuous specification through schema 
    categorical_columns = []
    ordinal_columns = [] # range(0,len(data))

    return dataset, dataset_document, {'categorical_columns': categorical_columns, 'ordinal_columns': ordinal_columns}

def release_data(release_dataset_name, dataset_type, details, budget, auth_users):
    # Create the new details
    # TODO: Add type inference
    dataset_to_release = {
        "dataset_name": release_dataset_name,
        "dataset_type": dataset_type,
        dataset_type: details,
        "budget": budget,
        "authorized_users": auth_users
    }

    response = DATASET_CLIENT.release(dataset_to_release)
    return response.dataset_name == release_dataset_name


if __name__ == "__main__":
    # Example run args: "iris" "PrivBNSynthesizer" 20 3.0
    # NOTE: PrivBNSynthesizer actually ignores sample_size, 
    # returns synthetic_samples_size == real_dataset_size
    dataset_name = sys.argv[1]
    synthesizer_name = sys.argv[2]
    sample_size = sys.argv[3]
    budget = sys.argv[4]
    release_dataset_name = sys.argv[5]

    with mlflow.start_run():
        dataset, dataset_document, synth_schema = load_data(dataset_name, budget)

        # Collect from the schema

        # Ensure the C dependencies are compiled
        if synthesizer_name == 'PrivBNSynthesizer':
            try:
                assert os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)),"privbayes","privBayes.bin"))
            except:
                raise ValueError('You must compile the PrivBayes C dependencies. Run "make compile" in privbayes/.')
                # NOTE: Ideally, would spin up a subprocess to do this, but issues with WSL and Popen.
                # process = subprocess.Popen(["make -f "+os.path.join(os.path.dirname(os.path.abspath(__file__)),"privbayes","makefile")], stdout=subprocess.STDOUT)
                # if process.wait() != 0:
                #     raise ValueError('Issue with PrivBayes C dependencies')
            
        # Try to get an instance of synthesizer
        try:
            synthesizer = SUPPORTED_SYNTHESIZERS[synthesizer_name]()
        except:
            raise ValueError('Specified synthesizer is not supported.')
        
        # TODO: Add check to validate dataset.to_numpy
        synthesizer.fit(dataset.to_numpy(), synth_schema['categorical_columns'], synth_schema['categorical_columns'])
        synthetic_data = synthesizer.sample(int(sample_size))
        
        # Create new synthetic dataframe
        df = pd.DataFrame(synthetic_data, 
            index=dataset.index,
            columns=dataset.columns)
        
        # Retrieve dataset details
        details = getattr(dataset_document, dataset_document.dataset_type)

        # Release dataset first, if successful, add the dataframe csv to the path, as well as schema 
        if release_data(release_dataset_name, "local_csv", details.copy(), budget, []):
            # TODO: Only supports csv scenario as of now
            df.to_csv(os.path.join(os.path.dirname(details["local_csv_path"]), release_dataset_name), index=False)
            with open(os.path.join(os.path.dirname(details["local_csv_path"]), release_dataset_name + "_schema"), 'w') as yaml_path:
                yaml.dump(os.path.join(os.path.dirname(details["local_csv_path"]), dataset_name + '.yaml'), yaml_path, default_flow_style=False)

        with open("result.json", "w") as stream:
            json.dump({"released_dataset_name": release_dataset_name}, stream)
        mlflow.log_artifact("result.json")
        
