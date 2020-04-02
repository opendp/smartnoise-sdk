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

# List of supported DP synthesizers
SUPPORTED_SYNTHESIZERS = {'PrivBNSynthesizer': PrivBNSynthesizer}

def load_data(dataset_name, budget):
    """
    Only works with categorical/ordinal columns as of now
    
    SQL scenario...?
    """
    # Load dataset from service (dataset is pd.DataFrame)
    dataset_document = get_dataset_client().read(dataset_name, budget)
    dataset = load_dataset(dataset_document)

    # NOTE: As of right now, any data clipping per schema is not
    # required for the supported synthetic data scenarios

    # TODO: Support categorical, ordinal and continuous specification through schema 
    categorical_columns = []
    ordinal_columns = [] # range(0,len(data))

    return dataset, dataset_document.budget, {'categorical_columns': categorical_columns, 'ordinal_columns': ordinal_columns}

if __name__ == "__main__":
    # Example run args: "iris" "PrivBNSynthesizer" 20 3.0
    # NOTE: PrivBNSynthesizer actually ignores sample_size, 
    # returns synthetic_samples_size == real_dataset_size
    dataset_name = sys.argv[1]
    synthesizer_name = sys.argv[2]
    sample_size = sys.argv[3]
    budget = sys.argv[4]

    with mlflow.start_run():
        dataset, budget, synth_schema = load_data(dataset_name, budget)

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

        # NOTE: Two options here.
        # 1. CURRENTLY IMPLEMENTED: Can default return the synthetic data in the payload.
        # 2. Can default release a new dataset, and return name.
        
        df = pd.DataFrame(synthetic_data, 
            index=dataset.index,
            columns=dataset.columns)
        with open("result.json", "w") as stream:
            json.dump(df.to_dict(), stream)
        mlflow.log_artifact("result.json")
