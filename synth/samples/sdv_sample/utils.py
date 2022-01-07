import subprocess
import os

import numpy as np
import pandas as pd

def retrieve_PUMS_data_categorical():
    git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

    csv_path = os.path.join(git_root_dir, os.path.join("datasets", "PUMS_pid.csv"))

    df = pd.read_csv(csv_path)
    sample_size = len(df)
    df_non_continuous = df[['sex','educ','race','married']].copy()
    return df, df_non_continuous, sample_size

def return_PUMS_metadata():
    meta = {
        "tables":{
            "pums":{
                "primary_key":"pid",
                "fields":{
                    'pid':{
                    'type': 'id', 
                    'subtype': 'integer'
                    },
                    "sex":{
                    'type': 'boolean'
                    },
                    "educ":{
                    "type":"categorical"
                    },
                    "race":{
                    "type":"categorical"
                    },
                    "married":{
                    "type":"categorical"
                    }
                }
            }
        }
    }

    return meta