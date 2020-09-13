import sys
import os
import time
import mlflow
import json

import conf

from load_data import load_data

from synthesis import run_all_synthesizers
from evaluate import run_ml_eval, run_wasserstein, run_pMSE

def run(epsilons, run_name, flags, dataset):
    loaded_datasets = load_data(dataset)
    data_dicts = run_all_synthesizers(loaded_datasets, epsilons)
    if 'wasserstein' in flags:
        run_wasserstein(data_dicts, 50, run_name)
    if 'pmse' in flags:
        run_pMSE(data_dicts, run_name)
    if 'ml_eval' in flags:
        results = run_ml_eval(data_dicts, epsilons, run_name)
        print(results)
    with open('artifact.json', 'w') as f:
        json.dump(results, f)
    mlflow.log_artifact("artifact.json", "results.json")

flag_options = ['wasserstein', 'ml_eval', 'pmse']

if __name__ == "__main__":
    # TODO: Add epsilon flag to specify epsilons pre run
    args = sys.argv
    epsilons = [0.01, 0.1, 0.5, 1.0, 3.0, 6.0, 9.0]
    dataset = args[1]

    if len(args) > 2:
        if args[2] == 'all' or args == None:
            flags = flag_options
        else:
            flags = args[2]
    else:
        flags = flag_options


    with mlflow.start_run(run_name="test"):
        mlflow.log_param("epsilons", str(epsilons))
        mlflow.log_param("dataset", dataset)
        mlflow.log_param("flags", str(flags))
        run(epsilons=epsilons, run_name='test', flags=flags, dataset=dataset)
