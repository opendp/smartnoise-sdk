import sys
import os
import time
import mlflow

import conf

from load_data import load_data

from synthesis import run_all_synthesizers
from evaluate import run_ml_eval, run_wasserstein, run_pMSE

def run(epsilons, run_name, flags):
    loaded_datasets = load_data()
    data_dicts = run_all_synthesizers(loaded_datasets, epsilons)
    if 'wasserstein' in flags:
        run_wasserstein(data_dicts, 50, run_name)
    if 'pmse' in flags:
        run_pMSE(data_dicts, run_name)
    if 'ml_eval' in flags:
        results = run_ml_eval(data_dicts, epsilons, run_name)
        print(results)
    # TODO: Maybe dump the results?
    # with open('artifact.json', 'w') as f:
    #     json.dump(results, f, cls=JSONEncoder)

flag_options = ['wasserstein', 'ml_eval', 'sra', 'pmse']

if __name__ == "__main__":
    # TODO: Add epsilon flag to specify epsilons pre run
    args = sys.argv
    exp_name = args[1]
    mlflow.create_experiment(exp_name)
    if args[2] == 'all' or args == None:
        flags = flag_options
    else: 
        flags = args[1:]
    #mlflow.set_tracking_uri('file:/mlflowrun')
    with mlflow.start_run(run_name="test"):
        # os.environ['MLFLOW_RUN_ID'] = mlflow.active_run().info.run_id
        # assert(os.environ.get('MLFLOW_RUN_ID'))
        # print(os.environ['MLFLOW_RUN_ID'])
        run(epsilons=[1.0], run_name='test', flags=flags)
        # [0.01, 0.1, 1.0, 9.0, 45.0, 95.0] , 10.0, 100.0 , 10.0, 50.0, 100.0