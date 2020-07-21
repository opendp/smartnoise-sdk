import sys
import mlflow

import conf

from load_data import load_data

from synthesis import run_all_synthesizers
from evaluate import run_ml_eval, run_wasserstein, run_pMSE

def run(epsilons, flags):
    loaded_datasets = load_data()
    data_dicts = run_all_synthesizers(loaded_datasets, epsilons)
    run_wasserstein(data_dicts, 50)
    run_pMSE(data_dicts)
    results = run_ml_eval(data_dicts, epsilons)
    print(results)
    # TODO: Maybe dump the results?
    # with open('artifact.json', 'w') as f:
    #     json.dump(results, f, cls=JSONEncoder)

flag_options = ['wasserstein', 'ml_eval', 'sra', 'pmse']

if __name__ == "__main__":
    # TODO: Add epsilon flag to specify epsilons pre run
    args = sys.argv

    if args[1] == 'all' or args == None:
        flags = flag_options
    else: 
        flags = args[1:]
    #mlflow.set_tracking_uri('file:/mlflowrun')
    with mlflow.start_run(run_name="dpsdgym"):
        run(epsilons=[0.01, 0.1, 1.0], flags=flags)
        # [0.01, 0.1, 1.0, 9.0, 45.0, 95.0]