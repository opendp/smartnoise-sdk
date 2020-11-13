import mlflow
import json
import os

from load_data import load_data

from synthesis import run_all_synthesizers
from evaluate import run_ml_eval, run_wasserstein, run_pMSE, run_sra


def run(epsilons, run_name, metric, dataset, result_path):

	filepath = os.path.join(result_path, run_name + "_result.json")
	os.makedirs(os.path.dirname(filepath), exist_ok=True)

	with mlflow.start_run(run_name=run_name):

	    mlflow.log_param("epsilons", str(epsilons))
	    mlflow.log_param("dataset", dataset)
	    mlflow.log_param("metric", str(metric))
	    
	    loaded_datasets = load_data(dataset)

	    data_dicts = run_all_synthesizers(loaded_datasets, epsilons, result_path, run_name)
	    
	    if 'wasserstein' in metric:
	        run_wasserstein(data_dicts, 100, run_name)

	    if 'pmse' in metric:
	        run_pMSE(data_dicts, run_name)
	    
	    if 'ml_eval' in metric:
	        results = run_ml_eval(data_dicts, epsilons, run_name)
	        
	        if 'sra' in metric:
	            results = run_sra(results)

	    with open(filepath, 'w') as f:
	        json.dump(results, f)
	    mlflow.log_artifact(filepath)
	    print(results)