import mlflow
import mlflow.sklearn
import json
import sys
import logging

import numpy as np
import pandas as pd

from burdock.client import get_dataset_client
from burdock.data.adapters import load_metadata, load_dataset

from dp_sgd import DPSGDClassifier, DPSGDRegressor

if __name__ == "__main__":
    dataset_name = sys.argv[1]
    budget = float(sys.argv[2])
    # The following input should be either "R" for regressor, or "C" for classifier
    reg_or_class = sys.argv[3]
    # We expect the next two inputs in the following format: json.dumps([col, names, here]) (e.g. '["a","b"]')
    x_features = json.loads(sys.argv[4])
    y_targets = json.loads(sys.argv[5])

    with mlflow.start_run(run_name="diffpriv_sgd"):
        # Log mlflow attributes for mlflow UI
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("budget", budget)
        mlflow.log_param("reg_or_class", reg_or_class)
        mlflow.log_param("x_features", x_features)
        mlflow.log_param("y_targets", y_targets)

        if reg_or_class == "R":
            use_regressor = True
        elif reg_or_class == "C":
            use_regressor = False
        else:
            raise ValueError(reg_or_class +
                             " is not a valid choice. Please use 'R' to do regression, and 'C' for classification")

        dataset_document = get_dataset_client().read(dataset_name, budget)
        dataset = load_dataset(dataset_document)
        schema = load_metadata(dataset_document)

        # Find X and y values from dataset
        X = dataset[x_features]
        y = dataset[y_targets]

        # Find ranges for X and ranges for y
        table_name = dataset_name + "." + dataset_name
        x_range_dict = dict([(col, schema[table_name][col].maxval - schema[table_name][col].minval)
                             for col in x_features])
        y_range_dict = dict([(col, schema[table_name][col].maxval - schema[table_name][col].minval)
                             for col in y_targets])
        x_range = pd.Series(data=x_range_dict)
        y_range = pd.Series(data=y_range_dict)

        data_range = pd.DataFrame([[schema[table_name][col].minval, schema[table_name][col].maxval] for col in
                                   (x_features+y_targets)], index=(x_features+y_targets)).transpose()

        if use_regressor:
            model = DPSGDRegressor().fit(X, y, data_range, budget)
        else:
            model = DPSGDClassifier().fit(X, y, data_range, budget)

        # Save model for access through mlflow ui
        mlflow.sklearn.log_model(model, "model")

        results = {
            "run_id": mlflow.active_run().info.run_id,
            "model_name": "diffpriv_linreg"
        }
        with open("result.json", "w") as stream:
            json.dump(results, stream)
        mlflow.log_artifact("result.json")

        # Save model for access through mlflow ui
        mlflow.sklearn.log_model(model, "model")

        results = {
            "run_id": mlflow.active_run().info.run_id,
            "model_name": "diffpriv_linreg"
        }
        with open("result.json", "w") as stream:
            json.dump(results, stream)
        mlflow.log_artifact("result.json")

