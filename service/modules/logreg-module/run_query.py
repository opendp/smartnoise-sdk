import mlflow
import mlflow.sklearn
import json
import sys
import logging

import numpy as np
import pandas as pd

from burdock.client import get_dataset_client
from burdock.data.adapters import load_reader, load_metadata, load_dataset

from diffprivlib.mechanisms import Vector
from diffprivlib.utils import PrivacyLeakWarning, DiffprivlibCompatibilityWarning, warn_unused_args
from diffprivlib.tools.utils import mean
from diffprivlib.models import LogisticRegression

if __name__ == "__main__":
    dataset_name = sys.argv[1] # string
    budget = float(sys.argv[2]) # any number

    # We expect the next two inputs in the following format: json.dumps([col, names, here]) (e.g. '["a","b"]')
    x_features = json.loads(sys.argv[3]) # features
    y_targets = json.loads(sys.argv[4]) # class labels

    with mlflow.start_run(run_name="diffpriv_logreg"):
        dataset_document = get_dataset_client().read(dataset_name, budget)
        dataset = load_dataset(dataset_document)
        schema = load_metadata(dataset_document)

        # use column names to get X and y from dataset to pass to LogisticRegression 
        X = dataset[x_features]
        y = np.ravel(dataset[y_targets]) # use ravel to convert the column vector to a 1d array to avoid issues later using fit

        # TODO change how we find data norm (bad practice here in calculating the true norm; will be changed to use one specified in the schema)
        norms = np.linalg.norm(dataset, axis=1) # norms for each column
        max_norm = np.amax(norms)
        logging.warning('Currently calculating the data norm instead of using schema-specified value. This is bad practice, and will eventually be changed') # provide a warning about bad practice

        model = LogisticRegression(data_norm=max_norm, epsilon=budget).fit(X, y)

        # log attributes
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("budget", budget)
        mlflow.log_param("x_features", x_features)
        mlflow.log_param("y_targets", y_targets)
        mlflow.sklearn.log_model(model, "model")