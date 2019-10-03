import os
from flask import abort


DATASETS = {"example": {"type": "local_csv",
                        "local_path": os.path.join(os.path.dirname(__file__), "datasets", "example.csv"),
                        "key": "csv_details"}}


def read(info):
    dataset_name = info["dataset_name"]

    if dataset_name not in DATASETS:
        abort(400, "Dataset id {} not found.".format(dataset_name))
    dataset = DATASETS[dataset_name]

    return {"dataset_type": dataset["type"], dataset["key"]: dataset}
