import os
from flask import abort


DATASETS = {"0": {"type": "local_csv",
                  "local_path": os.path.abspath("datasets/example.csv"),
                  "key": "csv_details"}}


def read(info):
    dataset_id = info["dataset_id"]

    if dataset_id not in DATASETS:
        abort(400, "Dataset id {} not found.".format(dataset_id))
    dataset = DATASETS[dataset_id]

    return {"dataset_type": dataset["type"], dataset["key"]: dataset}
