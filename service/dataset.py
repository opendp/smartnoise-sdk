import os
from flask import abort
from secrets import get as secrets_get


DATASETS = {"example": {"type": "local_csv",
                        "local_path": os.path.join(os.path.dirname(__file__), "datasets", "example.csv"),
                        "key": "csv_details"},
            "demo_dataverse": {"type": "dataverse",
                               "local_metadata_path": os.path.join(os.path.dirname(__file__),
                                                                   "datasets",
                                                                   "dataverse",
                                                                   "demo_dataverse.yml"),
                               "key": "dataverse_details",
                               "host": "https://demo.dataverse.org/api/access/datafile/395811"}}


def read(info):
    dataset_name = info["dataset_name"]

    if dataset_name not in DATASETS:
        abort(400, "Dataset id {} not found.".format(dataset_name))
    dataset = DATASETS[dataset_name]
    if dataset["type"] == "dataverse":
        dataset["token"] = secrets_get(name="dataverse:{}".format(info["dataset_name"]))["value"]

    return {"dataset_type": dataset["type"], dataset["key"]: dataset}
