#%%
import os
import json
from flask import abort
from secrets import get as secrets_get
from secrets import put as secrets_put


DATASETS = {"example": {"type": "local_csv",
                        "local_path": os.path.join(os.path.dirname(__file__), "datasets", "example.csv"),
                        "key": "csv_details",
                        "budget":3.0},
            "demo_dataverse": {"type": "dataverse",
                               "local_metadata_path": os.path.join(os.path.dirname(__file__),
                                                                   "datasets",
                                                                   "dataverse",
                                                                   "demo_dataverse.yml"),
                               "key": "dataverse_details",
                               "host": "https://demo.dataverse.org/api/access/datafile/395811",
                               "budget":3.0}}

# Do we need to add an ability to register these?
KNOWN_DATASET_KEYS = ["csv_details", "dataverse_details"]

def read(info):
    dataset_name = info["dataset_name"]

    if dataset_name not in DATASETS:
        abort(400, "Dataset id {} not found.".format(dataset_name))
    
    dataset = DATASETS[dataset_name]

    # Validate the secret, extract token
    if dataset["type"] == "dataverse":
        dataset["token"] = secrets_get(name="dataverse:{}".format(info["dataset_name"]))["value"]

    # Check/Decrement the budget before returning dataset
    # Unclear what behaviour budget decrementing should have
    # - should it check the type of query, and decrement accordingly?
    adjusted_budget = dataset["budget"] - info["budget"]
    if adjusted_budget >= 0.0:
        dataset["budget"] = adjusted_budget
    else:
        abort(412, "Not enough budget for read. Remaining budget: {}".format(dataset_name))

    return {"dataset_type": dataset["type"], dataset["key"]: dataset}

#%%
def register(dataset):
    dataset_name = dataset["dataset_name"]

    if dataset_name in DATASETS:
        abort(401, "Dataset id {} already exists. Identifies must be unique".format(dataset_name))

    # Add key if possible
    if dataset["key"] not in KNOWN_DATASET_KEYS:
        abort(402, "Given key was {}, must be either csv_details or dataverse_details.".format(str(dataset["key"])))
    
    # Add budget if possible 
    if dataset["budget"]:
        b = float(dataset["budget"])
        if b <= 0.0: abort(403, "Budget must be greater than 0.")
        dataset["budget"] = b
    else:
        abort(403, "Must specify a budget")
    
     # Add type if possible
    if not (dataset["type"] is "local_csv" or "dataverse"):
        abort(405, "Given type was {}, must be either local_csv or dataverse.".format(str(dataset["type"])))

    # Add checks
    if "local_path" in dataset:
        # Local dataset
        if not os.path.isfile(dataset["local_path"]):
            abort(406, "Local file path {} does not exist.".format(str(dataset["local_path"])))
    elif "schema" in dataset:
        # Remote dataset
        if dataset["schema"]:
            try:
                dataset["schema"] = json.dumps(dataset["schema"])
            except:
                abort(407, "Schema {} must be valid json.".format(str(dataset["schema"])))
        else:
            abort(414, "Schema must exist.")
        
        # Specify host
        if not dataset["host"]:
            abort(408, "Must specify host, {} is malformed.".format(str(dataset["host"])))
    else:
        abort(409, "Dataset must specify either local_path or local_metadata_path.")
    
    if dataset["type"] == "dataverse":
        # There's a more elegant way to do this
        try:
            # get throws exception if non-existant secret
            secrets_get(name="dataverse:{}".format(info["dataset_name"]))["value"]
            abort(410, "Duplicated secret - not allowed to register the same dataset twice.")
        except:
            # TODO: Temp secret - not sure where to grab it from
            sec = {"name":dataset_name, "value":0}
            secrets_put(sec)
    
    # If everything looks good, register it.
    DATASETS[dataset_name] = dataset
        
#%%
new_dataset = {
    "dataset_name": "new",
    "type": "dataverse",
    "host": "https://me.com",
    "schema": '{"fake_schema": "schema"}',
    "budget": 3.0,
    "key": "dataverse_details"
}

register(new_dataset)
print(DATASETS)

