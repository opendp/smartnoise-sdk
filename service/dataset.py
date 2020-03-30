#%%
import os
import json
from collections import defaultdict
from flask import abort
from flask import request
from secrets import get as secrets_get
from secrets import put as secrets_put

DATASETS = {"example": {
                        "dataset_name": "example",
                        "dataset_type": "csv_details",
                        "csv_details": {
                            "local_path": os.path.join(os.path.dirname(__file__), "datasets", "example.csv")
                        },
                        "release_cost":1.0,
                        "budget":3.0},
            "iris": {
                        "dataset_name": "iris",
                        "dataset_type": "csv_details",
                        "csv_details": {
                            "local_path": os.path.join(os.path.dirname(__file__), "datasets", "iris.csv")
                        },
                        "release_cost":10.0,
                        "budget":300.0},
            "demo_dataverse": {
                        "dataset_name": "demo_dataverse",
                        "dataset_type": "dataverse_details",
                        "dataverse_details": {
                            "local_metadata_path": os.path.join(os.path.dirname(__file__),
                                                                "datasets",
                                                                "dataverse",
                                                                "demo_dataverse.yml"),
                            "host": "https://demo.dataverse.org/api/access/datafile/395811",
                            "schema": '{"fake":"schema"}'
                        },
                        "release_cost":1.0,
                        "budget":3.0,
                        "authorized_users":['mock_user_guid']}}

RELEASED_DATASETS = {"example_released_csv": {
                        "dataset_name": "example_released_csv",
                        "dataset_type": "csv_details",
                        "csv_details": {
                            "local_path": os.path.join(os.path.dirname(__file__), "datasets", "example.csv")
                        },
                        "release_cost":1.0,
                        "budget":10.0,
                        "authorized_users":['mock_user_guid']},
                    "demo_released_dataverse": {
                        "dataset_name": "demo_released_dataverse",
                        "dataset_type": "dataverse_details",
                        "dataverse_details": {
                            "local_metadata_path": os.path.join(os.path.dirname(__file__),
                                                                "datasets",
                                                                "dataverse",
                                                                "demo_dataverse.yml"),
                            "host": "https://demo.dataverse.org/api/access/datafile/395811",
                            "schema": '{"fake":"schema"}'
                        },
                        "release_cost":1.0,
                        "budget":4.0,
                        "authorized_users":['mock_user_guid']}}

KNOWN_DATASET_TYPE_KEYS = ["csv_details", "dataverse_details"]

# Construct nested default dict
int_default_dict = lambda: defaultdict(int)

# Track dataset released to known users
# Tracking Spec:
# USERS_RELEASED -> {dataset -> {user -> count}}
# If count is 0, user has never been given access to this dset
# If count is 1, user has access to this dataset, but has never read from it
# Else, count == (# of reads + 1)
USERS_RELEASED = defaultdict(int_default_dict) 
USERS_RELEASED['example_released_csv']['mock_user_guid'] += 1

# NOTE: This could be tracked inside datasets as well, although
# potentially safer to only track owners service side
OWNERS = defaultdict(int_default_dict) 
OWNERS['mock_user_guid']['example_released_csv'] += 1

def _read_helper(dataset_request, dataset_storage):
    dataset_name = dataset_request["dataset_name"]

    if dataset_name not in dataset_storage:
        abort(400, "Dataset id {} not found.".format(dataset_name))
        
    dataset = dataset_storage[dataset_name]

    # Validate the secret, extract token
    try:
        if dataset["dataset_type"] == "dataverse_details":
            dataset[dataset["dataset_type"]]["token"] = secrets_get(name="dataverse:{}".format(dataset_request["dataset_name"]))["value"]
    except:
        # TODO: Temp fix for testing - Do better cleanup if secret missing
        # dataset["dataset_type"]["token"] = {'name':dataset_name,'value':42}
        pass

    return dataset

def read(dataset_request):
    """
    Read from private dataset

    :param info: The dataset to read and budget to use.
    :type info: dict {"dataset_name": str, "budget":int}
    :return: A dataset document that contains the type and info of the dataset
    :rtype: dict{"dataset_type": str, dataset_key: dict}
    """
    dataset = _read_helper(dataset_request, DATASETS)

    # Decrement budget if possible, private read called from module
    adjusted_budget = dataset["budget"] - dataset_request["budget"]
    if adjusted_budget >= 0.0:
        dataset["budget"] = adjusted_budget
    else:
        abort(412, "Not enough budget for read. Remaining budget: {}".format(dataset["budget"]))

    return dataset

def readreleased(dataset_request):
    """
    Read from a released dataset, if authorized
    
    Check request header for client_guid

    :param info: The dataset to read and budget to use
    :type info: dict {"dataset_name": str, "budget":int}
    :return: A dataset document that contains the type and info of the dataset
    :rtype: dict{"dataset_type": str, dataset_key: dict}
    """
    dataset = _read_helper(dataset_request, RELEASED_DATASETS)
    dataset_name = dataset["dataset_name"]

    client_guid = request.headers.get('client_guid')

    # If client owns a released dataset, treat readreleased as
    # private read
    if client_guid in OWNERS:
        if dataset_request["dataset_name"] in OWNERS[client_guid]:
            USERS_RELEASED[dataset_name][client_guid] += 1
            return dataset

    # Checks to see if authorized users specified
    if 'authorized_users' in dataset and len(dataset['authorized_users']) > 0:
        if client_guid not in dataset['authorized_users']:
            abort(404, "User not authorized to access this dataset.")

    # Track readrelease
    USERS_RELEASED[dataset_name][client_guid] += 1
    
    return dataset

def release(release_request):
    """Releases a new dataset

    :param release_request: The dataset to read and budget to use.
    :type release_request: dict(DatasetDocument) 
    :return: A dataset document that contains the type and info of the dataset
    :rtype: dict(DatasetDocument) 
    """
    # Call helper
    released_dataset = _release_register_helper(release_request, RELEASED_DATASETS, True)

    print(RELEASED_DATASETS.keys())

    return released_dataset

#%%
def register(dataset):
    """Register a new dataset

    :param dataset: The dataset to read and budget to use.
    :type dataset: dict(DatasetDocument) 
    :return: A dataset document that contains the type and info of the dataset
    :rtype: dict(DatasetDocument) 
    """

    # Call helper
    dataset = _release_register_helper(dataset, DATASETS, False)

    print(DATASETS.keys())

    return {"result": dataset["dataset_name"]}

def _release_register_helper(dataset_request, dataset_storage, release_check):
    # Dataset name, for convenience
    dataset_name = dataset_request["dataset_name"]

    if release_check:
        # Decrement budget by release cost, if possible
        adjusted_budget = dataset_request["budget"] - dataset_request["release_cost"]
        if adjusted_budget >= 0.0:
            dataset_request["budget"] = adjusted_budget
        else:
            abort(412, "Not enough budget for release. Remaining budget: {}".format(dataset_request["budget"]))

    # Check secret here, to make sure this comes from service

    # Check duplicate
    if dataset_name in dataset_storage:
        abort(401, "Dataset id {} already exists. Identifiers must be unique".format(dataset_name))

    # Check key
    if dataset_request["dataset_type"] not in KNOWN_DATASET_TYPE_KEYS:
        abort(402, "Given type was {}, must be either csv_details or dataverse_details.".format(str(dataset_request["dataset_type"])))

    # Type specific registration
    if dataset_request["dataset_type"] == "csv_details":
        # Local dataset
        if not os.path.isfile(dataset_request["csv_details"]["local_path"]):
            abort(406, "Local file path {} does not exist.".format(str(dataset_request["csv_details"]["local_path"])))
    elif dataset_request["dataset_type"] == "dataverse_details":
        # Validate Json schema
        if dataset_request["dataverse_details"]["schema"]:
            try:
                dataset_request["dataverse_details"]["schema"] = json.dumps(dataset_request["dataverse_details"]["schema"])
            except:
                abort(407, "Schema {} must be valid json.".format(str(dataset_request["dataverse_details"]["schema"])))
        else:
            abort(414, "Schema must exist.")

        # Specify host
        if not dataset_request["dataverse_details"]["host"]:
            abort(408, "Must specify host, {} is malformed.".format(str(dataset_request["dataverse_details"]["host"])))

    # Track owner
    client_guid = request.headers.get('client_guid')
    OWNERS[client_guid][dataset_name] += 1

    # Before we register/release officially,
    # Track user dataset registrations (count)
    USERS_RELEASED[dataset_name][client_guid] += 1

    # If everything looks good, register it.
    dataset_storage[dataset_name] = dataset_request

    return dataset_request