#%%
import os
import json
from collections import defaultdict
from flask import abort
from flask import request
from secrets import get as secrets_get
from secrets import put as secrets_put

DATASETS = {"example_csv": {
                        "dataset_name": "example_csv",
                        "dataset_type": "csv_details",
                        "csv_details": {
                            "local_path": os.path.join(os.path.dirname(__file__), "datasets", "example.csv")
                        },
                        "budget":3.0,
                        "authorized_users":['mock_creds']},
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
                        "budget":3.0,
                        "authorized_users":['mock_creds']}}

RELEASED_DATASETS = {"example_released_csv": {
                        "dataset_name": "example_released_csv",
                        "dataset_type": "csv_details",
                        "csv_details": {
                            "local_path": os.path.join(os.path.dirname(__file__), "datasets", "example.csv")
                        },
                        "budget":10.0,
                        "authorized_users":['cb20a2f9-d97b-42b5-9e74-f9ed05b43839', 'mock_creds']},
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
                        "budget":10.0,
                        "authorized_users":['mock_creds']}}

KNOWN_DATASET_TYPE_KEYS = ["csv_details", "dataverse_details"]

KNOWN_USERS = {'ab30c2f7-d97b-42b5-9e74-f9ed05b43839': defaultdict(int), 'mock_creds': defaultdict(int, {'example_released_csv': 1})}

# NOTE: This could be tracked inside datasets as well, although
# potentially safer to only track owners service side
OWNERS = {'ab30c2f7-d97b-42b5-9e74-f9ed05b43839': ["example_csv"]}

def _read_helper(dataset_request, dataset_storage):
    dataset_name = dataset_request["dataset_name"]

    if dataset_name not in dataset_storage:
        abort(400, "Dataset id {} not found.".format(dataset_name))
        
    dataset = dataset_storage[dataset_name]

    # Validate the secret, extract token
    try:
        if dataset["dataset_type"] == "dataverse_details":
            dataset["dataset_type"]["token"] = secrets_get(name="dataverse:{}".format(dataset_request["dataset_name"]))["value"]
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
    return _read_helper(dataset_request, DATASETS)

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

    client_guid = request.headers.get('client_guid')

    # If client owns a released dataset, treat readreleased as
    # private read
    if client_guid in OWNERS:
        if dataset_request["dataset_name"] in OWNERS[client_guid]:
            return dataset

    # NOTE: Should not happen
    if 'authorized_users' not in dataset:
        abort(411, "Released dataset must specify authorized users.")

    if client_guid not in dataset['authorized_users']:
        abort(404, "User not authorized to access this dataset.")

    # Check/Decrement the budget before returning dataset
    # Note: We are guaranteed that client_guid in KNOWN_USERS
    if client_guid not in KNOWN_USERS:
        abort(419, "Something went seriously wrong. Check registration validity.")

    # Decrement budget if this is a users first read
    if KNOWN_USERS[client_guid][dataset["dataset_name"]] == 1:
        adjusted_budget = dataset["budget"] - dataset_request["budget"]
        if adjusted_budget >= 0.0:
            dataset["budget"] = adjusted_budget
        else:
            abort(412, "Not enough budget for read. Remaining budget: {}".format(dataset["budget"]))
    elif KNOWN_USERS[client_guid][dataset["dataset_name"]] == 0:
        abort(418, "Client does not have access to this dataset.")

    # Increment user count for every read. Reads past the first
    # do not incur budget for now.
    KNOWN_USERS[client_guid][dataset["dataset_name"]] += 1
    
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

def _release_register_helper(dataset_request, dataset_storage, track_users):
    # Dataset name, for convenience
    dataset_name = dataset_request["dataset_name"]

    # Check secret here, to make sure this comes from service

    # Check duplicate
    if dataset_name in dataset_storage:
        abort(401, "Dataset id {} already exists. Identifiers must be unique".format(dataset_name))

    # Check key
    if dataset_request["dataset_type"] not in KNOWN_DATASET_TYPE_KEYS:
        abort(402, "Given type was {}, must be either csv_details or dataverse_details.".format(str(dataset_request["dataset_type"])))
    
    # Check budget 
    if dataset_request["budget"]:
        if dataset_request["budget"] <= 0.0: abort(403, "Budget must be greater than 0.")
    else:
        abort(403, "Must specify a budget")

    # Type specific registration
    if dataset_request["dataset_type"] == "csv_details":
        # Local dataset
        if not os.path.isfile(dataset_request["csv_details"]["local_path"]):
            abort(406, "Local file path {} does not exist.".format(str(dataset_request["dataset_type"])))
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
    if client_guid in OWNERS:
        OWNERS[client_guid].append(dataset_name)
    else:
        OWNERS[client_guid] = [dataset_name]

    # This flag is currently only on for released datasets
    if track_users:
        # Before we register/release officially,
        # Track user dataset registrations (count)
        # Tracking Spec:
        # UserDict -> {User -> {dataset -> count}}
        # If count is 0, User has never been given access to this dset
        # If count is 1, User has access to this dataset, but has never read from it
        # Else, count == (# of reads + 1)
        if dataset_request["authorized_users"]:
            for user in dataset_request["authorized_users"]:
                if user not in KNOWN_USERS:
                    KNOWN_USERS[user] = defaultdict(int)
                
                KNOWN_USERS[user][dataset_name] += 1

    # If everything looks good, register it.
    dataset_storage[dataset_name] = dataset_request

    return dataset_request