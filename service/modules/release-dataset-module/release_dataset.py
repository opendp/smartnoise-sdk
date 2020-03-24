import mlflow
import json
import sys
import os

import pandas as pd

from opendp.whitenoise.client import get_dataset_client
from opendp.whitenoise.data.adapters import load_reader, load_metadata, load_dataset
from opendp.whitenoise.sql.private_reader import PrivateReader
from pandasql import sqldf


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    dataset_type = sys.argv[2]
    query_budget = sys.argv[3]
    release_budget = sys.argv[4]
    released_dataset_name = sys.argv[5]
    query = sys.argv[6]
    
    with mlflow.start_run():
        dataset_document = get_dataset_client().read(dataset_name)

        dataset = load_dataset(dataset_document)
        reader = load_reader(dataset_document)
        schema = load_metadata(dataset_document)
        # NOTE: Is query_budget equivalent to epsilon of DP mechanism here?
        private_reader = PrivateReader(reader, schema, query_budget)
        rowset = private_reader.execute(query)

        released_dataset_document = get_dataset_client().release(dataset_name, release_budget)
        

        result = {"query_result": rowset}
        df = pd.DataFrame(rowset[1:], columns=rowset[0])
        with open("result.json", "w") as stream:
            json.dump(df.to_dict(), stream)
        mlflow.log_artifact("result.json")
        
    
    client = get_dataset_client()
    response = client.release(dataset_name)
    print(response)
    dataset_document = client.readreleased(dataset_name, 1.0)
    df = load_dataset(dataset_document)

    ###
    """
    Module outline:
    Client:
    submits module
    passes in an input private dataset, budget for the action, and a unique name for the released dataset, type?
    expects an output with result.json {"released_dataset_name": "name"}
    can now call read_released("name") # this won't cost the uploader budget because creating it already deducted it(hence the id comment above since any other user would be charged budget)
    Module:
    reads private data in
    creates something from it that is differentially private
    registers a new Dataset, ReleasedDataset, that points to this new type of dataset that costs budget to read but is not private.
    This means Modules will need to create a local csv(for testing) or a local csv released dataset(for now since it is easier to test against)
    """

    ###

    

    
