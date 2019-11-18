import mlflow
import json
import sys

from burdock.client import get_dataset_client
from burdock.data.adapters import load_reader, load_metadata
from burdock.query.sql.private.query import PrivateQuery
from pandasql import sqldf


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    budget = float(sys.argv[2])
    query = sys.argv[3]

    with mlflow.start_run():
        dataset_document = get_dataset_client().read(dataset_name, budget)
        reader = load_reader(dataset_document)
        schema = load_metadata(dataset_document)
        private_reader = PrivateQuery(reader, schema, budget)
        rowset = private_reader.execute(query)

        result = {"query_result": rowset}

        with open("result.json", "w") as stream:
            json.dump(result, stream)
        mlflow.log_artifact("result.json")
