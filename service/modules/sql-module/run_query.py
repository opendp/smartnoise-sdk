import mlflow
import json
import sys

import pandas as pd

from opendp.smartnoise.client import get_dataset_client
from opendp.smartnoise.data.adapters import load_reader, load_metadata
from opendp.smartnoise.sql import PrivateReader


if __name__ == "__main__":
    dataset_name = sys.argv[1]
    budget = float(sys.argv[2])
    query = sys.argv[3]

    with mlflow.start_run():
        dataset_document = get_dataset_client().read(dataset_name, budget)
        reader = load_reader(dataset_document)
        metadata = load_metadata(dataset_document)

        budget_per_column = budget / PrivateReader.get_budget_multiplier(metadata,
                                                                         reader,
                                                                         query)
        private_reader = PrivateReader(reader, metadata, budget_per_column)

        rowset = private_reader.execute(query)
        result = {"query_result": rowset}
        df = pd.DataFrame(rowset[1:], columns=rowset[0])

        with open("result.json", "w") as stream:
            json.dump(df.to_dict(), stream)
        mlflow.log_artifact("result.json")
