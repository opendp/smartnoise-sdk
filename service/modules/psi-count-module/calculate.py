import mlflow
import json
import sys

from statistic import Count
from burdock.client import get_dataset_client
from burdock.data.adapters import load_dataset


if __name__ == "__main__":
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "example"
    column_name = sys.argv[2] if len(sys.argv) > 1 else "a"
    budget = float(sys.argv[3]) if len(sys.argv) > 1 else 1


    with mlflow.start_run():
        df = load_dataset(get_dataset_client().read(dataset_name, budget))
        statistic = Count(column_name, budget).release(df)

        with open("result.json", "w") as stream:
            json.dump(statistic.as_dict(), stream)
        mlflow.log_artifact("result.json")
