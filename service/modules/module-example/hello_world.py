import json
import sys

import mlflow

if __name__ == "__main__":
    with mlflow.start_run():
        text = sys.argv[1] if len(sys.argv) > 1 else "Default text"
        print("Ran the example privacy module with text {}".format(text))
        result_path = "result.json"
        with open(result_path, "w") as stream:
            json.dump({"text": text}, stream)
        mlflow.log_artifact(result_path)
