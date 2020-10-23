import mlflow
from conf import KNOWN_DATASETS

mlflow.set_experiment("smartnoise_synth_data_eval")
for i in range(10):
    for dataset in KNOWN_DATASETS:
        mlflow.projects.run(".", parameters={"dataset": dataset},
                            synchronous=False)
