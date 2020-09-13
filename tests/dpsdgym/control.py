import mlflow

mlflow.set_experiment("synth_data_evals")

datasets = ['adult','mushroom','shopping','car']
for dataset in datasets:
    mlflow.projects.run(".", parameters={"dataset": dataset}, backend="azureml",
                        backend_config={"COMPUTE": "cpu-cluster"},
                        synchronous=False)
