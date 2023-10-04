from sneval import Metric
import json
import os
import subprocess
import sneval.metrics.compare as CompareModule
from .metrics.compare.base import CompareMetric
import inspect

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

class Evaluate:
    def __init__(
            self, 
            original_dataset,
            synthetic_datasets,
            *ignore, 
            workload=[],
            run_len=2,
            timeout=None,
            max_retry=3,
            max_errors=50,
            output_path=os.path.join(git_root_dir, "eval/evaluate_output.json")
        ):
        self.original_dataset = original_dataset
        self.synthetic_datasets = synthetic_datasets
        self.workload = workload
        self.run_len = run_len
        self.timeout = timeout
        self.max_retry = max_retry
        self.max_errors = max_errors
        self.output_path = output_path
        self.error_count = 0

    def _load_previous_results(self):
        """
        Load results from previous runs if available.
        """
        try:
            with open(self.output_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
        
    def _save_intermediate_results(self, result):
        """
        Save results incrementally after each metric computation.
        """
        current_results = self._load_previous_results()
        current_results += result
        with open(self.output_path, 'w') as f:
            json.dump(current_results, f, indent=4)

    def _is_metric_computed(self, name, params):
        """
        Check if a metric is already computed by comparing with previous results.
        """
        previous_results = self._load_previous_results()
        for result in previous_results:
            if result.get('name') == name and result.get('parameters') == params:
                return True
        return False

    def _compute_metric(self, name, params):
        metric_instance = Metric.create(name, **params)
        result = []
        for synth_dataset in self.synthetic_datasets:
            result.append(metric_instance.compute(self.original_dataset, synth_dataset))
        return result
    
    def run(self):
        metric_names = [name for name, obj in inspect.getmembers(CompareModule) if inspect.isclass(obj) and name != "CompareMetric"]
        for wl in self.workload:
            names = wl.get("metrics", metric_names)
            params = {}
            params["categorical_columns"] = wl.get("categorical_columns")
            if params["categorical_columns"] is None:
                continue
            params["measure_sum_columns"] = wl.get("measure_sum_columns")
            params["edges"] = wl.get("edges", [1, 10, 100, 1000, 10000, 100000])
            params["unknown_keyword"] = wl.get("unknown_keyword", "Unknown")

            for name in names:
                cls = getattr(CompareModule, name)
                if not issubclass(cls, CompareMetric):
                    continue

                if name in ["MeanAbsoluteError", "MeanProportionalError"]:
                    if params["measure_sum_columns"] is None:
                        continue
                    new_params = {k: params[k] for k in ["categorical_columns", "measure_sum_columns", "edges"] if k in params}
                elif name in ["MeanAbsoluteErrorInCount", "MeanProportionalErrorInCount"]:
                    new_params = {k: params[k] for k in ["categorical_columns", "edges"] if k in params}
                elif name == "FabricatedCombinationCount":
                    new_params = {k: params[k] for k in ["categorical_columns", "unknown_keyword"] if k in params} 
                else:
                    new_params = {"categorical_columns": params["categorical_columns"]} 

                # is_metric_defined = name in vars(CompareModule) and isinstance(vars(CompareModule)[name], type)
                if self._is_metric_computed(name, new_params):
                    continue  # Skip this metric and move to the next

                try:
                    result = self._compute_metric(name, new_params)
                    self._save_intermediate_results(result)
                except Exception as e:
                    self.error_count += 1
                    error_result = {
                        "name": name,
                        "parameters": new_params,
                        "value": None,
                        "error": str(e)
                    }
                    self._save_intermediate_results([error_result])
                if self.error_count > self.max_errors:
                    raise Exception(f"Exceeded the maximum error limit of {self.max_errors}")