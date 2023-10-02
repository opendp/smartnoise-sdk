import importlib
from sneval import Metric
from sneval.metrics import CompareMetric
import json
import os
import subprocess

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

class Evaluate:
    def __init__(
            self, 
            original_dataset,
            synthetic_dataset,
            *ignore, 
            workload=[],
            run_len=2,
            timeout=None,
            max_retry=3,
            max_errors=50,
            output_path="evaluate.json"
        ):
        self.original_dataset = original_dataset
        self.synthetic_dataset = synthetic_dataset
        self.workload = workload
        self.run_len = run_len
        self.timeout = timeout
        self.max_retry = max_retry
        self.max_errors = max_errors
        self.output_path = os.path.join(git_root_dir, os.path.join("eval", output_path))
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
        current_results.append(result)
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
        if not isinstance(metric_instance, CompareMetric):
            raise ValueError("Metric {} requires only one dataset.".format(name))
        return metric_instance.compute(self.original_dataset, self.synthetic_dataset)

    def run(self):
        for item in self.workload:
            name = item["metric"]
            params = item["params"]

            if self._is_metric_computed(name, params):
                continue  # Skip this metric and move to the next

            try:
                result = self._compute_metric(name, params)
                self._save_intermediate_results(result)
            except Exception as e:
                self.error_count += 1
                error_result = {
                    "name": name,
                    "parameters": params,
                    "value": None,
                    "error": str(e)
                }
                self._save_intermediate_results(error_result)
            if self.error_count > self.max_errors:
                raise Exception(f"Exceeded the maximum error limit of {self.max_errors}")