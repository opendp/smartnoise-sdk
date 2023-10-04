from sneval import Metric
import json
import os
import subprocess
import sneval.metrics.basic as BasicModule
from .metrics.basic.base import SingleColumnMetric, MultiColumnMetric
import inspect

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

class Analyze:
    def __init__(
            self, 
            dataset, 
            *ignore, 
            workload=[],
            run_len=2,
            timeout=None,
            max_retry=3,
            max_errors=50,
            output_path=os.path.join(git_root_dir, "eval/analyze_output.json")
        ):
        self.dataset = dataset
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
        return metric_instance.compute(self.dataset)
    
    def run(self):
        metric_names = [name for name, obj in inspect.getmembers(BasicModule) 
                        if inspect.isclass(obj) and not name in ["SingleColumnMetric", "MultiColumnMetric"]]
        for wl in self.workload:
            names = wl.get("metrics", metric_names)
            params = {}
            params["column_name"] = wl.get("column_name")
            params["column_names"] = wl.get("column_names")
            if params["column_name"] is None and params["column_names"] is None:
                continue

        for name in names:
            # params = params["params"]
            cls = getattr(BasicModule, name)
            if issubclass(cls, SingleColumnMetric):
                new_params = {k: params[k] for k in ["column_name"] if k in params}
            elif issubclass(cls, MultiColumnMetric):
                new_params = {k: params[k] for k in ["column_names"] if k in params}
            else:
                continue

            # is_metric_defined = name in vars(BasicModule) and isinstance(vars(BasicModule)[name], type)
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
                self._save_intermediate_results(error_result)
            if self.error_count > self.max_errors:
                raise Exception(f"Exceeded the maximum error limit of {self.max_errors}")
