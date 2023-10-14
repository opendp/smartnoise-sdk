from sneval import Metric
import json
import os
import subprocess
import sneval.metrics.basic as BasicModule
from .metrics.basic.base import SingleColumnMetric, MultiColumnMetric
import inspect
import csv

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
            output_path=os.path.join(git_root_dir, "eval")
        ):
        self.dataset = dataset
        self.workload = workload
        self.run_len = run_len
        self.timeout = timeout
        self.max_retry = max_retry
        self.max_errors = max_errors
        self.error_count = 0
        self.json_output_path = os.path.join(output_path, "analyze_output.json")
        self.csv_output_path = os.path.join(output_path, "analyze_output.csv")

    def _load_previous_results(self):
        """
        Load results from previous runs if available.
        """
        try:
            with open(self.json_output_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
        
    def _save_intermediate_results(self, result):
        """
        Save results incrementally after each metric computation.
        """
        current_results = self._load_previous_results()
        current_results.append(result)
        with open(self.json_output_path, 'w') as f:
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
        
        json_data = self._load_previous_results()
        csv_data = {}
        for entry in json_data:
            metric_name = entry['name']

            # Retrieve the parameters.
            params = entry.get('parameters', {})
            column_names = params.get('column_names')
            single_column = params.get('column_name')

            if column_names is not None and column_names != "null":
                column_identifier = ', '.join(sorted(column_names))  # This treats the list as a single identifier.
            elif single_column is not None and single_column != "null":
                column_identifier = single_column
            else:
                continue
            
            if column_identifier not in csv_data:
                csv_data[column_identifier] = {}

            # Constructing metric parameters representation
            paras_repr = '; '.join([
                f"{k}={v}" for k, v in params.items() if k not in ('column_names', 'column_name')])
            # Create a unique metric descriptor for each metric consisting of its name and parameters representation
            metric_descriptor = f"{metric_name} ({paras_repr})" if paras_repr else metric_name

            csv_data[column_identifier][metric_descriptor] = 'N/A' if 'error' in entry else entry.get('value', 'N/A')
        
        headers = ['columns'] + sorted(list(set(metric_descriptor for data in csv_data.values() for metric_descriptor in data)))
        with open(self.csv_output_path, 'w', newline='') as csvfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames=headers)
            csvwriter.writeheader()

            for column_identifier, metrics in csv_data.items():
                row = {'columns': column_identifier}
                row.update(metrics)  # Fill in the metric values for this column identifier.
                csvwriter.writerow(row)
