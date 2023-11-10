from sneval import Metric
import json
import os
import subprocess
from sneval.dataset import Dataset
import sneval.metrics.compare as CompareModule
from .metrics.compare.base import CompareMetric
import inspect
import csv
from itertools import combinations
import gc

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

class Evaluate:
    """The Evaluate class is is used to compare the original dataset with one or more synthetic
        datasets or private synopses, to understand the utility impact of the privacy mitigations.

        :param original_dataset: The original dataset to compare against. Must be a Dataset object, wrapping a Spark DataFrame.
        :param synthetic_datasets: A list of synthetic datasets to compare against. Each dataset must be a Dataset object, wrapping a Spark DataFrame.
        :param workload: By default, Analyze will analyze one-way and two way marginals,
            if you want to analyze specific marginals, you can pass them in as a list
            of tuples, with each tuple containing the column names to include in the marginal.
        :type workload: list, optional
        :param metrics: If not specified, Analyze will compute a default set of metrics. To specify
            a specific set of metrics, pass in as JSON here. See the documentation for more details.
        :type metrics: dict, optional
        :param run_len: The maximum marginal width to analyze. Defaults to 2. You may set this to
            zero if you don't want to measure any marginal-based metrics, or if you only want to measure
            the marginals you specified in the workload parameter.
        :type run_len: int, optional
        :param timeout: The maximum amount of time to spend computing all metrics. Defaults to None,
            which means no timeout.
        :type timeout: int, optional
        :param max_retry: The maximum number of times to retry a metric computation if it fails.
            Defaults to 3.
        :type max_retry: int, optional
        :param max_errors: The maximum number of errors to allow before giving up. Defaults to 100.
        :type max_errors: int, optional

    """

    def __init__(
            self, 
            original_dataset : Dataset,
            synthetic_datasets : list[Dataset],
            *ignore, 
            workload=[],
            metrics={},
            run_len=2,
            timeout=None,
            max_retry=3,
            max_errors=100,
            output_path=os.path.join(git_root_dir, "eval")
        ):
        self.original_dataset = original_dataset
        self.synthetic_datasets = synthetic_datasets
        self.workload = workload
        self.metrics = metrics
        self.run_len = run_len
        self.timeout = timeout
        self.max_retry = max_retry
        self.max_errors = max_errors
        self.json_output_path = os.path.join(output_path, "evaluate_output.json")
        self.csv_output_path = os.path.join(output_path, "evaluate_output.csv")
        self.error_count = 0

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
        current_results += result
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
        if self._is_metric_computed(name, params):
            pass
        else:
            metric_instance = Metric.create(name, **params)
            for i, synth_dataset in enumerate(self.synthetic_datasets):
                try:
                    res = metric_instance.compute(self.original_dataset, synth_dataset)
                    res['data pair'] = f'0 - {i}'
                    self._save_intermediate_results([res])

                    # If datasets are cached, unpersist them
                    self.original_dataset.source.unpersist()
                    synth_dataset.source.unpersist()
                except Exception as e:
                    self.error_count += 1
                    error_result = {
                        "name": name,
                        "parameters": params,
                        "value": None,
                        "error": str(e),
                        "data_pair": f'0 - {i}'
                    }
                    self._save_intermediate_results([error_result])          
                if self.error_count > self.max_errors:
                    raise Exception(f"Exceeded the maximum error limit of {self.max_errors}")
    
    def run(self):
        """Run the analysis.  This will compute all metrics specified in the constructor, and
            store the results as JSON
        """
        metric_names = [name for name, obj in inspect.getmembers(CompareModule) if inspect.isclass(obj) and name != "CompareMetric"]
        for wl in self.workload:
            names = wl.get("metrics", metric_names)

            param_list = []
            if not wl:  # do a default 2-way computation
                param_list.append({"categorical_columns": self.original_dataset.categorical_columns})
                n_way = self.run_len
                while n_way >= 1:
                    current_combs = [list(combo) for combo in combinations(self.original_dataset.categorical_columns, n_way)]
                    for col_comb in current_combs:
                        param_list.append({"categorical_columns": col_comb})
                    n_way -= 1
            else:
                params = {}
                params["categorical_columns"] = wl.get("categorical_columns")
                if params["categorical_columns"] is None:
                    continue
                params["measure_sum_columns"] = wl.get("measure_sum_columns")
                params["edges"] = wl.get("edges", [1, 10, 100, 1000, 10000, 100000])
                params["unknown_keyword"] = wl.get("unknown_keyword", "Unknown")
                param_list.append(params)


            for par in param_list:
                for name in names:
                    cls = getattr(CompareModule, name)
                    if not issubclass(cls, CompareMetric):
                        continue
                    new_par = {"categorical_columns": par["categorical_columns"]}
                    if name in ["MeanAbsoluteError", "MeanProportionalError"]:
                        if par.get("measure_sum_columns") is None:
                            continue
                        new_par["measure_sum_columns"] = par.get("measure_sum_columns")
                        new_par["edges"] = par.get("edges", [1, 10, 100, 1000, 10000, 100000])
                    elif name in ["MeanAbsoluteErrorInCount", "MeanProportionalErrorInCount"]:
                        new_par["edges"] = par.get("edges", [1, 10, 100, 1000, 10000, 100000])
                    elif name == "FabricatedCombinationCount":
                        new_par["unknown_keyword"] = par.get("unknown_keyword", "Unknown")
                    else:
                        pass
                    self._compute_metric(name, new_par)
        
        json_data = self._load_previous_results()
        csv_data = {}
        for entry in json_data:
            metric_name = entry['name']
            data_pair = entry['data pair']
            if data_pair not in csv_data:
                csv_data[data_pair] = {}

            # Retrieve the parameters.
            params = entry.get('parameters', {})
            categorical_columns = params.get('categorical_columns')
            measure_sum_columns = params.get('measure_sum_columns')

            if categorical_columns is not None and len(categorical_columns) > 0:
                column_identifier = ', '.join(sorted(categorical_columns))
            else:
                continue

            if column_identifier not in csv_data[data_pair]:
                csv_data[data_pair][column_identifier] = {}

            # Constructing new metric descriptors
            metric_descriptor = metric_name
            if measure_sum_columns is not None and len(measure_sum_columns) > 0:
                metric_descriptor = f"{metric_descriptor}_{measure_sum_columns[0]}"  # Only one measure/sum column will be taken
            
            edges = params.get('edges')
            metric_descriptor_list = []
            if edges is not None and len(edges) > 0:
                metric_descriptor_list.append(f'{metric_descriptor} (bin < {edges[0]})')
                for i in range(1, len(edges)):
                    metric_descriptor_list.append(f'{metric_descriptor} (bin [{edges[i-1]}, {edges[i]}))')
                metric_descriptor_list.append(f'{metric_descriptor} (bin >= {edges[-1]})')
            
            if len(metric_descriptor_list) > 0:  
                for i, descriptor in enumerate(metric_descriptor_list):
                    csv_data[data_pair][column_identifier][descriptor] = 'N/A' if 'error' in entry else entry.get('value').get(f'Bin {i}', 'N/A')
            else:
                csv_data[data_pair][column_identifier][metric_descriptor] = 'N/A' if 'error' in entry else entry.get('value', 'N/A')
        
        headers = ['data pair', 'columns'] + sorted(list(set(metric_descriptor for data_pairs in csv_data.values() for columns in data_pairs.values() for metric_descriptor in columns)))
        with open(self.csv_output_path, 'w', newline='') as csvfile:
            csvwriter = csv.DictWriter(csvfile, fieldnames=headers)
            csvwriter.writeheader()
            for data_pair, data in csv_data.items():
                row = {'data pair': data_pair} 
                for column_identifier, metrics in data.items():
                    row.update({'columns': column_identifier})
                    row.update(metrics)  # Fill in the metric values for this column identifier.
                    csvwriter.writerow(row)