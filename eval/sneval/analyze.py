from sneval import Metric, Dataset
import json
import os
import subprocess
import sneval.metrics.basic as BasicModule
from .metrics.basic.base import SingleColumnMetric, MultiColumnMetric
import inspect
import csv
from itertools import combinations

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

class Analyze:
    """The Analysis class is used to analyze a dataset to provide information 
        useful for planning a privacy mitigation. You pass in a dataset, call run(),
        and then review the results.
        
        :param dataset:  The dataset to analyze. Must be a Dataset object, wrapping a Spark DataFrame.
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
            dataset : Dataset, 
            *ignore, 
            workload=[],
            metrics={},
            run_len=2,
            timeout=None,
            max_retry=3,
            max_errors=100,
            output_path="eval_output"
        ):
        self.dataset = dataset
        self.workload = workload
        self.run_len = run_len if run_len <= 3 else 2  # do 3-way computations at most
        self.timeout = timeout
        self.max_retry = max_retry
        self.max_errors = max_errors
        self.error_count = 0
        self.metrics = metrics
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
        if self._is_metric_computed(name, params):
            pass
        else:
            try:
                metric_instance = Metric.create(name, **params)
                res = metric_instance.compute(self.dataset)
                self._save_intermediate_results(res)

                # If datasets are cached, unpersist them
                if self.dataset.source.is_cached:
                    self.dataset.source.unpersist()
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
    
    def run(self):
        """Run the analysis.  This will compute all metrics specified in the constructor, and
            store the results as JSON
        """
        metric_names = [name for name, obj in inspect.getmembers(BasicModule) 
                        if inspect.isclass(obj) and not name in ["SingleColumnMetric", "MultiColumnMetric"]]
        
        def generate_params(column_names, metric_name):
            if metric_name in ("BelowKCombs", "BelowKCount"):
                return [{"column_names": column_names, "k": k} for k in (5, 10)]
            return [{"column_names": column_names}]
        
        for wl in self.workload:
            names = wl.get("metrics", metric_names)

            param_list = []
            if not wl:  # do a default 1-way and 2-way computation
                param_list.append({"column_names": self.dataset.categorical_columns})
                n_way = self.run_len
                while n_way >= 1:
                    if n_way >= 2:  # 2-way or 3-way metric computation
                        current_combs = [list(combo) for combo in combinations(self.dataset.categorical_columns, n_way)]
                        for col_comb in current_combs:
                            param_list.append({"column_names": col_comb})
                    else:  # 1-way metric computation
                        for col in (self.dataset.categorical_columns + self.dataset.measure_columns + [self.dataset.count_column]):  
                            param_list.append({"column_name": col})
                    n_way -= 1
            else:
                if wl.get("column_names") is not None:
                    param_list.append({"column_names": wl.get("column_names")})
                if wl.get("column_name") is not None:    
                    param_list.append({"column_name": wl.get("column_name")})
                

            for par in param_list:
                for name in names:
                    new_pars = []
                    cls = getattr(BasicModule, name)
                    if issubclass(cls, SingleColumnMetric):
                        if "column_name" not in par:
                            continue
                        new_pars.append(par)
                    elif issubclass(cls, MultiColumnMetric):
                        if "column_names" not in par:
                            continue
                        new_pars.extend(generate_params(par["column_names"], name))
                    else:
                        continue

                    # is_metric_defined = name in vars(BasicModule) and isinstance(vars(BasicModule)[name], type)
                    for new_par in new_pars:
                        self._compute_metric(name, new_par)
        
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
