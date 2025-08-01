from sneval import Metric
import json
from sneval.dataset import Dataset
import sneval.metrics.compare as CompareModule
from .metrics.compare.base import CompareMetric
import inspect
from itertools import combinations

class Evaluate:
    """The Evaluate class is used to compare the original dataset with one or more synthetic
        datasets or private synopses, to understand the utility impact of the privacy mitigations.

        :param original_dataset: The original dataset to compare against. Must be a Dataset object, wrapping a Spark DataFrame.
        :param synthetic_datasets: A list of synthetic datasets to compare against. Each dataset must be a Dataset object, wrapping a Spark DataFrame.
        :param workload: By default, Analyze will analyze one-way and two way marginals,
            if you want to analyze specific marginals, you can pass them in as a list
            of dicts, with each dict indicating the column name(s) and the metric 
            parameters for analysis.
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
            workload=None,
            metrics=None,
            run_len=2,
            timeout=None,
            max_retry=3,
            max_errors=200,
        ):
        self.original_dataset = original_dataset
        self.synthetic_datasets = synthetic_datasets
        self.workload = workload if workload is not None else [{}]
        self.metrics = metrics if metrics is not None else {}
        self.run_len = run_len
        self.timeout = timeout
        self.max_retry = max_retry
        self.max_errors = max_errors
        self.error_count = 0
        self._computed_cache = set()

    def _cache_key(self, name, params):
        """
        Convert metric name and params into a unique, hashable key.
        Sort keys to avoid ordering issues in dicts.
        """
        key_str = json.dumps({"name": name, "params": params}, sort_keys=True)
        return key_str

    def _compute_metric(self, name, params):
        results = []
        metric_instance = Metric.create(name, **params)

        for i, synth_dataset in enumerate(self.synthetic_datasets):
            cache_key = self._cache_key(name, params)
            if cache_key in self._computed_cache:
                continue
            
            try:
                res = metric_instance.compute(self.original_dataset, synth_dataset)
                res['data pair'] = f'0 - {i}'
                results.append(res)
            except Exception as e:
                self.error_count += 1
                results.append({
                    "name": name,
                    "parameters": params,
                    "value": None,
                    "error": str(e),
                    "data pair": f'0 - {i}'
                })         
                if self.error_count > self.max_errors:
                    raise Exception(f"Exceeded the maximum error limit of {self.max_errors}")
            finally:
                self._computed_cache.add(cache_key)

                # Unpersist if cached
                if self.original_dataset.source.is_cached:
                    self.original_dataset.source.unpersist()
                if synth_dataset.source.is_cached:
                    synth_dataset.source.unpersist()
        return results
    
    def run(self):
        """Run the evaluation and return results as a list of dicts."""
        metric_names = [name for name, obj in inspect.getmembers(CompareModule) if inspect.isclass(obj) and name != "CompareMetric"]
        all_results = []

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
                if wl.get("categorical_columns") is None:
                    continue
                param_list.append({
                    "categorical_columns": wl["categorical_columns"],
                    "measure_sum_columns": wl.get("measure_sum_columns"),
                    "edges": wl.get("edges", [1, 10, 100, 1000, 10000, 100000]),
                    "unknown_keyword": wl.get("unknown_keyword", "Unknown")
                })


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

                    results = self._compute_metric(name, new_par)
                    all_results.extend(results)
        
        return all_results