from sneval import Metric, Dataset
import sneval.metrics.basic as BasicModule
from .metrics.basic.base import SingleColumnMetric, MultiColumnMetric
import inspect
import json
from itertools import combinations


class Analyze:
    """The Analysis class is used to analyze a dataset to provide information 
        useful for planning a privacy mitigation. You pass in a dataset, call run(),
        and then review the results.
        
        :param dataset:  The dataset to analyze. Must be a Dataset object, wrapping a Spark DataFrame.
        :param workload: By default, Analyze will analyze one-way and two way marginals,
            if you want to analyze specific marginals, you can pass them in as a list
            of dicts, with each dict indicating the column name(s) for analysis.
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
            workload=None,
            metrics=None,
            run_len=2,
            timeout=None,
            max_retry=3,
            max_errors=100
        ):
        self.dataset = dataset
        self.workload = workload if workload is not None else [{}]
        self.metrics = metrics if metrics is not None else {}
        self.run_len = run_len if run_len <= 3 else 2  # do 3-way computations at most
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
        cache_key = self._cache_key(name, params)
        if cache_key in self._computed_cache:
            return None
    
        try:
            metric_instance = Metric.create(name, **params)
            res = metric_instance.compute(self.dataset)

            if self.dataset.source.is_cached:
                self.dataset.source.unpersist()

            self._computed_cache.add(cache_key)
            return res
        except Exception as e:
            self.error_count += 1
            if self.error_count > self.max_errors:
                raise Exception(f"Exceeded the maximum error limit of {self.max_errors}")  
            return {
                "name": name,
                "parameters": params,
                "value": None,
                "error": str(e)
            }  
    
    def run(self):
        """Run the analysis and return the list of metric results."""
        metric_names = [name for name, obj in inspect.getmembers(BasicModule) 
                        if inspect.isclass(obj) and not name in ["SingleColumnMetric", "MultiColumnMetric"]]
        all_results = []

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

                    for new_par in new_pars:
                        result = self._compute_metric(name, new_par)
                        if result:
                            all_results.append(result)
        return all_results