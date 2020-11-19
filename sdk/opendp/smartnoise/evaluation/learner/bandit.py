import os
import numpy as np
import pandas as pd 

from opendp.smartnoise.evaluation.params._learner_params import LearnerParams
from opendp.smartnoise.evaluation.learner.util import create_simulated_dataset, generate_neighbors, generate_query, write_to_csv

from opendp.smartnoise.evaluation.params._learner_params import LearnerParams
from opendp.smartnoise.evaluation.params._privacy_params import PrivacyParams
from opendp.smartnoise.evaluation.params._eval_params import EvaluatorParams
from opendp.smartnoise.evaluation.params._dataset_params import DatasetParams
from opendp.smartnoise.evaluation.evaluator._dp_evaluator import DPEvaluator
from opendp.smartnoise.sql import PandasReader
from dp_singleton_query import DPSingletonQuery


class Bandit():
    def __init__(self):
        self.pp = PrivacyParams(epsilon=1.0)
        self.ev = EvaluatorParams(repeat_count=100)
        self.dd = DatasetParams(dataset_size=500)
        self.pa = DPSingletonQuery()

    def learn(self, querypool, export_as_csv=False):
        output = []
        for i in range(len(querypool)):
            df, metadata = create_simulated_dataset(self.dd.dataset_size, "dataset")
            d1_dataset, d2_dataset, d1_metadata, d2_metadata = generate_neighbors(df, metadata)
            d1 = PandasReader(d1_metadata, d1_dataset)
            d2 = PandasReader(d2_metadata, d2_dataset)
            eval = DPEvaluator()
            pa = DPSingletonQuery()
            key_metrics = eval.evaluate([d1_metadata, d1], [d2_metadata, d2], pa, querypool[i], self.pp, self.ev)
            if key_metrics['__key__'].dp_res is None:
                dp_res = key_metrics['__key__'].dp_res
                error =  key_metrics['__key__'].error
                output.append({"query":querypool[i], "dpresult": dp_res, "jensen_shannon_divergence":None, "error": error})
            else:
                res_list = []
                for key, metrics in key_metrics.items():
                    dp_res = metrics.dp_res
                    js_res = metrics.jensen_shannon_divergence
                    res_list.append([dp_res, js_res])
                dp_res = np.all(np.array([res[0] for res in res_list]))
                js_res = (np.array([res[1] for res in res_list])).max()
                output.append({"query":querypool[i], "dpresult": dp_res,"jensen_shannon_divergence": js_res, "error":None})
        if export_as_csv:
            write_to_csv('Bandit.csv', output, flag='bandit')
        else:
            return output
