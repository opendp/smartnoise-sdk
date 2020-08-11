import pandas as pd
import numpy as np
import math

class DPSampleLibrary:
    """
    Sample DP library class implementing DP functions (analysis)
    """
    def dp_count(self, df, privacy_params, eval_params):
        """
        Returns repeatedly applied differentially private noisy responses while 
        counting rows of a dataset
        """
        delta = 1/(len(df) * math.sqrt(len(df)))
        sigmacnt = math.sqrt(privacy_params.t)*((math.sqrt(math.log(1/delta)) + math.sqrt(math.log((1/delta)) + privacy_params.epsilon)) / (math.sqrt(2)*privacy_params.epsilon))
        dp_noise = (len(df) + np.random.normal(0, sigmacnt, eval_params.repeat_count)).tolist()
        return {"__key__" : dp_noise}

    def dp_sum(self, df, privacy_params, eval_params):
        """
        Returns repeatedly applied differentially private noisy response to 
        summing rows of a numerical column in a dataset
        """
        res = {}
        for colname in list(df):
            delta = 1/(len(df) * math.sqrt(len(df)))
            M = abs(max(df[colname]) - min(df[colname]))
            sigmasum = math.sqrt(privacy_params.t)*M*((math.sqrt(math.log(1/delta)) + math.sqrt(math.log((1/delta)) + privacy_params.epsilon)) / (math.sqrt(2)*privacy_params.epsilon))
            dp_noise = (df[colname].sum() + np.random.normal(0, sigmasum, eval_params.repeat_count)).tolist()
            res[colname] = dp_noise
        return res