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
        sigmacnt = math.sqrt(privacy_params.t)*((math.sqrt(math.log(1/delta)) + math.sqrt(math.log((1/delta)) + eval_params.repeat_count)) / (math.sqrt(2)*eval_params.repeat_count))
        dp_noise = (len(df) + np.random.normal(0, sigmacnt, eval_params.repeat_count)).tolist()
        return {"__key__" : dp_noise}