import os
import time

import numpy as np
import pandas as pd

import mlflow

from joblib import Parallel, delayed

import conf

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

def run_synthesis(synthesis_args):
    """
    A parallel run of the synthesis step

    :param synthesis_args: 
        n = name of the synthesizer, 
        s = synthesizer object,
        synth_args = dictionary of hyperparams for the synthesizer
        d = name of dataset
        e = epsilon value 
        datasets = dataset dictionary 
        cat_cols = list of categorical columns in dataset d
    :type synthesis_args: tuple
    :return: (n, d, str(e), sampled = synthesized data of size len(d))
    :rtype: tuple
    """
    n, s, synth_args, d, e, datasets, cat_cols = synthesis_args
    with mlflow.start_run(nested=True):
        start_time = time.time()
        synth = s(epsilon=float(e), **synth_args)
        d_copy = datasets[d]["data"].copy()
        sampled = synth.fit_sample(d_copy,categorical_columns=cat_cols.split(','))
        end_time = time.time()
        mlflow.set_tags({"synthesizer": type(synth),
                         "args": str(synth_args),
                         "epsilon": str(e),
                         "dataset": str(d),
                         "duration_seconds": str(end_time - start_time)})
        print(datasets[d]["name"] + ' finished. Epsilon: ' + str(e))
        datasets[d][n][str(e)] = sampled
    return (n, d, str(e), sampled)

def run_all_synthesizers(datasets, epsilons, save_models_path, run_name):
    """
    Run each synthesizer on each dataset for specified epsilons

    :param datasets: dictionary with real datasets
    :type datasets: dict
    :param synthesizers: list of synthesizers used, often KNOWN_SYNTHESIZERS
    :type synthesizers: list
    :param epsilons: Epsilons used in the synthesis
    :type epsilons: list
    :return: dictionary of both real and synthetic data for each dataset
    :rtype: dict
    """
    synthesizer_runs = []
    for n, s in conf.SYNTHESIZERS:
        print('Synthesizer: '+ str(n))
        for d in datasets:
            datasets[d][n] = {}

            if d in conf.SYNTH_SETTINGS[n]:
                synth_args = conf.SYNTH_SETTINGS[n][d]
            else:
                synth_args = conf.SYNTH_SETTINGS[n]['default']

            for e in epsilons:
                a_run = (n, s, synth_args, d, e, datasets, datasets[d]["categorical_columns"])
                synthesizer_runs.append(a_run)

        # TODO: This needs to be moved out, and parallelized further
        start = time.time()
        job_num = len(datasets) * len(epsilons)
        results = Parallel(n_jobs=job_num, verbose=1, backend="loky")(
            map(delayed(run_synthesis), synthesizer_runs))
        end = time.time() - start
        for n, d, e, sampled in results:
            datasets[d][n][e] = sampled

        print('Synthesis for ' + str(n) + ' finished in ' + str(end))

    return datasets
