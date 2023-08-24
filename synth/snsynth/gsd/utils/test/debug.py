import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import subprocess
from snsynth import Synthesizer
from snsynth.gsd import GSDSynthesizer
import time


from load_data import load_data
from sklearn.model_selection import train_test_split




if __name__ == "__main__":
    adult_path = 'adult.csv'
    datasets = load_data(['adult'])

    adult_df = datasets['adult']['data']

    target = datasets['adult']['target']
    categorical_columns = datasets['adult']['categorical_columns'].split(',')
    print(adult_df.columns)
    print(categorical_columns)

    # Create config file. Note that we know the lower bound of each ordinal feature is 0.
    ordinal_columns = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'fnlwgt']
    continuous_columns = []
    config = {}
    for c in adult_df.columns:
        if c in categorical_columns:
            config[c] = {'type': 'string'}
        else:
            config[c] = {'type': 'int', 'lower': 0, 'upper': adult_df[c].max()}

    # Split into train/test sets for machine learning evaluation.
    adult_df_train, adult_df_test = train_test_split(adult_df, test_size=0.2)
    # Still need to implement
    epsilon = 1.0

    synth = GSDSynthesizer(epsilon, 1e-5, verbose=True)
    synth.fit(adult_df_train, meta_data=config,
              N_prime=5000)

    max_error = np.abs(synth.stat_fn(synth.data.to_numpy()) - synth.stat_fn(synth.sync_data.to_numpy())).max()
    print(f'Statistical error:', max_error)

    adult_sync_df = synth.sample()
    os.makedirs('downloaded_datasets', exist_ok=True)
    adult_sync_df.to_csv(f'downloaded_datasets/adult_sync_{epsilon:.2f}.csv')

