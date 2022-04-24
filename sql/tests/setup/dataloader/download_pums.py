import os
import subprocess
import pandas as pd
import random
import numpy as np

root_url = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()


def _download_file(url, local_file):
    try:
        from urllib import urlretrieve
    except ImportError:
        from urllib.request import urlretrieve
    urlretrieve(url, local_file)

def download_pums():
    pums_csv_path = os.path.join(root_url,"datasets", "PUMS.csv")
    pums_pid_csv_path = os.path.join(root_url,"datasets", "PUMS_pid.csv")
    pums_large_csv_path = os.path.join(root_url,"datasets", "PUMS_large.csv")
    pums_dup_csv_path = os.path.join(root_url,"datasets", "PUMS_dup.csv")
    pums_null_csv_path = os.path.join(root_url,"datasets", "PUMS_null.csv")
    
    if not os.path.exists(pums_large_csv_path):
        print("Downloading PUMS large with 1.2 million rows")
        pums_large_url = "https://raw.githubusercontent.com/opendifferentialprivacy/dp-test-datasets/master/data/PUMS_california_demographics/data.csv"
        _download_file(pums_large_url, pums_large_csv_path)
    if not os.path.exists(pums_csv_path):
        print("Downloading PUMS small with 1000 rows")
        pums_url = "https://raw.githubusercontent.com/opendifferentialprivacy/dp-test-datasets/master/data/PUMS_california_demographics_1000/data.csv"
        _download_file(pums_url, pums_csv_path)
    if not os.path.exists(pums_pid_csv_path):
        print("Generating PUMS small with private_key")
        df = pd.read_csv(pums_csv_path)
        df_pid = df.assign(pid = [i for i in range(1, 1001)])
        df_pid.to_csv(pums_pid_csv_path, index=False)
    if not os.path.exists(pums_dup_csv_path):
        print("Generating PUMS small with 1-3X duplicate IDs")
        random.seed(1011)
        df_pid = pd.read_csv(pums_pid_csv_path)
        for _ in range(2):
            n_rows = df_pid.shape[0]
            for n in range(n_rows):
                row = df_pid.iloc[n]
                if row['sex'] == 1.0:
                    p = 0.22
                else:
                    p = 0.56
                if random.random() < p:
                    df_pid.loc[df_pid.shape[0]] = row
        df_pid = df_pid.astype(int)
        df_pid.to_csv(pums_dup_csv_path, index=False)
    if not os.path.exists(pums_null_csv_path):
        print("Generating PUMS small with ~40% rows with NULL and duplicate IDs")
        random.seed(1011)
        df_dup = pd.read_csv(pums_dup_csv_path)
        colnames = list(df_dup.columns)
        p = [random.choice([1.0, 1.5, 2.0]) for _ in range(len(colnames))]
        p = np.array(p) / np.sum(p)
        for idx, row in df_dup.iterrows():
            if random.random() < 0.38:
                col = np.random.choice(colnames, 1, p=p)
                df_dup.at[idx, col] = pd.NA
        df_dup.to_csv(pums_null_csv_path, index=False)
