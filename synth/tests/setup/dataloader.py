import os
import subprocess
import sklearn.datasets
import pandas as pd
import random

from snsql.metadata import CollectionMetadata
from snsql.metadata.collection import Table, Float, String

root_url = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

pums_csv_path = os.path.join(root_url,"datasets", "PUMS.csv")
pums_pid_csv_path = os.path.join(root_url,"datasets", "PUMS_pid.csv")
pums_large_csv_path = os.path.join(root_url,"datasets", "PUMS_large.csv")
pums_dup_csv_path = os.path.join(root_url,"datasets", "PUMS_dup.csv")

pums_schema_path = os.path.join(root_url,"datasets", "PUMS.yaml")
pums_large_schema_path = os.path.join(root_url,"datasets", "PUMS_large.yaml")
pums_pid_schema_path = os.path.join(root_url,"datasets", "PUMS_pid.yaml")
pums_schema_path = os.path.join(root_url,"datasets", "PUMS.yaml")
pums_dup_schema_path = os.path.join(root_url,"datasets", "PUMS_dup.yaml")


def _download_file(url, local_file):
    try:
        from urllib import urlretrieve
    except ImportError:
        from urllib.request import urlretrieve
    urlretrieve(url, local_file)

def find_ngrams(input_list, n):
    return input_list if n == 1 else list(zip(*[input_list[i:] for i in range(n)]))

def download_data_files():
    iris_dataset_path = os.path.join(root_url,"datasets", "iris.csv")
    if not os.path.exists(iris_dataset_path):
        sklearn_dataset = sklearn.datasets.load_iris()
        sklearn_df = pd.DataFrame(data=sklearn_dataset.data, columns=sklearn_dataset.feature_names)
        sklearn_df.to_csv(iris_dataset_path)


    iris_schema_path = os.path.join(root_url,"datasets", "iris.yaml")
    if not os.path.exists(iris_schema_path):
        iris = Table("iris", "iris", [
                    Float("sepal length (cm)", 4, 8),
                    Float("sepal width (cm)", 2, 5),
                    Float("petal length (cm)", 1, 7),
                    Float("petal width (cm)", 0, 3)
        ], 150)
        schema = CollectionMetadata([iris], "csv")
        schema.to_file(iris_schema_path, "iris")

    if not os.path.exists(pums_csv_path) or not os.path.exists(pums_pid_csv_path) or not os.path.exists(pums_dup_csv_path) or not os.path.exists(pums_large_csv_path):
        pums_url = "https://raw.githubusercontent.com/opendifferentialprivacy/dp-test-datasets/master/data/PUMS_california_demographics_1000/data.csv"
        pums_large_url = "https://raw.githubusercontent.com/opendifferentialprivacy/dp-test-datasets/master/data/PUMS_california_demographics/data.csv"
        _download_file(pums_url, pums_csv_path)
        _download_file(pums_large_url, pums_large_csv_path)
        df = pd.read_csv(pums_csv_path)
        df_pid = df.assign(pid = [i for i in range(1, 1001)])
        df_pid.to_csv(pums_pid_csv_path, index=False)

    if not os.path.exists(pums_dup_csv_path):
        random.seed(1011)
        df_pid = pd.read_csv(pums_pid_csv_path)
        new_records = []
        for _ in range(2):
            for idx, row in df_pid.iterrows():
                if row['sex'] == 1.0:
                    p = 0.22
                else:
                    p = 0.56
                if random.random() < p:
                    new_records.append(row)
        for row in new_records:
            df_pid = df_pid.append(row)
        df_pid = df_pid.astype(int)
        df_pid.to_csv(pums_dup_csv_path, index=False)

    reddit_dataset_path = os.path.join(root_url,"datasets", "reddit.csv")
    if not os.path.exists(reddit_dataset_path):
        import re
        reddit_url = "https://github.com/joshua-oss/differentially-private-set-union/raw/master/data/clean_askreddit.csv.zip"
        reddit_zip_path = os.path.join(root_url,"datasets", "askreddit.csv.zip")
        datasets = os.path.join(root_url,"datasets")
        clean_reddit_path = os.path.join(datasets, "clean_askreddit.csv")
        _download_file(reddit_url, reddit_zip_path)
        from zipfile import ZipFile
        with ZipFile(reddit_zip_path) as zf:
            zf.extractall(datasets)
        reddit_df = pd.read_csv(clean_reddit_path, index_col=0)
        reddit_df = reddit_df.sample(frac=0.01)
        reddit_df['clean_text'] = reddit_df['clean_text'].astype(str)
        reddit_df.loc[:,'clean_text'] = reddit_df.clean_text.apply(lambda x : str.lower(x))
        reddit_df.loc[:,'clean_text'] = reddit_df.clean_text.apply(lambda x : " ".join(re.findall('[\w]+', x)))
        reddit_df['ngram'] = reddit_df['clean_text'].map(lambda x: find_ngrams(x.split(" "), 2))
        rows = list()
        for row in reddit_df[['author', 'ngram']].iterrows():
            r = row[1]
            for ngram in r.ngram:
                rows.append((r.author, ngram))
        ngrams = pd.DataFrame(rows, columns=['author', 'ngram'])
        ngrams.to_csv(reddit_dataset_path)


    reddit_schema_path = os.path.join(root_url,"datasets", "reddit.yaml")
    if not os.path.exists(reddit_schema_path):
        reddit = Table("reddit", "reddit",  [
                    String("author", card=10000, is_key=True),
                    String("ngram", card=10000)
        ], 500000, None, False, max_ids=500)
        schema = CollectionMetadata([reddit], "csv")
        schema.to_file(reddit_schema_path, "reddit")

