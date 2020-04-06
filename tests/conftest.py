import os
import subprocess
import sys
import time
import sklearn.datasets
import pandas as pd


from opendp.whitenoise.metadata import CollectionMetadata
from opendp.whitenoise.metadata.collection import Table, Float, String

from subprocess import Popen, PIPE
from threading import Thread

import pytest

from requests import Session

from opendp.whitenoise.client import _get_client
from opendp.whitenoise.client.restclient.rest_client import RestClient
from opendp.whitenoise.client.restclient.models.secret import Secret
DATAVERSE_TOKEN_ENV_VAR = "WHITENOISE_DATAVERSE_TEST_TOKEN"

# Add the utils directory to the path
root_url = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()
sys.path.append(os.path.join(root_url, "utils"))

from service_utils import run_app  # NOQA

iris_dataset_path = os.path.join(root_url, "service", "datasets", "iris.csv")
if not os.path.exists(iris_dataset_path):
    sklearn_dataset = sklearn.datasets.load_iris()
    sklearn_df = pd.DataFrame(data=sklearn_dataset.data, columns=sklearn_dataset.feature_names)
    sklearn_df.to_csv(iris_dataset_path)


iris_schema_path = os.path.join(root_url, "service", "datasets", "iris.yaml")
if not os.path.exists(iris_schema_path):
    iris = Table("iris", "iris", 150, [
                Float("sepal length (cm)", 4, 8),
                Float("sepal width (cm)", 2, 5),
                Float("petal length (cm)", 1, 7),
                Float("petal width (cm)", 0, 3)
    ])
    schema = CollectionMetadata([iris], "csv")
    schema.to_file(iris_schema_path, "iris")

def find_ngrams(input_list, n):
    return input_list if n == 1 else list(zip(*[input_list[i:] for i in range(n)]))

def _download_file(url, local_file):
    try:
        from urllib import urlretrieve
    except ImportError:
        from urllib.request import urlretrieve
    urlretrieve(url, local_file)

reddit_dataset_path = os.path.join(root_url, "service", "datasets", "reddit.csv")
if not os.path.exists(reddit_dataset_path):
    import re
    reddit_url = "https://github.com/heyyjudes/differentially-private-set-union/raw/master/data/clean_askreddit.csv.zip"
    reddit_zip_path = os.path.join(root_url, "service", "datasets", "askreddit.csv.zip")
    datasets = os.path.join(root_url, "service", "datasets")
    clean_reddit_path = os.path.join(datasets, "clean_askreddit.csv")
    _download_file(reddit_url, reddit_zip_path)
    from zipfile import ZipFile
    with ZipFile(reddit_zip_path) as zf:
        zf.extractall(datasets)
    reddit_df = pd.read_csv(clean_reddit_path, index_col=0)
    reddit_df = reddit_df.sample(frac=0.05)
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


reddit_schema_path = os.path.join(root_url, "service", "datasets", "reddit.yaml")
if not os.path.exists(reddit_schema_path):
    reddit = Table("reddit", "reddit", 500000, [
                String("author", card=10000, is_key=True),
                String("ngram", card=10000)
    ])
    schema = CollectionMetadata([reddit], "csv")
    schema.to_file(reddit_schema_path, "reddit")

@pytest.fixture(scope="session")
def client():
    url = os.environ.get("WHITENOISE_SERVICE_URL", "localhost")
    port = int(os.environ.get("WHITENOISE_SERVICE_PORT", 5001))

    client = _get_client()
    if DATAVERSE_TOKEN_ENV_VAR in os.environ:
        import pdb; pdb.set_trace()
        client.secretsput(Secret(name="dataverse:{}".format("demo_dataverse"),
                                 value=os.environ[DATAVERSE_TOKEN_ENV_VAR]))
    return client

def test_client(client):
    pass
