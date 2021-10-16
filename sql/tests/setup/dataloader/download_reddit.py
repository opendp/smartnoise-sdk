import os
import subprocess
import pandas as pd

root_url = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()

def _download_file(url, local_file):
    try:
        from urllib import urlretrieve
    except ImportError:
        from urllib.request import urlretrieve
    urlretrieve(url, local_file)

def find_ngrams(input_list, n):
    return input_list if n == 1 else list(zip(*[input_list[i:] for i in range(n)]))

def download_reddit():
    reddit_dataset_path = os.path.join(root_url,"datasets", "reddit.csv")
    if not os.path.exists(reddit_dataset_path):
        print("Downloading reddit test dataset")
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