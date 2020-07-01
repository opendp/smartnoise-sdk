import conf

import numpy as np
import pandas as pd

def load_data(req_datasets=[]):
    import requests
    import io
    import json
    # Returns a dictionary of datasets
    # TODO: Add memory check to make sure to not overwhelm computer
    # TODO: Add safety checks here
    if not req_datasets:
        req_datasets = conf.KNOWN_DATASETS

    with open('datasets.json') as j:
        dsets = j.read()
    archive = json.loads(dsets)

    loaded_datasets = {}

    def retrieve_dataset(dataset):
        r = requests.post(dataset['url'])
        if r.ok:
            data = r.content.decode('utf8')
            sep = dataset['sep']
            df = pd.read_csv(io.StringIO(data), names=dataset['columns'].split(','), sep=sep, index_col=False)
            if dataset['header'] == "t":
                df = df.iloc[1:]
            return df

        raise "Unable to retrieve dataset: " + dataset

    def select_column(scol):
        # Zero indexed, inclusive
        return scol.split(',')

    def encode_categorical(df,dataset):
        from sklearn.preprocessing import LabelEncoder

        encoders = {}
        if dataset['categorical_columns'] != "":
            for column in select_column(dataset['categorical_columns']):
                encoders[column] = LabelEncoder()
                df[column] = encoders[column].fit_transform(df[column])

        return {"data": df, "target": dataset['target'], "name": dataset['name']}


    for d in req_datasets:
        df = retrieve_dataset(archive[d])
        encoded_df_dict = encode_categorical(df, archive[d])
        loaded_datasets[d] = encoded_df_dict

    # Return dictionary of pd dataframes
    return loaded_datasets