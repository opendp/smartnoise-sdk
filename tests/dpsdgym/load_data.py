import conf

import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE

import zipfile

# NOTE: Temporary
# We add a memory cap here for now, which
# forces a subsampling of particularly large
# datasets in order to not overwhelm
# joblib 
#
# The largest included dataset is adult, which
# is ~4000000 bytes 
MEM_CAP = 1000000 #1500000

def load_data():
    """
    Takes in optional dataset list. Otherwise grabs them
    from the conf.py file.

    Returns a dictionary of datasets.
    {
        'dset': pd.DataFrame,
        'dset': pd.DataFrame
    }
    """
    import requests
    import io
    import json

    req_datasets = conf.KNOWN_DATASETS

    with open('datasets.json') as j:
        dsets = j.read()
    archive = json.loads(dsets)

    loaded_datasets = {}

    def retrieve_dataset(dataset):
        if dataset['zipped'] == 'f':
            r = requests.post(dataset['url'])
            if r.ok:
                data = r.content.decode('utf8')
                df = pd.read_csv(io.StringIO(data), names=dataset['columns'].split(','), sep=dataset['sep'], index_col=False)
                if dataset['header'] == "t":
                    df = df.iloc[1:]
                return df

            raise "Unable to retrieve dataset: " + dataset
        else:
            r = requests.get(dataset['url'])
            z = zipfile.ZipFile(io.BytesIO(r.content))
            df = pd.read_csv(z.open(dataset['zip_name']), names=dataset['columns'].split(','), sep=dataset['sep'], index_col=False)
            if dataset['header'] == "t":
                df = df.iloc[1:]
            return df

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

        df = df.apply(pd.to_numeric, errors='ignore')
        data_mem = df.memory_usage(index=True).sum()
        print("Memory consumed by " + dataset['name'] + ":" + str(data_mem))
        if data_mem > MEM_CAP:
            print("Memory use too high with " + dataset['name'] + ", subsampling to:" + str(MEM_CAP))
            reduct_ratio = MEM_CAP / data_mem
            subsample_count = int(len(df.index) * reduct_ratio)
            df = df.sample(n=subsample_count)
            print("Memory consumed by " + dataset['name'] + ":" + str(df.memory_usage(index=True).sum()))

        # If our dataset is imbalanced, let's fix it
        # here before we go on to eval
        
        # Make a new df made of all the columns, except the target class
        # if dataset['imbalanced'] == 't':
        #     X = df.loc[:, df.columns != dataset['target']]
        #     y = df.loc[:, df.columns == dataset['target']]
        #     sm = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=42)
        #     X_smote, y_smote = sm.fit_resample(X, y)
        #     X_smote[dataset['target']] = y_smote
        #     df = X_smote

        return {"data": df, "target": dataset['target'], "name": dataset['name'], "categorical_columns": dataset['categorical_columns']}

    for d in req_datasets:
        df = retrieve_dataset(archive[d])
        encoded_df_dict = encode_categorical(df, archive[d]) 
        loaded_datasets[d] = encoded_df_dict

    # Return dictionary of pd dataframes
    return loaded_datasets