import os.path
import pandas as pd


def load_data(req_datasets):
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

    os.makedirs('downloaded_datasets', exist_ok=True)
    with open('datasets.json') as j:
        dsets = j.read()
    archive = json.loads(dsets)

    loaded_datasets = {}

    def retrieve_dataset(dataset):
        r = requests.get(dataset['url'])
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

        df = df.apply(pd.to_numeric, errors='ignore')
        data_mem = df.memory_usage(index=True).sum()
        print("Memory consumed by " + dataset['name'] + ":" + str(data_mem))
        return {"data": df, "target": dataset['target'], "name": dataset['name'], "categorical_columns": dataset['categorical_columns']}

    for d in req_datasets:
        data_name = archive[d]['name']
        data_path = f"downloaded_datasets/{data_name}.csv"
        if os.path.exists(data_path):
            print(f'loading {data_path}')
            df = pd.read_csv(data_path, index_col=0)
        else:
            df = retrieve_dataset(archive[d])
            print(f'saving {data_path}')
            df.to_csv(data_path)
        encoded_df_dict = encode_categorical(df, archive[d]) 
        loaded_datasets[d] = encoded_df_dict

    # Return dictionary of pd dataframes
    return loaded_datasets