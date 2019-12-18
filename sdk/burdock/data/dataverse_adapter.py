import os
import tempfile
import requests

import pandas as pd

from burdock.query.sql import MetadataLoader
from burdock.query.sql.reader import CSVReader

from .dataset_adapter import DatasetAdapter


class DataverseAdapter(DatasetAdapter):
    KEY = "dataverse"

    @staticmethod
    def validate_document(document):
        if document.dataverse_details is None:
            raise Exception("Malformed details.")

    @staticmethod
    def _load_df(dataset_document):
        response = requests.get(dataset_document.dataverse_details.host, headers={"X-Dataverse-key":dataset_document.dataverse_details.token})
        response.raise_for_status()

        temp_dir = tempfile.gettempdir()
        path = os.path.join(temp_dir, "data.tsv")
        with open(path, "w") as stream:
            stream.write(response.text)

        return pd.read_csv(path, sep='\t')

    @staticmethod
    def _load_metadata(dataset_document):
        return MetadataLoader(dataset_document.dataverse_details.local_metadata_path).read_schema()

    @staticmethod
    def _load_reader(dataset_document):
        return CSVReader(DataverseAdapter.load_metadata(dataset_document),
                         DataverseAdapter.load_df(dataset_document))
