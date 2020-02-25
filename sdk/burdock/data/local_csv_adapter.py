import pandas as pd

from burdock.query.sql import MetadataLoader
from burdock.query.sql.reader import PandasReader

from .dataset_adapter import DatasetAdapter


class LocalCSVAdapter(DatasetAdapter):
    KEY = "local_csv"

    @staticmethod
    def validate_document(document):
        if document.csv_details is None:
            raise Exception("Malformed details.")

    def _load_df(dataset_document):
        return pd.read_csv(dataset_document.csv_details.local_path)

    def _load_metadata(dataset_document):
        return MetadataLoader(dataset_document.csv_details.local_path.split(".")[0] + ".yaml").read_schema()

    def _load_reader(dataset_document):
        return PandasReader(LocalCSVAdapter.load_metadata(dataset_document),
                               LocalCSVAdapter.load_df(dataset_document))
