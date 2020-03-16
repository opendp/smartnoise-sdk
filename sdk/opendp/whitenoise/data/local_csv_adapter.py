import pandas as pd

from opendp.whitenoise.metadata.collection import CollectionMetadata
from opendp.whitenoise.sql import PandasReader

from .dataset_adapter import DatasetAdapter


class LocalCSVAdapter(DatasetAdapter):
    KEY = "csv_details"

    @staticmethod
    def validate_document(document):
        if document.csv_details is None:
            raise Exception("Malformed details.")

    def _load_df(dataset_document):
        return pd.read_csv(dataset_document.csv_details.local_path)

    def _load_metadata(dataset_document):
        return CollectionMetadata.from_file(dataset_document.csv_details.local_path.split(".")[0] + ".yaml")

    def _load_reader(dataset_document):
        return PandasReader(LocalCSVAdapter.load_metadata(dataset_document),
                               LocalCSVAdapter.load_df(dataset_document))
