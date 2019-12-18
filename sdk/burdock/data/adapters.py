import os
import tempfile
import requests

import pandas as pd

from burdock.query.sql import MetadataLoader
from burdock.query.sql.reader import CSVReader


class DatasetAdapter(object):
    _dataset_adapters = {}
    _metadata_adapters = {}
    _reader_adapters = {}

    @staticmethod
    def load_df(dataset_document):
        if dataset_document.dataset_type in DatasetAdapter._dataset_adapters:
            return DatasetAdapter._dataset_adapters[dataset_document.dataset_type](dataset_document)
        else:
            # TODO: formalize exceptions
            raise Exception("UnsupportedDatasetDocumentType {} was found"
                            ", supported types include {}.".format(
                                dataset_document.dataset_type, DatasetAdapter._dataset_adapters.keys()))

    @staticmethod
    def load_metadata(dataset_document):
        if dataset_document.dataset_type in DatasetAdapter._metadata_adapters:
            return DatasetAdapter._metadata_adapters[dataset_document.dataset_type](dataset_document)
        else:
            # TODO: formalize exceptions
            raise Exception("UnsupportedDatasetDocumentType {} was found"
                            ", supported types include {}.".format(
                                dataset_document.dataset_type, DatasetAdapter._dataset_adapters.keys()))

    @staticmethod
    def load_reader(dataset_document):
        if dataset_document.dataset_type in DatasetAdapter._reader_adapters:
            return DatasetAdapter._reader_adapters[dataset_document.dataset_type](dataset_document)
        else:
            # TODO: formalize exceptions
            raise Exception("UnsupportedDatasetDocumentType {} was found"
                            ", supported types include {}.".format(
                                dataset_document.dataset_type, DatasetAdapter._reader_adapters.keys()))

def register_adapter(dataset_type, dataset_loader, dataset_metadata_loader, dataset_reader_loader=None):
    if dataset_type not in DatasetAdapter._dataset_adapters:
        DatasetAdapter._dataset_adapters[dataset_type] = dataset_loader
        DatasetAdapter._metadata_adapters[dataset_type] = dataset_metadata_loader
        if dataset_reader_loader is not None:
            DatasetAdapter._reader_adapters[dataset_type] = dataset_reader_loader
    else:
        raise Exception("Dataset type {} has already been "
                        "used to register an adapter.".format(dataset_type))
        # TODO: formalize exceptions


def load_dataset(dataset_document):
    """
    rtype: pandas dataframe
    """
    return DatasetAdapter.load_df(dataset_document)


def load_metadata(dataset_document):
    """
    rtype: MetadataLoader
    """
    return DatasetAdapter.load_metadata(dataset_document)


def load_reader(dataset_document):
    """
    rtype: BaseReader
    """
    return DatasetAdapter.load_reader(dataset_document)


_CSV_DETAILS_KEY = "local_csv"

def _csv_details_adapter(dataset_document):
    if dataset_document.csv_details is None:
        raise Exception("Malformed csv details.")
    else:
        return pd.read_csv(dataset_document.csv_details.local_path)

def _csv_metadata_adapter(dataset_document):
    if dataset_document.csv_details is None:
        raise Exception("Malformed csv details.")
    else:
        return MetadataLoader(dataset_document.csv_details.local_path.split(".")[0] + ".yaml").read_schema()

def _csv_reader_adapter(dataset_document):
    if dataset_document.csv_details is None:
        raise Exception("Malformed csv details.")
    else:
        return CSVReader(_csv_metadata_adapter(dataset_document),
                         _csv_details_adapter(dataset_document))

_DATAVERSE_DETAILS_KEY = "dataverse"

def _dataverse_details_adapter(dataset_document):
    if dataset_document.dataverse_details is None:
        raise Exception("Malformed csv details.")
    else:
        response = requests.get(dataset_document.dataverse_details.host, headers={"X-Dataverse-key":dataset_document.dataverse_details.token})
        response.raise_for_status()
        temp_dir = tempfile.gettempdir()
        path = os.path.join(temp_dir, "data.tsv")
        with open(path, "w") as stream:
            stream.write(response.text)
        return pd.read_csv(path, sep='\t')


def _dataverse_metadata_adapter(dataset_document):
    if dataset_document.dataverse_details is None:
        raise Exception("Malformed dataverse details.")
    else:
        return MetadataLoader(dataset_document.dataverse_details.local_metadata_path).read_schema()


def _dataverse_reader_adapter(dataset_document):
    if dataset_document.dataverse_details is None:
        raise Exception("Malformed dataverse details.")
    else:
        return CSVReader(_dataverse_metadata_adapter(dataset_document),
                         _dataverse_details_adapter(dataset_document))


register_adapter(_CSV_DETAILS_KEY, _csv_details_adapter, _csv_metadata_adapter, _csv_reader_adapter)
register_adapter(_DATAVERSE_DETAILS_KEY, _dataverse_details_adapter, _dataverse_metadata_adapter, _dataverse_reader_adapter)
