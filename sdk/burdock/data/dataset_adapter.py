import os
import tempfile
import requests

import pandas as pd

from burdock.query.sql import MetadataLoader
from burdock.query.sql.reader import CSVReader


class DatasetAdapter(object):
    KEY = "default"

    @classmethod
    def load_df(cls, dataset_document):
        cls.validate_document(dataset_document)
        return cls._load_df(dataset_document)

    @classmethod
    def load_metadata(cls, dataset_document):
        cls.validate_document(dataset_document)
        return cls._load_metadata(dataset_document)

    @classmethod
    def load_reader(cls, dataset_document):
        cls.validate_document(dataset_document)
        return cls._load_reader(dataset_document)
