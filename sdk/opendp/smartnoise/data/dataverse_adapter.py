import os
import tempfile
import requests
import urllib.parse as parser

import pandas as pd

from opendp.smartnoise.metadata import CollectionMetadata
from opendp.smartnoise.sql.reader import PandasReader

from .dataset_adapter import DatasetAdapter


def _make_doi_host(host, doi):
    doi = doi.replace("doi:", "")
    return "{}/api/access/datafile/:persistentId/?persistentId=doi:{}".format(host, doi)


def dataverse_loader(host, doi=None, token=None):
    host = host if doi is None else _make_doi_host(host, doi)
    kwargs = {}
    if token is not None:
        kwargs["headers"] = {"X-Dataverse-key": token}
    response = requests.get(host, **kwargs)
    response.raise_for_status()

    temp_dir = tempfile.gettempdir()
    path = os.path.join(temp_dir, "data.tsv")
    with open(path, "w") as stream:
        stream.write(response.text)
    return pd.read_csv(path, sep="\t")


def dataverse_uri_loader(dv_uri):
    uri = parser.urlparse(dv_uri)
    try:
        scheme = uri.scheme
        host = uri.netloc
        path = uri.path
        kv = uri.query.split("&")
    except Exception as e:
        print(str(e))

    doi_key = "persistentId"
    token_key = "token"
    kv_arr = [pair.split("=") for pair in kv]
    kv_dict = dict(kv_arr)
    doi = kv_dict[doi_key]
    token = kv_dict[token_key]
    url = "https://" + host + path + "?" + doi_key + "=" + doi
    response = (
        requests.get(url, headers={"X-Dataverse-Key": "%s" % token}) if token else requests.get(url)
    )
    response.raise_for_status()

    temp_dir = tempfile.gettempdir()
    path = os.path.join(temp_dir, "data.tsv")
    with open(path, "w") as stream:
        stream.write(response.text)
    return pd.read_csv(path, sep="\t")


class DataverseAdapter(DatasetAdapter):
    KEY = "dataverse_details"

    @staticmethod
    def validate_document(document):
        if document.dataverse_details is None:
            raise Exception("Malformed details.")

    @staticmethod
    def _load_df(dataset_document):
        return dataverse_loader(
            dataset_document.dataverse_details.host, token=dataset_document.dataverse_details.token
        )

    @staticmethod
    def _load_metadata(dataset_document):
        return CollectionMetadata.from_file(dataset_document.dataverse_details.local_metadata_path)

    @staticmethod
    def _load_reader(dataset_document):
        return PandasReader(
            DataverseAdapter.load_df(dataset_document),
            DataverseAdapter.load_metadata(dataset_document),
        )
