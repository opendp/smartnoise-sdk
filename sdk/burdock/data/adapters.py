import pandas as pd


class DatasetAdapter(object):
    _adapters = {}

    @staticmethod
    def load(dataset_document):
        if dataset_document.dataset_type in DatasetAdapter._adapters:
            return DatasetAdapter._adapters[dataset_document.dataset_type](dataset_document)
        else:
            # TODO: formalize exceptions
            raise Exception("UnsupportedDatasetDocumentType {} was found"
                            ", supported types include {}.".format(
                                dataset_document.dataset_type, DatasetAdapter._adapters.keys()))


def register_adapter(dataset_type, func):
    if dataset_type not in DatasetAdapter._adapters:
        DatasetAdapter._adapters[dataset_type] = func
    else:
        raise Exception("Dataset type {} has already been "
                        "used to register an adapter.".format(dataset_type))
        # TODO: formalize exceptions


def load_dataset(dataset_document):
    """
    rtype: pandas dataframe
    """
    return DatasetAdapter.load(dataset_document)


_CSV_DETAILS_KEY = "local_csv"


def _csv_details_adapter(dataset_document):
    if dataset_document.csv_details is None:
        raise Exception("Malformed csv details.")
    else:
        return pd.read_csv(dataset_document.csv_details.local_path)


register_adapter(_CSV_DETAILS_KEY, _csv_details_adapter)
