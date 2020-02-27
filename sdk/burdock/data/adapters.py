from .local_csv_adapter import LocalCSVAdapter
from .dataverse_adapter import DataverseAdapter


class DatasetAdapterLoader(object):
    _adapters = {}

    @staticmethod
    def load_df(dataset_document):
        if dataset_document.dataset_type in DatasetAdapterLoader._adapters:
            return DatasetAdapterLoader._adapters[dataset_document.dataset_type].load_df(dataset_document)
        else:
            # TODO: formalize exceptions
            raise Exception("UnsupportedDatasetDocumentType {} was found"
                            ", supported types include {}.".format(
                                dataset_document.dataset_type,
                                DatasetAdapterLoader._adapters.keys()))

    @staticmethod
    def load_metadata(dataset_document):
        if dataset_document.dataset_type in DatasetAdapterLoader._adapters:
            return DatasetAdapterLoader._adapters[dataset_document.dataset_type].load_metadata(dataset_document)
        else:
            # TODO: formalize exceptions
            raise Exception("UnsupportedDatasetDocumentType {} was found"
                            ", supported types include {}.".format(
                                dataset_document.dataset_type,
                                DatasetAdapterLoader._adapters.keys()))

    @staticmethod
    def load_reader(dataset_document):
        if dataset_document.dataset_type in DatasetAdapterLoader._adapters:
            return DatasetAdapterLoader._adapters[dataset_document.dataset_type].load_reader(dataset_document)
        else:
            # TODO: formalize exceptions
            raise Exception("UnsupportedDatasetDocumentType {} was found"
                            ", supported types include {}.".format(
                                dataset_document.dataset_type,
                                DatasetAdapterLoader._adapters.keys()))


def register_adapter(adapter):
    if adapter.KEY == "default":
        raise Exception("Invalid key for adapter, adapter.KEY needs to be overrided")
    elif adapter.KEY not in DatasetAdapterLoader._adapters:
        DatasetAdapterLoader._adapters[adapter.KEY] = adapter
    else:
        raise Exception("Dataset type {} has already been "
                        "used to register an adapter.".format(adapter.KEY))


def load_dataset(dataset_document):
    """
    rtype: pandas dataframe
    """
    return DatasetAdapterLoader.load_df(dataset_document)


def load_metadata(dataset_document):
    """
    rtype: MetadataLoader
    """
    return DatasetAdapterLoader.load_metadata(dataset_document)


def load_reader(dataset_document):
    """
    rtype: Reader
    """
    return DatasetAdapterLoader.load_reader(dataset_document)


register_adapter(LocalCSVAdapter)
register_adapter(DataverseAdapter)
