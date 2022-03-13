"""Metadata module."""

from snsynth.sdv.metadata.dataset import Metadata
from snsynth.sdv.metadata.errors import MetadataError, MetadataNotFittedError
from snsynth.sdv.metadata.table import Table

__all__ = (
    'Metadata',
    'MetadataError',
    'MetadataNotFittedError',
    'Table'
)
