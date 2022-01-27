"""Models for tabular data."""

from snsynth.sdv.tabular.copulas import GaussianCopula
from snsynth.sdv.tabular.ctgan import CTGAN, TVAE

__all__ = (
    'GaussianCopula',
    'CTGAN',
    'TVAE',
)
