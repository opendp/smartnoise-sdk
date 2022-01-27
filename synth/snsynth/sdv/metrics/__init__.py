"""Metrics to evaluate quality of Synthetic Data.

This subpackage exists only to enable importing sdmetrics as part of sdv.
"""

from snsynth.sdv.metrics import factory, tabular, timeseries

__all__ = [
    'factory',
    'tabular',
    'timeseries',
]
