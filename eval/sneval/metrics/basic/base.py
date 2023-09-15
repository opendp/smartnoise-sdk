from ..base import Metric
from ...dataset import Dataset

class SingleColumnMetric(Metric):
    def __init__(self, column_name):
        self.column_name = column_name
    def param_names(self):
        return super().param_names() + ["column_name"]
    @property
    def width(self):
        return 1
    def compute(self, data : Dataset) -> dict:
        raise NotImplementedError

class MultiColumnMetric(Metric):
    def __init__(self, column_names):
        self.column_names = column_names
    def param_names(self):
        return super().param_names() + ["column_names"]
    @property
    def width(self):
        return len(self.column_names)
    def compute(self, data : Dataset) -> dict:
        raise NotImplementedError
    
class BinaryClassificationMetric(Metric):
    def __init__(self, label_column, prediction_column):
        self.label_column = label_column
        self.prediction_column = prediction_column
    def param_names(self):
        return super().param_names() + ["label_column", "prediction_column"]
    def compute(self, data : Dataset) -> dict:
        raise NotImplementedError