from ..base import Metric

class SingleColumnMetric(Metric):
    def __init__(self, column_name):
        self.column_name = column_name
    @property
    def width(self):
        return 1
    def compute(self, data):
        raise NotImplementedError

class MultiColumnMetric(Metric):
    def __init__(self, column_names):
        self.column_names = column_names
    @property
    def width(self):
        return len(self.column_names)
    def compute(self, data):
        raise NotImplementedError