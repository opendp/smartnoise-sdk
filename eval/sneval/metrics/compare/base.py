from ..base import Metric
from ...dataset import Dataset

class CompareMetric(Metric):
    def __init__(self, categorical_columns=[]):
        self.categorical_columns = categorical_columns
    def param_names(self):
        return super().param_names() + ["categorical_columns"]
    def validate(self, original : Dataset, synthetic : Dataset):
        if not original.matches(synthetic):
            raise ValueError("Original and synthetic datasets must have the same schema.")
        if not set(self.categorical_columns).issubset(set(original.categorical_columns)):
            raise ValueError("Columns {} are not categorical.".format(self.categorical_columns))
    def compute(self, original : Dataset, synthetic : Dataset) -> dict:
        raise NotImplementedError