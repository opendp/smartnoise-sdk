from .base import CompareMetric

class MeanAbsoluteError(CompareMetric):
    def __init__(self, categorical_columns=[], measure_columns=[]):
        if len(measure_columns) != 1:
            raise ValueError("MeanAbsoluteError requires exactly one measure column.")
        if len(categorical_columns) == 0:
            raise ValueError("MeanAbsoluteError requires at least one categorical column. Use all categorical columns if you want all aggregates measured.")
        super().__init__(categorical_columns, measure_columns)
    def compute(self, original, synthetic):
        value_dict = self.to_dict()
        
