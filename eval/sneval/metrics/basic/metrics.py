
from .base import SingleColumnMetric, MultiColumnMetric
from pyspark.sql import functions as F

class Cardinality(SingleColumnMetric):
    # column must be categorical
    def __init__(self, column_name):
        super().__init__(column_name)
    def compute(self, data):
        if self.column_name not in data.categorical_columns:
            raise ValueError("Column {} is not categorical.".format(self.column_name))
        return data.source.select(self.column_name).distinct().count()

class Entropy(SingleColumnMetric):
    # column must be categorical
    def __init__(self, column_name):
        super().__init__(column_name)

class Mean(SingleColumnMetric):
    # column must be numerical
    def __init__(self, column_name):
        super().__init__(column_name)

class Median(SingleColumnMetric):
    # column must be numerical
    def __init__(self, column_name):
        super().__init__(column_name)

class Variance(SingleColumnMetric):
    # column must be numerical
    def __init__(self, column_name):
        super().__init__(column_name)

class StandardDeviation(SingleColumnMetric):
    # column must be numerical
    def __init__(self, column_name):
        super().__init__(column_name)

class Skewness(SingleColumnMetric):
    # column must be numerical
    def __init__(self, column_name):
        super().__init__(column_name)

class Kurtosis(SingleColumnMetric):
    # column must be numerical
    def __init__(self, column_name):
        super().__init__(column_name)

class Range(SingleColumnMetric):
    # column must be numerical
    def __init__(self, column_name):
        super().__init__(column_name)

class DiscreteMutualInformation(MultiColumnMetric):
    # columns must be categorical
    def __init__(self, column_names):
        super().__init__(column_names)

class Dimensionality(MultiColumnMetric):
    # columns must be categorical
    def __init__(self, column_name):
        super().__init__(column_name)

class BelowK(MultiColumnMetric):
    # columns must be categorical
    def __init__(self, column_names, k):
        if len(column_names) == 0:
            raise ValueError("BelowK requires at least one column.")
        super().__init__(column_names)
        self.k = k
    def param_names(self):
        return super().param_names() + ["k"]
    def compute(self, data):
        if not set(self.column_names).issubset(set(data.categorical_columns)):
            raise ValueError("Columns {} are not categorical.".format(self.column_names))
        if data.count_column is not None:
            return data.source.groupBy(*self.column_names).agg(F.sum(data.count_column).alias("count_below_k")).filter(f"count_below_k < {self.k}").count()
        elif data.id_column is not None:
            return data.source.groupBy(*self.column_names).agg(F.countDistinct(data.id_column).alias("count_below_k")).filter(f"count_below_k < {self.k}").count()
        else:
            return data.source.groupBy(*self.column_names).agg(F.count('*').alias("count_below_k")).filter(f"count_below_k < {self.k}").count()

class DistinctCount(MultiColumnMetric):
    # columns must be categorical
    def __init__(self, column_names):
        super().__init__(column_names)
    def compute(self, data):
        if not set(self.column_names).issubset(set(data.categorical_columns)):
            raise ValueError("Columns {} are not categorical.".format(self.column_names))
        return data.source.select(self.column_names).distinct().count()
