
class Metric:
    pass

class SingleColumnMetric(Metric):
    def __init__(self, column_name):
        self.column_name = column_name

class MultiColumnMetric(Metric):
    def __init__(self, column_names):
        self.column_names = column_names

class Cardinality(SingleColumnMetric):
    # column must be categorical
    def __init__(self, column_name):
        super().__init__(column_name)

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