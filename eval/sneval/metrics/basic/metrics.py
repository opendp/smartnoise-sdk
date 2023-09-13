
from sneval.dataset import Dataset
from .base import SingleColumnMetric, MultiColumnMetric
from ...dataset import Dataset
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
    def compute(self, data : Dataset) -> dict:
        if not data.is_aggregated:
            if self.column_name not in data.categorical_columns:
                raise ValueError("Column {} is not categorical.".format(self.column_name))
            total_count = data.source.count()
            grouped_data = data.source.groupBy(self.column_name).agg(F.count('*').alias("count")) \
                .withColumn("probability", F.col("count") / total_count)
        else:
            if data.count_column is None:
                raise ValueError("Dataset is aggregated but has no count column.")
            total_count = data.source.select(F.sum(data.count_column)).collect()[0][0]
            grouped_data = data.source.groupBy(self.column_name).agg(F.sum(data.count_column).alias("count_by_category")) \
                .withColumn("probability", F.col("count_by_category") / total_count)
        value = - grouped_data.select(F.sum(F.when(F.col("probability") != 0, F.col("probability") * F.log2(F.col("probability"))).otherwise(0)).alias("entropy")).collect()[0]["entropy"]
        response = self.to_dict()
        response["value"] = value
        return response


class Mean(SingleColumnMetric):
    # column must be numerical
    def __init__(self, column_name):
        super().__init__(column_name)
    def compute(self, data : Dataset) -> dict:
        if not data.is_aggregated:
            if self.column_name not in data.measure_columns:
                raise ValueError("Column {} is not numerical.".format(self.column_name))
            value = data.source.agg(F.sum(self.column_name).alias("sum"), F.count('*').alias("count")).select(F.col("sum") / F.col("count")).collect()[0][0]
        else:
            if data.count_column is None:
                raise ValueError("Dataset is aggregated but has no count column.")
            value = data.source.agg(F.sum(self.column_name).alias("sum"), F.sum(data.count_column).alias("count")).select(F.col("sum") / F.col("count")).collect()[0][0]
        response = self.to_dict()
        response["value"] = value
        return response


class Median(SingleColumnMetric):
    # column must be numerical
    def __init__(self, column_name):
        super().__init__(column_name)
    def compute(self, data : Dataset):
        if not data.is_aggregated:
            if self.column_name not in data.measure_columns:
                raise ValueError("Column {} is not numerical.".format(self.column_name))
            return data.source.approxQuantile(self.column_name, [0.5], 0.001)[0]
        else:
            raise ValueError("Median is not available for aggregated dataset.")


class Variance(SingleColumnMetric):
    # column must be numerical
    def __init__(self, column_name):
        super().__init__(column_name)
    def compute(self, data: Dataset):
        if not data.is_aggregated:
            if self.column_name not in data.measure_columns:
                raise ValueError("Column {} is not numerical.".format(self.column_name))
            return data.source.select(F.variance(self.column_name)).collect()[0][0]
        else:
            raise ValueError("Variance is not available for aggregated dataset.")
                  

class StandardDeviation(SingleColumnMetric):
    # column must be numerical
    def __init__(self, column_name):
        super().__init__(column_name)
    def compute(self, data: Dataset):
        if not data.is_aggregated:
            if self.column_name not in data.measure_columns:
                raise ValueError("Column {} is not numerical.".format(self.column_name))
            return data.source.select(F.stddev(self.column_name)).collect()[0][0]
        else:
            raise ValueError("Standard deviation is not available for aggregated dataset.")
        

class Skewness(SingleColumnMetric):
    # column must be numerical
    def __init__(self, column_name):
        super().__init__(column_name)
    def compute(self, data: Dataset):
        if not data.is_aggregated:
            if self.column_name not in data.measure_columns:
                raise ValueError("Column {} is not numerical.".format(self.column_name))
            return data.source.select(F.skewness(self.column_name)).collect()[0][0]
        else:
            raise ValueError("Skewness is not available for aggregated dataset.")
        

class Kurtosis(SingleColumnMetric):
    # column must be numerical
    def __init__(self, column_name):
        super().__init__(column_name)
    def compute(self, data: Dataset):
        if not data.is_aggregated:
            if self.column_name not in data.measure_columns:
                raise ValueError("Column {} is not numerical.".format(self.column_name))
            return data.source.select(F.kurtosis(self.column_name)).collect()[0][0]
        else:
            raise ValueError("Kurtosis is not available for aggregated dataset.")


class Range(SingleColumnMetric):
    # column must be numerical
    def __init__(self, column_name):
        super().__init__(column_name)
    def compute(self, data: Dataset):
        if not data.is_aggregated:
            if self.column_name not in data.measure_columns:
                raise ValueError("Column {} is not numerical.".format(self.column_name))
            return (data.source.select(F.min(self.column_name)).collect()[0][0], data.source.select(F.max(self.column_name)).collect()[0][0])
        else:
            raise ValueError("Range is not available for aggregated dataset.")


class DiscreteMutualInformation(MultiColumnMetric):
    # columns must be categorical
    def __init__(self, column_names):
        if len(column_names) != 2:
            raise ValueError("DiscreteMutualInformation requires two columns.")
        super().__init__(column_names)
    def compute(self, data):
        if not set(self.column_names).issubset(set(data.categorical_columns)):
            raise ValueError("Columns {} are not categorical.".format(self.column_names))
        
        c1, c2 = self.column_names[0], self.column_names[1]
        if data.count_column is not None:
            total_count = data.source.select(F.sum(data.count_column)).collect()[0][0]
            joint_probs = data.source.groupBy(*self.column_names).agg(F.sum(data.count_column).alias("count_joint")).withColumn("joint_prob", F.col("count_joint") / total_count)
            marginal_probs_c1 = data.source.groupBy(c1).agg(F.sum(data.count_column).alias("count_c1")).withColumn("marginal_prob_c1", F.col("count_c1") / total_count)
            marginal_probs_c2 = data.source.groupBy(c2).agg(F.sum(data.count_column).alias("count_c2")).withColumn("marginal_prob_c2", F.col("count_c2") / total_count)
        else:
            total_count = data.source.count()
            joint_probs = data.source.groupBy(*self.column_names).agg(F.count('*').alias("count_joint")).withColumn("joint_prob", F.col("count_joint") / total_count)
            marginal_probs_c1 = data.source.groupBy(c1).agg(F.count('*').alias("count_c1")).withColumn("marginal_prob_c1", F.col("count_c1") / total_count)
            marginal_probs_c2 = data.source.groupBy(c2).agg(F.count('*').alias("count_c2")).withColumn("marginal_prob_c2", F.col("count_c2") / total_count)

        mutual_information = joint_probs.join(marginal_probs_c1, c1, "inner") \
            .join(marginal_probs_c2, c2, "inner") \
            .withColumn("mutual_info", (F.col("joint_prob") * F.log2(F.col("joint_prob") / (F.col("marginal_prob_c1") * F.col("marginal_prob_c2")))))
        return mutual_information.selectExpr("sum(mutual_info)").collect()[0][0]


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
