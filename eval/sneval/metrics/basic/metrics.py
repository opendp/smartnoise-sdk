
from sneval.dataset import Dataset
from .base import SingleColumnMetric, MultiColumnMetric, BinaryClassificationMetric
from ...dataset import Dataset
from pyspark.sql import functions as F
from pyspark.ml.evaluation import BinaryClassificationEvaluator

class Cardinality(SingleColumnMetric):
    # column must be categorical
    def __init__(self, column_name):
        super().__init__(column_name)
    def compute(self, data):
        if self.column_name not in data.categorical_columns:
            raise ValueError("Column {} is not categorical.".format(self.column_name))
        response = self.to_dict()
        response["value"] = data.source.select(self.column_name).distinct().count()
        return response

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
        if data.is_aggregated:
            raise ValueError("Median is not available for aggregated dataset.")
        if self.column_name not in data.measure_columns:
            raise ValueError("Column {} is not numerical.".format(self.column_name))
        response = self.to_dict()
        response["value"] = data.source.approxQuantile(self.column_name, [0.5], 0.001)[0]
        return response

class Variance(SingleColumnMetric):
    # column must be numerical
    def __init__(self, column_name):
        super().__init__(column_name)
    def compute(self, data: Dataset):
        if data.is_aggregated:
            raise ValueError("Variance is not available for aggregated dataset.")
        if self.column_name not in data.measure_columns:
            raise ValueError("Column {} is not numerical.".format(self.column_name))
        response = self.to_dict()
        response["value"] = data.source.select(F.variance(self.column_name)).collect()[0][0]
        return response
                  
class StandardDeviation(SingleColumnMetric):
    # column must be numerical
    def __init__(self, column_name):
        super().__init__(column_name)
    def compute(self, data: Dataset):
        if data.is_aggregated:
            raise ValueError("Standard deviation is not available for aggregated dataset.")
        if self.column_name not in data.measure_columns:
            raise ValueError("Column {} is not numerical.".format(self.column_name))
        response = self.to_dict()
        response["value"] = data.source.select(F.stddev(self.column_name)).collect()[0][0]
        return response

class Skewness(SingleColumnMetric):
    # column must be numerical
    def __init__(self, column_name):
        super().__init__(column_name)
    def compute(self, data: Dataset):
        if data.is_aggregated:
            raise ValueError("Skewness is not available for aggregated dataset.")
        if self.column_name not in data.measure_columns:
            raise ValueError("Column {} is not numerical.".format(self.column_name))
        response = self.to_dict()
        response["value"] = data.source.select(F.skewness(self.column_name)).collect()[0][0]
        return response

class Kurtosis(SingleColumnMetric):
    # column must be numerical
    def __init__(self, column_name):
        super().__init__(column_name)
    def compute(self, data: Dataset):
        if data.is_aggregated:
            raise ValueError("Kurtosis is not available for aggregated dataset.")
        if self.column_name not in data.measure_columns:
            raise ValueError("Column {} is not numerical.".format(self.column_name))
        response = self.to_dict()
        response["value"] = data.source.select(F.kurtosis(self.column_name)).collect()[0][0]
        return response

class Range(SingleColumnMetric):
    # column must be numerical
    def __init__(self, column_name):
        super().__init__(column_name)
    def compute(self, data: Dataset):
        if self.column_name not in data.measure_columns:
            raise ValueError("Column {} is not numerical.".format(self.column_name))
        response = self.to_dict()
        response["value"] = (data.source.select(F.min(self.column_name)).collect()[0][0], data.source.select(F.max(self.column_name)).collect()[0][0])
        return response

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
        response = self.to_dict()
        response["value"] = mutual_information.selectExpr("sum(mutual_info)").collect()[0][0]
        return response

class Dimensionality(MultiColumnMetric):
    # columns must be categorical
    def __init__(self, column_names):
        if len(column_names) == 0:
            raise ValueError("Dimensionality requires at least one column.")
        super().__init__(column_names)
    def compute(self, data):
        value = 1
        for col in self.column_names:
            unique_count = data.source.select(col).distinct().count()
            value *= unique_count
        response = self.to_dict()
        response["value"] = value
        return response

class Sparsity(MultiColumnMetric):
    # columns must be categorical
    def __init__(self, column_names):
        if len(column_names) == 0:
            raise ValueError("Sparsity requires at least one column.")
        super().__init__(column_names)
        self.dimensionality = Dimensionality(column_names)
        self.distinct_count = DistinctCount(column_names)
    def compute(self, data):
        response = self.to_dict()
        response["value"] = self.distinct_count.compute(data)["value"] / self.dimensionality.compute(data)["value"]
        return response

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
        value = 0
        if data.count_column is not None:
            value = data.source.groupBy(*self.column_names).agg(F.sum(data.count_column).alias("count_below_k")).filter(f"count_below_k < {self.k}").count()
        elif data.id_column is not None:
            value = data.source.groupBy(*self.column_names).agg(F.countDistinct(data.id_column).alias("count_below_k")).filter(f"count_below_k < {self.k}").count()
        else:
            value = data.source.groupBy(*self.column_names).agg(F.count('*').alias("count_below_k")).filter(f"count_below_k < {self.k}").count()
        response = self.to_dict()
        response["value"] = value
        return response

class RowCount(MultiColumnMetric):
    # columns must be categorical
    def __init__(self, column_names):
        if len(column_names) == 0:
            raise ValueError("RowCount requires at least one column.")
        super().__init__(column_names)
    def compute(self, data):
        if not set(self.column_names).issubset(set(data.categorical_columns)):
            raise ValueError("Columns {} are not categorical.".format(self.column_names))
        response = self.to_dict()
        response["value"] = data.source.select(self.column_names).count()
        return response

class DistinctCount(MultiColumnMetric):
    # columns must be categorical
    def __init__(self, column_names):
        if len(column_names) == 0:
            raise ValueError("DistinctCount requires at least one column.")
        super().__init__(column_names)
    def compute(self, data):
        if not set(self.column_names).issubset(set(data.categorical_columns)):
            raise ValueError("Columns {} are not categorical.".format(self.column_names))
        response = self.to_dict()
        response["value"] = data.source.select(self.column_names).distinct().count()
        return response

class BelowKPercentage(MultiColumnMetric):
    # columns must be categorical
    def __init__(self, column_names, k):
        if len(column_names) == 0:
            raise ValueError("BelowKPercentage requires at least one column.")
        super().__init__(column_names)
        self.blow_k = BelowK(column_names, k)
        self.distinct_count = DistinctCount(column_names)
    def param_names(self):
        return super().param_names() + ["k"]
    def compute(self, data):
        if not set(self.column_names).issubset(set(data.categorical_columns)):
            raise ValueError("Columns {} are not categorical.".format(self.column_names))
        response = self.to_dict()
        response["value"] = self.blow_k.compute(data)["value"] / self.distinct_count.compute(data)["value"] * 100
        return response

class MostLinkable(MultiColumnMetric):
    # columns must be categorical
    def __init__(self, column_names, linkable_k=10, top_n=5):
        if len(column_names) == 0:
            raise ValueError("MostLinkable requires at least one column.")
        super().__init__(column_names)
        self.linkable_k = linkable_k
        self.top_n = top_n
    def param_names(self):
        return super().param_names() + ["linkable_k", "top_n"]
    def compute(self, data):
        if not set(self.column_names).issubset(set(data.categorical_columns)):
            raise ValueError("Columns {} are not categorical.".format(self.column_names))
        linkable_counts_dict = {}
        for col in self.column_names:
            if data.count_column is not None:
                linkable_df = data.source.groupBy(col).agg(F.sum(data.count_column).alias("count_below_k")).filter(f"count_below_k < {self.linkable_k}")
            elif data.id_column is not None:
                linkable_df = data.source.groupBy(col).agg(F.countDistinct(data.id_column).alias("count_below_k")).filter(f"count_below_k < {self.linkable_k}")
            else:
                linkable_df = data.source.groupBy(col).agg(F.count('*').alias("count_below_k")).filter(f"count_below_k < {self.linkable_k}")         
            total_linkable_count = linkable_df.agg(F.sum("count_below_k").alias("total_count_below_k")).collect()[0]["total_count_below_k"]
            if total_linkable_count:
                linkable_counts_dict[col] = total_linkable_count
        most_linkable_columns = sorted(linkable_counts_dict.items(), key=lambda x: x[1], reverse=True)[:self.top_n]
        response = self.to_dict()
        response["value"] = most_linkable_columns
        return response

class RedactedRowCount(MultiColumnMetric):
    # columns must be categorical
    def __init__(self, column_names, redacted_keyword="Unknown"):
        if len(column_names) == 0:
            raise ValueError("RedactedRowCount requires at least one column.")
        super().__init__(column_names)
        self.redacted_keyword = redacted_keyword
    def param_names(self):
        return super().param_names() + ["redacted_keyword"]
    def compute(self, data):
        if not set(self.column_names).issubset(set(data.categorical_columns)):
            raise ValueError("Columns {} are not categorical.".format(self.column_names))
        
        # Create an additional column that counts the number of "unknown" values per row
        df_with_unknown_count = data.source.withColumn("unknown_count", sum(F.when(F.col(c) == self.redacted_keyword, 1).otherwise(0) for c in self.column_names))
        
        # Count the number of rows with partly unknown values (some, but not all columns are "unknown")
        partly_redacted = df_with_unknown_count.filter((F.col("unknown_count") > 0) & (F.col("unknown_count") < len(self.column_names))).count()
        # Count the number of rows with fully unknown values (all columns are "unknown")
        fully_redacted = df_with_unknown_count.filter(F.col("unknown_count") == len(self.column_names)).count()

        response = self.to_dict()
        response["value"] = {"partly redacted row count": partly_redacted, "fully redacted row count": fully_redacted}
        return response
    
class AUCMetric(BinaryClassificationMetric):
    def __init__(self, label_column, prediction_column):
        super().__init__(label_column, prediction_column)
    def compute(self, data):
        response = self.to_dict()
        evaluator = BinaryClassificationEvaluator(rawPredictionCol=self.prediction_column,
                                                  labelCol=self.label_column,
                                                  metricName="areaUnderROC")
        response["value"] = evaluator.evaluate(data.source)
        return response