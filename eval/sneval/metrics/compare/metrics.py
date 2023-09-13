from .base import CompareMetric
from pyspark.sql import functions as F

class MeanAbsoluteError(CompareMetric):
    def __init__(self, categorical_columns=[], measure_columns=[]):
        if len(measure_columns) != 1:
            raise ValueError("MeanAbsoluteError requires exactly one measure column.")
        if len(categorical_columns) == 0:
            raise ValueError("MeanAbsoluteError requires at least one categorical column. Use all categorical columns if you want all aggregates measured.")
        super().__init__(categorical_columns, measure_columns)
    def compute(self, original, synthetic):
        self.validate(original, synthetic)
        value_dict = self.to_dict()

        target_col = self.measure_columns[0]
        original_df = original.source.select(self.categorical_columns + [target_col]) \
            .groupby(self.categorical_columns).agg(F.sum(target_col).alias(target_col))
        synthetic_df = synthetic.source.select(self.categorical_columns + [target_col]) \
            .groupby(self.categorical_columns).agg(F.sum(target_col).alias(target_col)) \
            .withColumnRenamed(target_col, target_col + "_2")
        joined = original_df.join(synthetic_df, on=self.categorical_columns, how="left")
        joined = joined.fillna({target_col + "_2": 0})
        # Compute absolute differences
        joined = joined.withColumn("abs_diff", F.abs(F.col(target_col) - F.col(target_col + "_2")))

        value_dict["value"] = joined.rdd.map(lambda row: (tuple(row[col] for col in self.categorical_columns), row["abs_diff"])).collectAsMap()
        return value_dict
    
class SuppressedCombinationCount(CompareMetric):
    def __init__(self, categorical_columns=[]):
        if len(categorical_columns) == 0:
            raise ValueError("SuppressedCombinationCount requires at least one categorical column. Use all categorical columns if you want all aggregates measured.")
        super().__init__(categorical_columns)
    def compute(self, original, synthetic):
        self.validate(original, synthetic)
        value_dict = self.to_dict()

        # Subtract synthetic from original based on dimension columns only
        suppressed_combinations_df = original.source.select(self.categorical_columns).subtract(synthetic.source.select(self.categorical_columns))
        value_dict["value"] = suppressed_combinations_df.count()
        return value_dict
    
class FabricatedCombinationCount(CompareMetric):
    def __init__(self, categorical_columns=[]):
        if len(categorical_columns) == 0:
            raise ValueError("FabricatedCombinationCount requires at least one categorical column. Use all categorical columns if you want all aggregates measured.")
        super().__init__(categorical_columns)
    def compute(self, original, synthetic):
        self.validate(original, synthetic)
        value_dict = self.to_dict()

        # Subtract original from synthetic based on dimension columns only
        # Q: Should we consider a combination with "Unknown" values as fabricated?
        fabricated_combinations_df = synthetic.source.select(self.categorical_columns).subtract(original.source.select(self.categorical_columns))
        value_dict["value"] = fabricated_combinations_df.count()
        return value_dict
    