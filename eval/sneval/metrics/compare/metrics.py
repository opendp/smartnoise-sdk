from .base import CompareMetric
from pyspark.sql import functions as F

class MeanAbsoluteError(CompareMetric):
    def __init__(self, categorical_columns=[], measure_columns=[]):
        if len(measure_columns) != 1:
            raise ValueError("MeanAbsoluteError requires exactly one measure column.")
        if len(categorical_columns) == 0:
            raise ValueError("MeanAbsoluteError requires at least one categorical column. Use all categorical columns if you want all aggregates measured.")
        super().__init__(categorical_columns, measure_columns)
    def compute(self, original, synthetic, bin_size_lower=1, bin_size_upper=float('inf')):
        self.validate(original, synthetic)

        # excludes the combinations whose counts fall outside the specified range of bin size.
        bin_size_lower = max(1, bin_size_lower) 
        if bin_size_upper < bin_size_lower:
            raise ValueError("The upper bound of the target bin size range must be larger than or equal to the lower bound.")
        if original.count_column is not None:
            bin_filtered = original.source.groupBy(*self.categorical_columns).agg(F.sum(original.count_column).alias("total_count")) \
                .filter((F.col("total_count") >= bin_size_lower) & (F.col("total_count") <= bin_size_upper))
        elif original.id_column is not None:
            bin_filtered = original.source.groupBy(*self.categorical_columns).agg(F.countDistinct(original.id_column).alias("total_count")) \
                .filter((F.col("total_count") >= bin_size_lower) & (F.col("total_count") <= bin_size_upper))
        else:
            bin_filtered = original.source.groupBy(*self.categorical_columns).agg(F.count('*').alias("total_count")) \
                .filter((F.col("total_count") >= bin_size_lower) & (F.col("total_count") <= bin_size_upper))
        original_df = original.source.join(bin_filtered, on=self.categorical_columns, how="left").drop("total_count")

        measure_col = self.measure_columns[0]
        original_df = original_df.groupby(self.categorical_columns).agg(F.sum(measure_col).alias(measure_col))
        synthetic_df = synthetic.source.groupby(self.categorical_columns).agg(F.sum(measure_col).alias(measure_col + "_synth"))
        joined_df = original_df.join(synthetic_df, on=self.categorical_columns, how="left").fillna({measure_col + "_synth": 0})
        abs_diff_df = joined_df.withColumn("abs_diff", F.abs(F.col(measure_col) - F.col(measure_col + "_synth")))

        value_dict = self.to_dict()
        value_dict["value"] = abs_diff_df.agg(F.avg("abs_diff")).collect()[0][0]
        # value_dict["value"] = abs_diff_df.rdd.map(lambda row: (tuple(row[col] for col in self.categorical_columns), row["abs_diff"])).collectAsMap()
        return value_dict

class MeanProportionalError(CompareMetric):
    def __init__(self, categorical_columns=[], measure_columns=[]):
        if len(measure_columns) != 1:
            raise ValueError("MeanProportionalError requires exactly one measure column.")
        if len(categorical_columns) == 0:
            raise ValueError("MeanProportionalError requires at least one categorical column. Use all categorical columns if you want all aggregates measured.")
        super().__init__(categorical_columns, measure_columns)
    def compute(self, original, synthetic, bin_size_lower=1, bin_size_upper=float('inf')):
        self.validate(original, synthetic)

        # excludes the combinations whose counts fall outside the specified range of bin size.
        bin_size_lower = max(1, bin_size_lower) 
        if bin_size_upper < bin_size_lower:
            raise ValueError("The upper bound of the target bin size range must be larger than or equal to the lower bound.")
        if original.count_column is not None:
            bin_filtered = original.source.groupBy(*self.categorical_columns).agg(F.sum(original.count_column).alias("total_count")) \
                .filter((F.col("total_count") >= bin_size_lower) & (F.col("total_count") <= bin_size_upper))
        elif original.id_column is not None:
            bin_filtered = original.source.groupBy(*self.categorical_columns).agg(F.countDistinct(original.id_column).alias("total_count")) \
                .filter((F.col("total_count") >= bin_size_lower) & (F.col("total_count") <= bin_size_upper))
        else:
            bin_filtered = original.source.groupBy(*self.categorical_columns).agg(F.count('*').alias("total_count")) \
                .filter((F.col("total_count") >= bin_size_lower) & (F.col("total_count") <= bin_size_upper))
        original_df = original.source.join(bin_filtered, on=self.categorical_columns, how="left").drop("total_count")

        measure_col = self.measure_columns[0]
        original_df = original_df.groupby(self.categorical_columns).agg(F.sum(measure_col).alias(measure_col))
        synthetic_df = synthetic.source.groupby(self.categorical_columns).agg(F.sum(measure_col).alias(measure_col + "_synth"))
        joined_df = original_df.join(synthetic_df, on=self.categorical_columns, how="left").fillna({measure_col + "_synth": 0})
        mpe_df = joined_df.withColumn("mpe_part", (F.col(measure_col) - F.col(measure_col + "_synth")) / F.col(measure_col))

        value_dict = self.to_dict()
        value_dict["value"] = mpe_df.agg(F.sum("mpe_part") * 100 / F.count('*')).collect()[0][0]
        return value_dict
        
class SuppressedCombinationCount(CompareMetric):
    def __init__(self, categorical_columns=[]):
        if len(categorical_columns) == 0:
            raise ValueError("SuppressedCombinationCount requires at least one categorical column. Use all categorical columns if you want all aggregates measured.")
        super().__init__(categorical_columns)
    def compute(self, original, synthetic):
        self.validate(original, synthetic)

        # Subtract synthetic from original based on dimension columns only
        suppressed_combinations_df = original.source.select(self.categorical_columns).subtract(synthetic.source.select(self.categorical_columns))
        value_dict = self.to_dict()
        value_dict["value"] = suppressed_combinations_df.count()
        return value_dict
    
class FabricatedCombinationCount(CompareMetric):
    def __init__(self, categorical_columns=[]):
        if len(categorical_columns) == 0:
            raise ValueError("FabricatedCombinationCount requires at least one categorical column. Use all categorical columns if you want all aggregates measured.")
        super().__init__(categorical_columns)
    def compute(self, original, synthetic):
        self.validate(original, synthetic)

        # Subtract original from synthetic based on dimension columns only
        # Q: Should we consider a combination with "Unknown" values as fabricated?
        fabricated_combinations_df = synthetic.source.select(self.categorical_columns).subtract(original.source.select(self.categorical_columns))
        value_dict = self.to_dict()
        value_dict["value"] = fabricated_combinations_df.count()
        return value_dict
    