from .base import CompareMetric
from pyspark.sql import functions as F
from functools import reduce

def get_mean(data, categorical_columns, value_column):
    if data.count_column is not None:
        df = data.source.groupBy(categorical_columns).agg(F.sum(value_column).alias("total_value"), F.sum(data.count_column).alias("total_count")) \
            .withColumn("avg_value", F.col("total_value") / F.col("total_count")).drop("total_value")
    elif data.id_column is not None:
        df = data.source.groupBy(categorical_columns).agg(F.sum(value_column).alias("total_value"), F.countDistinct(data.id_column).alias("total_count")) \
            .withColumn("avg_value", F.col("total_value") / F.col("total_count")).drop("total_value")
    else:
        df = data.source.groupBy(categorical_columns).agg(F.sum(value_column).alias("total_value"), F.count('*').alias("total_count")) \
            .withColumn("avg_value", F.col("total_value") / F.col("total_count")).drop("total_value")
    return df

def get_count(data, categorical_columns):
    if data.count_column is not None:
        df = data.source.groupBy(categorical_columns).agg(F.sum(data.count_column).alias("total_count"))
    elif data.id_column is not None:
        df = data.source.groupBy(categorical_columns).agg(F.countDistinct(data.id_column).alias("total_count"))
    else:
        df = data.source.groupBy(categorical_columns).agg(F.count('*').alias("total_count"))
    return df

class MeanAbsoluteError(CompareMetric):
    def __init__(self, categorical_columns=[], measure_columns=[], sum_columns=[], edges=[1, 10, 100, 1000]):
        if len(measure_columns) + len(sum_columns) == 0 or len(measure_columns) + len(sum_columns) > 2:
            raise ValueError("MeanAbsoluteError requires exactly one measure and/or one sum column.")
        if len(categorical_columns) == 0:
            raise ValueError("MeanAbsoluteError requires at least one categorical column. Use all categorical columns if you want all aggregates measured.")
        super().__init__(categorical_columns, measure_columns, sum_columns)
        self.edges = edges
    def compute(self, original, synthetic):
        self.validate(original, synthetic)
        
        orig_value_column = self.sum_columns[0] if original.is_aggregated else self.measure_columns[0]
        original_df = get_mean(original, self.categorical_columns, orig_value_column).withColumnRenamed("avg_value", "orig_avg_value").withColumnRenamed("total_count", "orig_total_count")
        bin_expr = F.expr('CASE ' + ' '.join([f'WHEN orig_total_count BETWEEN {self.edges[i]} AND {self.edges[i+1]} THEN {i+1}' for i in range(len(self.edges)-1)]) + ' END as bin_number')
        original_df = original_df.withColumn("bin_number", bin_expr)

        synth_value_column = self.sum_columns[0] if synthetic.is_aggregated else self.measure_columns[0]
        synthetic_df = get_mean(synthetic, self.categorical_columns, synth_value_column).withColumnRenamed("avg_value", "synth_avg_value").drop("total_count")

        joined_df = original_df.join(synthetic_df, on=self.categorical_columns, how="left").fillna({"synth_avg_value": 0})
        abs_diff_df = joined_df.withColumn("abs_diff_value", F.abs(F.col("orig_avg_value") - F.col("synth_avg_value")))
        abs_diff_df = abs_diff_df.groupBy("bin_number").agg(F.avg("abs_diff_value").alias("abs_diff_value"))

        abs_diff_dict = {row["bin_number"]: row["abs_diff_value"] for row in abs_diff_df.collect()}
        value_dict = self.to_dict()
        value_dict["value"] = {f"Bin {bin+1}": abs_diff_dict.get(bin+1, 'NA') for bin in range(len(self.edges)-1)}
        return value_dict

class MeanAbsoluteErrorInCount(CompareMetric):
    def __init__(self, categorical_columns=[], edges=[1, 10, 100, 1000]):
        if len(categorical_columns) == 0:
            raise ValueError("MeanAbsoluteErrorInCount requires at least one categorical column. Use all categorical columns if you want all aggregates measured.")
        super().__init__(categorical_columns)
        self.edges = edges
    def compute(self, original, synthetic):
        self.validate(original, synthetic)
        
        original_df = get_count(original, self.categorical_columns).withColumnRenamed("total_count", "orig_total_count")
        bin_expr = F.expr('CASE ' + ' '.join([f'WHEN orig_total_count BETWEEN {self.edges[i]} AND {self.edges[i+1]} THEN {i+1}' for i in range(len(self.edges)-1)]) + ' END as bin_number')
        original_df = original_df.withColumn("bin_number", bin_expr)

        synthetic_df = get_count(synthetic, self.categorical_columns).withColumnRenamed("total_count", "synth_total_count")

        joined_df = original_df.join(synthetic_df, on=self.categorical_columns, how="left").fillna({"synth_total_count": 0})
        abs_diff_df = joined_df.withColumn("abs_diff_count", F.abs(F.col("orig_total_count") - F.col("synth_total_count")))
        abs_diff_df = abs_diff_df.groupBy("bin_number").agg(F.avg("abs_diff_count").alias("abs_diff_count"))

        abs_diff_dict = {row["bin_number"]: row["abs_diff_count"] for row in abs_diff_df.collect()}
        value_dict = self.to_dict()
        value_dict["value"] = {f"Bin {bin+1}": abs_diff_dict.get(bin+1, 'NA') for bin in range(len(self.edges)-1)}
        return value_dict       

class MeanProportionalError(CompareMetric):
    def __init__(self, categorical_columns=[], measure_columns=[], sum_columns=[], edges=[1, 10, 100, 1000]):
        if len(measure_columns) + len(sum_columns) == 0 or len(measure_columns) + len(sum_columns) > 2:
            raise ValueError("MeanProportionalError requires exactly one measure column.")
        if len(categorical_columns) == 0:
            raise ValueError("MeanProportionalError requires at least one categorical column. Use all categorical columns if you want all aggregates measured.")
        super().__init__(categorical_columns, measure_columns, sum_columns)
        self.edges = edges
    def compute(self, original, synthetic):
        self.validate(original, synthetic)

        orig_value_column = self.sum_columns[0] if original.is_aggregated else self.measure_columns[0]
        original_df = get_mean(original, self.categorical_columns, orig_value_column).withColumnRenamed("avg_value", "orig_avg_value").withColumnRenamed("total_count", "orig_total_count")
        bin_expr = F.expr('CASE ' + ' '.join([f'WHEN orig_total_count BETWEEN {self.edges[i]} AND {self.edges[i+1]} THEN {i+1}' for i in range(len(self.edges)-1)]) + ' END as bin_number')
        original_df = original_df.withColumn("bin_number", bin_expr)

        synth_value_column = self.sum_columns[0] if synthetic.is_aggregated else self.measure_columns[0]
        synthetic_df = get_mean(synthetic, self.categorical_columns, synth_value_column).withColumnRenamed("avg_value", "synth_avg_value").drop("total_count")
    
        joined_df = original_df.join(synthetic_df, on=self.categorical_columns, how="left").fillna({"synth_avg_value": 0})
        mpe_df = joined_df.withColumn("mpe_part", (F.col("orig_avg_value") - F.col("synth_avg_value")) / F.col("orig_avg_value"))
        mpe_df = mpe_df.groupBy("bin_number").agg((F.sum("mpe_part") * 100 / F.count('*')).alias("mpe_value"))

        mpe_dict = {row["bin_number"]: row["mpe_value"] for row in mpe_df.collect()}
        value_dict = self.to_dict()
        value_dict["value"] = {f"Bin {bin+1}": mpe_dict.get(bin+1, 'NA') for bin in range(len(self.edges)-1)}
        return value_dict

class MeanProportionalErrorInCount(CompareMetric):
    def __init__(self, categorical_columns=[], edges=[1, 10, 100, 1000]):
        if len(categorical_columns) == 0:
            raise ValueError("MeanProportionalErrorInCount requires at least one categorical column. Use all categorical columns if you want all aggregates measured.")
        super().__init__(categorical_columns)
        self.edges = edges
    def compute(self, original, synthetic):
        self.validate(original, synthetic)
        
        original_df = get_count(original, self.categorical_columns).withColumnRenamed("total_count", "orig_total_count")
        bin_expr = F.expr('CASE ' + ' '.join([f'WHEN orig_total_count BETWEEN {self.edges[i]} AND {self.edges[i+1]} THEN {i+1}' for i in range(len(self.edges)-1)]) + ' END as bin_number')
        original_df = original_df.withColumn("bin_number", bin_expr)

        synthetic_df = get_count(synthetic, self.categorical_columns).withColumnRenamed("total_count", "synth_total_count")

        joined_df = original_df.join(synthetic_df, on=self.categorical_columns, how="left").fillna({"synth_total_count": 0})
        mpe_df = joined_df.withColumn("mpe_part", (F.col("orig_total_count") - F.col("synth_total_count")) / F.col("orig_total_count"))
        mpe_df = mpe_df.groupBy("bin_number").agg((F.sum("mpe_part") * 100 / F.count('*')).alias("mpe_count"))

        mpe_dict = {row["bin_number"]: row["mpe_count"] for row in mpe_df.collect()}
        value_dict = self.to_dict()
        value_dict["value"] = {f"Bin {bin+1}": mpe_dict.get(bin+1, 'NA') for bin in range(len(self.edges)-1)}
        return value_dict

class SuppressedCombinationCount(CompareMetric):
    def __init__(self, categorical_columns=[]):
        if len(categorical_columns) == 0:
            raise ValueError("SuppressedCombinationCount requires at least one categorical column. Use all categorical columns if you want all aggregates measured.")
        super().__init__(categorical_columns)
    def compute(self, original, synthetic):
        self.validate(original, synthetic)

        synthetic_df = synthetic.source.select(self.categorical_columns).distinct()
        original_df = original.source.select(self.categorical_columns).distinct()
        
        value_dict = self.to_dict()
        # Subtract synthetic from original based on dimension columns only
        value_dict["value"] = original_df.subtract(synthetic_df).count()
        return value_dict
    
class FabricatedCombinationCount(CompareMetric):
    def __init__(self, categorical_columns=[], unknown_keyword="Unknown"):
        if len(categorical_columns) == 0:
            raise ValueError("FabricatedCombinationCount requires at least one categorical column. Use all categorical columns if you want all aggregates measured.")
        super().__init__(categorical_columns)
        self.unknown_keyword = unknown_keyword
    def compute(self, original, synthetic):
        self.validate(original, synthetic)

        synthetic_df = synthetic.source.select(self.categorical_columns).distinct()
        original_df = original.source.select(self.categorical_columns).distinct()

        # Count a combination with "unknown" values as fabricated?
        synthetic_unknown = synthetic_df.filter(" or ".join(["{} = '{}'".format(c, self.unknown_keyword) for c in synthetic_df.columns]))
        synthetic_no_unknown = synthetic_df.subtract(synthetic_unknown)

        # Generate custom matching conditions for rows with "unknown" values and other rows
        # 1. For rows with "unknown" values, subtract them by comparing only the columns not equal to "unknown".
        # 2. For other rows, perform a normal subtract. 
        def generate_conditions(columns, unknown_value):
            conditions = [F.when((F.col("df1." + c) == unknown_value) | (F.col("df1." + c) == F.col("df2." + c)), True).otherwise(False) for c in columns]
            return reduce(lambda x, y: x & y, conditions)
        conditions = generate_conditions(self.categorical_columns, self.unknown_keyword)
        # Subtract rows with "unknown" in synthetic_unknown based on the custom conditions
        matches_unknown = synthetic_unknown.alias("df1").crossJoin(original_df.alias("df2")).filter(conditions)
        fabricated_part1 = synthetic_unknown.join(matches_unknown.select(*["df1." + c for c in self.categorical_columns]), on=self.categorical_columns, how="left_anti")

        # Perform normal subtract for rows without "unknown"
        fabricated_part2 = synthetic_no_unknown.subtract(original_df)

        value_dict = self.to_dict()
        value_dict["value"] = fabricated_part1.unionByName(fabricated_part2).count()
        return value_dict
    