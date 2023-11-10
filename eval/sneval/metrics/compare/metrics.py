"""
CompareMetrics Module
=====================

This module contains a collection of classes for calculating various comparison metrics
between two datasets. These metrics help assess the quality and differences between 
the datasets in terms of specified categorical and numerical attributes.

Classes:
--------
- `MeanAbsoluteError`: Calculate the Mean Absolute Error (MAE) between two numerical columns.
- `MeanAbsoluteErrorInCount`: Calculate the MAE based on the count of data points.
- `MeanProportionalError`: Calculate the Mean Proportional Error (MPE) between two numerical columns.
- `MeanProportionalErrorInCount`: Calculate the MPE based on the count of data points.
- `SuppressedCombinationCount`: Calculate the count of suppressed combinations in specified categorical columns.
- `FabricatedCombinationCount`: Calculate the count of fabricated combinations in specified categorical columns.
"""

from .base import CompareMetric
from pyspark.sql import functions as F
from functools import reduce

def get_mean(data, categorical_columns, value_column):
    if data.count_column is not None:
        total_count_agg = F.sum(data.count_column).alias("total_count")
    elif data.id_column is not None:
        total_count_agg = F.countDistinct(data.id_column).alias("total_count")
    else:
        total_count_agg = F.count('*').alias("total_count")

    df = (data.source.groupBy(categorical_columns)
          .agg(F.sum(value_column).alias("total_value"), total_count_agg)
          .withColumn("avg_value", F.col("total_value") / F.col("total_count"))
          .drop("total_value"))
    return df

def get_count(data, categorical_columns):
    if data.count_column is not None:
        total_count_agg = F.sum(data.count_column).alias("total_count")
    elif data.id_column is not None:
        total_count_agg = F.countDistinct(data.id_column).alias("total_count")
    else:
        total_count_agg = F.count('*').alias("total_count")

    df = data.source.groupBy(categorical_columns).agg(total_count_agg)
    return df

class MeanAbsoluteError(CompareMetric):
    """
    Compute the Mean Absolute Error (MAE) metric based on the comparison of
    measure or sum columns within specified categorical columns.

    This metric calculates the Mean Absolute Error between the values of a specified
    measure/sum column in two datasets (e.g., original and synthetic). It groups data 
    points into bins based on the total count of specified categorical columns and 
    computes the mean value of the measure/sum column's values within each bin. It then 
    computes the average absolute difference between the mean values of the measure/sum 
    column in the two datasets for each bin size.

    :param categorical_columns: List of categorical columns to group data by.
    :type categorical_columns: list, optional
    :param measure_sum_columns: List of measure or sum columns to compare.
                                Only one column should be provided.
    :type measure_sum_columns: list, optional
    :param edges: List of bin edges for splitting data points based on total count.
                  Default bin edges are [1, 10, 100, 1000, 10000, 100000].
    :type edges: list, optional

    :raises ValueError: If there are issues with the provided columns or input data.

    Example usage:

    .. code-block:: python

        mae_metric = Metric.create("MeanAbsoluteError", categorical_columns=["category"], measure_sum_columns=['value'])
        result = mae_metric.compute(original_data, synthetic_data)
    """
    def __init__(self, categorical_columns=[], measure_sum_columns=[], edges=[1, 10, 100, 1000, 10000, 100000]):
        if len(measure_sum_columns) != 1:
            raise ValueError("MeanAbsoluteError requires exactly one measure or one sum column.")
        if len(categorical_columns) == 0:
            raise ValueError("MeanAbsoluteError requires at least one categorical column. Use all categorical columns if you want all aggregates measured.")
        super().__init__(categorical_columns)
        self.edges = edges
        self.measure_sum_columns = measure_sum_columns
    def param_names(self):
        return super().param_names() + ["measure_sum_columns", "edges"]
    def compute(self, original, synthetic):
        """
        Compute the MAE between the original and synthetic datasets.

        :param original: The original dataset.
        :type original: Dataset
        :param synthetic: The synthetic dataset.
        :type synthetic: Dataset
        :return: Dictionary containing the computed MAE values for each size of bin.
        :rtype: dict
        """
        self.validate(original, synthetic)
        
        if original.is_aggregated and not set(self.measure_sum_columns).issubset(set(original.sum_columns)):
            raise ValueError("Make sure column {} is summed up for aggregated dataset.".format(self.measure_sum_columns))
        if not original.is_aggregated and not set(self.measure_sum_columns).issubset(set(original.measure_columns)):
            raise ValueError("Column {} is not numerical.".format(self.measure_sum_columns))
        value_column = self.measure_sum_columns[0]

        original_df = get_mean(original, self.categorical_columns, value_column).withColumnRenamed("avg_value", "orig_avg_value").withColumnRenamed("total_count", "orig_total_count")
        # Create the first condition for values less than the smallest edge
        bin_expr = 'CASE WHEN orig_total_count < {} THEN 0 '.format(self.edges[0])
        for i in range(len(self.edges)-1):
            bin_expr += 'WHEN orig_total_count >= {} AND orig_total_count < {} THEN {} '.format(self.edges[i], self.edges[i+1], i+1)
        # Add the last condition for values greater than or equal to the largest edge
        bin_expr += 'WHEN orig_total_count >= {} THEN {} '.format(self.edges[-1], len(self.edges))
        bin_expr += 'END as bin_number'
        original_df = original_df.withColumn("bin_number", F.expr(bin_expr))

        synthetic_df = get_mean(synthetic, self.categorical_columns, value_column).withColumnRenamed("avg_value", "synth_avg_value").drop("total_count")

        joined_df = original_df.join(synthetic_df, on=self.categorical_columns, how="left").fillna({"synth_avg_value": 0})
        abs_diff_df = joined_df.withColumn("abs_diff_value", F.abs(F.col("orig_avg_value") - F.col("synth_avg_value")))
        abs_diff_df = abs_diff_df.groupBy("bin_number").agg(F.avg("abs_diff_value").alias("abs_diff_value"))

        abs_diff_dict = {row["bin_number"]: row["abs_diff_value"] for row in abs_diff_df.collect()}
        value_dict = self.to_dict()
        value_dict["value"] = {f"Bin {bin}": abs_diff_dict.get(bin, 'NA') for bin in range(len(self.edges)+1)}
        return value_dict

class MeanAbsoluteErrorInCount(CompareMetric):
    """
    Compute the Mean Absolute Error (MAE) metric based on the count of data points
    within specified categorical columns.

    This metric calculates the Mean Absolute Error between the counts of data points
    within the specified categorical columns of two datasets. It groups the data points
    into bins based on the total count of the categorical columns and computes the 
    absolute difference in counts within each bin. It then computes the average 
    absolute difference for each bin size.

    :param categorical_columns: List of categorical columns to group data by.
    :type categorical_columns: list, optional
    :param edges: List of bin edges for splitting data points based on total count.
                  Default bin edges are [1, 10, 100, 1000, 10000, 100000].
    :type edges: list, optional

    :raises ValueError: If there are issues with the provided columns or input data.

    Example usage:

    .. code-block:: python

        mae_count_metric = Metric.create("MeanAbsoluteErrorInCount", categorical_columns=["category"])
        result = mae_count_metric.compute(original_data, synthetic_data)
    """
    def __init__(self, categorical_columns=[], edges=[1, 10, 100, 1000, 10000, 100000]):
        if len(categorical_columns) == 0:
            raise ValueError("MeanAbsoluteErrorInCount requires at least one categorical column. Use all categorical columns if you want all aggregates measured.")
        super().__init__(categorical_columns)
        self.edges = edges
    def param_names(self):
        return super().param_names() + ["edges"]
    def compute(self, original, synthetic):
        """
        Compute the MAE of count values within specified categorical columns between the 
        original and synthetic datasets.

        :param original: The original dataset.
        :type original: Dataset
        :param synthetic: The synthetic dataset.
        :type synthetic: Dataset
        :return: Dictionary containing the computed MAE values for each size of bin.
        :rtype: dict
        """        
        self.validate(original, synthetic)
        
        original_df = get_count(original, self.categorical_columns).withColumnRenamed("total_count", "orig_total_count")

        conditions = [f'WHEN orig_total_count >= {self.edges[i]} AND orig_total_count < {self.edges[i+1]} THEN {i+1}' for i in range(len(self.edges)-1)]
        conditions.insert(0, f'CASE WHEN orig_total_count < {self.edges[0]} THEN 0')
        conditions.append(f'WHEN orig_total_count >= {self.edges[-1]} THEN {len(self.edges)}')
        bin_expr = ' '.join(conditions) + ' END as bin_number'

        original_df = original_df.withColumn("bin_number", F.expr(bin_expr))
   
        synthetic_df = get_count(synthetic, self.categorical_columns).withColumnRenamed("total_count", "synth_total_count")

        original_df = original_df.repartition(*self.categorical_columns)
        synthetic_df = synthetic_df.repartition(*self.categorical_columns)
        joined_df = original_df.join(synthetic_df, on=self.categorical_columns, how="left").fillna({"synth_total_count": 0})
    
        abs_diff_df = (joined_df.withColumn("abs_diff_count", F.abs(F.col("orig_total_count") - F.col("synth_total_count")))
                       .groupBy("bin_number")
                       .agg(F.avg("abs_diff_count").alias("abs_diff_count")))

        abs_diff_dict = {row["bin_number"]: row["abs_diff_count"] for row in abs_diff_df.collect()} 
        value_dict = self.to_dict()
        value_dict["value"] = {f"Bin {bin}": abs_diff_dict.get(bin, 'NA') for bin in range(len(self.edges)+1)}
        return value_dict

class MeanError(CompareMetric):
    """
    The `MeanError` class is designed to calculate the average error between
    the original and synthetic datasets for specified categorical columns.

    The `compute` method performs the calculation in three steps:
    1. Count the occurrences of each category in the specified columns for both datasets.
    2. Compute the difference in counts for each category between the two datasets.
    3. Average these differences to provide the mean error across all categories.

    :param categorical_columns: List of categorical columns to group data by.
    :type categorical_columns: list, optional

    :raises ValueError: If there are issues with the provided columns or input data.

    Example usage:

    .. code-block:: python

        mae_count_metric = Metric.create("MeanError", categorical_columns=["category"])
        result = mae_count_metric.compute(original_data, synthetic_data)
    """
    def __init__(self, categorical_columns=[]):
        if len(categorical_columns) == 0:
            raise ValueError("MeanError requires at least one categorical column.")
        super().__init__(categorical_columns)
    def compute(self, original, synthetic):
        """
        Computes the mean error between original and synthetic datasets for the specified 
        categorical columns.

        :param original: The original dataset.
        :type original: Dataset
        :param synthetic: The synthetic dataset.
        :type synthetic: Dataset
        :return: .
        :rtype: dict
        """        
        self.validate(original, synthetic)
        
        original_df = get_count(original, self.categorical_columns).withColumnRenamed("total_count", "orig_total_count")
        synthetic_df = get_count(synthetic, self.categorical_columns).withColumnRenamed("total_count", "synth_total_count")

        original_df = original_df.repartition(*self.categorical_columns)
        synthetic_df = synthetic_df.repartition(*self.categorical_columns)
        joined_df = original_df.join(synthetic_df, on=self.categorical_columns, how="left").fillna({"synth_total_count": 0})
    
        value_dict = self.to_dict()
        value_dict["value"] = joined_df.withColumn("diff_count", F.col("synth_total_count") - F.col("orig_total_count")).agg(F.avg("diff_count")).collect()[0][0]
        return value_dict

class MeanProportionalError(CompareMetric):
    """
    Compute the Mean Proportional Error (MPE) metric based on the comparison of
    measure or sum columns within specified categorical columns.

    This metric calculates the Mean Proportional Error by comparing the values of a 
    specified measure/sum column between two datasets within specified categorical 
    columns. It categorizes data points into bins based on the total count of the 
    categorical columns and computes the mean value of the measure/sum column's 
    values within each bin. It then computes the average proportional error in 
    percentage between the mean measure/sum values for each bin size.

    :param categorical_columns: List of categorical columns to group data by.
    :type categorical_columns: list, optional
    :param measure_sum_columns: List of measure or sum columns to compare.
                                Only one column should be provided.
    :type measure_sum_columns: list, optional
    :param edges: List of bin edges for splitting data points based on total count.
                  Default bin edges are [1, 10, 100, 1000, 10000, 100000].
    :type edges: list, optional

    :raises ValueError: If there are issues with the provided columns or input data.

    Example usage:

    .. code-block:: python

        mpe_metric = Metric.create("MeanProportionalError", categorical_columns=["category"], measure_sum_columns=['value'])
        result = mpe_metric.compute(original_data, synthetic_data)
    """
    def __init__(self, categorical_columns=[], measure_sum_columns=[], edges=[1, 10, 100, 1000, 10000, 100000]):
        if len(measure_sum_columns) != 1:
            raise ValueError("MeanProportionalError requires exactly one measure or one sum column.")
        if len(categorical_columns) == 0:
            raise ValueError("MeanProportionalError requires at least one categorical column. Use all categorical columns if you want all aggregates measured.")
        super().__init__(categorical_columns)
        self.edges = edges
        self.measure_sum_columns = measure_sum_columns
    def param_names(self):
        return super().param_names() + ["measure_sum_columns", "edges"]
    def compute(self, original, synthetic):
        """
        Compute the MPE between the original and synthetic datasets.

        :param original: The original dataset.
        :type original: Dataset
        :param synthetic: The synthetic dataset.
        :type synthetic: Dataset
        :return: Dictionary containing the computed MPE values for each size of bin.
        :rtype: dict
        """
        self.validate(original, synthetic)

        if original.is_aggregated and not set(self.measure_sum_columns).issubset(set(original.sum_columns)):
            raise ValueError("Make sure column {} is summed up for aggregated dataset.".format(self.measure_sum_columns))
        if not original.is_aggregated and not set(self.measure_sum_columns).issubset(set(original.measure_columns)):
            raise ValueError("Column {} is not numerical.".format(self.measure_sum_columns))
        value_column = self.measure_sum_columns[0]

        original_df = get_mean(original, self.categorical_columns, value_column).withColumnRenamed("avg_value", "orig_avg_value").withColumnRenamed("total_count", "orig_total_count")
        # Create the first condition for values less than the smallest edge
        bin_expr = 'CASE WHEN orig_total_count < {} THEN 0 '.format(self.edges[0])
        for i in range(len(self.edges)-1):
            bin_expr += 'WHEN orig_total_count >= {} AND orig_total_count < {} THEN {} '.format(self.edges[i], self.edges[i+1], i+1)
        # Add the last condition for values greater than or equal to the largest edge
        bin_expr += 'WHEN orig_total_count >= {} THEN {} '.format(self.edges[-1], len(self.edges))
        bin_expr += 'END as bin_number'
        original_df = original_df.withColumn("bin_number", F.expr(bin_expr))

        synthetic_df = get_mean(synthetic, self.categorical_columns, value_column).withColumnRenamed("avg_value", "synth_avg_value").drop("total_count")
    
        joined_df = original_df.join(synthetic_df, on=self.categorical_columns, how="left").fillna({"synth_avg_value": 0})
        mpe_df = (joined_df.withColumn("mpe_part", (F.col("synth_avg_value") - F.col("orig_avg_value")) / F.col("orig_avg_value"))
                  .groupBy("bin_number")
                  .agg((F.sum("mpe_part") * 100 / F.count('*')).alias("mpe_value")))

        mpe_dict = {row["bin_number"]: row["mpe_value"] for row in mpe_df.collect()}
        value_dict = self.to_dict()
        value_dict["value"] = {f"Bin {bin}": mpe_dict.get(bin, 'NA') for bin in range(len(self.edges)+1)}
        return value_dict

class MeanProportionalErrorInCount(CompareMetric):
    """
    Compute the Mean Proportional Error (MPE) metric based on the count of data points
    within specified categorical columns.

    This metric calculates the Mean Proportional Error between the counts of data points
    within the specified categorical columns of two datasets. It groups the data points
    into bins based on the total count of the categorical columns and computes the average 
    proportional error in percentage between the counts for each size of bin.

    :param categorical_columns: List of categorical columns to group data by.
    :type categorical_columns: list, optional
    :param edges: List of bin edges for splitting data points based on total count.
                  Default bin edges are [1, 10, 100, 1000, 10000, 100000].
    :type edges: list, optional

    :raises ValueError: If there are issues with the provided columns or input data.

    Example usage:

    .. code-block:: python

        mpe_count_metric = Metric.create("MeanProportionalErrorInCount", categorical_columns=["category"])
        result = mpe_count_metric.compute(original_data, synthetic_data)
    """
    def __init__(self, categorical_columns=[], edges=[1, 10, 100, 1000, 10000, 100000]):
        if len(categorical_columns) == 0:
            raise ValueError("MeanProportionalErrorInCount requires at least one categorical column. Use all categorical columns if you want all aggregates measured.")
        super().__init__(categorical_columns)
        self.edges = edges
    def param_names(self):
        return super().param_names() + ["edges"]
    def compute(self, original, synthetic):
        """
        Compute the MPE of count values within specified categorical columns between the 
        original and synthetic datasets.

        :param original: The original dataset.
        :type original: Dataset
        :param synthetic: The synthetic dataset.
        :type synthetic: Dataset
        :return: Dictionary containing the computed MPE values for each size of bin.
        :rtype: dict
        """     
        self.validate(original, synthetic)
        
        original_df = get_count(original, self.categorical_columns).withColumnRenamed("total_count", "orig_total_count")
        conditions = [f'WHEN orig_total_count >= {self.edges[i]} AND orig_total_count < {self.edges[i+1]} THEN {i+1}' for i in range(len(self.edges)-1)]
        conditions.insert(0, f'CASE WHEN orig_total_count < {self.edges[0]} THEN 0')
        conditions.append(f'WHEN orig_total_count >= {self.edges[-1]} THEN {len(self.edges)}')
        bin_expr = ' '.join(conditions) + ' END as bin_number'

        original_df = original_df.withColumn("bin_number", F.expr(bin_expr))
   
        synthetic_df = get_count(synthetic, self.categorical_columns).withColumnRenamed("total_count", "synth_total_count")

        original_df = original_df.repartition(*self.categorical_columns)
        synthetic_df = synthetic_df.repartition(*self.categorical_columns)
        joined_df = original_df.join(synthetic_df, on=self.categorical_columns, how="left").fillna({"synth_total_count": 0})
    
        mpe_df = (joined_df.withColumn("mpe_part", (F.col("synth_total_count") - F.col("orig_total_count")) / F.col("orig_total_count"))
                  .groupBy("bin_number")
                  .agg((F.sum("mpe_part") * 100 / F.count('*')).alias("mpe_count")))

        mpe_dict = {row["bin_number"]: row["mpe_count"] for row in mpe_df.collect()}
        value_dict = self.to_dict()
        value_dict["value"] = {f"Bin {bin}": mpe_dict.get(bin, 'NA') for bin in range(len(self.edges)+1)}
        return value_dict

class SuppressedCombinationCount(CompareMetric):
    """
    Calculate the count of suppressed combinations within specified categorical columns.

    This metric calculates the count of combinations within the specified categorical columns
    that are present in the original dataset but are suppressed or missing in the synthetic dataset.
    A suppressed combination refers to a unique set of values across the categorical columns that
    exists in the original dataset but not in the synthetic dataset.

    :param categorical_columns: List of categorical columns to consider.
    :type categorical_columns: list, optional

    :raises ValueError: If there are issues with the provided columns or input data.

    Example usage:

    .. code-block:: python

        suppressed_count_metric = Metric.create("SuppressedCombinationCount", categorical_columns=["category"])
        result = suppressed_count_metric.compute(original_data, synthetic_data)
    """
    def __init__(self, categorical_columns=[]):
        if len(categorical_columns) == 0:
            raise ValueError("SuppressedCombinationCount requires at least one categorical column. Use all categorical columns if you want all aggregates measured.")
        super().__init__(categorical_columns)
    def compute(self, original, synthetic):
        """
        Computes the count of suppressed combinations.

        :param original: The original dataset.
        :type original: Dataset
        :param synthetic: The synthetic dataset.
        :type synthetic: Dataset
        :return: Dictionary containing the count of suppressed combinations within the specified categorical columns.
        :rtype: dict
        """    
        self.validate(original, synthetic)

        synthetic_df = synthetic.source.select(self.categorical_columns).distinct()
        original_df = original.source.select(self.categorical_columns).distinct()
        
        value_dict = self.to_dict()
        value_dict["value"] = original_df.subtract(synthetic_df).count()
        return value_dict
    
class FabricatedCombinationCount(CompareMetric):
    """
    Calculate the count of fabricated combinations within specified categorical columns.

    This metric calculates the count of combinations within the specified categorical columns
    that are present in the synthetic dataset but are not present in the original dataset.
    A fabricated combination refers to a unique set of values across the categorical columns that
    exists in the synthetic dataset but not in the original dataset.

    :param categorical_columns: List of categorical columns to consider.
    :type categorical_columns: list, optional
    :param unknown_keyword: Keyword to represent "unknown" values in the data.
                            Default is "Unknown."
    :type unknown_keyword: str, optional

    :raises ValueError: If there are issues with the provided columns or input data.

    Example usage:

    .. code-block:: python

        fabricated_count_metric = Metric.create("FabricatedCombinationCount", categorical_columns=["category"])
        result = fabricated_count_metric.compute(original_data, synthetic_data)

    Note: For rows with "unknown" values, this metric ignores the "unknown" columns 
    and only compares the columns not equal to "unknown."
    """
    def __init__(self, categorical_columns=[], unknown_keyword="Unknown"):
        if len(categorical_columns) == 0:
            raise ValueError("FabricatedCombinationCount requires at least one categorical column. Use all categorical columns if you want all aggregates measured.")
        super().__init__(categorical_columns)
        self.unknown_keyword = unknown_keyword
    def param_names(self):
        return super().param_names() + ["unknown_keyword"]
    def compute(self, original, synthetic):
        """
        Computes the count of fabricated combinations.

        :param original: The original dataset.
        :type original: Dataset
        :param synthetic: The synthetic dataset.
        :type synthetic: Dataset
        :return: Dictionary containing the count of fabricated combinations within the specified categorical columns.
        :rtype: dict
        """  
        self.validate(original, synthetic)

        ''' ## This is a more complex version for a nuanced comparision. Will require significant memory
        original_df = original.source.select(self.categorical_columns).distinct()
        synthetic_df = synthetic.source.select(self.categorical_columns).distinct()

        # Separate synthetic rows with 'unknown'
        synthetic_unknown = synthetic_df.filter(" or ".join(["{} = '{}'".format(c, self.unknown_keyword) for c in synthetic_df.columns]))
        synthetic_no_unknown = synthetic_df.subtract(synthetic_unknown)
        synthetic_unknown.show(50)

        # Generate custom matching conditions for rows with "unknown" values and other rows
        # 1. For rows with "unknown" values, subtract them by comparing only the columns not equal to "unknown".
        # 2. For other rows, perform a normal subtract. 
        conditions = reduce(lambda x, y: x & y, [(F.col("df1." + c) == self.unknown_keyword) | (F.col("df1." + c) == F.col("df2." + c)) for c in self.categorical_columns])
        fabricated_part1 = synthetic_unknown.alias("df1").join(original_df.alias("df2"), on=conditions, how="left_anti")

        # Normal subtract for rows without "unknown"
        fabricated_part2 = synthetic_no_unknown.subtract(original_df)

        value_dict = self.to_dict()
        value_dict["value"] = fabricated_part1.unionByName(fabricated_part2).count()
        return value_dict
        '''

        ## This is a simplified version for limited memory
        original_df = original.source.select(self.categorical_columns).distinct()
        synthetic_df = synthetic.source.select(self.categorical_columns).distinct()
        
        value_dict = self.to_dict()
        value_dict["value"] = synthetic_df.subtract(original_df).count()
        return value_dict