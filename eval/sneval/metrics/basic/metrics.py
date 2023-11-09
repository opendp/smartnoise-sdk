
from sneval.dataset import Dataset
from .base import SingleColumnMetric, MultiColumnMetric, BinaryClassificationMetric
from ...dataset import Dataset
from pyspark.sql import functions as F
from pyspark.ml.evaluation import BinaryClassificationEvaluator

class Cardinality(SingleColumnMetric):
    """
    Calculate the cardinality (number of unique values) within a categorical column.

    This metric calculates the number of unique values within a specified categorical column.
    It is particularly useful for understanding the diversity or uniqueness of values in a
    categorical attribute.

    :param column_name: The name of the categorical column to calculate cardinality for.
    :type column_name: str

    :raises ValueError: If the provided column is not categorical.

    Example usage:

    .. code-block:: python

        # Create an instance of the Cardinality metric for the 'category' column
        cardinality_metric = Metric.create("Cardinality", column_name="Category")

        # Compute the cardinality value for the 'category' column in the dataset
        result = cardinality_metric.compute(dataset)

        print(result)  # {'name': 'Cardinality', 'value': 5}
    """
    # column must be categorical
    def __init__(self, column_name):
        super().__init__(column_name)
    def compute(self, data):
        """
        Compute the cardinality of a categorical column in the dataset.

        :param data: The dataset containing the categorical column.
        :type data: Dataset
        :returns: A dictionary containing the computed cardinality value.
        :rtype: dict
        """
        if self.column_name not in data.categorical_columns:
            raise ValueError("Column {} is not categorical.".format(self.column_name))
        response = self.to_dict()
        response["value"] = data.source.select(self.column_name).distinct().count()
        return response

class Entropy(SingleColumnMetric):
    """
    Calculate the entropy of a categorical column.

    This metric calculates the entropy of a specified categorical column, which measures the degree of randomness
    or uncertainty in the distribution of values within the column. Entropy provides insights into how uniformly
    data points are distributed among different categories within the column.

    :param column_name: The name of the categorical column to calculate entropy for.
    :type column_name: str

    :raises ValueError: If the provided column is not categorical or if the dataset is not properly aggregated.

    Example usage:

    .. code-block:: python

        entropy_metric = Metric.create("Entropy", column_name="Category")
        result = entropy_metric.compute(dataset)
    """
    # column must be categorical
    def __init__(self, column_name):
        super().__init__(column_name)
    def compute(self, data : Dataset) -> dict:
        """
        Compute the entropy of a categorical column in the dataset.

        :param data: The dataset containing the categorical column.
        :type data: Dataset
        :returns: A dictionary containing the computed entropy value.
        :rtype: dict
        """
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
    """
    Calculate the mean (average) value of a numerical column.

    This metric calculates the mean (average) value of a specified numerical column.
    The mean provides insights into the central tendency or typical value within the column.

    :param column_name: The name of the numerical column to calculate the mean for.
    :type column_name: str

    :raises ValueError: If the provided column is not numerical or if the dataset is not properly aggregated.

    Example usage:

    .. code-block:: python

        mean_metric = Metric.create("Mean", column_name="value")
        result = mean_metric.compute(dataset)
    """
    # column must be numerical
    def __init__(self, column_name):
        super().__init__(column_name)
    def compute(self, data : Dataset) -> dict:
        """
        Compute the mean value of a numerical column in the dataset.

        :param data: The dataset containing the numerical column.
        :type data: Dataset
        :returns: A dictionary containing the computed mean value.
        :rtype: dict
        """
        if not data.is_aggregated:
            if self.column_name not in data.measure_columns and self.column_name not in data.count_column:
                raise ValueError("Column {} is not numerical.".format(self.column_name))
            value = data.source.agg(F.sum(self.column_name).alias("sum"), F.count('*').alias("count")).select(F.col("sum") / F.col("count")).collect()[0][0]
        else:
            if data.count_column is None:
                raise ValueError("Dataset is aggregated but has no count column.")
            if self.column_name not in data.sum_columns:
                raise ValueError("Column {} is not numerical.".format(self.column_name))
            value = data.source.agg(F.sum(self.column_name).alias("sum"), F.sum(data.count_column).alias("count")).select(F.col("sum") / F.col("count")).collect()[0][0]
        response = self.to_dict()
        response["value"] = value
        return response

class Median(SingleColumnMetric):
    """
    Calculate the median value of a numerical column.

    This metric calculates the median value of a specified numerical column. The median is the
    middle value in a dataset, separating the higher half from the lower half. It provides
    insights into the central tendency of the data, especially in the presence of outliers.

    :param column_name: The name of the numerical column to calculate the median for.
    :type column_name: str

    :raises ValueError: If the provided column is not numerical.

    Example usage:

    .. code-block:: python

        # Create an instance of the Median metric for the 'income' column
        median_metric = Metric.create("Median", column_name="income")

        # Compute the median value for the 'income' column in the dataset
        result = median_metric.compute(dataset)

        print(result)  # {'name': 'Median', 'value': 45000.0}
    """
    def __init__(self, column_name):
        super().__init__(column_name)
    def compute(self, data : Dataset):
        """
        Compute the median value of a numerical column in the dataset.

        :param data: The dataset to compute the median for.
        :type data: Dataset
        :return: A dictionary containing the computed median value.
        :rtype: dict
        """
        if self.column_name not in data.measure_columns and self.column_name not in data.count_column:
            raise ValueError("Column {} is not numerical.".format(self.column_name))
        response = self.to_dict()
        response["value"] = data.source.approxQuantile(self.column_name, [0.5], 0.001)[0]
        return response

class Variance(SingleColumnMetric):
    """
    Calculate the variance of a numerical column.

    This metric calculates the variance of a specified numerical column. Variance measures
    how much the values in the column differ from the mean. It provides insights into the
    dispersion or spread of the data.

    :param column_name: The name of the numerical column to calculate the variance for.
    :type column_name: str

    :raises ValueError: If the provided column is not numerical or if the dataset is missing required columns.

    Example usage:

    .. code-block:: python

        # Create an instance of the Variance metric for the 'income' column
        variance_metric = Metric.create("Variance", column_name="income")

        # Compute the variance for the 'income' column in the dataset
        result = variance_metric.compute(dataset)

        print(result)  # {'name': 'Variance', 'value': 2500000.0}

    """
    def __init__(self, column_name):
        super().__init__(column_name)
    def compute(self, data: Dataset):
        """
        Compute the variance of a numerical column in the dataset.

        :param data: The dataset to compute the variance for.
        :type data: Dataset
        :return: A dictionary containing the computed variance value.
        :rtype: dict
        """
        if self.column_name not in data.measure_columns and self.column_name not in data.count_column:
            raise ValueError("Column {} is not numerical.".format(self.column_name))
        response = self.to_dict()
        response["value"] = data.source.select(F.variance(self.column_name)).collect()[0][0]
        return response
                  
class StandardDeviation(SingleColumnMetric):
    """
    Calculate the standard deviation of a numerical column.

    This metric calculates the standard deviation of a specified numerical column.
    Standard deviation measures the amount of variation or dispersion in the data.
    It indicates how spread out the values are around the mean.

    :param column_name: The name of the numerical column to calculate the standard deviation for.
    :type column_name: str

    :raises ValueError: If the provided column is not numerical or if the dataset is missing required columns.

    Example usage:

    .. code-block:: python

        # Create an instance of the StandardDeviation metric for the 'income' column
        stddev_metric = Metric.create("StandardDeviation", column_name="income")

        # Compute the standard deviation for the 'income' column in the dataset
        result = stddev_metric.compute(dataset)

        print(result)  # {'name': 'StandardDeviation', 'value': 1581.1388300841898}
    """
    def __init__(self, column_name):
        super().__init__(column_name)
    def compute(self, data: Dataset):
        """
        Compute the standard deviation of a numerical column in the dataset.

        :param data: The dataset to compute the standard deviation for.
        :type data: Dataset
        :return: A dictionary containing the computed standard deviation value.
        :rtype: dict
        """
        if self.column_name not in data.measure_columns and self.column_name not in data.count_column:
            raise ValueError("Column {} is not numerical.".format(self.column_name))
        response = self.to_dict()
        response["value"] = data.source.select(F.stddev(self.column_name)).collect()[0][0]
        return response

class Percentiles(SingleColumnMetric):
    """
    Calculate the percentiles of a numerical column.

    This metric calculates specific percentiles of a specified numerical column.
    Percentiles are measures that indicate the value below which a given percentage 
    of observations in a group of observations fall. They are useful for understanding
    the distribution and dispersion of the data.

    :param column_name: The name of the numerical column to calculate the percentiles for.
    :type column_name: str
    :param percentiles: List of desired percentile values as decimals between 0 and 1.
                        Default percentiles are [0.25, 0.50, 0.75], representing the 
                        25th, 50th, and 75th percentiles, respectively.
                        Default percentiles are [0.25, 0.50, 0.75].
    :type percentiles: list, optional

    :raises ValueError: If the provided column is not numerical or if the dataset is missing required columns.

    Example usage:

    .. code-block:: python

        # Create an instance of the Percentiles metric for the 'income' column
        stddev_metric = Metric.create("Percentiles", column_name="income")

        # Compute the percentiles for the 'income' column in the dataset
        result = stddev_metric.compute(dataset)

        print(result)  # {'name': 'Percentiles', 'value': {P-0.25: 25000, P-0.50: 52000, P-0.75: 86500}}
    """
    def __init__(self, column_name, percentiles=[0.25, 0.50, 0.75]):
        super().__init__(column_name)
        self.percentiles = percentiles
    def param_names(self):
        return super().param_names() + ["percentiles"]
    def compute(self, data: Dataset):
        """
        Compute the percentiles of a numerical column in the dataset.

        This function calculates the specified percentiles for the numerical column
        defined in the class instance, providing a view of the distribution of the column's data.
        
        :param data: The dataset to compute the percentiles for.
        :type data: Dataset
        :return: A dictionary containing the computed percentile value for the specified column.
        :rtype: dict
        """
        if self.column_name not in data.measure_columns and self.column_name not in data.count_column:
            raise ValueError("Column {} is not numerical.".format(self.column_name))
        response = self.to_dict()
        result = data.source.agg(F.percentile_approx(F.col(self.column_name), self.percentiles)).collect()[0][0]
        response["value"] = {f"P-{p}": result[i] for i, p in enumerate(self.percentiles)}
        return response

class Skewness(SingleColumnMetric):
    """
    Calculate the skewness of a numerical column.

    This metric calculates the skewness of a specified numerical column.
    Skewness is a measure of the asymmetry of the probability distribution of a real-valued random variable.
    A negative skewness indicates a left-skewed distribution, while a positive skewness indicates a right-skewed distribution.

    :param column_name: The name of the numerical column to calculate skewness for.
    :type column_name: str

    :raises ValueError: If the provided column is not numerical or if the dataset is missing required columns.

    Example usage:

    .. code-block:: python

        # Create an instance of the Skewness metric for the 'income' column
        skewness_metric = Metric.create("Skewness", column_name="income")

        # Compute the skewness for the 'income' column in the dataset
        result = skewness_metric.compute(dataset)

        print(result)  # {'name': 'Skewness', 'value': 1.3217279838342987}
    """
    def __init__(self, column_name):
        super().__init__(column_name)
    def compute(self, data: Dataset):
        """
        Compute the skewness of a numerical column in the dataset.

        :param data: The dataset to compute the skewness for.
        :type data: Dataset
        :return: A dictionary containing the computed skewness value.
        :rtype: dict
        """
        if self.column_name not in data.measure_columns and self.column_name not in data.count_column:
            raise ValueError("Column {} is not numerical.".format(self.column_name))
        response = self.to_dict()
        response["value"] = data.source.select(F.skewness(self.column_name)).collect()[0][0]
        return response

class Kurtosis(SingleColumnMetric):
    """
    Calculate the kurtosis of a numerical column.

    This metric calculates the kurtosis of a specified numerical column.
    Kurtosis is a measure of the "tailedness" of the probability distribution of a real-valued random variable.
    A positive kurtosis indicates heavy tails (more extreme values), while a negative kurtosis indicates light tails (fewer extreme values).

    :param column_name: The name of the numerical column to calculate kurtosis for.
    :type column_name: str

    :raises ValueError: If the provided column is not numerical or if the dataset is missing required columns.

    Example usage:

    .. code-block:: python

        # Create an instance of the Kurtosis metric for the 'income' column
        kurtosis_metric = Metric.create("Kurtosis", column_name="income")

        # Compute the kurtosis for the 'income' column in the dataset
        result = kurtosis_metric.compute(dataset)

        print(result)  # {'name': 'Kurtosis', 'value': 2.8739356846481846}
    """
    def __init__(self, column_name):
        super().__init__(column_name)
    def compute(self, data: Dataset):
        """
        Compute the kurtosis of a numerical column in the dataset.

        :param data: The dataset to compute the kurtosis for.
        :type data: Dataset
        :return: A dictionary containing the computed kurtosis value.
        :rtype: dict
        """
        if self.column_name not in data.measure_columns and self.column_name not in data.count_column:
            raise ValueError("Column {} is not numerical.".format(self.column_name))
        response = self.to_dict()
        response["value"] = data.source.select(F.kurtosis(self.column_name)).collect()[0][0]
        return response

class Range(SingleColumnMetric):
    """
    Calculate the range of a numerical column.

    This metric calculates the range of a specified numerical column, which is the difference
    between the maximum and minimum values in the column.

    :param column_name: The name of the numerical column to calculate the range for.
    :type column_name: str

    :raises ValueError: If the provided column is not numerical or if the dataset is missing required columns.

    Example usage:

    .. code-block:: python

        # Create an instance of the Range metric for the 'age' column
        range_metric = Metric.create("Range", column_name="age")

        # Compute the range for the 'age' column in the dataset
        result = range_metric.compute(dataset)

        print(result)  # {'name': 'Range', 'value': (25, 75)}
    """
    def __init__(self, column_name):
        super().__init__(column_name)
    def compute(self, data: Dataset):
        """
        Compute the range of a numerical column in the dataset.

        :param data: The dataset to compute the range for.
        :type data: Dataset
        :return: A dictionary containing a tuple of the minimum and maximum values in the column.
        :rtype: dict
        """
        if self.column_name not in data.measure_columns and self.column_name not in data.count_column:
            raise ValueError("Column {} is not numerical.".format(self.column_name))
        response = self.to_dict()
        response["value"] = (data.source.select(F.min(self.column_name)).collect()[0][0], data.source.select(F.max(self.column_name)).collect()[0][0])
        return response

class DiscreteMutualInformation(MultiColumnMetric):
    """
    Calculate the discrete mutual information between two categorical columns.

    This metric calculates the discrete mutual information between two specified categorical columns.
    Mutual information measures the degree of statistical dependence between two variables and is often
    used to assess the information shared between categorical attributes.

    :param column_names: A list containing two column names for which mutual information will be computed.
    :type column_names: list

    :raises ValueError: If the wrong number of columns are provided or if the columns are not categorical.

    Example usage:

    .. code-block:: python

        # Create an instance of the DiscreteMutualInformation metric for the 'A' and 'B' columns
        mi_metric = Metric.create("DiscreteMutualInformation", column_names=["A", "B"])

        # Compute the mutual information between columns 'A' and 'B' in the dataset
        result = mi_metric.compute(dataset)

        print(result)  # {'name': 'DiscreteMutualInformation', 'value': 0.1234}
    """
    def __init__(self, column_names):
        if len(column_names) != 2:
            raise ValueError("DiscreteMutualInformation requires two columns.")
        super().__init__(column_names)
    def compute(self, data):
        """
        Compute the discrete mutual information between two categorical columns in the dataset.

        :param data: The dataset to compute mutual information for.
        :type data: Dataset
        :return: A dictionary containing the computed mutual information value between the columns.
        :rtype: dict
        """
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
    """
    Calculate the dimensionality (number of possible unique combinations) of categorical columns.

    This metric calculates the dimensionality, which represents the number of unique combinations
    of values across the specified categorical columns. It provides insights into the diversity and
    complexity of the categorical attributes.

    :param column_names: A list containing the column names for which dimensionality will be computed.
    :type column_names: list

    :raises ValueError: If no columns are provided.

    Example usage:

    .. code-block:: python

        # Create an instance of the Dimensionality metric for columns 'A' and 'B'
        dimensionality_metric = Metric.create("Dimensionality", column_names=["A", "B"])

        # Compute the dimensionality of the columns in the dataset
        result = dimensionality_metric.compute(dataset)

        print(result)  # {'name': 'Dimensionality', 'value': 15}
    """
    def __init__(self, column_names):
        if len(column_names) == 0:
            raise ValueError("Dimensionality requires at least one column.")
        super().__init__(column_names)
    def compute(self, data):
        """
        Compute the dimensionality based on unique combinations of values in the specified columns.

        :param data: The dataset to compute dimensionality for.
        :type data: Dataset
        :return: A dictionary containing the computed dimensionality value.
        :rtype: dict
        """
        value = 1
        for col in self.column_names:
            unique_count = data.source.select(col).distinct().count()
            value *= unique_count
        response = self.to_dict()
        response["value"] = value
        return response

class Sparsity(MultiColumnMetric):
    """
    Calculate the sparsity (proportion of distinct combinations presented in the data) of categorical columns.

    This metric calculates the sparsity, which represents the proportion of distinct combinations presented 
    over the total possible combinations (dimensionality) within the specified categorical columns. It provides 
    insights into how densely populated the categorical attributes are with unique combinations.

    :param column_names: A list containing the column names for which sparsity will be computed.
    :type column_names: list

    :raises ValueError: If no columns are provided.

    Example usage:

    .. code-block:: python

        # Create an instance of the Sparsity metric for columns 'A' and 'B'
        sparsity_metric = Metric.create("Sparsity", column_names=["A", "B"])

        # Compute the sparsity of the columns in the dataset
        result = sparsity_metric.compute(dataset)

        print(result)  # {'name': 'Sparsity', 'value': 0.6}
    """
    def __init__(self, column_names):
        if len(column_names) == 0:
            raise ValueError("Sparsity requires at least one column.")
        super().__init__(column_names)
        self.dimensionality = Dimensionality(column_names)
        self.distinct_count = DistinctCount(column_names)
    def compute(self, data):
        """
        Compute the sparsity based on the proportion of distinct combinations in the specified columns.

        :param data: The dataset to compute sparsity for.
        :type data: Dataset
        :return: A dictionary containing the computed sparsity value, 
                which is the ratio of distinct values to the dimensionality of the specified columns.
        :rtype: dict
        """
        response = self.to_dict()
        response["value"] = self.distinct_count.compute(data)["value"] / self.dimensionality.compute(data)["value"]
        return response

class BelowKCombs(MultiColumnMetric):
    """
    Calculate the count of categorical combinations that occur fewer than or equal to 'k' times.

    This metric calculates the count and percentage of distinct combinations within the specified 
    categorical columns that occur below a specified threshold (k). It is useful for identifying 
    the number of unique combinations that are relatively less frequent in the data.

    :param column_names: A list containing the column names for which the count will be calculated.
    :type column_names: list
    :param k: The threshold value. Combinations occurring less than 'k' times are counted.
    :type k: int, optional (default=10)

    :raises ValueError: If no columns are provided.

    Example usage:

    .. code-block:: python

        # Create an instance of the BelowKCombs metric for columns 'A' and 'B' with a threshold of 5
        below_k_metric = Metric.create("BelowKCombs", column_names=["A", "B"], k=5)

        # Compute the count of combinations occurring below the threshold in the dataset
        result = below_k_metric.compute(dataset)

        print(result)  # {'name': 'BelowKCombs', 'value': {'Count': 7, 'Percentage': 10.2}}
    """
    def __init__(self, column_names, k=10):
        if len(column_names) == 0:
            raise ValueError("BelowKCombs requires at least one column.")
        super().__init__(column_names)
        self.k = k
    def param_names(self):
        return super().param_names() + ["k"]
    def compute(self, data):
        """
        Compute the count and percentage of categorical combinations occurring below the specified threshold 'k'.

        :param data: The dataset to compute the count for.
        :type data: Dataset
        :return: A dictionary containing the computed count of combinations below the threshold 'k'.
        :rtype: dict
        """
        if not set(self.column_names).issubset(set(data.categorical_columns)):
            raise ValueError("Columns {} are not categorical.".format(self.column_names))
        
        if data.count_column is not None:
            below_k_comb_count = data.source.groupBy(*self.column_names).agg(F.sum(data.count_column).alias("agg_count")).filter(f"agg_count < {self.k}").count()
        elif data.id_column is not None:
            below_k_comb_count = data.source.groupBy(*self.column_names).agg(F.countDistinct(data.id_column).alias("agg_count")).filter(f"agg_count < {self.k}").count()
        else:
            below_k_comb_count = data.source.groupBy(*self.column_names).agg(F.count('*').alias("agg_count")).filter(f"agg_count < {self.k}").count()
        below_k_comb_count = below_k_comb_count or 0

        distinct_count = data.source.select(self.column_names).distinct().count()
        response = self.to_dict()
        response["value"] = {"Count": below_k_comb_count, "Percentage": below_k_comb_count / distinct_count * 100}
        return response

class BelowKCount(MultiColumnMetric):
    """
    Aggregate the count for a specificed combinations that occur fewer than or equal to 'k' times.

    This metric computes the sum of counts and its percentage for combinations from the specified categorical 
    columns that appear less frequently than or equal to a given threshold (k). It provides insights into the 
    cumulative distribution of rarer combinations in the dataset.

    :param column_names: A list containing the column names for which the count will be calculated.
    :type column_names: list
    :param k: The threshold value. Combinations occurring less than 'k' times are counted.
    :type k: int, optional (default=10)

    :raises ValueError: If no columns are provided.

    Example usage:

    .. code-block:: python

        # Create an instance of the BelowKCount metric for columns 'A' and 'B' with a threshold of 5
        below_k_metric = Metric.create("BelowKCount", column_names=["A", "B"], k=5)
        result = below_k_metric.compute(dataset)
        print(result)  # {'name': 'BelowKCount', 'value': {'Count': 1052, 'Percentage': 8.1}}
    """
    def __init__(self, column_names, k=10):
        if len(column_names) == 0:
            raise ValueError("BelowKCount requires at least one column.")
        super().__init__(column_names)
        self.k = k
    def param_names(self):
        return super().param_names() + ["k"]
    def compute(self, data):
        """
        Aggregate the count for a specificed combinations that occur fewer than or equal to 'k' times 
        and compute its percentage.

        :param data: The dataset to compute the count for.
        :type data: Dataset
        :return: A dictionary containing the computed count of combinations below the threshold 'k'.
        :rtype: dict
        """
        if not set(self.column_names).issubset(set(data.categorical_columns)):
            raise ValueError("Columns {} are not categorical.".format(self.column_names))

        if data.count_column is not None:
            below_k_agg = data.source.groupBy(*self.column_names).agg(F.sum(data.count_column).alias("agg_count")) \
                        .filter(f"agg_count < {self.k}") \
                        .select(F.sum("agg_count")).collect()
            total_count = data.source.select(F.sum(data.count_column)).collect()[0][0]
        else:
            below_k_agg = data.source.groupBy(*self.column_names).agg(F.count('*').alias("agg_count")) \
                        .filter(f"agg_count < {self.k}") \
                        .select(F.sum("agg_count")).collect()
            total_count = data.source.count()
        below_k_agg_count = below_k_agg[0][0] if below_k_agg and below_k_agg[0][0] is not None else 0
        response = self.to_dict()
        response["value"] = {"Count": below_k_agg_count, "Percentage": below_k_agg_count / total_count * 100}
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
    """
    Calculate the count of distinct combinations within specified categorical columns.

    This metric calculates the count of distinct combinations of values within the specified
    categorical columns. It provides insights into the variety and diversity of unique combinations
    present in the data.

    :param column_names: A list containing the column names for which the distinct count will be calculated.
    :type column_names: list

    :raises ValueError: If no columns are provided.

    Example usage:

    .. code-block:: python

        # Create an instance of the DistinctCount metric for columns 'A' and 'B'
        distinct_count_metric = Metric.create("DistinctCount", column_names=["A", "B"])

        # Compute the count of distinct combinations within the specified columns in the dataset
        result = distinct_count_metric.compute(dataset)

        print(result)  # {'name': 'DistinctCount', 'value': 42}
    """
    def __init__(self, column_names):
        if len(column_names) == 0:
            raise ValueError("DistinctCount requires at least one column.")
        super().__init__(column_names)
    def compute(self, data):
        """
        Compute the count of distinct combinations within the specified categorical columns.

        :param data: The dataset to compute the count for.
        :type data: Dataset
        :return: A dictionary containing the computed count of distinct combinations.
        :rtype: dict
        """
        if not set(self.column_names).issubset(set(data.categorical_columns)):
            raise ValueError("Columns {} are not categorical.".format(self.column_names))
        response = self.to_dict()
        response["value"] = data.source.select(self.column_names).distinct().count()
        return response

class SingledOutCount(MultiColumnMetric):
    """
    Calculate the count of combinations that are singled out or unique within the specified categorical columns.

    This metric determines the number of unique combinations of specified categorical columns. It's useful
    for identifying the combinations that have a unique occurrence in the dataset.

    :param column_names: A list containing the column names for which the unique combination count will be calculated.
    :type column_names: list

    :raises ValueError: If no columns are provided.

    Example usage:

    .. code-block:: python

        # Create an instance of the SingledOutCount metric for columns 'A' and 'B'
        unique_count_metric = Metric.create("SingledOutCount", column_names=["A", "B"])

        # Compute the count of unique combinations within the specified columns in the dataset
        result = unique_count_metric.compute(dataset)

        print(result)  # {'name': 'SingledOutCount', 'value': {'Count': 7, 'Percentage': 14.0}}
    """
    def __init__(self, column_names):
        if len(column_names) == 0:
            raise ValueError("SingledCount requires at least one column.")
        super().__init__(column_names)
    def compute(self, data):
        """
        Compute the count of combinations that are singled out or unique within the specified categorical columns.

        :param data: The dataset to compute the count for.
        :type data: Dataset
        :return: A dictionary containing the computed count of unique combinations.
        :rtype: dict
        """
        if not set(self.column_names).issubset(set(data.categorical_columns)):
            raise ValueError("Columns {} are not categorical.".format(self.column_names))

        if data.count_column is not None:
            unique_agg_count = data.source.groupBy(*self.column_names).agg(F.sum(data.count_column).alias("agg_count")).filter(f"agg_count == 1").count()
            total_count = data.source.select(F.sum(data.count_column)).collect()[0][0]
        else:
            unique_agg_count = data.source.groupBy(*self.column_names).agg(F.count('*').alias("agg_count")).filter(f"agg_count == 1").count()
            total_count = data.source.count()
        unique_agg_count = unique_agg_count or 0
        response = self.to_dict()
        response["value"] = {"Count": unique_agg_count, "Percentage": unique_agg_count / total_count * 100}
        return response

class MostLinkable(MultiColumnMetric):
    """
    Calculate the most linkable categorical columns.

    This metric calculates the most linkable categorical column by identifying columns where
    a high proportion of distinct values appear fewer than 'linkable_k' times. Linkable data 
    might be used to make inferences about individuals, even without direct identification. 
    So this metric provides insights into columns that potentially pose privacy concerns.

    :param column_names: A list containing the column names to evaluate for linkability.
    :type column_names: list
    :param linkable_k: The threshold value for counting linkable values. Values appearing fewer than 'linkable_k' times are considered.
    :type linkable_k: int, optional

    :raises ValueError: If no columns are provided.

    Example usage:

    .. code-block:: python

        # Create an instance of the MostLinkable metric for columns 'A' and 'B' with a linkable threshold of 10
        most_linkable_metric = Metric.create("MostLinkable", column_names=["A", "B"], linkable_k=10)

        # Compute the most linkable columns in the dataset
        result = most_linkable_metric.compute(dataset)

        print(result)  # {'name': 'MostLinkable', 'value': {'A': 25}}

    The `compute` method returns a dictionary containing the computed most linkable categorical column
    and its respective counts of distinct values that appear fewer than 'linkable_k' times.
    """
    def __init__(self, column_names, linkable_k=10):
        if len(column_names) == 0:
            raise ValueError("MostLinkable requires at least one column.")
        super().__init__(column_names)
        self.linkable_k = linkable_k
        # self.top_n = top_n
    def param_names(self):
        return super().param_names() + ["linkable_k"]
    def compute(self, data):
        """
        Compute the most linkable categorical column.

        :param data: The dataset to compute the most linkable column for.
        :type data: Dataset

        :return: A dictionary containing the computed most linkable categorical column and its counts.
        :rtype: dict

        :raises ValueError: If no columns are provided or if the columns are not categorical.
        """
        if not set(self.column_names).issubset(set(data.categorical_columns)):
            raise ValueError("Columns {} are not categorical.".format(self.column_names))
        linkable_counts_dict = {}
        for col in self.column_names:
            if data.count_column is not None:
                linkable_df = data.source.groupBy(col).agg(F.sum(data.count_column).alias("agg_count")).filter(f"agg_count < {self.linkable_k}")
            elif data.id_column is not None:
                linkable_df = data.source.groupBy(col).agg(F.countDistinct(data.id_column).alias("agg_count")).filter(f"agg_count < {self.linkable_k}")
            else:
                linkable_df = data.source.groupBy(col).agg(F.count('*').alias("agg_count")).filter(f"agg_count < {self.linkable_k}")

            linkable_counts_dict[col] = linkable_df.select(F.sum("agg_count")).collect()[0][0] or 0
        
        # most_linkable_columns = dict(sorted(linkable_counts_dict.items(), key=lambda x: x[1], reverse=True)[:self.top_n])
        most_linkable_columns = dict([sorted(linkable_counts_dict.items(), key=lambda x: x[1], reverse=True)[0]])
        response = self.to_dict()
        response["value"] = most_linkable_columns
        return response

class RedactedRowCount(MultiColumnMetric):
    """
    Calculate the count of rows with redacted or partially redacted values.

    This metric calculates the count of rows where values in the specified categorical columns are redacted
    or partially redacted. It helps in identifying the extent of redacted data in the dataset.

    :param column_names: A list containing the column names to evaluate for redacted values.
    :type column_names: list
    :param keyword: The keyword used to represent redacted or unknown values in the data.
                             Default is "Unknown."
    :type keyword: str, optional

    :raises ValueError: If no columns are provided.

    Example usage:

    .. code-block:: python

        # Create an instance of the RedactedRowCount metric for columns 'A' and 'B' with a redacted keyword of "Unknown"
        redacted_count_metric = Metric.create("RedactedRowCount", column_names=["A", "B"], keyword="Unknown")

        # Compute the count of rows with redacted values in the dataset
        result = redacted_count_metric.compute(dataset)

        print(result)  # {'name': 'RedactedRowCount', 'value': {'partly redacted': 20, 'fully redacted': 10}}

    The `compute` method returns a dictionary containing the computed counts of rows with redacted or partially
    redacted values in the specified categorical columns.
    """
    def __init__(self, column_names, keyword="Unknown"):
        if len(column_names) == 0:
            raise ValueError("RedactedRowCount requires at least one column.")
        super().__init__(column_names)
        self.keyword = keyword
    def param_names(self):
        return super().param_names() + ["keyword"]
    def compute(self, data):
        """
        Compute the count of rows with redacted or partially redacted values.

        :param data: The dataset to compute the count of redacted rows for.
        :type data: Dataset

        :return: A dictionary containing the computed counts of rows with redacted or partially redacted values.
        :rtype: dict

        :raises ValueError: If no columns are provided or if the columns are not categorical.
        """
        if not set(self.column_names).issubset(set(data.categorical_columns)):
            raise ValueError("Columns {} are not categorical.".format(self.column_names))
        
        # Create an additional column that counts the number of "unknown" values per row
        df_with_unknown_count = data.source.withColumn("unknown_count", sum(F.when(F.col(c) == self.keyword, 1).otherwise(0) for c in self.column_names))
        
        # Count the number of rows with partly unknown values (some, but not all columns are "unknown")
        partly_redacted = df_with_unknown_count.filter((F.col("unknown_count") > 0) & (F.col("unknown_count") < len(self.column_names))).count()
        # Count the number of rows with fully unknown values (all columns are "unknown")
        fully_redacted = df_with_unknown_count.filter(F.col("unknown_count") == len(self.column_names)).count()

        response = self.to_dict()
        response["value"] = {"partly redacted": partly_redacted, "fully redacted": fully_redacted}
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