import pyspark
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, FloatType, IntegerType, LongType

class Dataset:
    """A Dataset wraps an existing Spark DataFrame and provides important metadata.

        :param source: The Spark DataFrame to wrap. The DataFrame must be either row-level, where each row
            represents separate observation, or aggregated, where each row represents a group of observations.
        :param categorical_columns: A list of categorical columns in the dataset.
        :param measure_columns: A list of measure columns in the dataset. Measure columns
            are numeric columns that are not aggregated.
        :param sum_columns: A list of sum columns in the dataset. Sum columns are summed measures in
            aggregated datasets.
        :param avg_columns: A list of average columns in the dataset. Average columns are averaged measures
            in aggregated datasets.
        :param count_column: The name of the column that contains the count of rows in the dataset.
            Count column is only needed in aggregated datasets, and is required if sum_columns or avg_columns
            are specified.
        :param id_column: The name of the column that contains the unique identifier representing an
            individual. If not specified, the dataset is assumed to be row-level privacy.
        :param idx: An optional index to allow multiple runs of a privacy algorithm to be compared

    """
    def __init__(self, 
                 source : pyspark.sql.dataframe.DataFrame, 
                 *ignore, 
                 categorical_columns=[],
                 measure_columns=[],
                 sum_columns=[],
                 avg_columns=[],
                 count_column=None,
                 id_column=None,
                 idx=None
                ):
        self.source = source
        
        # must be a dataframe
        if not isinstance(source, pyspark.sql.dataframe.DataFrame):
            raise TypeError("Source must be a Spark DataFrame.")

        # must be either aggregated or not, but no mix
        if len(sum_columns) > 0 or len(avg_columns) > 0:
            if count_column is None:
                raise ValueError("Must specify count column if sum or avg columns are specified.")
            if len(measure_columns) > 0:
                raise ValueError("Cannot specify both measure and sum/avg columns.")
        if count_column is not None and id_column is not None:
            raise ValueError("Dataset has a pre-aggregated count column, so it cannot have an id column.")

        all_columns = categorical_columns + measure_columns + sum_columns + avg_columns
        if count_column is not None:
            all_columns.append(count_column)
        if id_column is not None:
            all_columns.append(id_column)

        # all columns must be described, and no superfluous descriptions
        for column in all_columns:
            if column not in source.columns:
                raise ValueError("Column {} not found in dataset.".format(column))
            
        for column in source.columns:
            if column not in all_columns:
                raise ValueError("Column {} in source dataframe is not described.  Please drop the column if not needed, or specify how it should be treated.".format(column))
            
        # measures and aggregated measures must be numeric
        for column in measure_columns + sum_columns + avg_columns:
            if not isinstance(source.schema[column].dataType, (DoubleType, FloatType, IntegerType, LongType)):
                raise TypeError("Column {} is not numeric.".format(column))

        self.count_column = count_column
        self.categorical_columns = categorical_columns
        self.sum_columns = sum_columns
        self.measure_columns = measure_columns
        self.avg_columns = avg_columns
        self.id_column = id_column
        self.idx_ = idx

    @property
    def idx(self):
        """The index of the dataset.  This is an optional index that can be used to compare
            multiple runs of a privacy algorithm.  If not specified, the index will be the id 
            of the source dataframe.
        """
        if self.idx_ is None:
            return id(self.source)
        else:
            return self.idx_
    def __hash__(self):
        return hash(self.id)
    @property
    def is_aggregated(self):
        """True if the dataset is aggregated, False otherwise."""
        return self.count_column is not None

    @property
    def is_row_privacy(self):
        """True if the dataset is row-level privacy, False otherwise."""
        if self.is_aggregated:
            return False
        if self.id_column is None:
            return True
        else:
            return False
    
    def aggregate(self) -> 'Dataset':
        """Aggregate the dataset if it is not already aggregated.  If the dataset is already aggregated,
            this function returns the dataset unchanged. Returns a new Dataset object over the
            aggregated dataframe."""
        if self.is_aggregated:
            return self
        grouped = self.source.groupBy(self.categorical_columns)
        assert self.count_column is None
        count_column = "count"
        if count_column in self.source.columns:
            count_column = "count_agg"
            while count_column in self.source.columns:
                count_column += "_"
        aggregations = []
        if self.id_column is None:
            aggregations.append(F.count("*").alias(count_column))
        else:
            aggregations.append(F.countDistinct(self.id_column).alias(count_column))
        
        for column in self.measure_columns:
            aggregations.append(F.sum(column).alias(column))

        df = grouped.agg(*aggregations)

        return Dataset(
            df, 
            categorical_columns=self.categorical_columns, 
            sum_columns=self.measure_columns, 
            count_column=count_column
            )

    def matches(self, other):
        """Returns True if the other dataset has the same metadata as this one. Matching datasets
            will not typically point to the same dataframe. Two different dataframes must have matching
            metadata in order to compare them.
        """
        if not isinstance(other, Dataset):
            return False
        return self.count_column == other.count_column and \
                set(self.categorical_columns) == set(other.categorical_columns) and \
                set(self.sum_columns) == set(other.sum_columns) and \
                set(self.measure_columns) == set(other.measure_columns) and \
                set(self.avg_columns) == set(other.avg_columns) and \
                self.id_column == other.id_column
    
