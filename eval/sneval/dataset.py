import pyspark
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, FloatType, IntegerType, LongType

class Dataset:
    """
        Source must be a Spark dataframe.
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
            raise TypeError("Source must be a Spark dataframe.")

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
        if self.idx_ is None:
            return id(self.source)
        else:
            return self.idx_
    def __hash__(self):
        return hash(self.id)
    @property
    def is_aggregated(self):
        return self.count_column is not None

    @property
    def is_row_privacy(self):
        if self.is_aggregated:
            return False
        if self.id_column is None:
            return True
        else:
            return False
    
    def aggregate(self):
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
        if not isinstance(other, Dataset):
            return False
        return self.count_column == other.count_column and \
                set(self.categorical_columns) == set(other.categorical_columns) and \
                set(self.sum_columns) == set(other.sum_columns) and \
                set(self.measure_columns) == set(other.measure_columns) and \
                set(self.avg_columns) == set(other.avg_columns) and \
                self.id_column == other.id_column
    
