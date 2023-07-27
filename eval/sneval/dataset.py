import pyspark

class Dataset:
    """
        Source must be a Spark dataframe.
    """
    def __init__(self, source):
        if not isinstance(source, pyspark.sql.dataframe.DataFrame):
            raise TypeError("Source must be a Spark dataframe.")
        self.source = source
