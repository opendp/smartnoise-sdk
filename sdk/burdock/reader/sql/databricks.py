import datetime
from pyspark.sql import Row
from pyspark.sql.types import BooleanType
from burdock.sql import CollectionMetadata
from burdock.sql import PrivateReader
from .spark import SparkReader
from burdock.mechanisms.gaussian import Gaussian

class DatabricksSparkReader:
    def __init__(self, sparkSession, meta_path):
        self.spark = sparkSession
        self.schema = CollectionMetadata.from_file(meta_path)
        dummyReader = SparkReader("", sparkSession, "")
        self.private_reader = PrivateReader(dummyReader, self.schema, 1.0)

    """
        Executes a raw SQL string against the Spark Hive database and returns spark dataframe.
    """
    def execute(self, query):
        if not isinstance(query, str):
            raise ValueError("Please pass strings to execute.  To execute ASTs, use execute_typed.")        
        
        max_contrib = self.private_reader.max_contrib
        epsilon = self.private_reader.epsilon
        interval_widths = self.private_reader.interval_widths

        subquery, fullquery = self.private_reader.rewrite(query)
        syms = subquery.all_symbols()
        numeric_cols = subquery.numeric_symbols()

        if numeric_cols is None:
          pass

        subquery_df = self.spark.sql(str(subquery))
        noisedDF = subquery_df.rdd.mapPartitions(applyNoises(numeric_cols, epsilon, max_contrib, interval_widths)).toDF()
        
        # censor dimensions for privacy
        if subquery.agg is not None:
            noisedDF = noisedDF.filter(noisedDF.keycount > max_contrib ** 2)

        noisedDF.createOrReplaceTempView("query_df_noised")
        noisedDF.cache()          

        q_result = self.spark.sql(str(fullquery))
        return q_result

"""
    Mappartition function as a trasnformer only accepts a function with iterator as the single parameter
    This is the way to make it accepting more parameters
"""
def applyNoises(numeric_cols, epsilon, max_contrib, interval_widths):    

    def applyNoise(data, numeric_symbol):
        final_output = []
        name, sym = numeric_symbol
        sens = sym.sensitivity() 

        # treat null as 0 and then add noise
        mechanism = Gaussian(epsilon, 10E-16, sens, max_contrib, interval_widths)
        noisedData = mechanism.release([v if v is not None else 0.0 for v in data], compute_accuracy=True).values

        # convert datatype 
        t = sym.type()
        if t == "string":
            final_output = [str(v) if v is not None else v for v in noisedData]
        elif t == "int":
            final_output = [int(v) if v is not None else v for v in noisedData]
        elif t == "float":
            final_output = [float(v) if v is not None else v for v in noisedData]
        elif t == "boolean":
            final_output = [bool(v) if v is not None else v for v in noisedData]
        elif t == "datetime":
            final_output = [datetime.datetime(v) if v is not None else None for v in noisedData]
        else:
            raise ValueError("Trying to load unknown type " + t)
        
        # clamp counts to 0
        if sens == 1:
            final_output = [v if v >=0 else 0 for v in final_output]

        return final_output

    def partition_func(list_of_items):
        noised_results = {}

        allRows = []
        for items in list_of_items:
            allRows.append(items) 

        for col in numeric_cols:            
            noised_results[col[0]] = applyNoise(list(map(lambda v : v[col[0]], allRows)), col)

        for j in range(0, len(allRows)):
            tmp_row = allRows[j].asDict()
            for col in numeric_cols:
                tmp_row[col[0]] = noised_results[col[0]][j]
            yield Row(**tmp_row)

    return partition_func