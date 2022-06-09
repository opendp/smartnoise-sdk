from .base import DbFactory, DbDataset

class SparkFactory(DbFactory):
    def __init__(self, engine="spark", user=None, host=None, port=None, 
        datasets={'PUMS': 'PUMS', 'PUMS_pid': 'PUMS_pid', 'PUMS_dup': 'PUMS_dup', 'PUMS_large': 'PUMS_large', 'PUMS_null' : 'PUMS_null'}):
        super().__init__(engine, user, host, port, datasets)
    def connect(self, dataset):
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()
            self.session = spark

            csv_paths = {
                'PUMS': self.pums_csv_path,
                'PUMS_pid': self.pums_pid_csv_path,
                'PUMS_dup': self.pums_dup_csv_path,
                'PUMS_null': self.pums_null_csv_path,
                'PUMS_large': self.pums_large_csv_path
            }

            if dataset in csv_paths:
                recordset = spark.read.load(csv_paths[dataset], format="csv", sep=",",inferSchema="true", header="true")
                if dataset == 'PUMS_large':
                    colnames = list(recordset.columns)
                    colnames[0] = "PersonID"
                    recordset = recordset.toDF(*colnames)
                recordset.createOrReplaceTempView(dataset)
                self.connections[dataset] = DbDataset(recordset, dataset)
            else:
                raise ValueError(f"We don't know how to connect to dataset {dataset} in Spark")
        except:
            print("Unable to connect to Spark test databases.  Make sure pyspark is installed.")
