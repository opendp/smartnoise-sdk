class Engine:
    PANDAS = "pandas"
    POSTGRES = "postgres"
    PRESTO = "Presto"
    SPARK = "Spark"
    SQL_SERVER = "SqlServer"

    known_engines = {PANDAS, POSTGRES, PRESTO, SPARK, SQL_SERVER}
