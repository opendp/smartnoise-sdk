class Engine:
    BIGQUERY = "BigQuery"
    PANDAS = "Pandas"
    POSTGRES = "Postgres"
    PRESTO = "Presto"
    SPARK = "Spark"
    SQL_SERVER = "SqlServer"

    known_engines = {BIGQUERY, PANDAS, POSTGRES, PRESTO, SPARK, SQL_SERVER}
    class_map = {BIGQUERY: "bigquery", PANDAS: "pandas", POSTGRES: "postgres", PRESTO: "presto", SPARK: "spark", SQL_SERVER: "sql_server"}