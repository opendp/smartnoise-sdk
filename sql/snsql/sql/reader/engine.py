class Engine:
    PANDAS = "Pandas"
    POSTGRES = "Postgres"
    PRESTO = "Presto"
    SPARK = "Spark"
    SQL_SERVER = "SqlServer"

    known_engines = {PANDAS, POSTGRES, PRESTO, SPARK, SQL_SERVER}
    class_map = {PANDAS: "pandas", POSTGRES: "postgres", PRESTO: "presto", SPARK: "spark", SQL_SERVER: "sql_server"}