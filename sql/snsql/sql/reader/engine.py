class Engine:
    BIGQUERY = "BigQuery"
    PANDAS = "Pandas"
    POSTGRES = "Postgres"
    PRESTO = "Presto"
    SPARK = "Spark"
    SQL_SERVER = "SqlServer"
    MYSQL = "MySql"
    SQLITE = "SQLite"

    known_engines = {BIGQUERY, PANDAS, POSTGRES, PRESTO, SPARK, SQL_SERVER, MYSQL, SQLITE}
    class_map = {BIGQUERY: "bigquery", PANDAS: "pandas", POSTGRES: "postgres", PRESTO: "presto", SPARK: "spark", SQL_SERVER: "sql_server", MYSQL: "mysql", SQLITE: "sqlite"}