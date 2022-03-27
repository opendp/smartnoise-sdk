class Probe:
    @classmethod
    def engine(cls, conn) -> str:
        """
        Probes a connection and tries to determine which engine it uses
        """
        conn_mod = conn.__class__.__module__
        conn_class = conn.__class__.__name__
        if (
            conn_mod == 'pandas.core.frame' and
            conn_class == 'DataFrame'
        ):
            return "pandas"
        if (
            conn_mod == 'pyspark.sql.session' and
            conn_class == 'SparkSession'
        ):
            return "spark"
        if (
            conn_mod == 'psycopg2.extensions' and
            conn_class == 'connection'
        ):
            return "postgres"
        if (
            conn_mod == 'google.cloud.bigquery.client' and
            conn_class == 'Client'
        ):
            return "bigquery"
        if (
            conn_mod == 'pyodbc' and
            conn_class == 'Connection'
        ):
            # probe for SQL Server or other database
            if cls.probe_sqlserver(conn):
                return "sqlserver"
            if cls.probe_postgres(conn):
                return "postgres"
            if cls.probe_mysql(conn):
                return "mysql"
            if cls.probe_sqlite(conn):
                return "sqlite"
    @classmethod
    def probe_sqlserver(cls, conn) -> bool:
        try:
            with conn.cursor() as cursor:
                res = cursor.execute("SELECT @@version")
                if cursor.description is None:
                    return False
                else:
                    res = [row for row in cursor]
                    res = str(res[0])
                    if "Microsoft SQL Server" in res:
                        return True
        except:
            pass
        return False
    @classmethod
    def probe_postgres(cls, conn) -> bool:
        try:
            with conn.cursor() as cursor:
                cursor.execute("select version();")
                if cursor.description is None:
                    return False
                else:
                    res = [row for row in cursor]
                    res = str(res[0])
                    if "PostgreSQL" in res:
                        return True
        except:
            pass
        return False
    @classmethod
    def probe_mysql(cls, conn) -> bool:
        try:
            with conn.cursor() as cursor:
                cursor.execute("select @@version_comment")
                if cursor.description is None:
                    return False
                else:
                    res = [row for row in cursor]
                    res = str(res[0])
                    if len(res) >= 1:
                        # this will error and pass to exception if result is not like "8.2.1"
                        vers = [int(v) for v in res.split('.')]
                        if len(vers) >= 1:
                            return True
        except:
            pass
        return False
