from .sql.private_reader import PrivateReader

def from_df(df, *ignore, privacy, metadata, **kwargs):
    """Open a private SQL connection against a Pandas DataFrame.

    .. code-block:: python

        from snsql import from_df, Privacy

        csv = 'datasets/PUMS.csv'
        pums = pd.read_csv(csv)
        metadata = 'datasets/PUMS.yaml'

        privacy = Privacy(epsilon=0.1, delta=1/10000)
        reader = from_df(pums, metadata=metadata, privacy=privacy)

        result = reader.execute('SELECT educ, COUNT(*) AS n FROM PUMS.PUMS GROUP BY educ')

    :param df: The Pandas DataFrame to be queried
    :param privacy: A Privacy object with the desired privacy parameters
    :param metadata: The metadata describing the data source.  This will typically be
        a path to a yaml metadata file, but can also be a dictionary.
    :returns: A PrivateReader that can be used to execute differentially private
        queries against the pandas dataframe.

    """
    return PrivateReader.from_connection(df, privacy=privacy, metadata=metadata, engine='pandas')

def from_connection(conn, *ignore, privacy, metadata, engine=None, **kwargs):
    """Open a private SQL connection against an established database connection.

    .. code-block:: python

        from snsql import from_connection, Privacy

        conn = pyodbc.connect(dsn)
        metadata = 'datasets/PUMS.yaml'
        privacy = Privacy(epsilon=0.1, delta=1/10000)
        reader = from_connection(conn, metadata=metadata, privacy=privacy)

        result = reader.execute('SELECT educ, COUNT(*) AS n FROM PUMS.PUMS GROUP BY educ')

    :param conn: An established database connection.  Can be pyodbc, psycopg2, spark, pandas, or presto.
    :param privacy: A Privacy object with the desired privacy parameters
    :param metadata: The metadata describing the data source.  This will typically be
        a path to a yaml metadata file, but can also be a dictionary.
    :param engine: Specifies the engine to use.  Can be 'sqlserver', 'postgres',
        'spark', 'pandas', or 'presto'.  If not supplied, from_connection will probe
        the supplied connection to decide what dialect to use.
    :returns: A PrivateReader that can be used to execute differentially private
        queries against the database.

    """
    return PrivateReader.from_connection(conn, privacy=privacy, metadata=metadata, engine=engine)