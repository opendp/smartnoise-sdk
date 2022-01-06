==============
SmartNoise SQL
==============

.. contents:: Table of Contents
  :local:
  :depth: 3

API Reference
=============
.. toctree::
  :glob:
  :maxdepth: 2

  API index <api/index>

Getting Started
===============

SmartNoise applies differential privacy by wrapping an existing database connection, intercepting queries, and ensuring results are private before returning results to the caller.

Querying a Pandas DataFrame
---------------------------

Use the ``from_df`` method to create a private reader that can issue queries against a pandas dataframe.

.. code-block:: python

  import snsql
  from snsql import Privacy
  import pandas as pd
  privacy = Privacy(epsilon=1.0, delta=0.01)

  csv_path = 'PUMS.csv'
  meta_path = 'PUMS.yaml'

  pums = pd.read_csv(csv_path)
  reader = snsql.from_df(pums, privacy=privacy, metadata=meta_path)

  result = reader.execute('SELECT sex, AVG(age) AS age FROM PUMS.PUMS GROUP BY sex')

Querying a SQL Database
-----------------------

Use ``from_connection`` to wrap an existing database connection.

.. code-block:: python

  import snsql
  from snsql import Privacy
  import psycopg2

  privacy = Privacy(epsilon=1.0, delta=0.01)
  meta_path = 'PUMS.yaml'

  pumsdb = psycopg2.connect(user='postgres', host='localhost', database='PUMS')
  reader = snsql.from_connection(pumsdb, privacy=privacy, metadata=meta_path)

  result = reader.execute('SELECT sex, AVG(age) AS age FROM PUMS.PUMS GROUP BY sex')

Querying a Spark DataFrame
--------------------------

Use ``from_connection`` to wrap a spark session.

.. code-block:: python

  import pyspark
  from pyspark.sql import SparkSession
  spark = SparkSession.builder.getOrCreate()
  from snsql import *

  pums = spark.read.load(...)  # load a Spark DataFrame
  pums.createOrReplaceTempView("PUMS_large")

  metadata = 'PUMS_large.yaml'

  private_reader = from_connection(
      spark, 
      metadata=metadata, 
      privacy=Privacy(epsilon=3.0, delta=1/1_000_000)
  )
  private_reader.reader.compare.search_path = ["PUMS"]

  res = private_reader.execute('SELECT COUNT(*) FROM PUMS_large')
  res.show()

When running a query against spark, the result of ``execute`` will be a spark DataFrame or RDD, which represents an execution plan.  The actual spark execution will not happen until the caller requests rows from the DataFrame, as in the ``res.show()`` above.

Metadata
========

The metadata is loaded from a file path, and describes important properties of the data source.

.. toctree::
  :glob:
  :maxdepth: 2

  Metadata <metadata.rst>

Advanced Usage
==============

.. toctree::
  :glob:
  :maxdepth: 2

  Advanced Usage <advanced.rst>


This is version |version| of the guides, last built on |today|.

.. |opendp-logo| image:: _static/images/opendp-logo.png
   :class: img-responsive
