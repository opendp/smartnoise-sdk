[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue)](https://www.python.org/)

<a href="https://smartnoise.org"><img src="https://github.com/opendp/smartnoise-sdk/raw/main/images/SmartNoise/SVG/Logo%20Mark_grey.svg" align="left" height="65" vspace="8" hspace="18"></a>

## SmartNoise SQL

Differentially private SQL queries.  Tested with:
* PostgreSQL
* SQL Server
* Spark
* Pandas (SQLite)
* PrestoDB
* BigQuery

SmartNoise is intended for scenarios where the analyst is trusted by the data owner.  SmartNoise uses the [OpenDP](https://github.com/opendp/opendp) library of differential privacy algorithms.

## Installation

```
pip install smartnoise-sql
```

## Querying a Pandas DataFrame

Use the `from_df` method to create a private reader that can issue queries against a pandas dataframe.

```python
import snsql
from snsql import Privacy
import pandas as pd
privacy = Privacy(epsilon=1.0, delta=0.01)

csv_path = 'PUMS.csv'
meta_path = 'PUMS.yaml'

pums = pd.read_csv(csv_path)
reader = snsql.from_df(pums, privacy=privacy, metadata=meta_path)

result = reader.execute('SELECT sex, AVG(age) AS age FROM PUMS.PUMS GROUP BY sex')
```

## Querying a SQL Database

Use `from_connection` to wrap an existing database connection.

```python
import snsql
from snsql import Privacy
import psycopg2

privacy = Privacy(epsilon=1.0, delta=0.01)
meta_path = 'PUMS.yaml'

pumsdb = psycopg2.connect(user='postgres', host='localhost', database='PUMS')
reader = snsql.from_connection(pumsdb, privacy=privacy, metadata=meta_path)

result = reader.execute('SELECT sex, AVG(age) AS age FROM PUMS.PUMS GROUP BY sex')
```

## Querying a Spark DataFrame

Use `from_connection` to wrap a spark session.

```python
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
```

## Privacy Cost

The privacy parameters epsilon and delta are passed in to the private connection at instantiation time, and apply to each computed column during the life of the session.  Privacy cost accrues indefinitely as new queries are executed, with the total accumulated privacy cost being available via the `spent` property of the connection's `odometer`:

```python
privacy = Privacy(epsilon=0.1, delta=10e-7)

reader = from_connection(conn, metadata=metadata, privacy=privacy)
print(reader.odometer.spent)  # (0.0, 0.0)

result = reader.execute('SELECT COUNT(*) FROM PUMS.PUMS')
print(reader.odometer.spent)  # approximately (0.1, 10e-7)
```

The privacy cost increases with the number of columns:

```python
reader = from_connection(conn, metadata=metadata, privacy=privacy)
print(reader.odometer.spent)  # (0.0, 0.0)

result = reader.execute('SELECT AVG(age), AVG(income) FROM PUMS.PUMS')
print(reader.odometer.spent)  # approximately (0.4, 10e-6)
```

The odometer is advanced immediately before the differentially private query result is returned to the caller.  If the caller wishes to estimate the privacy cost of a query without running it, `get_privacy_cost` can be used:

```python
reader = from_connection(conn, metadata=metadata, privacy=privacy)
print(reader.odometer.spent)  # (0.0, 0.0)

cost = reader.get_privacy_cost('SELECT AVG(age), AVG(income) FROM PUMS.PUMS')
print(cost)  # approximately (0.4, 10e-6)

print(reader.odometer.spent)  # (0.0, 0.0)
```

Note that the total privacy cost of a session accrues at a slower rate than the sum of the individual query costs obtained by `get_privacy_cost`.  The odometer accrues all invocations of mechanisms for the life of a session, and uses them to compute total spend.

```python
reader = from_connection(conn, metadata=metadata, privacy=privacy)
query = 'SELECT COUNT(*) FROM PUMS.PUMS'
epsilon_single, _ = reader.get_privacy_cost(query)
print(epsilon_single)  # 0.1

# no queries executed yet
print(reader.odometer.spent)  # (0.0, 0.0)

for _ in range(100):
    reader.execute(query)

epsilon_many, _ = reader.odometer.spent
print(f'{epsilon_many} < {epsilon_single * 100}')
```

## Histograms

SQL `group by` queries represent histograms binned by grouping key.  Queries over a grouping key with unbounded or non-public dimensions expose privacy risk. For example:

```sql
SELECT last_name, COUNT(*) FROM Sales GROUP BY last_name
```

In the above query, if someone with a distinctive last name is included in the database, that person's record might accidentally be revealed, even if the noisy count returns 0 or negative.  To prevent this from happening, the system will automatically censor dimensions which would violate differential privacy.

## Private Synopsis

A private synopsis is a pre-computed set of differentially private aggregates that can be filtered and aggregated in various ways to produce new reports.  Because the private synopsis is differentially private, reports generated from the synopsis do not need to have additional privacy applied, and the synopsis can be distributed without risk of additional privacy loss.  Reports over the synopsis can be generated with non-private SQL, within an Excel Pivot Table, or through other common reporting tools.

You can see a sample [notebook for creating private synopsis](samples/Synopsis.ipynb) suitable for consumption in Excel or SQL.

## Limitations

You can think of the data access layer as simple middleware that allows composition of `opendp` computations using the SQL language.  The SQL language provides a limited subset of what can be expressed through the full `opendp` library.  For example, the SQL language does not provide a way to set per-field privacy budget.

Because we delegate the computation of exact aggregates to the underlying database engines, execution through the SQL layer can be considerably faster, particularly with database engines optimized for precomputed aggregates.  However, this design choice means that analysis graphs composed with SQL language do not access data in the engine on a per-row basis.  Therefore, SQL queries do not currently support algorithms that require per-row access, such as quantile algorithms that use underlying values.  This is a limitation that future releases will relax for database engines that support row-based access, such as Spark.

The SQL processing layer has limited support for bounding contributions when individuals can appear more than once in the data.  This includes ability to perform reservoir sampling to bound contributions of an individual, and to scale the sensitivity parameter.  These parameters are important when querying reporting tables that might be produced from subqueries and joins, but require caution to use safely.

For this release, we recommend using the SQL functionality while bounding user contribution to 1 row.  The platform defaults to this option by setting `max_contrib` to 1, and should only be overridden if you know what you are doing.  Future releases will focus on making these options easier for non-experts to use safely.


## Communication

- You are encouraged to join us on [GitHub Discussions](https://github.com/opendp/opendp/discussions/categories/smartnoise)
- Please use [GitHub Issues](https://github.com/opendp/smartnoise-sdk/issues) for bug reports and feature requests.
- For other requests, including security issues, please contact us at [smartnoise@opendp.org](mailto:smartnoise@opendp.org).

## Releases and Contributing

Please let us know if you encounter a bug by [creating an issue](https://github.com/opendp/smartnoise-sdk/issues).

We appreciate all contributions. Please review the [contributors guide](../contributing.rst). We welcome pull requests with bug-fixes without prior discussion.

If you plan to contribute new features, utility functions or extensions, please first open an issue and discuss the feature with us.
