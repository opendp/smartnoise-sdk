# Running Unit Tests

To run unit tests, use a terminal at the root of the project you are working on (sql, synth, or evaluator):

```bash
pip install -r tests/requirements.txt
pytest tests
```

# Writing Unit Tests

Many tests use CSV datasets stored under the `datasets` folder.  Some of these files are large, and will be downloaded on first run of pytest.

The tests use a shared set of database connections, `test_databases`, loaded in a fixture.

Get a `PrivateReader` over PUMS_1000 using the `pandas` engine:

```python
privacy = Privacy(epsilon=1.0, delta=0.01)
reader = test_databases.get_private_reader(engine='pandas', database='PUMS')
res = reader.execute('COUNT(age) AS age_count FROM PUMS.PUMS')
assert(res['age_count'][0] > 500)
```

The fixture will cache connections for multiple database engines and databases, if they're available on the test machine.

If you want to access the cached connections outside of pytest, you can instantiate the `test_databases` manually:

```python
from tests.setup.dataloader.db import DbCollection
test_databases = DbCollection()
print(test_databases)
```

From the command line, you can get the same status by issuing `python tests/check_databases.py` from a command prompt in the SQL project.  Output will look like this.  If there are problems connecting to any of the configured test databases, these will print an error and display without the "connected" status:

```
None@pandas://None:None
	PUMS -> PUMS (connected)
	PUMS_pid -> PUMS_pid (connected)
	PUMS_dup -> PUMS_dup (connected)
postgres@postgres://localhost:5432
	PUMS -> pums (connected)
	PUMS_large -> pums (connected)
	PUMS_pid -> pums_pid (connected)
	PUMS_dup -> pums_pid (connected)
sa@sqlserver://tcp:10.0.0.199:1433
	PUMS -> pums (connected)
	PUMS_large -> pums (connected)
	PUMS_pid -> pums_pid (connected)
	PUMS_dup -> pums_pid (connected)
```

The current pre-cached datasets are:
* `PUMS`: The 1000-row sample of PUMS, using `row_privacy`.  When querying, use `PUMS.PUMS` as the table name.
* `PUMS_pid`: The 1000-row sample of PUMS, with an additional column, `pid` for a person primary key.  Does not use `row_privacy`.  When querying, use `PUMS.PUMS` as the table name.
* `PUMS_dup`: Oversample of PUMS_pid with ~1750 rows, each person appears 1-3 times.  Use this table for tests where reservoir needs to work.  When querying, use `PUMS.PUMS` as the table name.
* `PUMS_null`: PUMS_pid with ~40% of rows having one NULL value.  Use this table for tests where NULL behavior is important.  When querying, use `PUMS.PUMS` as the table name.
* `PUMS_large`: A sample of PUMS with 1.2 million records, and a person primary key.  When querying, use `PUMS.PUMS_large` as the table name.

The current supported engines are:
* `pandas` - does not include `PUMS_large`, due to memory constraints
* `postgres` 
* `sqlserver`
* `spark` - can use all datasets, but requires `export TEST_SPARK=1`
* `bigquery`

The service-hosted SQL engines, `pandas` and `sqlserver`, will be connected, if the connections are configured locally.  See this [for more information](tests/engines/README.md).

The GitHub Actions CI runners for SQL Server, Postgres, Spark and BigQuery will automatically install and run these engines.

Note that the `connections-unit.yml` can be configured to use different table names in the database, and the unit test fixture will automatically update test queries and metadata to use the appropriate table name.  In the default local and CI builds for Postgres, we map `PUMS_dup` to `pums.pums.pums_dup` to test three part names, and we install `PUMS_null` in postgres `public` schema and point to `pums` (with no schema) to test public schema search path resolution.  You can check the `connections-unit.yaml` under `tests/setup/postgres` to see the syntax.

## Test Against Multiple Engines

Unit tests which check query outputs should run on all available engines.  In normal development, this will be `pandas` only, but in CI will include `spark`, `postgres`, `sqlserver` and `bigquery`.  Developers can have all of these installed locally (except `bigquery` which will require GCP credentials), and `pandas` can be disabled with `export SKIP_PANDAS=1`.  If you have a unit test that requests pandas only, make sure the test checks for `None` from the reader, since `DbCollection` will not return pandas readers when `SKIP_PANDAS` is set.

Runs a test against all available engines that have PUMS_pid:

```python
readers = test_databases.get_private_readers(database='PUMS_pid', privacy=privacy)
print(f'We have {len(readers)} engines available to test against.')
for reader in readers:
    print(f'Querying engine {reader.engine}')
    res = reader.execute('SELECT COUNT(*) AS n FROM PUMS.PUMS')
    print(res)
    if reader.engine == "spark":
        res = test_databases.to_tuples(res)
    assert(n > 500)
```

Note that we select from `PUMS.PUMS` in the `PUMS_pid` database.  This is because the schemas are identical, apart from a hidden primary key.  All queries that work against one should work against the other.

In the example above, note the different code for `spark` engine.  This is required, because spark by default returns a `DataFrame` or `RDD`.  It's a great idea to tune your tests to work with spark, but there may be cases where you are sure your unit tests should never run in spark.  In these cases, you can simply skip the reader when cycling through:

```python
readers = test_databases.get_private_readers(database='PUMS_pid', privacy=privacy)
for reader in readers if reader.engine != 'spark':
    res = reader.execute('SELECT COUNT(*) AS n FROM PUMS.PUMS')
    n = res[0]
    assert(n > 100)
```

## Overrides

Some unit tests require overriding behavior specified in the metadata, such as `clamp_counts` or `censor_dims`.  These can be overridden when accessing the private readers, by passing in the values to the `overrides` argument:

```python
overrides = {'censor_dims': False, 'clamp_counts': False}
readers = test_databases.get_private_readers(database='PUMS_pid', privacy=privacy, overrides=overrides)
```